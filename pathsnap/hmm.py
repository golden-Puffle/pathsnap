"""HMM map matching with Viterbi decoding.

Hidden states   : for GPS point t, the candidate arcs of that point (<= k).
Emission        : log P(candidate is the true arc) from the trained classifier,
                  scaled by `emission_weight`.
Transition      : how plausible it is to move from candidate i at t-1 to
                  candidate j at t. With d_route = distance driven along the
                  road network between the two projected positions and
                  d_gps = straight-line distance between the two GPS points:
                      log P = -|d_route - d_gps| / beta
                  (Newson & Krumm 2009). Unreachable within `max_route_m` -> -inf.
Decoding        : Viterbi (max-product in log space).
Breaks          : if no transition from t-1 to t is feasible (a GPS gap or a
                  candidate set with no connected road), the chain is cut and
                  decoding restarts at t; the pieces are joined later by a
                  shortest path in `reconstruct_route`.
"""
import heapq
from dataclasses import dataclass

import numpy as np


@dataclass
class HMMParams:
    beta: float = 5.0             # metres; larger = more tolerant of detours
    emission_weight: float = 1.0
    max_route_m: float = 500.0    # Dijkstra search limit between consecutive points
    backward_tol_m: float = 15.0  # tolerated backward jitter along the same arc


class RoadGraph:
    """Directed graph over nodes with arc lengths, plus bounded Dijkstra with caching."""

    def __init__(self, arcs: np.ndarray, arc_len: np.ndarray, n_nodes: int):
        self.arcs = arcs
        self.arc_len = arc_len
        order = np.argsort(arcs[:, 0], kind="stable")
        self.out_arcs = order                                   # arc ids sorted by from-node
        self.out_ptr = np.searchsorted(arcs[order, 0], np.arange(n_nodes + 1))
        self._cache: dict[int, dict[int, float]] = {}

    def distances_from(self, src: int, limit: float) -> dict[int, float]:
        """Shortest driving distance from node `src` to every node within `limit` m."""
        hit = self._cache.get(src)
        if hit is not None:
            return hit
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for a in self.out_arcs[self.out_ptr[u]:self.out_ptr[u + 1]]:
                v = int(self.arcs[a, 1])
                nd = d + self.arc_len[a]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = dist
        return dist

    def shortest_arc_path(self, src: int, dst: int, limit: float = 5000.0) -> list[int] | None:
        """Arc ids of the shortest path from node src to node dst (None if unreachable)."""
        if src == dst:
            return []
        dist, parent = {src: 0.0}, {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist.get(u, np.inf):
                continue
            for a in self.out_arcs[self.out_ptr[u]:self.out_ptr[u + 1]]:
                v = int(self.arcs[a, 1])
                nd = d + self.arc_len[a]
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    parent[v] = int(a)
                    heapq.heappush(heap, (nd, v))
        if dst not in parent:
            return None
        path, v = [], dst
        while v != src:
            a = parent[v]
            path.append(a)
            v = int(self.arcs[a, 0])
        return path[::-1]


def _route_distance(g: RoadGraph, a: int, ta: float, b: int, tb: float, p: HMMParams) -> float:
    """Driving distance from position ta on arc a to position tb on arc b (inf if too far)."""
    la, lb = g.arc_len[a], g.arc_len[b]
    if a == b:
        delta = (tb - ta) * la
        if delta >= -p.backward_tol_m:
            return max(delta, 0.0)
    rest_a = (1.0 - ta) * la
    d_nodes = g.distances_from(int(g.arcs[a, 1]), p.max_route_m).get(int(g.arcs[b, 0]))
    if d_nodes is None:
        return np.inf
    return rest_a + d_nodes + tb * lb


def viterbi(states: list[dict], gps_step_m: np.ndarray, g: RoadGraph, p: HMMParams):
    """states[t] = {"arcs": (K,), "t_along": (K,), "logp": (K,)} for each GPS point.

    Returns (chosen index per point, break flags). break[t] = True means the
    chain was restarted at t because no transition from t-1 was feasible.
    """
    T = len(states)
    choice = np.zeros(T, dtype=int)
    breaks = np.zeros(T, dtype=bool)
    score = p.emission_weight * states[0]["logp"]
    back = [np.zeros(len(states[0]["arcs"]), dtype=int)]
    seg_start = 0

    def finish_segment(end, final_score):
        k = int(np.argmax(final_score))
        for t in range(end, seg_start - 1, -1):
            choice[t] = k
            k = back[t - seg_start][k] if t > seg_start else k

    for t in range(1, T):
        prev, cur = states[t - 1], states[t]
        trans = np.full((len(prev["arcs"]), len(cur["arcs"])), -np.inf)
        for i, (a, ta) in enumerate(zip(prev["arcs"], prev["t_along"])):
            for j, (b, tb) in enumerate(zip(cur["arcs"], cur["t_along"])):
                d = _route_distance(g, int(a), float(ta), int(b), float(tb), p)
                if np.isfinite(d):
                    trans[i, j] = -abs(d - gps_step_m[t]) / p.beta
        total = score[:, None] + trans
        best_prev = np.argmax(total, axis=0)
        best_val = total[best_prev, np.arange(len(cur["arcs"]))]
        if not np.isfinite(best_val).any():         # no feasible move: cut the chain here
            finish_segment(t - 1, score)
            breaks[t] = True
            seg_start = t
            back = [np.zeros(len(cur["arcs"]), dtype=int)]
            score = p.emission_weight * cur["logp"]
            continue
        back.append(best_prev)
        score = best_val + p.emission_weight * cur["logp"]

    finish_segment(T - 1, score)
    return choice, breaks


def reconstruct_route(point_arcs: np.ndarray, g: RoadGraph) -> tuple[list[int], int]:
    """Collapse per-point arcs into one driven route; fill gaps with shortest paths.
    Returns (route arc ids, number of gaps that could not be connected)."""
    route, unconnected = [], 0
    for a in point_arcs:
        a = int(a)
        if route and route[-1] == a:
            continue
        if route:
            prev = route[-1]
            if g.arcs[prev, 1] != g.arcs[a, 0]:
                path = g.shortest_arc_path(int(g.arcs[prev, 1]), int(g.arcs[a, 0]))
                if path is None:
                    unconnected += 1
                else:
                    route.extend(path)
        route.append(a)
    return route, unconnected
