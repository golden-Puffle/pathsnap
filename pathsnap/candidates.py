"""Candidate generation and feature engineering.

For every GPS point we take the K nearest road arcs (within RADIUS metres) as
candidates and describe each (point, candidate) pair with numeric features.
A classifier later learns P(candidate is the true arc | features).

Nothing here uses the reference route, so the same code runs at prediction time.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import LocalProjector, angle_diff_deg, bearing_deg, point_segment_distance
from .io import Trip

FEATURES = [
    "dist_m",            # perpendicular distance point -> arc
    "dist_minus_min",    # how much further than the closest candidate
    "dist_ratio",        # dist / closest-candidate dist
    "rank",              # 0 = closest candidate
    "n_cands",           # candidates found for this point (road density)
    "t_along",           # position of the closest point along the arc (0..1)
    "arc_len_m",         # arc length
    "heading_diff",      # angle between GPS heading and arc direction (0..180)
    "speed_mps",         # GPS speed around this point
    "dt_prev",           # seconds since the previous GPS point
    "from_degree",       # number of arcs touching the arc's start node
    "to_degree",         # number of arcs touching the arc's end node
    "conn_prev",         # candidate equals / continues the previous point's closest arc
    "conn_next",         # candidate equals / leads into the next point's closest arc
]


@dataclass
class TripGeometry:
    proj: LocalProjector
    track_xy: np.ndarray
    arc_a: np.ndarray
    arc_b: np.ndarray
    arc_len: np.ndarray


def trip_geometry(trip: Trip) -> TripGeometry:
    proj = LocalProjector.for_points(trip.track[:, :2])
    node_xy = proj.to_xy(trip.nodes)
    arc_a = node_xy[trip.arcs[:, 0]]
    arc_b = node_xy[trip.arcs[:, 1]]
    return TripGeometry(proj, proj.to_xy(trip.track[:, :2]), arc_a, arc_b,
                        np.linalg.norm(arc_b - arc_a, axis=1))


def _arc_sample_index(g: TripGeometry, step_m: float):
    """KD-tree over points sampled every `step_m` along each arc, so that long
    arcs are found even when the GPS point is far from their midpoint."""
    n = np.maximum(1, np.ceil(g.arc_len / step_m).astype(int)) + 1
    arc_ids = np.repeat(np.arange(len(n)), n)
    t = np.concatenate([np.linspace(0, 1, k) for k in n])
    pts = g.arc_a[arc_ids] + t[:, None] * (g.arc_b[arc_ids] - g.arc_a[arc_ids])
    return cKDTree(pts), arc_ids


def _gps_kinematics(track: np.ndarray, xy: np.ndarray):
    """Centred-difference heading (deg) and speed (m/s), plus dt to previous point."""
    T = len(xy)
    i_prev = np.maximum(np.arange(T) - 1, 0)
    i_next = np.minimum(np.arange(T) + 1, T - 1)
    dxy = xy[i_next] - xy[i_prev]
    dt = np.maximum(track[i_next, 2] - track[i_prev, 2], 1e-6)
    speed = np.linalg.norm(dxy, axis=1) / dt
    heading = bearing_deg(dxy[:, 0], dxy[:, 1])
    dt_prev = np.r_[0.0, np.diff(track[:, 2])]
    return heading, speed, dt_prev


def build_candidates(trip: Trip, k: int = 10, radius_m: float = 200.0,
                     step_m: float = 10.0, g: TripGeometry | None = None) -> pd.DataFrame:
    """One row per (GPS point, candidate arc) with the FEATURES columns."""
    g = g or trip_geometry(trip)
    tree, sample_arc = _arc_sample_index(g, step_m)
    T = len(g.track_xy)

    # 1. coarse search: arcs with a sample point within radius (+ half a step)
    hits = tree.query_ball_point(g.track_xy, r=radius_m + step_m / 2)
    obs = np.repeat(np.arange(T), [len(h) for h in hits])
    arcs = sample_arc[np.concatenate([np.asarray(h, dtype=int) for h in hits])] if len(obs) else np.empty(0, int)
    pairs = np.unique(np.stack([obs, arcs], axis=1), axis=0) if len(obs) else np.empty((0, 2), int)
    obs, arcs = pairs[:, 0], pairs[:, 1]

    # 2. exact distance, keep pairs within radius, then the k closest per point
    dist, t_along = point_segment_distance(g.track_xy[obs], g.arc_a[arcs], g.arc_b[arcs])
    keep = dist <= radius_m
    obs, arcs, dist, t_along = obs[keep], arcs[keep], dist[keep], t_along[keep]
    order = np.lexsort((dist, obs))
    obs, arcs, dist, t_along = obs[order], arcs[order], dist[order], t_along[order]
    first = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    group_start = np.repeat(first, np.diff(np.r_[first, len(obs)]))
    rank = np.arange(len(obs)) - group_start
    keep = rank < k
    obs, arcs, dist, t_along, rank = obs[keep], arcs[keep], dist[keep], t_along[keep], rank[keep]

    df = pd.DataFrame({"trip_id": trip.trip_id, "obs_idx": obs, "arc_id": arcs,
                       "dist_m": dist, "rank": rank, "t_along": t_along})

    # 3. features
    min_d = df.groupby("obs_idx")["dist_m"].transform("min")
    df["dist_minus_min"] = df["dist_m"] - min_d
    df["dist_ratio"] = df["dist_m"] / (min_d + 1.0)
    df["n_cands"] = df.groupby("obs_idx")["arc_id"].transform("size")
    df["arc_len_m"] = g.arc_len[arcs]

    heading, speed, dt_prev = _gps_kinematics(trip.track, g.track_xy)
    arc_vec = g.arc_b[arcs] - g.arc_a[arcs]
    arc_bear = bearing_deg(arc_vec[:, 0], arc_vec[:, 1])
    df["heading_diff"] = angle_diff_deg(heading[obs], arc_bear)
    df["speed_mps"] = speed[obs]
    df["dt_prev"] = dt_prev[obs]

    deg = np.bincount(trip.arcs.ravel(), minlength=len(trip.nodes))
    df["from_degree"] = deg[trip.arcs[arcs, 0]]
    df["to_degree"] = deg[trip.arcs[arcs, 1]]

    # connectivity with neighbouring points' closest arc (no labels used)
    closest = np.full(T, -1)
    top = df[df["rank"] == 0]
    closest[top["obs_idx"].to_numpy()] = top["arc_id"].to_numpy()
    prev_c = closest[np.maximum(obs - 1, 0)]
    next_c = closest[np.minimum(obs + 1, T - 1)]
    fr, to = trip.arcs[arcs, 0], trip.arcs[arcs, 1]
    df["conn_prev"] = ((arcs == prev_c) | ((prev_c >= 0) & (trip.arcs[prev_c, 1] == fr))).astype(np.int8)
    df["conn_next"] = ((arcs == next_c) | ((next_c >= 0) & (trip.arcs[next_c, 0] == to))).astype(np.int8)

    return df
