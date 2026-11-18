"""Per-point labels: which reference-route arc each GPS point lies on.

`.route` lists the arcs driven, in order, but not which GPS point belongs to
which arc. We align them with a monotonic dynamic programme: GPS point t is
assigned route position j(t), with j never moving backwards in time, and the
total point-to-arc distance is minimised. This respects driving order, so a
road visited twice (or a parallel road) cannot steal points from the wrong
part of the trip.
"""
import numpy as np

from .geo import point_segment_distance


def align_track_to_route(track_xy: np.ndarray, route: np.ndarray,
                         arc_a: np.ndarray, arc_b: np.ndarray):
    """Return (label_arc, label_dist_m, route_pos) for every GPS point.

    track_xy : (T, 2) GPS points in metres
    route    : (R,)   reference arc ids in driving order
    arc_a/b  : (M, 2) start/end coordinates of every arc in metres
    """
    T, R = len(track_xy), len(route)
    # distance of every point to every route arc: (T, R)
    d, _ = point_segment_distance(track_xy[:, None, :], arc_a[route][None], arc_b[route][None])
    d = d.astype(np.float32)

    cost = d[0].copy()
    back = np.zeros((T, R), dtype=np.int32)
    idx = np.arange(R, dtype=np.int32)
    for t in range(1, T):
        # best previous position j' <= j  (prefix minimum + its argmin)
        best_val = np.minimum.accumulate(cost)
        is_new_min = np.r_[True, cost[1:] < best_val[:-1]]
        best_arg = np.maximum.accumulate(np.where(is_new_min, idx, 0))
        back[t] = best_arg
        cost = d[t] + best_val

    pos = np.empty(T, dtype=np.int32)
    pos[-1] = int(np.argmin(cost))
    for t in range(T - 1, 0, -1):
        pos[t - 1] = back[t, pos[t]]

    label_arc = route[pos]
    label_dist = d[np.arange(T), pos]
    return label_arc, label_dist, pos
