"""Exploratory data analysis helpers (dataset guide, section 4).

Reads raw trips only; nothing is fitted here, so it is safe on any partition.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geo import LocalProjector, haversine_m, point_segment_distance
from .io import Trip


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def gps_to_route_distances(xy_pts, xy_nodes, arcs, route, chunk=256):
    """Distance (m) from every GPS point to the nearest arc of the reference route."""
    a = xy_nodes[arcs[route, 0]][None]
    b = xy_nodes[arcs[route, 1]][None]
    out = np.empty(len(xy_pts))
    for s in range(0, len(xy_pts), chunk):
        d, _ = point_segment_distance(xy_pts[s:s + chunk, None, :], a, b)
        out[s:s + chunk] = d.min(axis=1)
    return out


def network_stats(nodes: np.ndarray, arcs: np.ndarray):
    """Size and connectivity of one road network. Returns (stats dict, out-degree array)."""
    n, m = len(nodes), len(arcs)
    u, v = arcs[:, 0], arcs[:, 1]
    valid = (u >= 0) & (u < n) & (v >= 0) & (v < n)
    n_bad = int((~valid).sum())
    u, v = u[valid], v[valid]
    outdeg = np.bincount(u, minlength=n)
    indeg = np.bincount(v, minlength=n)
    g = coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)).tocsr()
    n_weak, lab_w = connected_components(g, directed=False)
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    key = u.astype(np.int64) * n + v
    has_rev = np.isin(v.astype(np.int64) * n + u, key)
    xy = LocalProjector.for_points(nodes).to_xy(nodes)
    alen = np.linalg.norm(xy[v] - xy[u], axis=1)
    return {
        "n_nodes": n, "n_arcs": m, "n_arcs_bad_node_ref": n_bad,
        "n_self_loops": int((u == v).sum()),
        "n_duplicate_arcs": int(len(key) - len(np.unique(key))),
        "n_isolated_nodes": int(((outdeg + indeg) == 0).sum()),
        "n_nodes_no_outgoing": int((outdeg == 0).sum()),
        "max_out_degree": int(outdeg.max()), "mean_out_degree": float(outdeg.mean()),
        "n_weak_components": int(n_weak),
        "largest_weak_component_frac": float(np.bincount(lab_w).max() / n),
        "n_strong_components": int(n_strong),
        "largest_strong_component_frac": float(np.bincount(lab_s).max() / n),
        "frac_arcs_with_reverse": float(has_rev.mean()),
        "arc_len_median_m": float(np.median(alen)), "arc_len_p95_m": float(np.percentile(alen, 95)),
        "arc_len_max_m": float(alen.max()),
    }, outdeg


def trip_summary(trip: Trip, trip_dir: Path, partition: str):
    """One inventory row for a trip + arrays needed for plots. Returns (row, extras)."""
    tid = trip.trip_id
    trk, nodes, arcs = trip.track, trip.nodes, trip.arcs
    lon, lat, t = trk[:, 0], trk[:, 1], trk[:, 2]
    dt = np.diff(t)
    proj = LocalProjector.for_points(trk[:, :2])
    xy = proj.to_xy(trk[:, :2])
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    hv = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    far = hv > 5
    pos = dt > 0
    speed_kmh = step[pos] / dt[pos] * 3.6
    d_end_h = haversine_m(lon[0], lat[0], lon[-1], lat[-1])
    d_end_p = float(np.linalg.norm(xy[-1] - xy[0]))

    row = {"trip_id": tid, "partition": partition}
    for ext in ("track", "nodes", "arcs", "route"):
        p = Path(trip_dir) / f"{tid}.{ext}"
        row[f"has_{ext}"] = p.exists()
        row[f"size_{ext}_kb"] = round(p.stat().st_size / 1024, 1) if p.exists() else 0.0
    row.update({
        "n_obs": len(trk),
        "n_nonfinite": int((~np.isfinite(trk)).sum()),
        "n_duplicate_rows": int(len(trk) - len(np.unique(trk, axis=0))),
        "ts_strictly_increasing": bool((dt > 0).all()),
        "n_nonpositive_dt": int((~pos).sum()),
        "dt_median_s": float(np.median(dt)), "dt_max_s": float(dt.max()),
        "frac_dt_1s": float((dt == 1).mean()),
        "n_gaps_gt_1s": int((dt > 1).sum()), "n_gaps_gt_5s": int((dt > 5).sum()),
        "n_coords_out_of_range": int(((np.abs(lon) > 180) | (np.abs(lat) > 90)).sum()),
        "lon_min": lon.min(), "lon_max": lon.max(), "lat_min": lat.min(), "lat_max": lat.max(),
        "extent_ns_km": float(haversine_m(lon.mean(), lat.min(), lon.mean(), lat.max()) / 1000),
        "extent_ew_km": float(haversine_m(lon.min(), lat.mean(), lon.max(), lat.mean()) / 1000),
        "duration_min": float((t[-1] - t[0]) / 60),
        "track_km": float(step.sum() / 1000),
        "max_step_m": float(step.max()),
        "mean_speed_kmh": float(speed_kmh.mean()), "max_speed_kmh": float(speed_kmh.max()),
        "proj_step_err_max_pct": float((np.abs(step - hv)[far] / hv[far]).max() * 100) if far.any() else 0.0,
        "proj_endpoint_err_pct": float(abs(d_end_p - d_end_h) / d_end_h * 100) if d_end_h > 0 else 0.0,
        "frac_gps_in_network_bbox": float(((lon >= nodes[:, 0].min()) & (lon <= nodes[:, 0].max()) &
                                           (lat >= nodes[:, 1].min()) & (lat <= nodes[:, 1].max())).mean()),
        "n_nodes": len(nodes), "n_arcs": len(arcs),
        "nodes_md5": md5_of(Path(trip_dir) / f"{tid}.nodes"),
        "arcs_md5": md5_of(Path(trip_dir) / f"{tid}.arcs"),
    })
    extras = {"dt": dt, "route_dist": None}
    row["route_len"] = 0
    if trip.route is not None:
        r = trip.route
        u, v = arcs[:, 0], arcs[:, 1]
        ids_ok = (r >= 0).all() and (r < len(arcs)).all()
        row["route_len"] = len(r)
        row["route_unique_arcs"] = int(len(np.unique(r)))
        row["route_ids_valid"] = bool(ids_ok)
        if ids_ok:
            xyn = proj.to_xy(nodes)
            row["route_breaks"] = int((v[r[:-1]] != u[r[1:]]).sum())
            row["route_km"] = float(np.linalg.norm(xyn[v[r]] - xyn[u[r]], axis=1).sum() / 1000)
            rd = gps_to_route_distances(xy, xyn, arcs, r)
            extras["route_dist"] = rd
            row["gps_to_route_median_m"] = float(np.median(rd))
            row["gps_to_route_p95_m"] = float(np.percentile(rd, 95))
            row["gps_to_route_max_m"] = float(rd.max())
    return row, extras