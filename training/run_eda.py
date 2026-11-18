"""Run the full EDA on data/train and data/student_test.

    python training/run_eda.py --data data --out results/eda

Writes (all from the real data):
    results/eda/trip_inventory.csv      one row per trip (counts, gaps, extent, route checks ...)
    results/eda/partition_summary.csv   totals per partition
    results/eda/network_summary.csv     size + connectivity per unique road network
    results/eda/data_quality.md         data-quality record (numbers generated, wording to review)
    results/eda/figures/*.png           plots for the report
    results/maps/eda_*.html             interactive Folium maps
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from pathsnap.eda import network_stats, trip_summary
from pathsnap.io import list_trips, load_trip
from pathsnap.maps import eda_trip_map

PARTS = ["train", "student_test"]
COL = {"train": "#1f77b4", "student_test": "#ff7f0e"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/eda")
    ap.add_argument("--maps", default="results/maps")
    ap.add_argument("--map_trips", default="", help="comma separated trip ids to map (default: first of each partition)")
    a = ap.parse_args()
    out, figs = Path(a.out), Path(a.out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    rows, dts, rds, tracks = [], {p: [] for p in PARTS}, {p: [] for p in PARTS}, {p: [] for p in PARTS}
    nets, net_trips, first_net = {}, {}, None
    trip_dirs = {}
    for part in PARTS:
        for d in list_trips(Path(a.data) / part):
            trip = load_trip(d, with_route=True)
            row, ex = trip_summary(trip, d, part)
            rows.append(row)
            trip_dirs[(part, trip.trip_id)] = d
            dts[part].append(ex["dt"])
            if ex["route_dist"] is not None:
                rds[part].append(ex["route_dist"])
            tracks[part].append(trip.track[:, :2])
            key = (row["nodes_md5"], row["arcs_md5"])
            net_trips.setdefault(key, []).append(trip.trip_id)
            if key not in nets:
                nets[key], outdeg = network_stats(trip.nodes, trip.arcs)
                if first_net is None:
                    first_net = (trip.nodes, trip.arcs, outdeg, trip.trip_id)
            print(f"[{part}] {trip.trip_id}: {row['n_obs']} obs, {row['track_km']:.1f} km", flush=True)

    inv = pd.DataFrame(rows)
    inv.to_csv(out / "trip_inventory.csv", index=False)

    # ---- partition + network summaries
    ps = inv.groupby("partition").agg(
        n_trips=("trip_id", "count"), total_obs=("n_obs", "sum"), obs_per_trip_mean=("n_obs", "mean"),
        obs_per_trip_min=("n_obs", "min"), obs_per_trip_max=("n_obs", "max"),
        total_track_km=("track_km", "sum"), trips_with_route=("has_route", "sum")).reset_index()
    ps.to_csv(out / "partition_summary.csv", index=False)
    ns = pd.DataFrame([{**s, "nodes_md5": k[0], "arcs_md5": k[1], "n_trips_using": len(net_trips[k]),
                        "example_trip": net_trips[k][0]} for k, s in nets.items()])
    ns.to_csv(out / "network_summary.csv", index=False)

    # ---- figures
    # 1 partition overview
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(ps.partition, ps.n_trips, color=[COL[p] for p in ps.partition])
    for i, v in enumerate(ps.n_trips):
        ax[0].text(i, v, str(v), ha="center", va="bottom")
    ax[0].set_title("Number of trips per partition")
    for p in PARTS:
        ax[1].hist(inv[inv.partition == p].n_obs, bins=20, alpha=0.6, label=p, color=COL[p])
    ax[1].set_title("GPS observations per trip"); ax[1].set_xlabel("observations"); ax[1].legend()
    fig.tight_layout(); fig.savefig(figs / "01_partition_overview.png", dpi=150); plt.close(fig)

    # 2 sampling intervals
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 10, "6-10"), (11, 1e9, ">10")]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    for k, p in enumerate(PARTS):
        dt = np.concatenate(dts[p]) if dts[p] else np.array([])
        cnt = [((dt >= lo) & (dt <= hi)).sum() for lo, hi, _ in bins]
        ax.bar(np.arange(len(bins)) + (k - 0.5) * w, cnt, w, label=f"{p} ({len(dt)} steps)", color=COL[p])
    ax.set_xticks(range(len(bins))); ax.set_xticklabels([b[2] for b in bins]); ax.set_yscale("log")
    ax.set_xlabel("time between consecutive GPS observations (s)"); ax.set_ylabel("count (log)")
    ax.set_title("Sampling intervals"); ax.legend()
    fig.tight_layout(); fig.savefig(figs / "02_sampling_intervals.png", dpi=150); plt.close(fig)

    # 3 length / duration / speed
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for col, title, axx in [("track_km", "Trajectory length (km)", ax[0]), ("duration_min", "Duration (min)", ax[1]),
                            ("mean_speed_kmh", "Mean speed (km/h)", ax[2])]:
        for p in PARTS:
            axx.hist(inv[inv.partition == p][col], bins=20, alpha=0.6, label=p, color=COL[p])
        axx.set_title(title)
    ax[0].legend(); fig.tight_layout(); fig.savefig(figs / "03_length_duration_speed.png", dpi=150); plt.close(fig)

    # 4 trajectories overview over the road network of the first trip
    nodes, arcs, outdeg, first_id = first_net
    fig, ax = plt.subplots(figsize=(7, 9))
    segs = np.stack([nodes[arcs[:, 0]], nodes[arcs[:, 1]]], axis=1)
    ax.add_collection(LineCollection(segs, colors="#cccccc", linewidths=0.3))
    for p in PARTS:
        for i, t in enumerate(tracks[p]):
            ax.plot(t[:, 0], t[:, 1], color=COL[p], lw=0.8, alpha=0.8, label=p if i == 0 else None)
    allp = np.concatenate([t for p in PARTS for t in tracks[p]])
    ax.set_xlim(allp[:, 0].min() - 0.02, allp[:, 0].max() + 0.02)
    ax.set_ylim(allp[:, 1].min() - 0.02, allp[:, 1].max() + 0.02)
    ax.set_aspect(1 / np.cos(np.radians(allp[:, 1].mean())))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend()
    ax.set_title(f"GPS trajectories (all trips) over the road network of trip {first_id}")
    fig.tight_layout(); fig.savefig(figs / "04_trajectories_overview.png", dpi=150); plt.close(fig)

    # 5 GPS -> reference route distance
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in PARTS:
        if rds[p]:
            d = np.concatenate(rds[p])
            ax.hist(d, bins=60, alpha=0.6, density=True, color=COL[p],
                    label=f"{p}: median {np.median(d):.1f} m, p95 {np.percentile(d, 95):.1f} m")
    ax.set_xlabel("distance from GPS point to nearest reference-route arc (m)"); ax.set_ylabel("density")
    ax.set_title("GPS noise relative to the reference route"); ax.legend()
    fig.tight_layout(); fig.savefig(figs / "05_gps_to_route_distance.png", dpi=150); plt.close(fig)

    # 6 network size + connectivity
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].hist(inv.n_nodes, bins=20, color="#2ca02c"); ax[0].set_title("Road-network nodes per trip")
    ax[1].hist(inv.n_arcs, bins=20, color="#9467bd"); ax[1].set_title("Road-network arcs per trip")
    dc = np.bincount(outdeg)
    ax[2].bar(range(len(dc)), dc, color="#8c564b"); ax[2].set_yscale("log")
    ax[2].set_title(f"Out-degree distribution (network of trip {first_id})"); ax[2].set_xlabel("out-degree")
    fig.tight_layout(); fig.savefig(figs / "06_network_size_connectivity.png", dpi=150); plt.close(fig)

    # 7 data-quality overview (number of affected trips)
    rb = inv.get("route_breaks", pd.Series(0, index=inv.index)).fillna(0)
    nsum = inv.merge(ns[["nodes_md5", "arcs_md5", "n_weak_components", "n_isolated_nodes", "n_arcs_bad_node_ref"]],
                     on=["nodes_md5", "arcs_md5"], how="left")
    issues = {
        "non-finite GPS values": (inv.n_nonfinite > 0).sum(),
        "duplicate GPS rows": (inv.n_duplicate_rows > 0).sum(),
        "timestamps not strictly increasing": (~inv.ts_strictly_increasing).sum(),
        "gap > 5 s between observations": (inv.n_gaps_gt_5s > 0).sum(),
        "coordinates out of range": (inv.n_coords_out_of_range > 0).sum(),
        "GPS outside network bounding box": (inv.frac_gps_in_network_bbox < 1).sum(),
        "no reference route file": (~inv.has_route).sum(),
        "reference route has breaks": (rb > 0).sum(),
        "network has isolated nodes": (nsum.n_isolated_nodes > 0).sum(),
        "network not weakly connected": (nsum.n_weak_components > 1).sum(),
        "arcs referencing invalid nodes": (nsum.n_arcs_bad_node_ref > 0).sum(),
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(list(issues.keys()), list(issues.values()), color="#d62728")
    for i, v in enumerate(issues.values()):
        ax.text(v, i, f" {v}", va="center")
    ax.invert_yaxis(); ax.set_xlabel(f"number of trips affected (of {len(inv)})"); ax.set_title("Data-quality checks")
    fig.tight_layout(); fig.savefig(figs / "07_data_quality.png", dpi=150); plt.close(fig)

    # ---- Folium maps
    ids = [s for s in a.map_trips.split(",") if s]
    if not ids:
        ids = [next(i for (p, i) in trip_dirs if p == part) for part in PARTS if any(p == part for (p, _) in trip_dirs)]
    for (p, tid), d in trip_dirs.items():
        if tid in ids:
            n = eda_trip_map(load_trip(d, with_route=True), Path(a.maps) / f"eda_{p}_{tid}.html", partition=p)
            print(f"map written: eda_{p}_{tid}.html ({n} arcs drawn)")

    # ---- data-quality record
    rt = inv[inv.has_route]
    allrd = np.concatenate([d for p in PARTS for d in rds[p]]) if any(rds[p] for p in PARTS) else np.array([0.0])
    lines = ["# Data-quality record (numbers generated by training/run_eda.py)", "",
             f"Trips analysed: {len(inv)} ({', '.join(f'{r.partition}: {r.n_trips}' for r in ps.itertuples())}).",
             f"Unique road networks: {len(ns)} (identical `.nodes` + `.arcs` files across trips share one network).", "",
             "| Check | Trips affected |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in issues.items()]
    lines += ["", "## Key measurements",
              f"- Sampling: median interval {inv.dt_median_s.median():.1f} s; {100 * inv.frac_dt_1s.mean():.1f}% of steps are exactly 1 s; "
              f"largest gap anywhere {inv.dt_max_s.max():.0f} s.",
              f"- Trajectory length {inv.track_km.min():.1f}-{inv.track_km.max():.1f} km; observations per trip {inv.n_obs.min()}-{inv.n_obs.max()}.",
              f"- Speed: mean of trip means {inv.mean_speed_kmh.mean():.0f} km/h; largest single-step speed {inv.max_speed_kmh.max():.0f} km/h.",
              f"- Local equirectangular projection vs haversine: worst per-step error {inv.proj_step_err_max_pct.max():.3f}%, "
              f"worst start-to-end error {inv.proj_endpoint_err_pct.max():.3f}%.",
              f"- Reference route (trips with route: {len(rt)}): GPS-to-route distance median {np.median(allrd):.1f} m, "
              f"95th percentile {np.percentile(allrd, 95):.1f} m, max {allrd.max():.1f} m.",
              f"- Network size per trip: {inv.n_nodes.min()}-{inv.n_nodes.max()} nodes, {inv.n_arcs.min()}-{inv.n_arcs.max()} arcs.",
              "", "## Assumptions and decisions (review and edit before submitting)",
              "- Column order is lon, lat, t for `.track` and lon, lat for `.nodes`, as documented; extents above are plausible for one region.",
              "- Distances are computed in local metres (equirectangular projection centred on each trip); degrees are never used as distances.",
              "- The reference route is treated as reference information, not error-free ground truth: GPS points do not lie exactly on it.",
              "- A GPS row does not correspond to a road-network row; the link between them is built later by candidate generation.",
              "- Partitions are never mixed: each trip belongs to exactly one partition."]
    (out / "data_quality.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nPartition summary:\n", ps.to_string(index=False))
    print("\nData-quality issues (trips affected):")
    for k, v in issues.items():
        print(f"  {k}: {v}")
    print(f"\nAll outputs in {out}/ and {a.maps}/")


if __name__ == "__main__":
    main()