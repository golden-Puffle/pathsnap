"""Build the candidate feature table for train and student_test.

Usage:
    python -m training.build_dataset --data data --out artifacts/candidates.parquet

Output: one row per (trip, GPS point, candidate arc) with the feature columns,
`label` (1 = candidate is the reference arc for that point) and `split`.
Also writes results/candidate_stats.csv (per-trip candidate miss rate etc.).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.candidates import build_candidates, trip_geometry
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import align_track_to_route


def process_trip(trip_dir: Path, split: str, k: int, radius: float):
    trip = load_trip(trip_dir)
    g = trip_geometry(trip)
    cands = build_candidates(trip, k=k, radius_m=radius, g=g)

    label_arc, label_dist, _ = align_track_to_route(g.track_xy, trip.route, g.arc_a, g.arc_b)
    cands["label"] = (cands["arc_id"].to_numpy() == label_arc[cands["obs_idx"].to_numpy()]).astype(np.int8)
    cands["true_arc"] = label_arc[cands["obs_idx"].to_numpy()]
    cands["split"] = split

    T = trip.n_points
    hit = np.zeros(T, bool)
    hit[cands.loc[cands["label"] == 1, "obs_idx"].to_numpy()] = True
    has_cand = np.zeros(T, bool)
    has_cand[cands["obs_idx"].unique()] = True
    true_rank = cands.loc[cands["label"] == 1].set_index("obs_idx")["rank"]
    stats = {
        "trip_id": trip.trip_id, "split": split, "n_points": T,
        "n_arcs": len(trip.arcs), "route_len": len(trip.route),
        "no_candidates": int((~has_cand).sum()),
        "miss_rate": float(1 - hit.mean()),
        "top1_nearest_acc": float((true_rank == 0).sum() / T),
        "label_dist_median_m": float(np.median(label_dist)),
        "label_dist_p95_m": float(np.percentile(label_dist, 95)),
    }
    return cands, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts/candidates.parquet")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--radius", type=float, default=200.0)
    args = ap.parse_args()

    frames, stats = [], []
    t0 = time.time()
    for split in ["train", "student_test"]:
        for trip_dir in list_trips(Path(args.data) / split):
            c, s = process_trip(trip_dir, split, args.k, args.radius)
            frames.append(c)
            stats.append(s)
            print(f"{split:12s} {s['trip_id']}  pts={s['n_points']:5d}  miss={s['miss_rate']:.3f}  "
                  f"nearest_acc={s['top1_nearest_acc']:.3f}  label_d50={s['label_dist_median_m']:.1f}m")

    df = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    st = pd.DataFrame(stats)
    Path("results").mkdir(exist_ok=True)
    st.to_csv("results/candidate_stats.csv", index=False)

    print(f"\nrows={len(df):,}  positives={df['label'].mean():.3%}  time={time.time() - t0:.0f}s")
    print(st.groupby("split")[["miss_rate", "top1_nearest_acc", "label_dist_median_m"]].mean())


if __name__ == "__main__":
    main()
