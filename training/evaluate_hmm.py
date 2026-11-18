"""Compare classifier-only decoding with classifier + HMM/Viterbi decoding.

Usage:
    python -m training.evaluate_hmm --split student_test
    python -m training.evaluate_hmm --split train --grid   # tune beta / emission weight on train trips

Point level : Top-1 accuracy of the arc chosen for each GPS point (points with
              no candidate count as wrong), sequence continuity.
Route level : length-weighted precision / recall / F1 of the reconstructed
              route against the reference route; unconnected gaps.
The reference route is loaded only here, after prediction, to score it.
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.candidates import trip_geometry
from pathsnap.hmm import HMMParams
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import align_track_to_route
from pathsnap.metrics import route_overlap, sequence_continuity
from pathsnap.predict import load_model, predict_trip
from training.train_models import BAD_TRIPS


def score_trip(trip_dir, bundle, params, use_hmm):
    t0 = time.time()
    pred = predict_trip(trip_dir, bundle, params, use_hmm=use_hmm)   # no route loaded
    elapsed = time.time() - t0

    ref = load_trip(trip_dir)                                          # reference, for scoring only
    g = trip_geometry(ref)
    label_arc, _, _ = align_track_to_route(g.track_xy, ref.route, g.arc_a, g.arc_b)
    correct = (pred["point_arcs"] == label_arc[pred["obs_idx"]]).sum()
    return {
        "trip_id": ref.trip_id,
        "decoder": "classifier+HMM" if use_hmm else "classifier-only",
        "top1_acc": correct / ref.n_points,
        "point_continuity": sequence_continuity(pred["point_arcs"], ref.arcs),
        **route_overlap(pred["route"], ref.route, g.arc_len),
        "unconnected_gaps": pred["unconnected_gaps"],
        "hmm_breaks": pred["n_breaks"],
        "seconds": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="student_test")
    ap.add_argument("--model", default="artifacts/final_model.joblib")
    ap.add_argument("--beta", type=float, default=HMMParams.beta)
    ap.add_argument("--emission-weight", type=float, default=HMMParams.emission_weight)
    ap.add_argument("--grid", action="store_true", help="grid-search beta and emission weight")
    ap.add_argument("--max-trips", type=int, default=None)
    args = ap.parse_args()

    bundle = load_model(args.model)
    trips = [t for t in list_trips(Path(args.data) / args.split) if t.name not in BAD_TRIPS]
    trips = trips[: args.max_trips] if args.max_trips else trips
    out = Path("results/hmm")
    out.mkdir(parents=True, exist_ok=True)

    if args.grid:
        rows = []
        for beta, w in itertools.product([2, 5, 10, 20], [0.5, 1.0, 2.0]):
            p = HMMParams(beta=beta, emission_weight=w)
            r = pd.DataFrame([score_trip(t, bundle, p, True) for t in trips])
            rows.append({"beta": beta, "emission_weight": w,
                         **r[["top1_acc", "route_f1", "point_continuity"]].mean().to_dict()})
            print(rows[-1], flush=True)
        grid = pd.DataFrame(rows).sort_values("route_f1", ascending=False)
        grid.to_csv(out / f"hmm_grid_{args.split}.csv", index=False)
        print("\n" + grid.round(4).to_string(index=False))
        return

    params = HMMParams(beta=args.beta, emission_weight=args.emission_weight)
    rows = []
    for t in trips:
        for use_hmm in (False, True):
            r = score_trip(t, bundle, params, use_hmm)
            rows.append(r)
            print(f"{r['trip_id']}  {r['decoder']:16s} top1={r['top1_acc']:.3f}  "
                  f"cont={r['point_continuity']:.3f}  routeF1={r['route_f1']:.3f}  "
                  f"gaps={r['unconnected_gaps']}  {r['seconds']:.1f}s", flush=True)
    per_trip = pd.DataFrame(rows)
    per_trip.to_csv(out / f"hmm_per_trip_{args.split}.csv", index=False)
    summary = per_trip.groupby("decoder")[["top1_acc", "point_continuity", "route_precision",
                                           "route_recall", "route_f1", "unconnected_gaps",
                                           "seconds"]].mean()
    summary.to_csv(out / f"hmm_summary_{args.split}.csv")
    print("\n" + summary.round(4).to_string())


if __name__ == "__main__":
    main()
