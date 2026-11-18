"""Show, dot by dot, where Viterbi overrules the classifier on one trip.

Usage:
    python -m training.demo_viterbi --trip data/student_test/00000091
    python -m training.demo_viterbi --trip data/student_test/00000091 --map results/maps/viterbi_00000091.html

For every GPS point where the classifier's top choice and the Viterbi choice
differ, it prints both arcs, the classifier probability of each, whether each
connects to the previous point's choice, and which one is the reference arc.
Optionally writes a Folium map with both routes.
"""
import argparse

import numpy as np

from pathsnap.candidates import trip_geometry
from pathsnap.hmm import HMMParams
from pathsnap.io import load_trip
from pathsnap.labels import align_track_to_route
from pathsnap.metrics import route_overlap
from pathsnap.predict import load_model, predict_trip


def connected(arcs, a, b):
    return a == b or arcs[a, 1] == arcs[b, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", default="data/student_test/00000091")
    ap.add_argument("--model", default="artifacts/final_model.joblib")
    ap.add_argument("--show", type=int, default=15, help="how many changed dots to print")
    ap.add_argument("--map", default=None, help="optional Folium HTML output path")
    args = ap.parse_args()

    bundle = load_model(args.model)
    params = HMMParams(beta=2.0, emission_weight=1.0)
    clf = predict_trip(args.trip, bundle, params, use_hmm=False)
    vit = predict_trip(args.trip, bundle, params, use_hmm=True)

    ref = load_trip(args.trip)               # reference, used only for scoring
    g = trip_geometry(ref)
    truth, _, _ = align_track_to_route(g.track_xy, ref.route, g.arc_a, g.arc_b)
    obs = clf["obs_idx"]
    c_arc, v_arc, t_arc = clf["point_arcs"], vit["point_arcs"], truth[obs]

    changed = np.flatnonzero(c_arc != v_arc)
    print(f"Trip {ref.trip_id}: {ref.n_points} GPS dots, Viterbi changed {len(changed)} of them\n")
    print(f"{'dot':>5} | {'classifier pick':>22} | {'Viterbi pick':>22} | correct")
    print("-" * 70)
    for i in changed[: args.show]:
        prev_v = v_arc[i - 1] if i > 0 else v_arc[i]
        c_ok = "connects" if connected(ref.arcs, prev_v, c_arc[i]) else "JUMP"
        v_ok = "connects" if connected(ref.arcs, prev_v, v_arc[i]) else "JUMP"
        winner = "Viterbi" if v_arc[i] == t_arc[i] else ("classifier" if c_arc[i] == t_arc[i] else "neither")
        print(f"{obs[i]:>5} | arc {c_arc[i]:>6} p={clf['point_proba'][i]:.2f} {c_ok:>8} | "
              f"arc {v_arc[i]:>6} p={vit['point_proba'][i]:.2f} {v_ok:>8} | {winner}")

    c_right = (c_arc[changed] == t_arc[changed]).sum()
    v_right = (v_arc[changed] == t_arc[changed]).sum()
    print(f"\nOn the {len(changed)} changed dots: classifier right {c_right}, Viterbi right {v_right}")
    for name, pred in (("classifier-only", clf), ("classifier+Viterbi", vit)):
        acc = (pred["point_arcs"] == t_arc).sum() / ref.n_points
        r = route_overlap(pred["route"], ref.route, g.arc_len)
        print(f"{name:20s} top1={acc:.3f}  route_precision={r['route_precision']:.3f}  "
              f"route_f1={r['route_f1']:.3f}  unconnected_gaps={pred['unconnected_gaps']}")

    if args.map:
        import folium
        nodes = ref.nodes
        m = folium.Map(location=[nodes[:, 1].mean(), nodes[:, 0].mean()], zoom_start=15,
                       tiles="cartodbpositron")
        layers = [("Reference route", ref.route, "#1baf7a", 7),
                  ("Classifier only", clf["route"], "#e34948", 4),
                  ("Classifier + Viterbi", vit["route"], "#2a78d6", 4)]
        for name, route, color, w in layers:
            fg = folium.FeatureGroup(name=name)
            for a in route:
                fr, to = ref.arcs[a]
                folium.PolyLine([[nodes[fr, 1], nodes[fr, 0]], [nodes[to, 1], nodes[to, 0]]],
                                color=color, weight=w, opacity=0.8).add_to(fg)
            fg.add_to(m)
        gps = folium.FeatureGroup(name="GPS points")
        for lon, lat, _ in ref.track[::3]:
            folium.CircleMarker([lat, lon], radius=2, color="#52514e", fill=True).add_to(gps)
        gps.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        m.save(args.map)
        print(f"\nmap written to {args.map}")


if __name__ == "__main__":
    main()
