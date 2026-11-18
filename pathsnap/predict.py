"""Full prediction pipeline for one trip: candidates -> classifier -> HMM/Viterbi -> route.

The reference route is never loaded here (load_trip(..., with_route=False)),
so evaluation labels cannot leak into predictions.
"""
import time
from pathlib import Path

import joblib
import numpy as np

from .candidates import build_candidates, trip_geometry
from .hmm import HMMParams, RoadGraph, reconstruct_route, viterbi
from .io import Trip, load_trip


def load_model(path: str | Path = "artifacts/final_model.joblib"):
    return joblib.load(path)


def predict_trip(trip: Trip | str | Path, bundle: dict, params: HMMParams | None = None,
                 use_hmm: bool = True, k: int = 10, radius_m: float = 200.0) -> dict:
    params = params or HMMParams()
    if not isinstance(trip, Trip):
        trip = load_trip(trip, with_route=False)
    timings = {}

    t0 = time.time()
    g = trip_geometry(trip)
    cands = build_candidates(trip, k=k, radius_m=radius_m, g=g)
    timings["candidates_s"] = time.time() - t0

    t0 = time.time()
    proba = bundle["model"].predict_proba(cands[bundle["features"]])[:, 1]
    cands["p"] = np.clip(proba, 1e-6, 1 - 1e-6)
    timings["classifier_s"] = time.time() - t0

    # points with at least one candidate, in time order
    groups = {o: grp for o, grp in cands.groupby("obs_idx", sort=True)}
    obs = np.array(sorted(groups))
    states = [{"arcs": groups[o]["arc_id"].to_numpy(),
               "t_along": groups[o]["t_along"].to_numpy(),
               "logp": np.log(groups[o]["p"].to_numpy())} for o in obs]

    graph = RoadGraph(trip.arcs, g.arc_len, len(trip.nodes))
    t0 = time.time()
    if use_hmm:
        step = np.r_[0.0, np.linalg.norm(np.diff(g.track_xy[obs], axis=0), axis=1)]
        choice, breaks = viterbi(states, step, graph, params)
    else:
        choice = np.array([int(np.argmax(s["logp"])) for s in states])
        breaks = np.zeros(len(states), dtype=bool)
    point_arcs = np.array([s["arcs"][c] for s, c in zip(states, choice)])
    route, unconnected = reconstruct_route(point_arcs, graph)
    timings["decode_s"] = time.time() - t0

    return {
        "trip_id": trip.trip_id,
        "obs_idx": obs,                       # GPS points that had candidates
        "point_arcs": point_arcs,             # chosen arc per GPS point
        "point_proba": np.array([np.exp(s["logp"][c]) for s, c in zip(states, choice)]),
        "route": route,                       # continuous arc sequence
        "n_breaks": int(breaks.sum()),
        "unconnected_gaps": unconnected,
        "n_points": trip.n_points,
        "timings": timings,
    }
