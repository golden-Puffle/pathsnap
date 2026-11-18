"""PathSnap API (Laptop 1).

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables:
    PATHSNAP_TRIPS_DIR  folder holding the evaluation trip folders
                        (default: data/student_test; point it at the evaluator trips for the demo)
    PATHSNAP_MODEL      model bundle (default: artifacts/final_model.joblib,
                        falls back to artifacts/models/lgbm.joblib)
    PATHSNAP_HMM_CONFIG HMM parameters JSON (default: artifacts/hmm_config.json)

Endpoints:
    GET  /health            status, model and trips folder in use
    GET  /trips             available trip ids (no ground truth)
    POST /predict           {"trip_id": "..."} -> predicted route + metrics

The prediction pipeline never sees the reference route. It is read only
afterwards, server-side, to compute the metrics; its geometry is never sent
to the UI.
"""
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from pathsnap.candidates import trip_geometry
from pathsnap.hmm import HMMParams
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import align_track_to_route
from pathsnap.metrics import route_overlap, sequence_continuity
from pathsnap.predict import load_model, predict_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("pathsnap")

TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", "data/student_test"))
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL", "artifacts/final_model.joblib"))
if not MODEL_PATH.exists():
    MODEL_PATH = Path("artifacts/models/lgbm.joblib")
HMM_CONFIG = Path(os.environ.get("PATHSNAP_HMM_CONFIG", "artifacts/hmm_config.json"))

BUNDLE = load_model(MODEL_PATH)
HMM = HMMParams(**json.loads(HMM_CONFIG.read_text())) if HMM_CONFIG.exists() else HMMParams()
log.info("model=%s  hmm=%s  trips_dir=%s", MODEL_PATH, HMM, TRIPS_DIR)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str
    use_hmm: bool = True


def _trip_dir(trip_id: str) -> Path:
    d = TRIPS_DIR / trip_id
    if not trip_id.isalnum() or not d.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown trip_id {trip_id!r}")
    return d


def _route_polylines(route, nodes, arcs):
    """Arc ids -> list of [lat, lon] polylines (split where arcs do not connect)."""
    lines, cur = [], []
    for a in route:
        fr, to = arcs[a]
        if not cur or cur[-1] != fr:
            if cur:
                lines.append(cur)
            cur = [int(fr)]
        cur.append(int(to))
    if cur:
        lines.append(cur)
    return [[[float(nodes[n, 1]), float(nodes[n, 0])] for n in line] for line in lines]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_PATH), "trips_dir": str(TRIPS_DIR),
            "hmm": HMM.__dict__}


@app.get("/trips")
def trips():
    ids = [p.name for p in list_trips(TRIPS_DIR)]
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    trip_dir = _trip_dir(req.trip_id)
    log.info("POST /predict trip=%s use_hmm=%s", req.trip_id, req.use_hmm)
    t0 = time.time()

    # 1. prediction: the route file is NOT loaded here
    trip = load_trip(trip_dir, with_route=False)
    pred = predict_trip(trip, BUNDLE, HMM, use_hmm=req.use_hmm)
    predict_s = time.time() - t0

    # 2. scoring: reference route read server-side only
    metrics = None
    ref = load_trip(trip_dir)
    if ref.route is not None:
        g = trip_geometry(ref)
        label_arc, _, _ = align_track_to_route(g.track_xy, ref.route, g.arc_a, g.arc_b)
        correct = int((pred["point_arcs"] == label_arc[pred["obs_idx"]]).sum())
        metrics = {
            "top1_accuracy": correct / ref.n_points,
            "point_continuity": sequence_continuity(pred["point_arcs"], ref.arcs),
            **route_overlap(pred["route"], ref.route, g.arc_len),
        }
        metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    total_s = time.time() - t0
    log.info("trip=%s points=%d route_arcs=%d predict=%.2fs total=%.2fs metrics=%s",
             req.trip_id, trip.n_points, len(pred["route"]), predict_s, total_s, metrics)

    return {
        "trip_id": req.trip_id,
        "decoder": "classifier+HMM" if req.use_hmm else "classifier-only",
        "n_points": trip.n_points,
        "gps_points": [[float(lat), float(lon)] for lon, lat in trip.track[:, :2]],
        "predicted_route": _route_polylines(pred["route"], trip.nodes, trip.arcs),
        "predicted_route_arc_ids": [int(a) for a in pred["route"]],
        "mean_point_confidence": round(float(np.mean(pred["point_proba"])), 4),
        "hmm_breaks": pred["n_breaks"],
        "unconnected_gaps": pred["unconnected_gaps"],
        "metrics": metrics,
        "timings_s": {**{k: round(v, 3) for k, v in pred["timings"].items()},
                      "predict_total": round(predict_s, 3), "request_total": round(total_s, 3)},
    }
