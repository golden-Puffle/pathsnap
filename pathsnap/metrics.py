"""Evaluation metrics shared by all models.

Row level  (each (point, candidate) pair is a binary example):
    macro F1, precision, recall, log loss, confusion matrix
Point level (pick the highest-scoring candidate per GPS point):
    Top-1 accuracy = chosen arc == reference arc
Sequence level:
    route continuity = share of consecutive points whose chosen arcs are the
    same arc or directly connected (end node of one = start node of the next)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def row_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    y_pred = (proba >= threshold).astype(int)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def top1_choice(df: pd.DataFrame, proba) -> pd.DataFrame:
    """Highest-probability candidate per (trip, point)."""
    d = df[["trip_id", "obs_idx", "arc_id", "true_arc"]].copy()
    d["p"] = proba
    idx = d.groupby(["trip_id", "obs_idx"])["p"].idxmax()
    return d.loc[idx].sort_values(["trip_id", "obs_idx"]).reset_index(drop=True)


def top1_accuracy(chosen: pd.DataFrame) -> float:
    return float((chosen["arc_id"] == chosen["true_arc"]).mean())


def route_overlap(pred_route, true_route, arc_len: np.ndarray) -> dict:
    """Length-weighted overlap between predicted and reference routes (unique arcs).
    precision = share of predicted length that is on the reference route,
    recall    = share of reference length that was predicted."""
    pred, true = set(map(int, pred_route)), set(map(int, true_route))
    inter = sum(arc_len[a] for a in pred & true)
    p_len = sum(arc_len[a] for a in pred)
    t_len = sum(arc_len[a] for a in true)
    prec = inter / p_len if p_len else 0.0
    rec = inter / t_len if t_len else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


def sequence_continuity(point_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Share of consecutive points whose arcs are the same or directly connected."""
    a = np.asarray(point_arcs)
    if len(a) < 2:
        return 1.0
    prev, cur = a[:-1], a[1:]
    return float(((prev == cur) | (arcs[prev, 1] == arcs[cur, 0])).mean())


_ARCS_CACHE: dict[str, np.ndarray] = {}


def _arcs_for(trip_id: str, data_dir: Path) -> np.ndarray:
    if trip_id not in _ARCS_CACHE:
        for split in ("train", "student_test"):
            p = data_dir / split / trip_id / f"{trip_id}.arcs"
            if p.exists():
                _ARCS_CACHE[trip_id] = np.loadtxt(p, dtype=int, ndmin=2)
                break
    return _ARCS_CACHE[trip_id]


def route_continuity(chosen: pd.DataFrame, data_dir: str | Path = "data") -> float:
    data_dir = Path(data_dir)
    ok = total = 0
    for tid, g in chosen.groupby("trip_id"):
        arcs = _arcs_for(tid, data_dir)
        a = g["arc_id"].to_numpy()
        prev, cur = a[:-1], a[1:]
        connected = (prev == cur) | (arcs[prev, 1] == arcs[cur, 0])
        ok += int(connected.sum())
        total += len(connected)
    return ok / max(total, 1)
