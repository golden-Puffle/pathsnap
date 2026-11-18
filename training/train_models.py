"""Train and compare candidate classifiers with trip-grouped cross-validation.

Usage:
    python -m training.train_models                    # all models in MODELS
    python -m training.train_models --models lr lgbm   # a subset

For each model:
  1. 5-fold GroupKFold on the training trips (a trip is never split across
     train and validation folds) -> CV macro F1, log loss, Top-1 per fold.
  2. Refit on all training trips -> evaluate on the 10 student_test trips.
Results: results/cv_folds.csv, results/model_comparison.csv,
         results/model_comparison.png; fitted models in artifacts/models/.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.candidates import FEATURES
from pathsnap.metrics import route_continuity, row_metrics, top1_accuracy, top1_choice

SEED = 42
# Trips whose GPS is offset 120-140 m from the reference route for the whole
# trip (see results/candidate_stats.csv): labels are unreliable, so excluded.
BAD_TRIPS = ["00000033", "00000054"]

# name -> (algorithm family, factory). Add new models here.
MODELS = {
    "lr": ("Linear", lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))),
    "rf": ("Bagging ensemble", lambda: RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, max_features="sqrt",
        n_jobs=-1, random_state=SEED)),
    "lgbm": ("Boosting", lambda: lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1)),
}


def subsample(df: pd.DataFrame, max_rows: int, seed: int = SEED) -> pd.DataFrame:
    """Row subsample for training only (validation always uses every row, so
    Top-1 sees all candidates of each point). Same seed for every model."""
    return df if len(df) <= max_rows else df.sample(max_rows, random_state=seed)


def fit_predict(name, train_df, eval_df):
    model = MODELS[name][1]()
    t0 = time.time()
    model.fit(train_df[FEATURES], train_df["label"])
    fit_s = time.time() - t0
    t0 = time.time()
    proba = model.predict_proba(eval_df[FEATURES])[:, 1]
    infer_s = time.time() - t0
    return model, proba, fit_s, infer_s


def evaluate(eval_df, proba, data_dir):
    m = row_metrics(eval_df["label"].to_numpy(), proba)
    chosen = top1_choice(eval_df, proba)
    m["top1_acc"] = top1_accuracy(chosen)
    m["continuity"] = route_continuity(chosen, data_dir)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--table", default="artifacts/candidates.parquet")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-train-rows", type=int, default=400_000)
    args = ap.parse_args()

    df = pd.read_parquet(args.table)
    train = df[(df["split"] == "train") & (~df["trip_id"].isin(BAD_TRIPS))].reset_index(drop=True)
    test = df[df["split"] == "student_test"].reset_index(drop=True)
    print(f"train rows={len(train):,} trips={train.trip_id.nunique()} | "
          f"test rows={len(test):,} trips={test.trip_id.nunique()}")

    Path("results").mkdir(exist_ok=True)
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    fold_rows, summary = [], []
    gkf = GroupKFold(n_splits=args.folds)

    for name in args.models:
        family = MODELS[name][0]
        print(f"\n=== {name} ({family})")
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, groups=train["trip_id"])):
            tr = subsample(train.iloc[tr_idx], args.max_train_rows)
            va = train.iloc[va_idx]
            _, proba, fit_s, infer_s = fit_predict(name, tr, va)
            m = evaluate(va, proba, args.data)
            fold_rows.append({"model": name, "fold": fold, "fit_s": fit_s, "infer_s": infer_s,
                              **{k: v for k, v in m.items() if k != "confusion"}})
            print(f"  fold {fold}: macroF1={m['macro_f1']:.4f} logloss={m['log_loss']:.4f} "
                  f"top1={m['top1_acc']:.4f} fit={fit_s:.1f}s")

        cv = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        model, proba, fit_s, infer_s = fit_predict(name, subsample(train, args.max_train_rows), test)
        t = evaluate(test, proba, args.data)
        joblib.dump({"model": model, "features": FEATURES}, f"artifacts/models/{name}.joblib")
        summary.append({
            "model": name, "family": family,
            "config": json.dumps(model.get_params(deep=False), default=str)[:300],
            "cv_macro_f1": cv["macro_f1"].mean(), "cv_macro_f1_std": cv["macro_f1"].std(),
            "cv_log_loss": cv["log_loss"].mean(), "cv_top1": cv["top1_acc"].mean(),
            "test_top1": t["top1_acc"], "test_macro_f1": t["macro_f1"],
            "test_precision": t["precision"], "test_recall": t["recall"],
            "test_log_loss": t["log_loss"], "test_continuity": t["continuity"],
            "test_confusion": json.dumps(t["confusion"]),
            "fit_s": fit_s, "infer_s_per_1k_points": infer_s / test.groupby(["trip_id", "obs_idx"]).ngroups * 1000,
        })
        print(f"  student_test: top1={t['top1_acc']:.4f} macroF1={t['macro_f1']:.4f} "
              f"logloss={t['log_loss']:.4f} continuity={t['continuity']:.4f}")

    # merge with earlier runs so models trained separately (e.g. by teammates) share one table
    folds_path, comp_path = Path("results/cv_folds.csv"), Path("results/model_comparison.csv")
    new_folds, new_comp = pd.DataFrame(fold_rows), pd.DataFrame(summary)
    if folds_path.exists():
        old = pd.read_csv(folds_path)
        new_folds = pd.concat([old[~old["model"].isin(args.models)], new_folds])
    if comp_path.exists():
        old = pd.read_csv(comp_path)
        new_comp = pd.concat([old[~old["model"].isin(args.models)], new_comp])
    new_folds.to_csv(folds_path, index=False)
    new_comp = new_comp.sort_values("cv_macro_f1", ascending=False)
    new_comp.to_csv(comp_path, index=False)
    plot_comparison(new_comp, "results/model_comparison.png")

    cols = ["model", "family", "cv_macro_f1", "cv_log_loss", "test_top1", "test_macro_f1",
            "test_log_loss", "test_continuity", "fit_s"]
    print("\n" + new_comp[cols].round(4).to_string(index=False))


def plot_comparison(comp: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, bar = "#0b0b0b", "#52514e", "#2a78d6"
    panels = [("cv_macro_f1", "CV macro F1 (5-fold, grouped by trip)"),
              ("test_top1", "Student-test Top-1 accuracy (per GPS point)"),
              ("test_log_loss", "Student-test log loss (lower is better)")]
    comp = comp.sort_values("cv_macro_f1")
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.6 * len(comp) + 1.6))
    for ax, (col, title) in zip(axes, panels):
        ax.barh(comp["model"], comp[col], color=bar, height=0.55)
        for y, v in enumerate(comp[col]):
            ax.text(v, y, f" {v:.3f}", va="center", color=ink, fontsize=9)
        ax.set_title(title, loc="left", color=ink, fontsize=10)
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(0, comp[col].max() * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
