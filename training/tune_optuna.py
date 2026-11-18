"""Bayesian hyperparameter optimisation of the selected model (LightGBM) with Optuna.

Usage:
    python -m training.tune_optuna --trials 40

Search: TPE sampler (Bayesian), objective = mean macro F1 over the same
5-fold GroupKFold (grouped by trip) used for model selection, on the same
400k-row training subsample, so tuned and untuned numbers are comparable.
A median pruner stops clearly bad trials after the first folds.

Outputs:
  results/tuning/optuna_trials.csv           every trial: params, score, time
  results/tuning/optuna_study.db             full Optuna study (SQLite)
  results/tuning/optimisation_history.png    score per trial + best so far
  results/tuning/param_importance.png        which hyperparameters mattered
  results/tuning/tuned_vs_untuned.csv        same metrics, same splits
  artifacts/final_model.joblib               tuned model refit on all train trips
  artifacts/lgbm_best_params.json
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.candidates import FEATURES
from pathsnap.metrics import row_metrics
from training.train_models import BAD_TRIPS, MODELS, SEED, evaluate, subsample

OUT = Path("results/tuning")


def make_model(params: dict) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1, **params)


def search_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def cv_score(model_factory, folds, trial=None):
    f1s, losses = [], []
    for i, (tr, va) in enumerate(folds):
        m = model_factory()
        m.fit(tr[FEATURES], tr["label"])
        p = m.predict_proba(va[FEATURES])[:, 1]
        r = row_metrics(va["label"].to_numpy(), p)
        f1s.append(r["macro_f1"])
        losses.append(r["log_loss"])
        if trial is not None:
            trial.report(float(np.mean(f1s)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="artifacts/candidates.parquet")
    ap.add_argument("--data", default="data")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--max-train-rows", type=int, default=400_000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.table)
    train = df[(df["split"] == "train") & (~df["trip_id"].isin(BAD_TRIPS))].reset_index(drop=True)
    test = df[df["split"] == "student_test"].reset_index(drop=True)

    # identical folds and subsamples to training/train_models.py
    folds = [(subsample(train.iloc[tr], args.max_train_rows), train.iloc[va])
             for tr, va in GroupKFold(n_splits=5).split(train, groups=train["trip_id"])]

    def objective(trial):
        params = search_space(trial)
        t0 = time.time()
        f1, f1_std, ll = cv_score(lambda: make_model(params), folds, trial)
        trial.set_user_attr("cv_f1_std", f1_std)
        trial.set_user_attr("cv_log_loss", ll)
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number:3d}  cv_macro_f1={f1:.4f}  logloss={ll:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        return f1

    study = optuna.create_study(
        study_name="lgbm_pathsnap", direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=f"sqlite:///{OUT / 'optuna_study.db'}", load_if_exists=True)
    t_search = time.time()
    study.optimize(objective, n_trials=args.trials)
    search_s = time.time() - t_search

    trials = study.trials_dataframe()
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best = study.best_params
    json.dump(best, open("artifacts/lgbm_best_params.json", "w"), indent=2)
    print(f"\nbest cv_macro_f1={study.best_value:.4f}\nbest params={best}")

    # ---- tuned vs untuned: same folds, same subsample, same test set
    rows = []
    for name, factory in [("untuned", MODELS["lgbm"][1]), ("tuned", lambda: make_model(best))]:
        f1, f1_std, ll = cv_score(factory, folds)
        model = factory()
        t0 = time.time()
        model.fit(*(lambda d: (d[FEATURES], d["label"]))(subsample(train, args.max_train_rows)))
        fit_s = time.time() - t0
        t0 = time.time()
        p = model.predict_proba(test[FEATURES])[:, 1]
        infer_s = time.time() - t0
        t = evaluate(test, p, args.data)
        n_pts = test.groupby(["trip_id", "obs_idx"]).ngroups
        rows.append({"model": f"lgbm_{name}", "cv_macro_f1": f1, "cv_macro_f1_std": f1_std,
                     "cv_log_loss": ll, "test_top1": t["top1_acc"], "test_macro_f1": t["macro_f1"],
                     "test_log_loss": t["log_loss"], "test_continuity": t["continuity"],
                     "fit_s": fit_s, "infer_s_per_1k_points": infer_s / n_pts * 1000})
        if name == "tuned":
            joblib.dump({"model": model, "features": FEATURES, "params": best},
                        "artifacts/final_model.joblib")
    comp = pd.DataFrame(rows)
    comp["search_minutes"] = [np.nan, search_s / 60]
    comp["n_trials"] = [np.nan, len(study.trials)]
    comp.to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    print("\n" + comp.round(4).to_string(index=False))

    plot_history(trials, OUT / "optimisation_history.png")
    try:
        imp = optuna.importance.get_param_importances(study)
        plot_importance(imp, OUT / "param_importance.png")
        pd.Series(imp).to_csv(OUT / "param_importance.csv", header=["importance"])
    except Exception as e:  # needs enough completed trials
        print("param importance skipped:", e)


def _style(ax):
    ax.grid(color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_history(trials: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    done = trials[trials["state"] == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.scatter(done["number"], done["value"], s=22, color="#86b6ef", label="trial (completed)")
    best = done.set_index("number")["value"].cummax()
    ax.step(best.index, best.values, where="post", color="#2a78d6", linewidth=2, label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV macro F1")
    ax.set_title("Optuna (TPE) optimisation history: LightGBM", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(imp: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = pd.Series(imp).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(s) + 1))
    ax.barh(s.index, s.values, color="#2a78d6", height=0.55)
    for y, v in enumerate(s.values):
        ax.text(v, y, f" {v:.2f}", va="center", fontsize=9)
    ax.set_title("Hyperparameter importance (fANOVA on completed trials)", loc="left", fontsize=10)
    ax.set_xlim(0, s.max() * 1.2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
