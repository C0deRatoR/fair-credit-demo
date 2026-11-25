# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas", "numpy", "scipy", "scikit-learn", "fairlearn",
#   "optuna", "mlflow", "matplotlib",
# ]
# ///
"""
Final-presentation experiments: EDA, two candidate models, Optuna tuning,
MLflow tracking, calibration and fairness, on a train/validation/test split.

Run with:  uv run experiments.py      (writes into out/)
MLflow UI: uv run --with mlflow mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from demo import RANDOM_STATE, ks_stat, load_data, prep_features, reweigh

OUT = Path("out")
OUT.mkdir(exist_ok=True)
N_TRIALS = 40


def metrics(y, pred, score, s) -> dict:
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, score),
        "ks": ks_stat(np.asarray(y), np.asarray(score)),
        "brier": brier_score_loss(y, score),
        "dp_gap": demographic_parity_difference(y, pred, sensitive_features=s),
        "eo_gap": equalized_odds_difference(y, pred, sensitive_features=s),
    }


def eda(df) -> dict:
    num = df.select_dtypes("number")
    num.describe().T.round(2).to_csv(OUT / "descriptive_stats.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = num.corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlation heatmap (numeric features)")
    fig.tight_layout(); fig.savefig(OUT / "correlation_heatmap.png", dpi=150); plt.close(fig)
    return {
        "rows": len(df), "columns": df.shape[1] - 2,
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "default_rate": round(df["target"].mean(), 3),
        "sex_counts": df["sex"].value_counts().to_dict(),
        "default_rate_by_sex": df.groupby("sex")["target"].mean().round(3).to_dict(),
    }


def main() -> None:
    df = load_data()
    summary = {"eda": eda(df)}
    X, y, s = prep_features(df)
    summary["eda"]["features_after_encoding"] = X.shape[1]

    # 60/20/20: validation drives model choice + Optuna; test is touched once at the end.
    X_tmp, X_te, y_tmp, y_te, s_tmp, s_te = train_test_split(
        X, y, s, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_tr, X_va, y_tr, y_va, s_tr, s_va = train_test_split(
        X_tmp, y_tmp, s_tmp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tmp)
    summary["split"] = {"train": len(X_tr), "validation": len(X_va), "test": len(X_te)}

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fair-credit")

    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "gradient_boosting_baseline": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    val_results = {}
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        m = metrics(y_va, model.predict(X_va), model.predict_proba(X_va)[:, 1], s_va)
        val_results[name] = m
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "split": "validation"})
            mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        }
        model = GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metric("val_auc", auc)
        return auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS)
    summary["optuna"] = {"trials": N_TRIALS, "best_val_auc": study.best_value,
                         "best_params": study.best_params,
                         "baseline_val_auc": val_results["gradient_boosting_baseline"]["auc"]}
    vals = [t.value for t in study.trials]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(vals, "o", alpha=0.5, label="trial AUC")
    ax.plot(np.maximum.accumulate(vals), "-", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("validation AUC"); ax.legend()
    ax.set_title("Optuna optimisation history")
    fig.tight_layout(); fig.savefig(OUT / "optuna_history.png", dpi=150); plt.close(fig)

    # Final test evaluation (test set used only here).
    tuned = GradientBoostingClassifier(random_state=RANDOM_STATE, **study.best_params)
    tuned.fit(X_tr, y_tr)
    weights = reweigh(y_tr.values, s_tr.values)
    tuned_rw = GradientBoostingClassifier(random_state=RANDOM_STATE, **study.best_params)
    tuned_rw.fit(X_tr, y_tr, sample_weight=weights)
    thr = ThresholdOptimizer(estimator=tuned_rw, constraints="equalized_odds",
                             predict_method="predict_proba", prefit=True)
    thr.fit(X_va, y_va, sensitive_features=s_va)  # fit cut-offs on validation, not test

    final_models = {
        "logistic_regression": candidates["logistic_regression"],
        "gb_baseline": candidates["gradient_boosting_baseline"],
        "gb_tuned": tuned,
        "gb_tuned_reweighed": tuned_rw,
    }
    test = {}
    for name, model in final_models.items():
        score = model.predict_proba(X_te)[:, 1]
        test[name] = (model.predict(X_te), score)
    rw_score = test["gb_tuned_reweighed"][1]
    test["gb_tuned_reweighed_threshold"] = (
        thr.predict(X_te, sensitive_features=s_te, random_state=RANDOM_STATE), rw_score)

    summary["test"] = {}
    for name, (pred, score) in test.items():
        m = metrics(y_te, pred, score, s_te)
        summary["test"][name] = {k: round(float(v), 3) for k, v in m.items()}
        summary["test"][name]["confusion_matrix"] = confusion_matrix(y_te, pred).tolist()
        with mlflow.start_run(run_name=f"test_{name}"):
            mlflow.log_param("model", name)
            mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for name in ["logistic_regression", "gb_baseline", "gb_tuned"]:
        frac, mean = calibration_curve(y_te, test[name][1], n_bins=8, strategy="quantile")
        ax.plot(mean, frac, "o-", label=f"{name} (Brier {summary['test'][name]['brier']})")
    ax.set_xlabel("mean predicted default probability"); ax.set_ylabel("observed default rate")
    ax.set_title("Reliability diagram (test set)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "reliability_diagram.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, name in zip(axes, ["gb_tuned", "gb_tuned_reweighed_threshold"]):
        cm = np.array(summary["test"][name]["confusion_matrix"])
        ax.imshow(cm, cmap="Reds")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], ["good", "default"]); ax.set_yticks([0, 1], ["good", "default"])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(name, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=150); plt.close(fig)

    summary["validation"] = {n: {k: round(float(v), 3) for k, v in m.items()}
                             for n, m in val_results.items()}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
