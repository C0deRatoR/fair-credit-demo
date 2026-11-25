# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "scipy",
#   "scikit-learn",
#   "fairlearn",
# ]
# ///
"""
Small working demo of the Slide 4 pipeline:
Loan Data -> Data Prep -> Baseline Model -> Fairness-Constrained Model

Dataset: German Credit ("credit-g" on OpenML) — a real credit-risk dataset
with a protected attribute (sex, derived from personal_status), used here
as the public proxy dataset for micro-finance-style credit scoring.

Run with:  uv run demo.py
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import train_test_split

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    data = fetch_openml(name="credit-g", version=1, as_frame=True)
    df = data.frame.copy()
    df["target"] = (df["class"] == "bad").astype(int)  # 1 = default risk
    df["sex"] = df["personal_status"].apply(
        lambda s: "female" if str(s).startswith("female") else "male"
    )
    return df.drop(columns=["class"])


def prep_features(df: pd.DataFrame):
    y = df["target"]
    sensitive = df["sex"]
    X = df.drop(columns=["target", "personal_status", "sex"])
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X, y, sensitive


def ks_stat(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    return ks_2samp(pos, neg).statistic


def reweigh(y: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
    """Kamiran-Calders reweighing: upweight under-represented (label, group) cells."""
    df = pd.DataFrame({"y": y, "s": sensitive})
    n = len(df)
    weights = np.ones(n)
    for (yv, sv), group in df.groupby(["y", "s"]):
        p_y = (df["y"] == yv).mean()
        p_s = (df["s"] == sv).mean()
        p_ys = len(group) / n
        weights[(df["y"] == yv) & (df["s"] == sv)] = (p_y * p_s) / p_ys
    return weights


def evaluate(name: str, y_true, y_pred, y_score, sensitive) -> None:
    auc = roc_auc_score(y_true, y_score)
    ks = ks_stat(y_true.values, np.asarray(y_score))
    recall_default = recall_score(y_true, y_pred)
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)
    print(f"\n--- {name} ---")
    print(f"  Accuracy metrics : AUC={auc:.3f}  KS={ks:.3f}  Recall(default)={recall_default:.3f}")
    print(f"  Fairness metrics : DemParityDiff={dpd:.3f}  EqualizedOddsDiff={eod:.3f}")


def main() -> None:
    df = load_data()
    X, y, sensitive = prep_features(df)
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    print("=" * 70)
    print("Credit Default Prediction under Fairness Constraints — mini demo")
    print(f"Rows: {len(df)}  |  Features: {X.shape[1]}  |  Protected attr: sex")
    print("=" * 70)

    # --- Baseline model (Slide 4: "Baseline Model") ---
    baseline = GradientBoostingClassifier(random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    base_score = baseline.predict_proba(X_test)[:, 1]
    base_pred = baseline.predict(X_test)
    evaluate("Baseline (no fairness constraint)", y_test, base_pred, base_score, s_test)

    # --- Fairness pre-processing: Reweighing ---
    weights = reweigh(y_train.values, s_train.values)
    fair_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    fair_model.fit(X_train, y_train, sample_weight=weights)
    fair_score = fair_model.predict_proba(X_test)[:, 1]
    fair_pred = fair_model.predict(X_test)
    evaluate("Reweighed model (pre-processing only)", y_test, fair_pred, fair_score, s_test)

    # --- Fairness post-processing: Threshold Adjustment on top of reweighed model ---
    postproc = ThresholdOptimizer(
        estimator=fair_model,
        constraints="equalized_odds",
        predict_method="predict_proba",
        prefit=True,
    )
    postproc.fit(X_train, y_train, sensitive_features=s_train)
    postproc_pred = postproc.predict(X_test, sensitive_features=s_test, random_state=RANDOM_STATE)
    evaluate(
        "Reweighing + Threshold Adjustment (full fairness pipeline)",
        y_test, postproc_pred, fair_score, s_test,
    )

    print(f"\nTest-set group sizes: {s_test.value_counts().to_dict()} (n={len(s_test)})")
    print("Note: this run is deterministic (fixed random_state throughout), but with only")
    print("~300 test rows split across 2 groups, fairness-difference estimates have real")
    print("sampling noise — a fairness technique is not guaranteed to reduce EVERY metric")
    print("on every dataset/split. That's exactly why Slide 4 tracks accuracy AND fairness")
    print("metrics together and reports both, rather than assuming the intervention 'worked'.")


if __name__ == "__main__":
    main()
