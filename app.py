"""
Tiny Streamlit UI over demo.py — click a button, watch the three-stage
fairness pipeline (Baseline -> Reweighed -> Reweighed+Threshold) run live.

Run with:
    uv run --with streamlit --with pandas --with numpy --with scipy \
        --with scikit-learn --with fairlearn streamlit run app.py
"""

import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer

from demo import RANDOM_STATE, ks_stat, load_data, prep_features, reweigh

st.set_page_config(page_title="Fair Credit Default Demo", page_icon="\U0001F4CA")


@st.cache_data
def get_data():
    df = load_data()
    return prep_features(df)


def run_pipeline(test_size: float):
    X, y, sensitive = get_data()
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    rows = []

    baseline = GradientBoostingClassifier(random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    base_score = baseline.predict_proba(X_test)[:, 1]
    base_pred = baseline.predict(X_test)
    rows.append(("Baseline", y_test, base_pred, base_score))

    weights = reweigh(y_train.values, s_train.values)
    fair_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    fair_model.fit(X_train, y_train, sample_weight=weights)
    fair_score = fair_model.predict_proba(X_test)[:, 1]
    fair_pred = fair_model.predict(X_test)
    rows.append(("Reweighed", y_test, fair_pred, fair_score))

    postproc = ThresholdOptimizer(
        estimator=fair_model, constraints="equalized_odds",
        predict_method="predict_proba", prefit=True,
    )
    postproc.fit(X_train, y_train, sensitive_features=s_train)
    postproc_pred = postproc.predict(X_test, sensitive_features=s_test, random_state=RANDOM_STATE)
    rows.append(("Reweighed + Threshold", y_test, postproc_pred, fair_score))

    records = []
    for name, y_true, y_pred, y_score in rows:
        records.append({
            "Stage": name,
            "AUC": round(roc_auc_score(y_true, y_score), 3),
            "KS": round(ks_stat(y_true.values, y_score), 3),
            "Recall (default)": round(recall_score(y_true, y_pred), 3),
            "Demographic Parity Diff": round(
                demographic_parity_difference(y_true, y_pred, sensitive_features=s_test), 3
            ),
            "Equalized Odds Diff": round(
                equalized_odds_difference(y_true, y_pred, sensitive_features=s_test), 3
            ),
        })
    return pd.DataFrame(records).set_index("Stage"), s_test.value_counts()


st.title("Credit Default Prediction — Fairness Pipeline Demo")
st.caption(
    "German Credit dataset (public proxy for micro-finance lending data) · "
    "protected attribute: sex"
)

test_size = st.slider("Test set size", 0.2, 0.4, 0.3, 0.05)

if st.button("Run pipeline", type="primary"):
    with st.spinner("Training baseline -> reweighed -> threshold-adjusted models..."):
        results, group_sizes = run_pipeline(test_size)

    st.subheader("Accuracy metrics")
    st.dataframe(results[["AUC", "KS", "Recall (default)"]], use_container_width=True)

    st.subheader("Fairness metrics")
    st.dataframe(results[["Demographic Parity Diff", "Equalized Odds Diff"]], use_container_width=True)

    st.bar_chart(results[["Demographic Parity Diff", "Equalized Odds Diff"]])

    st.caption(f"Test-set group sizes: {group_sizes.to_dict()}")
    st.info(
        "Fairness metrics don't necessarily shrink at every stage — with a small, "
        "noisy test split, a fairness technique can even look worse on one metric. "
        "That's why Slide 4 reports accuracy AND fairness together instead of "
        "assuming the intervention worked."
    )
else:
    st.write("Click **Run pipeline** to train the three models and compare metrics live.")
