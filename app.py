"""
Tiny Streamlit UI over demo.py — click a button, watch the three-stage
fairness pipeline (Baseline -> Reweighed -> Reweighed+Threshold) run live.

Run with:
    uv run --with streamlit --with pandas --with numpy --with scipy \
        --with scikit-learn --with fairlearn streamlit run app.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer

from demo import RANDOM_STATE, ks_stat, load_data, prep_features, reweigh

st.set_page_config(page_title="Fair Credit", page_icon="\U0001F7E5", layout="wide")


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
    # Per-applicant view: each test applicant ranked by the fair model's default probability.
    ranking = pd.DataFrame({
        "Applicant": X_test.index,
        "Sex": s_test.values,
        "Age": X_test["age"].values,
        "Loan amount": X_test["credit_amount"].values,
        "Baseline risk": base_score.round(3),
        "Fair risk": fair_score.round(3),
        "Flagged (final)": np.where(postproc_pred == 1, "Yes", "No"),
        "Actual": np.where(y_test.values == 1, "Defaulted", "Repaid"),
    }).sort_values("Fair risk", ascending=False)
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    # shortcut: illustrative tier cut-offs (0.30 ≈ base default rate), tune with the business before real use.
    ranking.insert(7, "Tier", pd.cut(ranking["Fair risk"], [0, 0.3, 0.6, 1],
                                     labels=["Approve", "Refer", "Decline"], include_lowest=True).astype(str))
    return pd.DataFrame(records).set_index("Stage"), s_test.value_counts(), ranking



# Overview copy is taken word for word from the "Fair Credit app design" PDF.
OVERVIEW_HTML = """
<div class="fc">
  <section class="fc-hero">
    <p class="fc-eyebrow">Credit risk · Fairness audit</p>
    <h1>Predict credit default&nbsp;— and show whether the model treats groups fairly.</h1>
    <p class="fc-lead">Fair Credit trains a credit-risk model on real lending data, then applies two
      fairness techniques in sequence. Every stage reports accuracy and fairness side by side, so you
      can see exactly what each intervention costs and gains.</p>
    <div class="fc-actions">
      <a class="fc-btn fc-btn-primary" href="?view=run">Run the pipeline <span aria-hidden="true">→</span></a>
      <a class="fc-btn" href="#how-it-works">How it works</a>
    </div>
  </section>

  <ol class="fc-flow">
    <li><span class="fc-label">IN</span><div><strong>Loan applications</strong><span class="fc-sub">German Credit · 1,000 applicants</span></div></li>
    <li><span class="fc-label">01</span><div><strong>Baseline model</strong><span class="fc-sub">Gradient boosting, no constraints</span></div></li>
    <li><span class="fc-label">02</span><div><strong>Reweighed</strong><span class="fc-sub">Pre-processing: balance training data</span></div></li>
    <li><span class="fc-label">03</span><div><strong>Reweighed + Threshold</strong><span class="fc-sub">Post-processing: group-aware cut-offs</span></div></li>
    <li><span class="fc-label">OUT</span><div><strong>Accuracy + fairness report</strong><span class="fc-sub">AUC, KS, recall, parity and odds gaps</span></div></li>
  </ol>

  <section class="fc-section" id="how-it-works">
    <div class="fc-split">
      <h2 class="fc-h2">Three stages, one question</h2>
      <p>Each stage builds on the last. Comparing them shows whether a fairness technique narrows the
        gap between groups — and what it does to predictive power.</p>
    </div>
    <div class="fc-stages">
      <article class="fc-stage"><span class="fc-num">01</span><h3>Baseline</h3>
        <p>A gradient-boosting classifier trained on the raw data. The protected attribute is removed
          from the inputs, but bias can still leak in through correlated features.</p></article>
      <article class="fc-stage"><span class="fc-num">02</span><h3>Reweighed</h3>
        <p>Kamiran–Calders reweighing gives each training example a weight so that outcome and group
          look statistically independent. Under-represented combinations count for more.</p></article>
      <article class="fc-stage"><span class="fc-num">03</span><h3>Reweighed + Threshold</h3>
        <p>A threshold optimizer sets a separate decision cut-off per group on top of the reweighed
          model, targeting equalized odds — similar error rates for everyone.</p></article>
    </div>
  </section>

  <section class="fc-section">
    <div class="fc-table-head"><h2 class="fc-h2">Accuracy</h2><span class="fc-note">higher is better</span></div>
    <dl class="fc-defs">
      <div><dt>AUC</dt><dd>How well the model ranks risky applicants above safe ones, from 0.5 (chance) to 1.0.</dd></div>
      <div><dt>KS</dt><dd>The largest separation between score distributions of defaulters and non-defaulters.</dd></div>
      <div><dt>Recall</dt><dd>Share of actual defaults the model catches.</dd></div>
    </dl>
    <div class="fc-table-head"><h2 class="fc-h2">Fairness</h2><span class="fc-note">closer to 0 is better</span></div>
    <dl class="fc-defs">
      <div><dt>Demographic parity gap</dt><dd>Difference in how often each group is flagged as high-risk, regardless of outcome.</dd></div>
      <div><dt>Equalized odds gap</dt><dd>Largest difference between groups in true-positive or false-positive rate — are mistakes shared equally?</dd></div>
    </dl>
  </section>

  <section class="fc-section fc-split">
    <h2 class="fc-h2">About the data</h2>
    <div>
      <p class="fc-strong-p">The <strong>German Credit</strong> dataset (OpenML <code>credit-g</code>) is a public
        stand-in for micro-finance lending data. Each row is an applicant; the target is whether they
        were rated a bad credit risk.</p>
      <p>The protected attribute is <strong>sex</strong>, derived from the applicant's personal status.
        With roughly 300 test rows split across two groups, fairness estimates carry real sampling
        noise — which is why every result is reported, not just the good ones.</p>
    </div>
  </section>

  <section class="fc-cta">
    <h2>Accuracy and fairness, reported together.</h2>
    <a class="fc-btn fc-btn-light" href="?view=run">Run the pipeline <span aria-hidden="true">→</span></a>
  </section>
</div>
"""


def top_bar(active: str) -> None:
    links = "".join(
        f'<a href="?view={view}"{" aria-current=page" if view == active else ""}>{label}</a>'
        for view, label in (("overview", "Overview"), ("run", "Run pipeline"))
    )
    st.html(
        '<div class="fc fc-top"><div class="fc-top-inner">'
        '<a class="fc-brand" href="?view=overview"><span class="fc-logo" aria-hidden="true"></span>Fair Credit</a>'
        f'<nav class="fc-nav" aria-label="Main">{links}</nav></div></div>'
    )


def table_html(title: str, note: str, df: pd.DataFrame) -> str:
    table = df.reset_index().to_html(index=False, border=0, classes="fc-table", float_format="{:.3f}".format)
    return (
        f'<div class="fc-table-head"><h2 class="fc-h2">{title}</h2><span class="fc-note">{note}</span></div>'
        f'<div class="fc-scroll">{table}</div>'
    )


def show_run() -> None:
    st.html(
        '<div class="fc"><section class="fc-hero fc-compact">'
        '<p class="fc-eyebrow">Baseline · Reweighed · Reweighed + Threshold</p>'
        "<h1>Run the pipeline</h1>"
        '<p class="fc-lead">German Credit dataset (public proxy for micro-finance lending data) · '
        "protected attribute: sex</p></section></div>"
    )

    with st.container(key="controls", horizontal=True, vertical_alignment="bottom", gap="medium"):
        test_size = st.slider("Test set size", 0.2, 0.4, 0.3, 0.05)
        clicked = st.button("Run pipeline", type="primary")

    if not clicked:
        st.html(
            '<p class="fc fc-empty">Click <strong>Run pipeline</strong> to train the three models '
            "and compare metrics live.</p>"
        )
        return

    try:
        with st.spinner("Training baseline → reweighed → threshold-adjusted models..."):
            results, group_sizes, ranking = run_pipeline(test_size)
    except OSError:  # URLError, HTTPError and timeouts from the first-run OpenML download
        logging.exception("OpenML download failed")
        st.error(
            "Couldn't download the German Credit dataset from OpenML. "
            "Check your internet connection and run the pipeline again."
        )
        return

    fairness = results[["Demographic Parity Diff", "Equalized Odds Diff"]]
    st.html(
        '<div class="fc">'
        + table_html("Accuracy", "higher is better", results[["AUC", "KS", "Recall (default)"]])
        + table_html("Fairness", "closer to 0 is better", fairness)
        + "</div>"
    )
    # Short legend labels so the chart legend fits at phone width.
    st.bar_chart(fairness.set_axis(["Parity gap", "Odds gap"], axis=1), horizontal=True, stack=False)

    groups = "".join(f"<div><dt>{g.capitalize()}</dt><dd>{n} applicants</dd></div>" for g, n in group_sizes.items())
    st.html(
        '<div class="fc">'
        '<div class="fc-table-head"><h2 class="fc-h2">Test-set group sizes</h2>'
        f'<span class="fc-note">n = {group_sizes.sum()}</span></div>'
        f'<dl class="fc-defs">{groups}</dl>'
        '<div class="fc-callout"><span class="fc-label">NOTE</span>'
        "<p>Fairness metrics don't necessarily shrink at every stage — with a small, noisy test split, "
        "a fairness technique can even look worse on one metric. That's why Slide 4 reports accuracy "
        "AND fairness together instead of assuming the intervention worked.</p></div></div>"
    )
    st.html(
        '<div class="fc"><div class="fc-table-head"><h2 class="fc-h2">Per-applicant risk ranking</h2>'
        '<span class="fc-note">riskiest first · approve &lt; 0.30 ≤ refer &lt; 0.60 ≤ decline</span></div></div>'
    )
    st.dataframe(ranking, hide_index=True, height=420)


st.html(Path(__file__).with_name("style.css"))
view = "run" if st.query_params.get("view") == "run" else "overview"
top_bar(view)
with st.container(key="page"):
    if view == "run":
        show_run()
    else:
        st.html(OVERVIEW_HTML)
