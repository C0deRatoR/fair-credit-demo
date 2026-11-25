# Fair Credit Default Demo

A credit-risk model on the German Credit dataset (OpenML `credit-g`) that runs in three stages: Baseline, then Reweighed, then Reweighed + Threshold. Each stage reports AUC, recall, KS, and the fairness gaps by sex (demographic parity and equalized odds).

## Run

Command-line version:

```sh
uv run demo.py
```

Streamlit UI:

```sh
uv run --with streamlit --with pandas --with numpy --with scipy \
    --with scikit-learn --with fairlearn streamlit run app.py
```

The dataset is downloaded from OpenML on the first run.
