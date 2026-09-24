"""
04_cox_models.py

Univariate and multivariate Cox proportional hazards models for new-onset
pneumonia (see paper's Tables 2-3).
"""

import argparse
from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter

DURATION_COL = "duration"
EVENT_COL = "new_pneumonia_status"

# gender is coded 1=female in the source data; the multivariate model uses
# male as the reference level, so it's flipped here (0=female, 1=male)
# -- this matches the reference-level choice used for the published Table 2.
MULTIVARIATE_COVARIATES = [
    "age", "covid_status", "gender", "CAD", "CHF", "CKD",
    "COPD", "diabetes", "hypertension", "MI", "smoking",
]
UNIVARIATE_COVARIATES = [
    "age", "covid_status", "gender", "CAD", "CHF", "CKD",
    "COPD", "diabetes", "hypertension", "MI", "obesity", "smoking",
]


def run_univariate(df: pd.DataFrame) -> pd.DataFrame:
    """Fits one Cox model per covariate; returns summary table."""
    rows = []
    for covariate in UNIVARIATE_COVARIATES:
        cph = CoxPHFitter()
        cph.fit(df, duration_col=DURATION_COL, event_col=EVENT_COL, formula=covariate)
        summary = cph.summary.reset_index()
        summary["covariate"] = covariate
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def run_multivariate(df: pd.DataFrame) -> CoxPHFitter:
    df = df.copy()
    df["gender"] = 1 - df["gender"]  # reference level -> male
    formula = " + ".join(MULTIVARIATE_COVARIATES)

    cph = CoxPHFitter()
    cph.fit(df, duration_col=DURATION_COL, event_col=EVENT_COL, formula=formula)
    return cph


def main():
    parser = argparse.ArgumentParser(description="Fit univariate and multivariate Cox models")
    parser.add_argument("--input", type=Path, default=Path("results/matched_binary_data.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    univariate_summary = run_univariate(df)
    univariate_summary.to_csv(args.out_dir / "cox_univariate.csv", index=False)

    multivariate_model = run_multivariate(df)
    multivariate_model.summary.to_csv(args.out_dir / "cox_multivariate.csv")
    multivariate_model.print_summary()


if __name__ == "__main__":
    main()
