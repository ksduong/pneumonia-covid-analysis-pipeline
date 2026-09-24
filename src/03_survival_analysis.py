"""
03_survival_analysis.py

Builds the final analytic (survival) table from the matched COVID+/COVID-
cohorts and plots the cumulative incidence of new-onset pneumonia, censored
at 46 months (see paper's Figure 2).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

FOLLOWUP_CUTOFF_MONTHS = 46
NEW_ONSET_MIN_DAYS = 30


def build_analytic_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the matched cohort and builds the table used for survival analysis: 
    one row per patient, with `duration` (months from index to event or censoring) 
    and `new_pneumonia_status` (0/1) column for KaplanMeierFitter / CoxPHFitter.
    """
    data = data.copy()
    for col in ["pneumonia_date", "covid_date", "index_date", "return_date", "death_date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")

    data = data.rename(columns={
        "follow_up_time_seconds": "obs_time_sec",
        "follow_up_time_days": "obs_time_days",
    })

    data = data.sort_values(["person_id", "pneumonia_date"]).drop_duplicates("person_id", keep="first")

    # unify index date across arms
    data["index"] = np.where(data["covid_status"] == 1, data["covid_date"], data["index_date"])

    # rederive new-onset status/date relative to the unified index
    days_to_condition = (data["pneumonia_date"] - data["index"]).dt.days
    is_new_onset = data["pneumonia_date"].notna() & (days_to_condition >= NEW_ONSET_MIN_DAYS)
    data["new_pneumonia_date"] = data["pneumonia_date"].where(is_new_onset)
    data["new_pneumonia_status"] = is_new_onset.astype(int)

    data["duration"] = np.where(
        data["new_pneumonia_status"] == 1,
        (data["new_pneumonia_date"] - data["index"]).dt.days / 30,
        (data["return_date"] - data["index"]).dt.days / 30,
    )
    data["duration"] = data["duration"].clip(upper=FOLLOWUP_CUTOFF_MONTHS)

    keep = ["person_id", "age", "covid_status", "new_pneumonia_status", "duration",
            "new_pneumonia_date", "index", "covid_date", "index_date", "return_date",
            "death_date", "gender", "hispanic", "non_hispanic", "asian", "black",
            "white", "other", "asthma", "CAD", "CHF", "CKD", "COPD", "diabetes",
            "hypertension", "MI", "obesity", "smoking", "obs_time_sec",
            "obs_time_days", "ps", "ps_logit"]
    return data[[c for c in keep if c in data.columns]]


def plot_cumulative_incidence(data: pd.DataFrame, out_path: Path):
    covid_positive = data[data["covid_status"] == 1]
    covid_negative = data[data["covid_status"] == 0]

    kmf_covid = KaplanMeierFitter()
    kmf_covid.fit(durations=covid_positive["duration"],
                  event_observed=covid_positive["new_pneumonia_status"], label="COVID+")

    kmf_control = KaplanMeierFitter()
    kmf_control.fit(durations=covid_negative["duration"],
                     event_observed=covid_negative["new_pneumonia_status"], label="COVID-")

    fig, ax = plt.subplots(figsize=(10, 6))
    kmf_covid.plot_cumulative_density(ax=ax, linewidth=2, ci_show=False, color="teal", label="COVID+")
    kmf_control.plot_cumulative_density(ax=ax, linewidth=2, ci_show=False, color="grey", label="COVID-")

    ax.set_xlabel("Time since index encounter (months)", fontsize=14)
    ax.set_ylabel("Cumulative Incidence", fontsize=14)
    ax.legend(fontsize=15)
    ax.tick_params(axis="both", labelsize=15)

    xticks = list(range(0, FOLLOWUP_CUTOFF_MONTHS + 5, 5))
    ax.set_xticks(xticks)
    ax.set_xlim([0, FOLLOWUP_CUTOFF_MONTHS + 4])

    ax.set_title("Cumulative Incidence Function for New-Onset Pneumonia", fontsize=17)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1, symbol="%"))

    add_at_risk_counts(kmf_covid, kmf_control, ax=ax, fontsize=12, ypos=-0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Survival / cumulative incidence analysis")
    parser.add_argument("--input", type=Path, default=Path("results/matching/matched_cohort_1to2.csv"),
                         help="Matched cohort from 02_propensity_matching.py")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    matched = pd.read_csv(args.input)
    analytic = build_analytic_table(matched)
    analytic.to_csv(args.out_dir / "matched_binary_data.csv", index=False)
    plot_cumulative_incidence(analytic, args.out_dir / "figures" / "cumulative_incidence.png")

    print(f"Analytic table: {len(analytic)} patients, "
          f"{analytic['new_pneumonia_status'].sum()} new-onset pneumonia events")


if __name__ == "__main__":
    main()
