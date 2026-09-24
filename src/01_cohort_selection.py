"""
01_cohort_selection.py

Builds the COVID+ and COVID- eligible cohorts (pre-matching): new-onset
pneumonia determination, demographics assembly, comorbidity assembly, and
hospitalization stratification (for COVID+).

+ and - arms differ only in which column anchors the "index date" 
(covid_date vs. index_date) and in the hospitalization stratification 
step, which only applies to the COVID+ arm.

Expects OMOP-derived CSV extracts as input (see sql/ for queries used).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from comorbidity_concepts import (
    ALL_COMORBIDITIES,
    ETHNICITY_HISPANIC_CONCEPT,
    ETHNICITY_NOT_HISPANIC_CONCEPT,
    GENDER_CONCEPT_MAP,
    RACE_CONCEPT_MAP,
    TOBACCO_USE_OBSERVATION_CONCEPT,
)

NEW_ONSET_MIN_DAYS = 30
HOSP_WINDOW_DAYS = 7
HOSP_VISIT_CONCEPTS = [262, 32037, 9201]  # inpatient / ER / hospital encounter


# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

def load_return_filtered_cohort(path: Path, covid_status: int) -> pd.DataFrame:
    """Load return-filtered extract (output of sql/01_cohort_identification.sql: 
    patients who returned to the health system >=30 days after their index date)."""
    df = pd.read_csv(path)
    for col in ["covid_date", "index_date", "return_date", "death_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["covid_status"] = covid_status
    return df


def index_column(covid_status: int) -> str:
    """COVID+ patients are indexed on their positive test date; COVID-
    patients are indexed on their first qualifying visit (index_date)."""
    return "covid_date" if covid_status == 1 else "index_date"


# ---------------------------------------------------------------------------
# Step 1: new-onset pneumonia determination
# ---------------------------------------------------------------------------

def flag_new_onset_pneumonia(cohort: pd.DataFrame, conditions: pd.DataFrame, index_col: str):
    """
    Assign each patient to one of three buckets relative to their index date:
      - prior pneumonia (excluded from the analytic cohort)
      - new-onset pneumonia (diagnosis >= NEW_ONSET_MIN_DAYS after index)
      - no pneumonia (no diagnosis, or one within the 30-day window
        immediately after index which is treated as related)
    """
    first_dx = (
        conditions.sort_values(["person_id", "condition_start_datetime"])
        .drop_duplicates("person_id", keep="first")
        .set_index("person_id")["condition_start_datetime"]
    )
    first_dx = pd.to_datetime(first_dx)

    cohort = cohort.copy()
    cohort["pneumonia_date"] = cohort["person_id"].map(first_dx)

    prior_pneumonia = cohort[cohort["pneumonia_date"] < cohort[index_col]]
    no_prior = cohort.loc[~cohort["person_id"].isin(prior_pneumonia["person_id"])].copy()

    diagnosed = no_prior[no_prior["person_id"].isin(first_dx.index)].copy()
    diagnosed["days_to_condition"] = (diagnosed["pneumonia_date"] - diagnosed[index_col]).dt.days
    new_onset_ids = set(diagnosed.loc[diagnosed["days_to_condition"] >= NEW_ONSET_MIN_DAYS, "person_id"])

    no_prior["new_onset_pneumonia"] = no_prior["person_id"].isin(new_onset_ids).astype(int)
    return no_prior, prior_pneumonia


# ---------------------------------------------------------------------------
# Step 2: demographics
# ---------------------------------------------------------------------------

def merge_demographics(cohort: pd.DataFrame, demo_raw: pd.DataFrame, index_col: str) -> pd.DataFrame:
    """
    Attach age (computed at index date from birth year/month), sex,
    race, and ethnicity from OMOP PERSON-table concept IDs.
    `demo_raw` columns expected: PERSON_ID, YEAR_OF_BIRTH, MONTH_OF_BIRTH,
    GENDER_CONCEPT_ID, ETHNICITY_CONCEPT_ID, RACE_CONCEPT_ID, DEATH_DATETIME.
    """
    demo = demo_raw.copy()
    demo["DEATH_DATETIME"] = pd.to_datetime(demo["DEATH_DATETIME"], errors="coerce")

    index_by_person = cohort.set_index("person_id")[index_col]
    demo["index_date"] = demo["PERSON_ID"].map(index_by_person)
    demo["age"] = (
        demo["index_date"].dt.year
        - demo["YEAR_OF_BIRTH"]
        - (demo["index_date"].dt.month < demo["MONTH_OF_BIRTH"]).astype(int)
    )

    demo["gender"] = demo["GENDER_CONCEPT_ID"].map(GENDER_CONCEPT_MAP)
    demo["hispanic"] = (demo["ETHNICITY_CONCEPT_ID"] == ETHNICITY_HISPANIC_CONCEPT).astype(int)
    demo["non_hispanic"] = (demo["ETHNICITY_CONCEPT_ID"] != ETHNICITY_HISPANIC_CONCEPT).astype(int)

    race_dummies = demo["RACE_CONCEPT_ID"].map(RACE_CONCEPT_MAP).fillna("other")
    for race in ["asian", "black", "white", "other"]:
        demo[race] = (race_dummies == race).astype(int)

    demo = demo.rename(columns={"PERSON_ID": "person_id", "DEATH_DATETIME": "death_date"})
    keep = ["person_id", "gender", "hispanic", "non_hispanic",
            "asian", "black", "white", "other", "death_date"]
    # `age` only pulled from demo table if input cohort doesn't already have one
    if "age" not in cohort.columns:
        keep.append("age")
    return cohort.merge(demo[keep], on="person_id", how="left")


# ---------------------------------------------------------------------------
# Step 3: comorbidities
# ---------------------------------------------------------------------------

def merge_comorbidities(cohort: pd.DataFrame, comorbidity_tables: dict, index_col: str) -> pd.DataFrame:
    """
    Attach 0/1 flag per comorbidity. Conditions only counted if earliest recorded 
    date is on/before index date -- conditions recorded after are set to null.

    `comorbidity_tables`: {condition_name: DataFrame[person_id, start_date]}
    """
    cohort = cohort.copy()
    for condition, table in comorbidity_tables.items():
        first_date = (
            table.sort_values(["person_id", "start_date"])
            .drop_duplicates("person_id", keep="first")
            .set_index("person_id")["start_date"]
        )
        first_date = pd.to_datetime(first_date)
        cohort[condition] = cohort["person_id"].map(first_date)
        after_index = cohort[condition] > cohort[index_col]
        cohort.loc[after_index, condition] = pd.NaT
        cohort[condition] = cohort[condition].notna().astype(int)

    # combined cardiovascular flag used in the published Table 1
    cv_cols = [c for c in ["CHF", "CAD", "MI"] if c in cohort.columns]
    if cv_cols:
        cohort["CHF_CAD_MI"] = cohort[cv_cols].eq(1).any(axis=1).astype(int)
    return cohort


# ---------------------------------------------------------------------------
# Step 4: hospitalization stratification (COVID+ arm only)
# ---------------------------------------------------------------------------

def stratify_hospitalization(cohort: pd.DataFrame, visits: pd.DataFrame, index_col: str) -> pd.DataFrame:
    """
    Flags patients hospitalized within HOSP_WINDOW_DAYS of their index date:
    (hospitalization "due to COVID-19").

    `visits`: VISIT_OCCURRENCE extract with VISIT_START_DATETIME,
    VISIT_END_DATETIME, VISIT_CONCEPT_ID, PERSON_ID.
    """
    visits = visits.dropna(subset=["VISIT_START_DATETIME", "VISIT_END_DATETIME"]).copy()
    visits["VISIT_START_DATETIME"] = pd.to_datetime(visits["VISIT_START_DATETIME"])

    hosp_visits = visits[
        visits["VISIT_CONCEPT_ID"].isin(HOSP_VISIT_CONCEPTS)
        & visits["PERSON_ID"].isin(cohort["person_id"])
    ].copy()

    index_by_person = cohort.set_index("person_id")[index_col]
    hosp_visits["index_date"] = hosp_visits["PERSON_ID"].map(index_by_person)

    window = pd.Timedelta(days=HOSP_WINDOW_DAYS)
    hosp_visits = hosp_visits[
        hosp_visits["VISIT_START_DATETIME"].between(
            hosp_visits["index_date"] - window, hosp_visits["index_date"] + window
        )
    ]

    cohort = cohort.copy()
    cohort["hospitalized"] = cohort["person_id"].isin(hosp_visits["PERSON_ID"]).astype(int)
    return cohort


# ---------------------------------------------------------------------------
# Table 1 summary
# ---------------------------------------------------------------------------

def summarize_table1(cohort: pd.DataFrame) -> pd.DataFrame:
    """Descriptive counts/percentages"""
    n = len(cohort)
    rows = [("Total persons", n, 100.0)]
    rows.append(("Age, mean (SD)", cohort["age"].mean(), cohort["age"].std()))
    for label, col in [("Male", "gender"), ("Hispanic", "hispanic"),
                       ("Black", "black"), ("White", "white"), ("Asian", "asian")]:
        if col in cohort.columns:
            count = (cohort[col] == 1).sum() if col != "gender" else (cohort[col] == "male").sum()
            rows.append((label, int(count), round(100 * count / n, 2)))
    for condition in list(ALL_COMORBIDITIES) + ["smoking"]:
        if condition in cohort.columns:
            count = int(cohort[condition].sum())
            rows.append((condition, count, round(100 * count / n, 2)))
    return pd.DataFrame(rows, columns=["characteristic", "n", "pct_or_value"])


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_cohort(covid_status: int, data_dir: Path) -> dict:
    index_col = index_column(covid_status)
    arm = "pos" if covid_status == 1 else "neg"

    returns = load_return_filtered_cohort(data_dir / f"returns_{arm}.csv", covid_status)
    conditions = pd.read_csv(data_dir / "pneumonia_condition_occurrence.csv")
    conditions = conditions.rename(columns={
        "PERSON_ID": "person_id", "CONDITION_START_DATETIME": "condition_start_datetime"
    })

    no_prior, prior_pneumonia = flag_new_onset_pneumonia(returns, conditions, index_col)

    demo_raw = pd.read_csv(data_dir / f"demographics_{arm}.csv")
    cohort = merge_demographics(no_prior, demo_raw, index_col)

    comorbidity_tables = {}
    for condition in ALL_COMORBIDITIES:
        raw = pd.read_csv(data_dir / f"comorbidity_{condition.lower()}.csv")
        comorbidity_tables[condition] = raw.rename(columns={
            "PERSON_ID": "person_id", "CONDITION_START_DATETIME": "start_date"
        })[["person_id", "start_date"]]
    smoking_raw = pd.read_csv(data_dir / "comorbidity_smoking.csv")
    comorbidity_tables["smoking"] = smoking_raw.rename(columns={
        "PERSON_ID": "person_id", "OBSERVATION_DATETIME": "start_date"
    })[["person_id", "start_date"]]

    cohort = merge_comorbidities(cohort, comorbidity_tables, index_col)

    if covid_status == 1:
        visits = pd.read_csv(data_dir / "visit_occurrence.csv")
        cohort = stratify_hospitalization(cohort, visits, index_col)

    return {
        "cohort": cohort,
        "prior_pneumonia_excluded": prior_pneumonia,
        "table1": summarize_table1(cohort),
    }


def main():
    parser = argparse.ArgumentParser(description="Build COVID+/COVID- analytic cohorts")
    parser.add_argument("--data-dir", type=Path,
                         default=Path(os.environ.get("PNEUMONIA_DATA_DIR", "data/sample")))
    parser.add_argument("--out-dir", type=Path, default=Path("results/cohort_selection"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for covid_status, label in [(1, "positive"), (0, "negative")]:
        result = build_cohort(covid_status, args.data_dir)
        result["cohort"].to_csv(args.out_dir / f"cohort_{label}.csv", index=False)
        result["table1"].to_csv(args.out_dir / f"table1_{label}.csv", index=False)
        print(f"{label} cohort: {len(result['cohort'])} patients "
              f"({result['cohort']['new_onset_pneumonia'].sum()} new-onset pneumonia)")


if __name__ == "__main__":
    main()
