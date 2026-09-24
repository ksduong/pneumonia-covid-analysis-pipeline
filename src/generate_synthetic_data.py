"""
Generate synthetic cohort data matching the schema of the real analysis pipeline.
All values fabricated by numpy's random generator.

Produces two files:
  data/sample/synthetic_raw_cohort.csv       -> input for 01_cohort_selection.py / 02_propensity_matching.py
  data/sample/synthetic_matched_analytic.csv -> input for 03_survival_analysis.py / 04_cox_models.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(seed=42)

N_POS = 500     # synthetic COVID+ cohort size (real: 39,805)
N_NEG = 1000    # synthetic COVID- cohort size (real: 79,610)

# prevalence targets pulled from Table 1; (positive_cohort_rate, negative_cohort_rate)
COMORBIDITY_RATES = {
    "obesity":      (0.2749, 0.1529),
    "smoking":      (0.1796, 0.1064),
    "hypertension": (0.3099, 0.1795),
    "diabetes":     (0.1759, 0.0912),
    "CKD":          (0.0893, 0.0337),
    "CAD":          (0.0883, 0.0430),  # CVD bucket in the paper; used as a stand-in for CAD/CHF/MI
    "asthma":       (0.1992, 0.0839),
    "COPD":         (0.0396, 0.0146),
}
RACE_RATES = {  # (black, white, hispanic, other)
    "pos": (0.3091, 0.1094, 0.4322, 0.0423),
    "neg": (0.2967, 0.1119, 0.4315, 0.0349),
}
MALE_RATE = {"pos": 0.3850, "neg": 0.4008}
AGE_PARAMS = {"pos": (41.43, 23.74), "neg": (41.27, 24.08)}

START_DATE = datetime(2020, 3, 1)
END_DATE = datetime(2024, 1, 31)


def _random_dates(n, start=START_DATE, end=END_DATE):
    delta_days = (end - start).days
    offsets = rng.integers(0, delta_days, size=n)
    return [start + timedelta(days=int(d)) for d in offsets]


def _make_cohort(n, status, person_id_start):
    status_key = "pos" if status == 1 else "neg"
    age_mean, age_sd = AGE_PARAMS[status_key]
    ages = np.clip(rng.normal(age_mean, age_sd, size=n), 0, 100).round().astype(int)

    black_r, white_r, hisp_r, other_r = RACE_RATES[status_key]
    race_roll = rng.random(n)
    black = (race_roll < black_r).astype(int)
    white = ((race_roll >= black_r) & (race_roll < black_r + white_r)).astype(int)
    hispanic = ((race_roll >= black_r + white_r) & (race_roll < black_r + white_r + hisp_r)).astype(int)
    other = ((race_roll >= 1 - other_r)).astype(int)
    non_hispanic = 1 - hispanic
    asian = np.zeros(n, dtype=int)  # not separately modeled; folded into "other" bucket for simplicity

    gender = (rng.random(n) < MALE_RATE[status_key]).astype(int)  # 1 = male

    index_dates = _random_dates(n)
    # COVID+ patients are indexed on their positive test, so covid_date must
    # equal index_date -- otherwise return_date can fall before covid_date and
    # produce negative follow-up durations downstream
    covid_dates = list(index_dates) if status == 1 else [pd.NaT] * n
    return_offsets = rng.integers(30, 900, size=n)
    return_dates = [idx + timedelta(days=int(o)) for idx, o in zip(index_dates, return_offsets)]
    death_dates = [pd.NaT] * n
    death_mask = rng.random(n) < 0.03
    for i in np.where(death_mask)[0]:
        death_dates[i] = return_dates[i] + timedelta(days=int(rng.integers(1, 60)))

    df = pd.DataFrame({
        "person_id": np.arange(person_id_start, person_id_start + n),
        "age": ages,
        "covid_status": status,
        "covid_date": covid_dates,
        "index_date": index_dates,
        "return_date": return_dates,
        "death_date": death_dates,
        "gender": gender,
        "hispanic": hispanic,
        "non_hispanic": non_hispanic,
        "asian": asian,
        "black": black,
        "white": white,
        "other": other,
    })

    for cond, (pos_r, neg_r) in COMORBIDITY_RATES.items():
        rate = pos_r if status == 1 else neg_r
        df[cond] = (rng.random(n) < rate).astype(int)
    for extra in ["CHF", "MI"]:
        df[extra] = (rng.random(n) < 0.03).astype(int)

    df["follow_up_time_days"] = (pd.to_datetime(df["return_date"]) - pd.to_datetime(df["index_date"])).dt.days
    df["follow_up_time_seconds"] = df["follow_up_time_days"] * 86400
    df["ps"] = rng.uniform(0.01, 0.5, size=n).round(4)
    df["ps_logit"] = np.log(df["ps"] / (1 - df["ps"])).round(4)
    return df


def build_raw_cohort():
    pos = _make_cohort(N_POS, status=1, person_id_start=10_000_000)
    neg = _make_cohort(N_NEG, status=0, person_id_start=20_000_000)
    return pd.concat([pos, neg], ignore_index=True)


def build_matched_analytic(raw_df):
    """Simulate the post-matching, post-outcome-assignment analytic table by sampling a 1:2 case:control
    set from the synthetic raw cohort and fabricating an outcome/duration.
    """
    pos = raw_df[raw_df.covid_status == 1].copy()
    neg_pool = raw_df[raw_df.covid_status == 0].copy()

    matched_rows = []
    for _, case in pos.iterrows():
        controls = neg_pool.sample(n=2, replace=len(neg_pool) < 2, random_state=int(case.person_id) % (2**32))
        matched_rows.append(case)
        matched_rows.extend([controls.iloc[0], controls.iloc[1]])
    matched = pd.DataFrame(matched_rows).reset_index(drop=True)

    # fabricate outcome + time-to-event, with a higher event rate for covid_status==1
    # mirrors paper's finding in direction; magnitude is illustrative
    event_rate = np.where(matched["covid_status"] == 1, 0.045, 0.018)
    matched["new_pneumonia_status"] = (rng.random(len(matched)) < event_rate).astype(int)
    max_duration = np.clip(matched["follow_up_time_days"] / 30.0, 0.5, 46)
    matched["duration"] = rng.uniform(0.5, 1, size=len(matched)) * max_duration
    # events need >=1 month (30 days) to qualify as "new onset"
    matched.loc[matched["new_pneumonia_status"] == 1, "duration"] = matched.loc[
        matched["new_pneumonia_status"] == 1, "duration"
    ].clip(lower=1.1)
    matched["duration"] = matched["duration"].round(2)

    index_date = pd.to_datetime(matched["covid_date"]).fillna(pd.to_datetime(matched["index_date"]))
    event_date = index_date + pd.to_timedelta((matched["duration"] * 30).round(), unit="D")
    matched["new_pneumonia_date"] = event_date.where(matched["new_pneumonia_status"] == 1, pd.NaT)
    return matched


def build_omop_style_extracts(raw_df: pd.DataFrame) -> dict:
    """
    Reverse-engineer raw_df into OMOP-style extracts that src/01_cohort_selection.py references.
    """
    from comorbidity_concepts import ALL_COMORBIDITIES, GENDER_CONCEPT_MAP  # noqa: F401 (documents source of truth)

    n = len(raw_df)
    extracts = {}

    # returns_{arm}.csv: base cohort, no comorbidity/demographic columns
    base_cols = ["person_id", "age", "covid_status", "covid_date", "index_date",
                 "return_date", "death_date", "follow_up_time_seconds", "follow_up_time_days"]
    for status, arm in [(1, "pos"), (0, "neg")]:
        extracts[f"returns_{arm}"] = raw_df.loc[raw_df.covid_status == status, base_cols].copy()

    # demographics_{arm}.csv: concept-coded, like OMOP's PERSON table would return
    race_concept_reverse = {"black": 8516, "white": 8527, "asian": 8515, "other": 0}
    for status, arm in [(1, "pos"), (0, "neg")]:
        sub = raw_df[raw_df.covid_status == status].copy()
        race_col = np.select(
            [sub["black"] == 1, sub["white"] == 1, sub["asian"] == 1],
            [race_concept_reverse["black"], race_concept_reverse["white"], race_concept_reverse["asian"]],
            default=race_concept_reverse["other"],
        )
        index_dates = pd.to_datetime(sub["covid_date"] if status == 1 else sub["index_date"])
        birth_year = index_dates.dt.year - sub["age"]
        demo = pd.DataFrame({
            "PERSON_ID": sub["person_id"],
            "YEAR_OF_BIRTH": birth_year,
            "MONTH_OF_BIRTH": rng.integers(1, 13, size=len(sub)),
            "GENDER_CONCEPT_ID": np.where(sub["gender"] == 1, 8507, 8532),  # 1=male here
            "ETHNICITY_CONCEPT_ID": np.where(sub["hispanic"] == 1, 38003563, 38003564),
            "RACE_CONCEPT_ID": race_col,
            "DEATH_DATETIME": sub["death_date"],
        })
        extracts[f"demographics_{arm}"] = demo

    # comorbidity_{condition}.csv: person_id + start date, only for flagged patients
    for condition in ALL_COMORBIDITIES:
        if condition not in raw_df.columns:
            continue
        flagged = raw_df[raw_df[condition] == 1].copy()
        index_dates = pd.to_datetime(np.where(flagged.covid_status == 1, flagged.covid_date, flagged.index_date))
        # comorbidity onset some time before the index date (preexisting condition)
        onset = pd.to_datetime(index_dates) - pd.to_timedelta(rng.integers(30, 2000, size=len(flagged)), unit="D")
        extracts[f"comorbidity_{condition.lower()}"] = pd.DataFrame({
            "PERSON_ID": flagged["person_id"],
            "CONDITION_START_DATETIME": onset,
        })

    # smoking -> OBSERVATION table
    flagged = raw_df[raw_df["smoking"] == 1].copy()
    index_dates = pd.to_datetime(np.where(flagged.covid_status == 1, flagged.covid_date, flagged.index_date))
    onset = pd.to_datetime(index_dates) - pd.to_timedelta(rng.integers(30, 2000, size=len(flagged)), unit="D")
    extracts["comorbidity_smoking"] = pd.DataFrame({
        "PERSON_ID": flagged["person_id"],
        "OBSERVATION_DATETIME": onset,
    })

    # visit_occurrence.csv: hosp visits for a subset of COVID+ patients
    hosp_concepts = [262, 32037, 9201]
    pos = raw_df[raw_df.covid_status == 1]
    hosp_sample = pos.sample(frac=0.15, random_state=1)  # ~15% hospitalized, in the ballpark of the paper
    covid_dates = pd.to_datetime(hosp_sample["covid_date"])
    visit_start = covid_dates + pd.to_timedelta(rng.integers(-3, 4, size=len(hosp_sample)), unit="D")
    visit_end = visit_start + pd.to_timedelta(rng.integers(1, 14, size=len(hosp_sample)), unit="D")
    extracts["visit_occurrence"] = pd.DataFrame({
        "PERSON_ID": hosp_sample["person_id"],
        "VISIT_CONCEPT_ID": rng.choice(hosp_concepts, size=len(hosp_sample)),
        "VISIT_START_DATETIME": visit_start,
        "VISIT_END_DATETIME": visit_end,
    })

    # pneumonia_condition_occurrence.csv: new-onset events
    matched = build_matched_analytic(raw_df)
    events = matched[matched["new_pneumonia_status"] == 1]
    extracts["pneumonia_condition_occurrence"] = pd.DataFrame({
        "PERSON_ID": events["person_id"],
        "CONDITION_START_DATETIME": events["new_pneumonia_date"],
    })

    return extracts


if __name__ == "__main__":
    raw = build_raw_cohort()

    omop_dir = Path("data/sample/omop_extracts")
    omop_dir.mkdir(parents=True, exist_ok=True)
    for name, df in build_omop_style_extracts(raw).items():
        df.to_csv(omop_dir / f"{name}.csv", index=False)
        print(f"Wrote omop_extracts/{name}.csv: {len(df)} rows")
