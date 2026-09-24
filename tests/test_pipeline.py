"""
Basic sanity tests, run against the synthetic data. 
`pytest tests/`
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlib import import_module

cohort_selection = import_module("01_cohort_selection")
matching = import_module("02_propensity_matching")
survival = import_module("03_survival_analysis")


def test_flag_new_onset_pneumonia_excludes_prior_condition():
    cohort = pd.DataFrame({
        "person_id": [1, 2],
        "covid_date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
    })
    conditions = pd.DataFrame({
        "person_id": [1],
        "condition_start_datetime": pd.to_datetime(["2020-06-01"]),  # before index -> prior
    })
    no_prior, prior = cohort_selection.flag_new_onset_pneumonia(cohort, conditions, "covid_date")
    assert list(prior["person_id"]) == [1]
    assert list(no_prior["person_id"]) == [2]
    assert no_prior["new_onset_pneumonia"].tolist() == [0]


def test_flag_new_onset_pneumonia_requires_30_day_gap():
    cohort = pd.DataFrame({
        "person_id": [1, 2],
        "covid_date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
    })
    conditions = pd.DataFrame({
        "person_id": [1, 2],
        "condition_start_datetime": pd.to_datetime(["2021-01-10", "2021-03-01"]),
    })
    no_prior, prior = cohort_selection.flag_new_onset_pneumonia(cohort, conditions, "covid_date")
    assert len(prior) == 0
    result = no_prior.set_index("person_id")["new_onset_pneumonia"]
    assert result[1] == 0  # only 9 days out -- doesn't qualify
    assert result[2] == 1  # ~59 days out -- qualifies


def test_merge_comorbidities_excludes_post_index_diagnoses():
    cohort = pd.DataFrame({
        "person_id": [1, 2],
        "covid_date": pd.to_datetime(["2021-06-01", "2021-06-01"]),
    })
    comorbidity_tables = {
        "diabetes": pd.DataFrame({
            "person_id": [1, 2],
            "start_date": pd.to_datetime(["2020-01-01", "2022-01-01"]),  # pt2's is after index
        })
    }
    result = cohort_selection.merge_comorbidities(cohort, comorbidity_tables, "covid_date")
    assert result.set_index("person_id")["diabetes"].to_dict() == {1: 1, 2: 0}


def test_match_1_to_2_produces_two_controls_per_case():
    rng = np.random.default_rng(0)
    n_pos, n_neg = 20, 100
    cohort = pd.DataFrame({
        "person_id": range(n_pos + n_neg),
        "covid_status": [1] * n_pos + [0] * n_neg,
        "age": rng.integers(20, 80, n_pos + n_neg),
        "gender": rng.integers(0, 2, n_pos + n_neg),
        "hispanic": rng.integers(0, 2, n_pos + n_neg),
        "non_hispanic": rng.integers(0, 2, n_pos + n_neg),
        "white": rng.integers(0, 2, n_pos + n_neg),
        "black": rng.integers(0, 2, n_pos + n_neg),
        "asian": rng.integers(0, 2, n_pos + n_neg),
        "other": rng.integers(0, 2, n_pos + n_neg),
        "follow_up_time_days": rng.integers(30, 900, n_pos + n_neg),
    })
    matched = matching.match_1_to_2(cohort)
    n_cases = (matched["covid_status"] == 1).sum()
    n_controls = (matched["covid_status"] == 0).sum()
    # not every case is guaranteed a match (caliper-dependent), but every
    # matched case should have brought exactly two controls with it
    assert n_controls == 2 * n_cases



def test_match_1_to_2_enforces_caliper():
    # two tight clusters on every feature: cases/controls inside a cluster
    # are near each other, but no control is near the lone outlier case
    rows = []
    for i in range(10):
        rows.append(dict(covid_status=1, age=40, follow_up_time_days=300))
    for i in range(40):
        rows.append(dict(covid_status=0, age=40, follow_up_time_days=300))
    rows.append(dict(covid_status=1, age=90, follow_up_time_days=1400))  # outlier case
    cohort = pd.DataFrame(rows)
    cohort["person_id"] = range(len(cohort))
    for col in ["gender", "hispanic", "non_hispanic", "white", "black", "asian", "other"]:
        cohort[col] = 0
    matched = matching.match_1_to_2(cohort)
    ps = matching._fit_propensity_scores(cohort)["ps"]
    caliper = np.std(ps) * matching.CALIPER_MULTIPLIER
    # every matched control is within the caliper of its own case
    cases = matched[matched["covid_status"] == 1]
    assert len(cases) > 0
    for _, case in cases.iterrows():
        for control_idx in case["matched"]:
            assert abs(case["ps"] - ps[control_idx]) <= caliper
    # the outlier case has no control within the caliper, so it stays unmatched
    assert len(cohort) - 1 not in set(cases["person_id"])


def test_events_after_cutoff_are_censored():
    data = pd.DataFrame({
        "person_id": [1, 2],
        "covid_status": [1, 1],
        "covid_date": ["2020-04-01", "2020-04-01"],
        "index_date": [None, None],
        "pneumonia_date": ["2021-04-01", "2024-12-01"],  # ~12 months vs ~56 months
        "return_date": ["2025-01-01", "2025-01-01"],
    })
    out = survival.build_analytic_table(data).set_index("person_id")
    assert out.loc[1, "new_pneumonia_status"] == 1
    assert out.loc[2, "new_pneumonia_status"] == 0  # event after 46 months -> censored
    assert out.loc[2, "duration"] == survival.FOLLOWUP_CUTOFF_MONTHS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
