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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
