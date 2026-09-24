"""
02_propensity_matching.py

Propensity-score matching of COVID+ patients to COVID- controls, on age,
sex, race/ethnicity, and observation time.

Two ratios were used over the course of this project:
  - 1:1 matching (`match_1_to_1`): initial approach
  - 1:2 matching (`match_1_to_2`): final method used in paper

Method: logistic-regression propensity score -> logit transform ->
nearest-neighbor candidate search (ball tree) -> greedy matching without
replacement, keeping only controls whose propensity score is within the
caliper (0.15 * SD of the propensity score) of the case's score.

Note: an earlier version passed the caliper to NearestNeighbors as
`radius=`, which `kneighbors()` ignores, so the caliper was never applied.
The caliper is now enforced explicitly in `_within_caliper()`.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

MATCH_FEATURES = [
    "follow_up_time_days", "age", "gender",
    "hispanic", "non_hispanic", "white", "black", "asian", "other",
]
CALIPER_MULTIPLIER = 0.15
N_NEIGHBORS = 200


def build_combined_cohort(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    """Stack the COVID+ and COVID- arms into one frame with a shared
    `follow_up_time_days` computed from each arm's own index date."""
    pos = pos.copy()
    neg = neg.copy()
    pos["gender"] = pos["gender"].map({"male": 0, "female": 1})
    neg["gender"] = neg["gender"].map({"male": 0, "female": 1})

    cohort = pd.concat([pos, neg], ignore_index=True)
    index_date = cohort["covid_date"].fillna(cohort.get("index_date"))
    cohort["follow_up_time_seconds"] = (
        pd.to_datetime(cohort["return_date"]) - pd.to_datetime(index_date)
    ).dt.total_seconds()
    cohort["follow_up_time_days"] = cohort["follow_up_time_seconds"] / 86400
    return cohort


def _fit_propensity_scores(cohort: pd.DataFrame) -> pd.DataFrame:
    cohort = cohort.copy()
    X = cohort[MATCH_FEATURES].fillna(cohort[MATCH_FEATURES].median())
    y = cohort["covid_status"].fillna(cohort["covid_status"].median())

    model = LogisticRegression(max_iter=800)
    model.fit(X, y)
    cohort["ps"] = model.predict_proba(X)[:, 1]
    cohort["ps_logit"] = cohort["ps"].apply(
        lambda p: math.log(p / (1 - p)) if 0 < p < 1 else np.nan
    )
    return cohort


def _within_caliper(cohort: pd.DataFrame, case_idx: int, control_idx: int, caliper: float) -> bool:
    """True if the control's propensity score is within `caliper` of the case's."""
    return abs(cohort.at[case_idx, "ps"] - cohort.at[control_idx, "ps"]) <= caliper


def match_1_to_1(cohort: pd.DataFrame) -> pd.DataFrame:
    cohort = _fit_propensity_scores(cohort)
    caliper = np.std(cohort["ps"]) * CALIPER_MULTIPLIER

    features = ["ps_logit"] + MATCH_FEATURES
    n_neighbors = min(N_NEIGHBORS, len(cohort))
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
    knn.fit(cohort[features])
    _, indexes = knn.kneighbors(cohort[features])

    exclude = set()

    def find_match(row):
        current = row.name
        for idx in indexes[current, :]:
            if idx in cohort.index and idx != current and row["covid_status"] == 1 \
                    and cohort.loc[idx, "covid_status"] == 0 and idx not in exclude \
                    and _within_caliper(cohort, current, idx, caliper):
                exclude.add(idx)
                return int(idx)
        return np.nan

    cohort["matched_1"] = cohort.apply(find_match, axis="columns")
    matched_cases = cohort.dropna(subset=["matched_1"])
    matched_controls = cohort.loc[matched_cases["matched_1"].astype(int)]
    return pd.concat([matched_cases, matched_controls], ignore_index=True)


def match_1_to_2(cohort: pd.DataFrame) -> pd.DataFrame:
    cohort = _fit_propensity_scores(cohort)
    caliper = np.std(cohort["ps"]) * CALIPER_MULTIPLIER

    features = ["ps_logit"] + MATCH_FEATURES
    n_neighbors = min(N_NEIGHBORS, len(cohort))
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
    knn.fit(cohort[features])
    _, indexes = knn.kneighbors(cohort[features])

    exclude = set()

    def find_two_matches(row):
        current = row.name
        matches = []
        if row["covid_status"] != 1:
            return np.nan
        for idx in indexes[current, :]:
            if idx in cohort.index and idx != current \
                    and cohort.loc[idx, "covid_status"] == 0 and idx not in exclude \
                    and _within_caliper(cohort, current, idx, caliper):
                exclude.add(idx)
                matches.append(int(idx))
                if len(matches) == 2:
                    break
        return matches if len(matches) == 2 else np.nan

    cohort["matched"] = cohort.apply(find_two_matches, axis="columns")
    matched_cases = cohort.dropna(subset=["matched"])
    control_ids = matched_cases["matched"].explode().astype(int).unique()
    matched_controls = cohort.loc[control_ids]
    return pd.concat([matched_cases, matched_controls], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Propensity-score match COVID+ to COVID- controls")
    parser.add_argument("--data-dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/matching"))
    parser.add_argument("--ratio", choices=["1:1", "1:2"], default="1:2",
                         help="Match ratio -- 1:2 is the final published method")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pos = pd.read_csv(args.data_dir / "cohort_positive.csv")
    neg = pd.read_csv(args.data_dir / "cohort_negative.csv")
    cohort = build_combined_cohort(pos, neg)

    matched = match_1_to_1(cohort) if args.ratio == "1:1" else match_1_to_2(cohort)
    matched.to_csv(args.out_dir / f"matched_cohort_{args.ratio.replace(':', 'to')}.csv", index=False)

    n_cases = (matched["covid_status"] == 1).sum()
    n_controls = (matched["covid_status"] == 0).sum()
    print(f"Matched ({args.ratio}): {n_cases} COVID+ cases, {n_controls} COVID- controls")


if __name__ == "__main__":
    main()
