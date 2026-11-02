"""Matching and weighting behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pywinratio import estimate_propensity_weights, propensity_match
from pywinratio.datasets import make_surgical_cohort
from pywinratio.match import _greedy, _optimal


def test_greedy_takes_the_next_available_control() -> None:
    treated = np.array([0.0, 0.01, 0.02])
    controls = np.array([0.0, 0.05, 0.06])
    pairs = _greedy(treated, controls, caliper=0.1, order="data", rng=np.random.default_rng(0))
    assert len(pairs) == 3


def test_optimal_maximizes_the_number_of_pairs() -> None:
    # chain where minimizing distance alone would leave a treated patient unmatched
    treated = np.array([0.0, 1.0])
    controls = np.array([0.9, 1.8])
    assert len(_optimal(treated, controls, caliper=1.0)) == 2


def test_matching_on_cohort_balances_and_reports() -> None:
    cohort = make_surgical_cohort(n=600, seed=3)
    m = propensity_match(cohort, "volume", "high", covariates=["age", "sex", "bmi", "asa3"], exact_cols=["period"])
    assert m.pairs > 50
    assert m.matched_df.groupby("PAIR_ID")["period"].nunique().eq(1).all()
    assert m.balance_after["abs_smd"].max() < m.balance_before["abs_smd"].max()
    summary = m.summary()
    assert summary["matched"].tolist() == [m.pairs, m.pairs]
    nearest = propensity_match(cohort, "volume", "high", covariates=["age", "sex", "bmi", "asa3"], method="nearest")
    assert nearest.pairs >= 0.9 * m.pairs


def test_three_arm_treatment_requires_control() -> None:
    df = pd.DataFrame({"g": ["a", "b", "c"] * 10, "x": np.arange(30.0)})
    with pytest.raises(ValueError, match="control"):
        propensity_match(df, "g", "a", covariates=["x"])
    with pytest.raises(ValueError, match="control"):
        estimate_propensity_weights(df, treatment="g", treated="a", covariates=["x"])
    m = propensity_match(df, "g", "a", covariates=["x"], control_value="b", caliper=float("inf"))
    assert set(m.matched_df["g"]) == {"a", "b"}


def test_supplied_propensity_scores() -> None:
    cohort = make_surgical_cohort(n=300, seed=5)
    cohort["ps"] = np.clip(cohort["age"] / 100, 0.05, 0.95)
    m = propensity_match(cohort, "volume", "high", ps_col="ps", covariates=["age"])
    assert m.model is None and m.pairs > 0
    w = estimate_propensity_weights(cohort, treatment="volume", treated="high", covariates=["age"], ps_col="ps")
    assert w.model is None


def test_overlap_weights_give_exact_mean_balance() -> None:
    cohort = make_surgical_cohort(n=500, seed=6)
    w = estimate_propensity_weights(
        cohort, treatment="volume", treated="high", covariates=["age", "sex", "bmi", "asa3", "malignant"]
    )
    assert w.balance_after["abs_smd"].max() < 1e-6
    assert w.rows_trimmed == 0


def test_default_caliper_uses_pooled_within_arm_sd() -> None:
    cohort = make_surgical_cohort(n=400, seed=7)
    m = propensity_match(cohort, "volume", "high", covariates=["age", "asa3"], trim_common_support=False)
    full = m.full_df
    pooled = np.sqrt((full.loc[full._T == 1, "_LPS"].var() + full.loc[full._T == 0, "_LPS"].var()) / 2)
    assert m.caliper == pytest.approx(0.2 * pooled)
