from __future__ import annotations

import pandas as pd
import pytest

from pywinratio import propensity_match


def test_propensity_match_creates_pair_ids() -> None:
    df = pd.DataFrame(
        {
            "treatment_group": ["HIGH", "HIGH", "HIGH", "OTHER", "OTHER", "OTHER"],
            "age_band": ["50-59", "60-69", "70-79", "50-59", "60-69", "70-79"],
            "sex": ["MALE", "FEMALE", "MALE", "MALE", "FEMALE", "MALE"],
            "asa_group": ["ASA2", "ASA3", "ASA3", "ASA2", "ASA3", "ASA3"],
        }
    )
    result = propensity_match(
        df,
        treat_col="treatment_group",
        treated_value="HIGH",
        covariates=["age_band", "sex", "asa_group"],
        method="nearest",
    )
    assert "PAIR_ID" in result.matched_df.columns
    assert result.pairs >= 1


def test_propensity_match_drops_missing_values_by_default() -> None:
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "age": [40, None, 60, 41, 51, 61],
            "sex": ["F", "M", "F", "F", "M", "F"],
        }
    )
    result = propensity_match(df, treat_col="group", treated_value="A", covariates=["age", "sex"])
    assert result.missing == "drop"
    assert result.rows_dropped_missing == 1
    assert result.full_df["age"].notna().all()


def test_propensity_match_requires_explicit_covariates() -> None:
    df = pd.DataFrame({"group": ["A", "A", "B", "B"], "outcome": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="specified explicitly"):
        propensity_match(df, treat_col="group", treated_value="A")
