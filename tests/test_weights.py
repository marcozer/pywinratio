from __future__ import annotations

import numpy as np
import pandas as pd

from pywinratio import balance_diagnostics, effective_sample_size, estimate_propensity_weights


def test_overlap_weights_are_bounded_and_report_ess() -> None:
    rng = np.random.default_rng(4)
    n = 120
    age = rng.normal(60, 10, n)
    treatment = np.where(age + rng.normal(0, 10, n) > 60, "A", "B")
    df = pd.DataFrame({"group": treatment, "age": age, "sex": rng.choice(["F", "M"], n)})
    result = estimate_propensity_weights(
        df,
        treatment="group",
        treated="A",
        covariates=["age", "sex"],
        estimand="overlap",
    )
    assert result.weighted_df["analysis_weight"].between(0, 1).all()
    assert all(value > 0 for value in result.ess_by_arm.values())
    assert result.balance_after["abs_smd"].max() < result.balance_before["abs_smd"].max()


def test_diagnostics_and_effective_sample_size() -> None:
    df = pd.DataFrame({"group": [1, 1, 0, 0], "x": [1.0, 2.0, 2.0, 3.0]})
    diagnostics = balance_diagnostics(df, treatment="group", treated=1, covariates=["x"])
    assert diagnostics.iloc[0]["smd"] < 0
    assert effective_sample_size([1, 1, 1, 1]) == 4.0


def test_simple_imputation_is_explicit_and_handles_missing_categories() -> None:
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "age": [40, 50, None, 42, 52, 62],
            "sex": ["F", None, "M", "F", "M", "F"],
        }
    )
    result = estimate_propensity_weights(
        df,
        treatment="group",
        treated="A",
        covariates=["age", "sex"],
        missing="simple",
        trim_common_support=False,
    )
    assert result.missing == "simple"
    assert len(result.weighted_df) == len(df)
