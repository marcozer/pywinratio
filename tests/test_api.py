"""High-level API, reporting, plotting, sensitivity, datasets and compatibility."""

from __future__ import annotations

import importlib
import json
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import pywinratio  # noqa: E402
from pywinratio import (  # noqa: E402
    Outcome,
    WinRatio,
    compare_hierarchies,
    estimate_propensity_weights,
    order_sensitivity,
    pairwise_win_ratios,
    plot_balance,
    plot_bootstrap,
    plot_forest,
    plot_tiers,
    propensity_match,
    simulate_power,
    win_ratio,
)
from pywinratio.datasets import make_heart_failure_trial, make_surgical_cohort  # noqa: E402

HF = [
    Outcome.time_to_event("time_death", "death", name="Death", terminal=True),
    Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization"),
    Outcome.continuous("kccq_change", name="KCCQ", favorable="higher", margin=5),
]
SURG = [
    Outcome.binary("death90", name="Death", terminal=True),
    Outcome.ordinal("clavien", name="Clavien-Dindo"),
    Outcome.ordinal("popf", name="POPF"),
    Outcome.binary("readmission", name="Readmission"),
    Outcome.continuous("los", name="LOS", margin=1),
]


@pytest.fixture(scope="module")
def trial() -> pd.DataFrame:
    return make_heart_failure_trial(n=300, seed=11)


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return make_surgical_cohort(n=500, seed=12)


def test_datasets_are_deterministic() -> None:
    pd.testing.assert_frame_equal(make_surgical_cohort(n=50, seed=1), make_surgical_cohort(n=50, seed=1))
    pd.testing.assert_frame_equal(make_heart_failure_trial(n=50, seed=1), make_heart_failure_trial(n=50, seed=1))


def test_result_object(trial: pd.DataFrame) -> None:
    res = win_ratio(trial, HF, group="arm", treatment="treatment", control="placebo", n_boot=200, n_perm=200)
    frame = res.to_frame()
    assert list(frame.index) == ["Win ratio", "Net benefit", "Win odds"]
    assert frame.loc["Win ratio", "ci_lower"] < res.wr < frame.loc["Win ratio", "ci_upper"]
    assert res.ci == (frame.loc["Win ratio", "ci_lower"], frame.loc["Win ratio", "ci_upper"])
    assert 0 < res.p_value <= 1
    tiers = res.tiers()
    assert tiers["resolved"].sum() + res.ties == pytest.approx(res.n_pairs)
    assert tiers.iloc[0]["entering"] == res.n_pairs
    assert "Win ratio" in res.report() and "Hierarchy" in str(res)
    assert res._repr_html_().startswith("<p>")
    payload = json.loads(res.to_json())
    assert payload["method"] == "all_pairs" and len(payload["tiers"]) == 3
    summary = res.summary()
    for key in ("wr", "ci_lower", "ci_upper", "n_boot", "net_benefit_ci_lower", "p_value"):
        assert key in summary


def test_matched_and_weighted_designs(cohort: pd.DataFrame) -> None:
    matching = propensity_match(cohort, "volume", "high", covariates=["age", "sex", "bmi", "asa3"])
    analysis = WinRatio(SURG, group="volume", treatment="high", control="low")
    matched = analysis.fit(matching.matched_df, pair_id="PAIR_ID", n_boot=100)
    assert matched.method == "matched"
    assert len(matched.pairs()) == matching.pairs
    weights = estimate_propensity_weights(cohort, treatment="volume", treated="high", covariates=["age", "sex"])
    weighted = analysis.fit(weights.weighted_df, weight_col="analysis_weight", n_boot=50)
    assert weighted.method == "weighted_gpc"
    assert weighted.inference["method"].endswith("fixed_weights")
    with pytest.raises(ValueError, match="not both"):
        analysis.fit(cohort, pair_id="x", weight_col="y")


def test_stratified_and_clustered_fit(cohort: pd.DataFrame) -> None:
    analysis = WinRatio(SURG, group="volume", treatment="high", control="low", strata="period")
    res = analysis.fit(cohort, cluster_col="hospital", cluster_within_arm=True, n_boot=100, n_perm=100)
    assert res.method == "stratified"
    assert set(res.strata()["period"]) == {"early", "late"}
    assert res.inference["method"] == "u_statistic_cluster"
    assert res.bootstrap["bootstrap_type"] == "cluster_within_arm"
    assert res.bootstrap["stratified_analysis"] is True
    assert res.permutation["permuted_units"] == "clusters"
    with pytest.raises(ValueError, match="requires cluster_col"):
        analysis.fit(cohort, cluster_within_arm=True)


def test_cluster_level_permutation(cohort: pd.DataFrame) -> None:
    res = win_ratio(cohort, SURG, group="volume", treatment="high", control="low", cluster="hospital", n_perm=200)
    assert res.permutation["method"] == "cluster_permutation"
    assert res.permutation["n_units"] == cohort["hospital"].nunique()


def test_sensitivity_helpers(cohort: pd.DataFrame) -> None:
    table = compare_hierarchies(
        cohort,
        {"main": SURG, "no LOS": SURG[:-1]},
        group="volume",
        treatment="high",
        control="low",
    )
    assert table["hierarchy"].tolist() == ["main", "no LOS"]
    orders = order_sensitivity(cohort, SURG[:4], group="volume", treatment="high", control="low", fixed=1)
    assert len(orders) == 6 and orders["is_prespecified"].sum() == 1
    three = cohort.assign(arm=np.where(cohort["age"] > 65, "old", cohort["volume"]))
    pw = pairwise_win_ratios(three, SURG[:2], group="arm", reference="low")
    assert set(pw["comparison"]) == {"high vs low", "old vs low"}
    assert (pw["p_value_adjusted"] >= pw["p_value"]).all()


def test_plots(trial: pd.DataFrame, cohort: pd.DataFrame) -> None:
    res = win_ratio(trial, HF, group="arm", treatment="treatment", control="placebo", n_boot=100)
    assert plot_tiers(res) is not None
    assert res.plot("bootstrap") is not None
    assert plot_bootstrap(res) is not None
    table = compare_hierarchies(cohort, {"a": SURG, "b": SURG[:2]}, group="volume", treatment="high", control="low")
    assert plot_forest(table, label="hierarchy") is not None
    m = propensity_match(cohort, "volume", "high", covariates=["age", "sex", "bmi"])
    assert plot_balance(m.balance_before, m.balance_after) is not None


def test_simulate_power_runs() -> None:
    analysis = WinRatio(HF[:1], group="arm", treatment="treatment", control="placebo")
    out = simulate_power(
        lambda rng: make_heart_failure_trial(n=100, seed=int(rng.integers(1 << 30))), analysis, n_sim=5
    )
    assert 0 <= out["power"] <= 1


def test_legacy_import_names_still_work() -> None:
    for name in ("winratio", "winratiopy"):
        sys.modules.pop(name, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module = importlib.import_module(name)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert module.compute_win_ratio is pywinratio.compute_win_ratio
        assert importlib.import_module(f"{name}.wr").compute_win_ratio is pywinratio.compute_win_ratio
