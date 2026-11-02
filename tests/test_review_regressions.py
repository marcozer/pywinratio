"""Regression tests for defects found by the independent statistical review of 0.3.0."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pywinratio import (
    MultiArmWinRatioConfig,
    Outcome,
    WinRatio,
    WinRatioConfig,
    add_pvalues_to_multi_arm_results,
    bootstrap_win_ratio_multi_arm,
    compute_weighted_gpc,
    permutation_test,
)

Y = [Outcome.continuous("y", favorable="higher")]


def _hospitals(seed: int, n_hosp: int = 12, per: int = 15, effect: float = 0.0) -> pd.DataFrame:
    """Cluster-level exposure with a strong hospital effect and two strata."""

    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_hosp):
        arm = "A" if h < n_hosp // 2 else "B"
        level = rng.normal(0, 1.0) + (effect if arm == "A" else 0.0)
        for i in range(per):
            rows.append({"arm": arm, "hosp": f"H{h}", "stratum": i % 2, "y": level + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_stratified_cluster_permutation_permutes_hospitals_and_keeps_its_size() -> None:
    analysis = WinRatio(Y, group="arm", treatment="A", control="B", strata=["stratum"])
    res = analysis.fit(_hospitals(0), cluster_col="hosp", n_perm=200)
    assert res.permutation["permuted_units"] == "clusters"
    rejections = 0
    for seed in range(60):
        r = analysis.fit(_hospitals(seed), cluster_col="hosp", n_perm=200, seed=seed)
        rejections += r.p_value < 0.05
    assert rejections / 60 <= 0.15


def test_patient_exposure_within_clusters_permutes_within_clusters() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"arm": rng.choice(["A", "B"], 200), "hosp": rng.choice(list("abcdefgh"), 200), "y": rng.normal(size=200)}
    )
    out = permutation_test(df, WinRatioConfig("arm", "A", "B", Y), cluster_col="hosp", n_perm=300)
    assert out["permuted_units"] == "patients within clusters"


def test_matched_pairs_with_clusters_flip_whole_clusters() -> None:
    pairs = pd.DataFrame(
        {
            "arm": ["A", "B"] * 40,
            "pair": np.repeat(np.arange(40), 2),
            "hosp": np.repeat(np.arange(40) % 8, 2),
            "y": np.arange(80.0),
        }
    )
    res = WinRatio(Y, group="arm", treatment="A", control="B").fit(
        pairs, pair_id="pair", cluster_col="hosp", n_perm=200
    )
    assert res.permutation["method"] == "sign_flip_cluster_of_pairs"
    assert res.permutation["n_units"] == 8
    with pytest.raises(ValueError, match="assigned pairs"):
        WinRatio(Y, group="arm", treatment="A", control="B").fit(
            pairs, pair_id="pair", cluster_col="hosp", cluster_within_arm=True
        )


def test_duplicate_row_labels_do_not_corrupt_cluster_variance_or_weights() -> None:
    df = _hospitals(3, effect=0.5)
    duplicated = pd.concat([df[df.arm == "A"].reset_index(drop=True), df[df.arm == "B"].reset_index(drop=True)])
    analysis = WinRatio(Y, group="arm", treatment="A", control="B")
    clean = analysis.fit(df.reset_index(drop=True), cluster_col="hosp")
    dup = analysis.fit(duplicated, cluster_col="hosp")
    assert dup.inference["se_log_wr"] == pytest.approx(clean.inference["se_log_wr"])
    assert dup.inference["n_clusters"] == 12
    duplicated = duplicated.assign(w=1.0)
    assert compute_weighted_gpc(duplicated, WinRatioConfig("arm", "A", "B", Y), weight_col="w")["wins"] == clean.wins


def test_win_odds_bootstrap_ignores_undefined_replicates() -> None:
    # four hospitals nested in arms with overlapping outcomes: a pooled cluster bootstrap
    # sometimes draws one arm only (undefined replicate) but never a net benefit of -1
    rng = np.random.default_rng(4)
    levels = {"H0": 1.0, "H1": 1.2, "H2": 0.0, "H3": 0.2}
    df = pd.DataFrame(
        [
            {"arm": "A" if h in ("H0", "H1") else "B", "hosp": h, "y": level + rng.normal(0, 1.0)}
            for h, level in levels.items()
            for _ in range(30)
        ]
    )
    res = WinRatio(Y, group="arm", treatment="A", control="B").fit(df, cluster_col="hosp", n_boot=500, seed=1)
    assert res.bootstrap["n_undefined"] > 0
    nb_lo = res.bootstrap["net_benefit_ci"][0]
    assert res.bootstrap["win_odds_ci"][0] == pytest.approx((1 + nb_lo) / (1 - nb_lo), rel=0.05)


def test_unestimable_variances_are_nan_with_a_warning() -> None:
    one_cluster_a = _hospitals(5).assign(hosp=lambda d: np.where(d.arm == "A", "H0", d.hosp))
    with pytest.warns(UserWarning, match="fewer than 2 clusters"):
        res = WinRatio(Y, group="arm", treatment="A", control="B").fit(one_cluster_a, cluster_col="hosp")
    assert np.isnan(res.inference["se_net_benefit"])
    single = pd.DataFrame({"arm": ["A", "B", "B"], "y": [1.0, 0.0, 2.0]})
    with pytest.warns(UserWarning, match="single patient"):
        res = WinRatio(Y, group="arm", treatment="A", control="B").fit(single)
    assert np.isnan(res.inference["net_benefit_ci"][0])


def test_cluster_robust_intervals_use_t_with_clusters_minus_one_df() -> None:
    df = _hospitals(6, effect=0.5)
    res = WinRatio(Y, group="arm", treatment="A", control="B").fit(df, cluster_col="hosp")
    assert res.inference["df"] == 11
    nb, se = res.net_benefit, res.inference["se_net_benefit"]
    from scipy.stats import t

    assert res.inference["net_benefit_ci"][1] == pytest.approx(min(1.0, nb + t.ppf(0.975, 11) * se))


def test_all_ties_give_undefined_intervals() -> None:
    df = pd.DataFrame({"arm": ["A"] * 5 + ["B"] * 5, "y": 1.0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = WinRatio(Y, group="arm", treatment="A", control="B").fit(df)
    assert np.isnan(res.inference["net_benefit_ci"][0])


def test_multi_arm_p_values_use_the_interval_level() -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"arm": np.repeat(["a", "b"], 60), "y": rng.normal(np.repeat([0.6, 0.0], 60), 1.0)})
    cfg = MultiArmWinRatioConfig("arm", ["a", "b"], outcomes=Y)
    p90 = add_pvalues_to_multi_arm_results(bootstrap_win_ratio_multi_arm(df, cfg, n_boot=500, alpha=0.10))
    p95 = add_pvalues_to_multi_arm_results(bootstrap_win_ratio_multi_arm(df, cfg, n_boot=500, alpha=0.05))
    assert p90["comparisons"][0]["p_value"] == pytest.approx(p95["comparisons"][0]["p_value"], rel=0.5)


def test_rows_without_pair_identifier_are_reported() -> None:
    df = pd.DataFrame({"arm": ["A", "B", "A"], "pair": [1, 1, None], "y": [1.0, 0.0, 3.0]})
    with pytest.warns(UserWarning, match="without a pair identifier"):
        WinRatio(Y, group="arm", treatment="A", control="B").fit(df, pair_id="pair")
