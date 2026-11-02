"""Analytic, bootstrap and permutation inference."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from pywinratio import (
    Outcome,
    WinRatio,
    WinRatioConfig,
    adjust_pvalues,
    bootstrap_p_value_from_samples,
    bootstrap_win_ratio,
    bootstrap_win_ratio_cluster,
    bootstrap_win_ratio_matched,
    logwr_wald_ci,
    matched_pair_inference,
    permutation_test,
    sample_size_win_ratio,
    win_statistics,
)
from pywinratio._bootstrap import extended_quantile, percentile_interval


def _two_arm(n: int, shift: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "arm": np.repeat(["A", "B"], n),
            "death": rng.binomial(1, np.repeat([0.15, 0.2], n)),
            "y": np.round(rng.normal(np.repeat([shift, 0.0], n), 1.0), 1),
            "site": rng.choice([f"S{i}" for i in range(8)], 2 * n),
        }
    )


TIERS = [Outcome.binary("death", terminal=True), Outcome.continuous("y", favorable="higher", margin=0.2)]


def test_win_statistics_identities() -> None:
    stats = win_statistics(30, 20, 100)
    assert stats["win_odds"] == pytest.approx((1 + stats["net_benefit"]) / (1 - stats["net_benefit"]))
    assert stats["win_probability"] == pytest.approx(0.55)
    assert win_statistics(5, 0, 10)["wr"] == np.inf


def test_u_statistic_se_matches_delong_formula() -> None:
    data = _two_arm(40, 0.3, seed=1)
    result = WinRatio(TIERS, group="arm", treatment="A", control="B").fit(data)
    # explicit structural-component (DeLong) computation
    from pywinratio._engine import pair_result_matrices

    a, b = data[data.arm == "A"], data[data.arm == "B"]
    w, lo = (m.astype(float) for m in pair_result_matrices(a, b, TIERS))
    pw, pl = w.mean(), lo.mean()
    va = np.cov(np.vstack([w.mean(1), lo.mean(1)]))
    vb = np.cov(np.vstack([w.mean(0), lo.mean(0)]))
    sigma = va / len(a) + vb / len(b)
    grad = np.array([1 / pw, -1 / pl])
    assert result.inference["se_log_wr"] == pytest.approx(float(np.sqrt(grad @ sigma @ grad)), rel=1e-10)


def test_u_statistic_interval_has_nominal_coverage() -> None:
    # true WR of the data-generating process from a very large sample
    big = _two_arm(3000, 0.3, seed=99)
    truth = WinRatio(TIERS, group="arm", treatment="A", control="B").fit(big).wr
    covered = 0
    sims = 150
    for seed in range(sims):
        res = WinRatio(TIERS, group="arm", treatment="A", control="B").fit(_two_arm(50, 0.3, seed))
        lo, hi = res.inference["wr_ci"]
        covered += lo <= truth <= hi
    assert 0.88 <= covered / sims <= 0.99


def test_matched_pair_variance_is_one_over_wins_plus_one_over_losses() -> None:
    results = [1] * 30 + [-1] * 20 + [0] * 10
    out = matched_pair_inference(results)
    assert out["se_log_wr"] == pytest.approx(np.sqrt(1 / 30 + 1 / 20))
    assert out["wr_ci"][0] == pytest.approx(logwr_wald_ci(30, 20)["ci"][0])


def test_cluster_robust_variance_with_singleton_clusters_is_close_to_iid() -> None:
    data = _two_arm(60, 0.2, seed=4).reset_index(drop=True)
    data["unit"] = np.arange(len(data))
    plain = WinRatio(TIERS, group="arm", treatment="A", control="B").fit(data)
    clustered = WinRatio(TIERS, group="arm", treatment="A", control="B").fit(data, cluster_col="unit")
    assert clustered.inference["se_log_wr"] == pytest.approx(plain.inference["se_log_wr"], rel=0.05)


def test_extended_quantile_keeps_infinite_replicates() -> None:
    samples = np.array([1.0, 2.0, 3.0, np.inf, np.inf])
    assert extended_quantile(samples, 0.975) == np.inf
    assert extended_quantile(np.array([1.0, 2.0, 3.0]), 0.5) == np.quantile([1.0, 2.0, 3.0], 0.5)
    assert percentile_interval(np.array([1.0, 2.0, np.nan]), 0.05)[0] == pytest.approx(1.025)


def test_small_matched_sample_gives_infinite_upper_limit() -> None:
    pairs = pd.DataFrame({"arm": ["A", "B"] * 7, "pair": np.repeat(np.arange(7), 2), "death": [0, 1] * 6 + [1, 0]})
    cfg = WinRatioConfig("arm", "A", "B", [Outcome.binary("death")], pair_strategy="matched", pair_id="pair")
    boot = bootstrap_win_ratio_matched(pairs, cfg, n_boot=500, seed=1)
    assert boot["n_infinite"] > 0
    assert boot["ci"][1] == np.inf


def test_bootstrap_is_reproducible_and_row_order_invariant() -> None:
    data = _two_arm(40, 0.3, seed=2)
    cfg = WinRatioConfig("arm", "A", "B", TIERS)
    first = bootstrap_win_ratio(data, cfg, n_boot=200, seed=5)
    second = bootstrap_win_ratio(data.sample(frac=1, random_state=3), cfg, n_boot=200, seed=5)
    assert first["ci"] == second["ci"]


def test_bootstrap_se_close_to_analytic_se() -> None:
    data = _two_arm(80, 0.3, seed=6)
    result = WinRatio(TIERS, group="arm", treatment="A", control="B").fit(data, n_boot=2000, seed=1)
    assert result.bootstrap["se_log_wr"] == pytest.approx(result.inference["se_log_wr"], rel=0.12)


def test_stratified_cluster_bootstrap_resamples_clusters_jointly() -> None:
    # every cluster appears in several strata; jointly resampled clusters keep their correlation
    rng = np.random.default_rng(0)
    rows = []
    for c in range(10):
        effect = rng.normal(0, 1.0)
        for s in range(3):
            for arm in ("A", "B"):
                for _ in range(4):
                    rows.append({"arm": arm, "site": f"C{c}", "stratum": s, "y": effect + rng.normal(0, 0.3)})
    data = pd.DataFrame(rows)
    cfg = WinRatioConfig("arm", "A", "B", [Outcome.continuous("y", favorable="higher")], strata=["stratum"])
    boot = bootstrap_win_ratio_cluster(data, cfg, cluster_col="site", n_boot=200, seed=1)
    assert boot["n_clusters"] == 10
    assert boot["bootstrap_type"] == "cluster"
    assert boot["stratified_analysis"] is True


def test_bootstrap_p_value_is_capped_and_uses_all_replicates() -> None:
    assert bootstrap_p_value_from_samples([2.0] * 50) == pytest.approx(2 / 51)
    assert bootstrap_p_value_from_samples([1.0] * 50) == 1.0
    assert bootstrap_p_value_from_samples([np.inf] * 9 + [0.5]) == pytest.approx(4 / 11)


def test_adjust_pvalues_methods() -> None:
    p = [0.01, 0.04, 0.03, np.nan]
    assert adjust_pvalues(p, "bonferroni")[:3] == pytest.approx([0.03, 0.12, 0.09])
    assert adjust_pvalues(p, "holm")[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert adjust_pvalues(p, "fdr_bh")[:3] == pytest.approx([0.03, 0.04, 0.04])
    assert np.isnan(adjust_pvalues(p)[3])


def test_permutation_test_matches_exact_enumeration() -> None:
    data = pd.DataFrame({"arm": ["A"] * 4 + ["B"] * 4, "y": [5.0, 3.0, 4.5, 2.0, 1.0, 2.5, 0.5, 3.5]})
    cfg = WinRatioConfig("arm", "A", "B", [Outcome.continuous("y", favorable="higher")])
    from pywinratio.permutation import _scores

    u = _scores(data, cfg)
    observed = u[:4].sum()
    sums = [u[list(c)].sum() for c in combinations(range(8), 4)]
    exact = np.mean(np.abs(sums) >= abs(observed) - 1e-9)
    mc = permutation_test(data, cfg, n_perm=20000, seed=1)
    assert mc["statistic"] == observed
    assert mc["p_value"] == pytest.approx(exact, abs=0.01)


def test_permutation_variance_matches_simulation() -> None:
    data = _two_arm(30, 0.0, seed=8)
    cfg = WinRatioConfig("arm", "A", "B", TIERS)
    out = permutation_test(data, cfg, n_perm=4000, seed=2)
    assert out["p_value_asymptotic"] == pytest.approx(out["p_value"], abs=0.05)


def test_matched_sign_flip_test() -> None:
    pairs = pd.DataFrame(
        {"arm": ["A", "B"] * 20, "pair": np.repeat(np.arange(20), 2), "death": [0, 1] * 15 + [1, 0] * 5}
    )
    cfg = WinRatioConfig("arm", "A", "B", [Outcome.binary("death")], pair_strategy="matched", pair_id="pair")
    out = permutation_test(pairs, cfg, n_perm=5000, seed=3)
    assert out["method"] == "sign_flip_matched_pairs"
    assert out["statistic"] == 10
    assert 0.02 < out["p_value"] < 0.06  # exact two-sided sign test p = 0.041


def test_sample_size_formula_achieves_target_power_by_simulation() -> None:
    plan = sample_size_win_ratio(1.6, p_tie=0.0, power=0.8)
    n = plan["n_a"]
    # continuous outcome without ties, location shift giving WR = 1.6
    # P(X > Y) = 1.6 / 2.6 for normal shift delta: Phi(delta / sqrt 2) = 0.6154
    from scipy.stats import norm

    delta = np.sqrt(2) * norm.ppf(1.6 / 2.6)
    rejections = 0
    sims = 200
    rng = np.random.default_rng(10)
    analysis = WinRatio([Outcome.continuous("y", favorable="higher")], group="arm", treatment="A", control="B")
    for _ in range(sims):
        y = np.concatenate([rng.normal(delta, 1, n), rng.normal(0, 1, plan["n_b"])])
        frame = pd.DataFrame({"arm": ["A"] * n + ["B"] * plan["n_b"], "y": y})
        rejections += analysis.fit(frame).p_value < 0.05
    assert 0.70 <= rejections / sims <= 0.90


def test_multi_arm_dictionary_api_with_holm() -> None:
    from pywinratio import (
        MultiArmWinRatioConfig,
        add_pvalues_to_multi_arm_results,
        bootstrap_win_ratio_multi_arm,
        compute_win_ratio_multi_arm,
    )

    rng = np.random.default_rng(3)
    data = pd.DataFrame({"arm": np.repeat(["a", "b", "c"], 40), "y": rng.normal(np.repeat([0.5, 0.0, 0.2], 40), 1)})
    cfg = MultiArmWinRatioConfig("arm", ["a", "b", "c"], outcomes=[Outcome.continuous("y", favorable="higher")])
    point = compute_win_ratio_multi_arm(data, cfg)
    assert point["summary"]["n_comparisons"] == 3
    boot = bootstrap_win_ratio_multi_arm(data, cfg, n_boot=200, seed=1)
    out = add_pvalues_to_multi_arm_results(boot)
    assert all(c["p_value_holm"] >= c["p_value"] for c in out["comparisons"])
    referenced = compute_win_ratio_multi_arm(
        data, MultiArmWinRatioConfig("arm", ["a", "b", "c"], reference_arm="b", outcomes=cfg.outcomes)
    )
    assert [(c["arm_a"], c["arm_b"]) for c in referenced["comparisons"]] == [("a", "b"), ("c", "b")]


def test_e_values() -> None:
    from pywinratio import compute_e_value, compute_e_value_for_ci

    assert compute_e_value(2.0)["e_value"] == pytest.approx(2 + np.sqrt(2))
    assert compute_e_value(0.5)["e_value"] == pytest.approx(compute_e_value(2.0)["e_value"])
    assert compute_e_value_for_ci(1.5, 0.9, 2.0)["e_value_ci"] == 1.0
    assert compute_e_value_for_ci(1.8, 1.2, 2.5)["e_value_ci"] == pytest.approx(compute_e_value(1.2)["e_value"])
