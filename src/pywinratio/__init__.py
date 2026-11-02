"""pywinratio: win ratio, net benefit and win odds for prioritized outcomes.

Hierarchical (prioritized) composite outcomes are compared patient-pair by
patient-pair: the most important outcome decides first, ties pass to the
next. pywinratio estimates the resulting win statistics for randomized and
observational studies, with matching, weighting, stratification, clustering,
censored time-to-event tiers and analytic, bootstrap or permutation
inference.

Quick start
-----------
>>> from pywinratio import Outcome, win_ratio
>>> from pywinratio.datasets import make_heart_failure_trial
>>> trial = make_heart_failure_trial()
>>> result = win_ratio(
...     trial,
...     [
...         Outcome.time_to_event("time_death", "death", name="Death", terminal=True),
...         Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization"),
...         Outcome.continuous("kccq_change", name="KCCQ", favorable="higher", margin=5),
...     ],
...     group="arm",
...     treatment="treatment",
...     control="placebo",
... )
>>> print(result)  # doctest: +SKIP
"""

from ._version import __version__
from .analysis import Outcome, WinRatio, WinRatioAnalysis, WinRatioResult, win_ratio
from .config import (
    MultiArmWinRatioConfig,
    WinRatioConfig,
    WinRatioOutcome,
    config_from_dict,
    multi_arm_config_from_dict,
)
from .diagnostics import (
    balance_diagnostics,
    effective_sample_size,
    propensity_overlap_coefficient,
)
from .gpc import bootstrap_weighted_gpc, compute_weighted_gpc
from .inference import (
    adjust_pvalues,
    compute_e_value,
    compute_e_value_for_ci,
    logwr_wald_ci,
    logwr_wald_p_value,
    matched_pair_inference,
    u_statistic_inference,
    win_statistics,
)
from .match import MatchResult, propensity_match
from .permutation import permutation_test
from .plotting import plot_balance, plot_bootstrap, plot_forest, plot_tiers
from .power import sample_size_win_ratio, simulate_power
from .resampling import bootstrap_propensity_matched_win_ratio, bootstrap_propensity_weighted_gpc
from .sensitivity import compare_hierarchies, order_sensitivity, pairwise_win_ratios
from .summary import (
    bootstrap_p_value_from_samples,
    paired_risk_difference_bootstrap,
    summarize_component_outcomes,
    summarize_wr_metrics_from_overall,
)
from .weights import WeightingResult, estimate_propensity_weights
from .wr import (
    add_pvalues_to_multi_arm_results,
    adjust_pvalues_holm,
    bootstrap_win_ratio,
    bootstrap_win_ratio_cluster,
    bootstrap_win_ratio_cluster_within_arm,
    bootstrap_win_ratio_matched,
    bootstrap_win_ratio_multi_arm,
    compare_patients,
    compute_pvalue_from_bootstrap,
    compute_win_ratio,
    compute_win_ratio_all_pairs,
    compute_win_ratio_matched,
    compute_win_ratio_multi_arm,
    pair_results,
)

__all__ = [
    "MatchResult",
    "MultiArmWinRatioConfig",
    "Outcome",
    "WeightingResult",
    "WinRatio",
    "WinRatioAnalysis",
    "WinRatioConfig",
    "WinRatioOutcome",
    "WinRatioResult",
    "__version__",
    "add_pvalues_to_multi_arm_results",
    "adjust_pvalues",
    "adjust_pvalues_holm",
    "balance_diagnostics",
    "bootstrap_p_value_from_samples",
    "bootstrap_propensity_matched_win_ratio",
    "bootstrap_propensity_weighted_gpc",
    "bootstrap_weighted_gpc",
    "bootstrap_win_ratio",
    "bootstrap_win_ratio_cluster",
    "bootstrap_win_ratio_cluster_within_arm",
    "bootstrap_win_ratio_matched",
    "bootstrap_win_ratio_multi_arm",
    "compare_hierarchies",
    "compare_patients",
    "compute_e_value",
    "compute_e_value_for_ci",
    "compute_pvalue_from_bootstrap",
    "compute_weighted_gpc",
    "compute_win_ratio",
    "compute_win_ratio_all_pairs",
    "compute_win_ratio_matched",
    "compute_win_ratio_multi_arm",
    "config_from_dict",
    "effective_sample_size",
    "estimate_propensity_weights",
    "logwr_wald_ci",
    "logwr_wald_p_value",
    "matched_pair_inference",
    "multi_arm_config_from_dict",
    "order_sensitivity",
    "pair_results",
    "paired_risk_difference_bootstrap",
    "pairwise_win_ratios",
    "permutation_test",
    "plot_balance",
    "plot_bootstrap",
    "plot_forest",
    "plot_tiers",
    "propensity_match",
    "propensity_overlap_coefficient",
    "sample_size_win_ratio",
    "simulate_power",
    "summarize_component_outcomes",
    "summarize_wr_metrics_from_overall",
    "u_statistic_inference",
    "win_ratio",
    "win_statistics",
]
