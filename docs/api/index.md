# API reference

```{eval-rst}
.. currentmodule:: pywinratio
```

## High-level interface

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   win_ratio
   WinRatioAnalysis
   WinRatioResult
   compare_patients
```

`WinRatio` is an alias of {class}`~pywinratio.WinRatioAnalysis` and `Outcome` an
alias of {class}`~pywinratio.WinRatioOutcome`.

## Configuration

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   WinRatioOutcome
   WinRatioConfig
   MultiArmWinRatioConfig
   config_from_dict
   multi_arm_config_from_dict
```

## Estimation

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   compute_win_ratio
   compute_win_ratio_all_pairs
   compute_win_ratio_matched
   compute_weighted_gpc
   compute_win_ratio_multi_arm
   pair_results
   win_statistics
```

## Inference

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   u_statistic_inference
   matched_pair_inference
   permutation_test
   bootstrap_win_ratio
   bootstrap_win_ratio_matched
   bootstrap_win_ratio_cluster
   bootstrap_win_ratio_cluster_within_arm
   bootstrap_weighted_gpc
   bootstrap_win_ratio_multi_arm
   bootstrap_propensity_matched_win_ratio
   bootstrap_propensity_weighted_gpc
   bootstrap_p_value_from_samples
   compute_pvalue_from_bootstrap
   adjust_pvalues
   adjust_pvalues_holm
   add_pvalues_to_multi_arm_results
   logwr_wald_ci
   logwr_wald_p_value
```

## Observational designs and diagnostics

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   propensity_match
   MatchResult
   estimate_propensity_weights
   WeightingResult
   balance_diagnostics
   effective_sample_size
   propensity_overlap_coefficient
   compute_e_value
   compute_e_value_for_ci
```

## Sensitivity analyses, power and reporting

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   compare_hierarchies
   order_sensitivity
   pairwise_win_ratios
   sample_size_win_ratio
   simulate_power
   summarize_component_outcomes
   summarize_wr_metrics_from_overall
   paired_risk_difference_bootstrap
```

## Plotting

Requires `matplotlib` (`pip install "pywinratio[plot]"`).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_tiers
   plot_forest
   plot_balance
   plot_bootstrap
```

## Example data

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   datasets.make_heart_failure_trial
   datasets.make_surgical_cohort
```
