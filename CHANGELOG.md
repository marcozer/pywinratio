# Changelog

## 0.3.0 (2026-11-08)

The package is now called pywinratio. It was previously distributed as
WinRatioPy. `import winratio` and `import winratiopy` still work but raise a
deprecation warning.

New:

- Censored time-to-event outcomes (`Outcome.time_to_event`), compared with the
  Gehan rule, with an optional margin, threshold or follow-up horizon.
- Analytic standard errors for every design: U-statistic variance for all pairs
  and strata, a cluster-robust version (Student's t with K − 1 degrees of
  freedom), and the multinomial variance for matched pairs.
- Net benefit and win odds are reported next to the win ratio everywhere.
- Permutation tests (Finkelstein–Schoenfeld). Patients, patients within
  clusters, whole clusters or matched pairs are permuted, depending on the design.
- `win_ratio()` and a richer result object: summary tables, a tier-by-tier
  breakdown, per-pair results, a text report, JSON export and plots.
- `compare_patients()`, to see how two patients are compared.
- Sensitivity analyses (`compare_hierarchies`, `order_sensitivity`), several
  arms with p-value adjustment (`pairwise_win_ratios`), sample size
  (`sample_size_win_ratio`) and simulated power (`simulate_power`).
- Two simulated example data sets in `pywinratio.datasets`.
- Matching and weighting can use propensity scores you supply (`ps_col`). They
  also accept a comparator arm when the exposure has more than two levels. The
  propensity-refit bootstraps can run in parallel (`n_jobs`).
- A `threshold` option for continuous and time-to-event tiers, with the same
  meaning as in BuyseTest.
- A validation folder comparing the results with BuyseTest and WINS, and new
  documentation.

Fixed (these can change results compared with 0.2):

- `missing_is="loss"` and `"win"` now apply to the patient whose value is
  missing. Before, they applied to arm A, so the result depended on which arm was
  called A.
- Bootstrap intervals keep replicates with an infinite win ratio. They used to
  be dropped, which made the upper limit too low.
- In stratified analyses, the cluster bootstrap now draws each cluster once for
  all strata instead of separately in each stratum. The old way underestimated
  the variance.
- Greedy matching takes the nearest control still available. It used to drop
  the treated patient when that control was already taken. Optimal matching now
  maximizes the number of pairs.
- The default caliper uses the pooled within-arm SD of the logit propensity
  score (Austin, 2011).
- The propensity model is no longer penalized, so overlap weights balance the
  covariate means exactly. Trimming is off by default for overlap weights.
- Standardized differences use the sample variance instead of the population
  variance.
- The bootstrap p-value is capped at 1 and uses every replicate.
- Bootstraps no longer depend on row order, and `bootstrap_win_ratio` handles
  matched designs.
- Bad inputs now raise an error instead of passing silently: non-0/1 binary
  outcomes, text in numeric outcomes, unknown configuration keys, missing strata
  or cluster labels, identical arms. Incomplete matched pairs give a warning.
- Variances that cannot be estimated (one patient or one cluster in an arm) are
  reported as missing with a warning, not as zero.
- Large data sets are compared in blocks. The pure-Python fallback above 5
  million pairs is gone (86 s before, under 1 s now), and the weighted bootstrap
  is about a hundred times faster.

Removed: the case study no longer has any code to export protected registry
data. It only uses synthetic data.

With the same data and the same pairs, point estimates without missing outcome
values are identical to 0.2.

## 0.2.2 (2026-07-24)

Synthetic distal pancreatectomy example aligned with the study's hierarchy;
order sensitivities, propensity-score overlap and matching-support output.

## 0.2.1

Center volume defined from a supplied field; graded complications and a 1-day
length-of-stay margin in the example hierarchy.

## 0.2.0

Renamed WinRatioPy. Binary, ordinal, continuous and terminal tiers;
matched-pair, all-pair and weighted analyses; subject, pair, cluster and
propensity-refit bootstraps; matching, weighting and balance diagnostics.

## 0.1.0

First version (`winratio`).
