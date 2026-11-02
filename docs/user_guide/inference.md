---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Inference

{meth}`WinRatio.fit() <pywinratio.WinRatioAnalysis.fit>` always computes
analytic standard errors, Wald intervals and p-values. Add `n_boot=` for
bootstrap percentile intervals and `n_perm=` for a permutation p-value.
{class}`~pywinratio.WinRatioResult` reports the bootstrap interval when one was
requested (otherwise the analytic one) and the permutation p-value when one was
requested (otherwise the Wald p-value); the method used is printed with every
table.

```{code-cell} ipython3
import pandas as pd
from pywinratio import Outcome, WinRatio
from pywinratio.datasets import make_heart_failure_trial

pd.set_option("display.precision", 4)
trial = make_heart_failure_trial()
analysis = WinRatio(
    [
        Outcome.time_to_event("time_death", "death", name="Death", terminal=True),
        Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization"),
        Outcome.continuous("kccq_change", name="KCCQ", favorable="higher", margin=5),
    ],
    group="arm", treatment="treatment", control="placebo",
)
result = analysis.fit(trial, n_boot=2000, n_perm=5000, seed=7)
```

## Which method?

| Design | Default analytic method | Bootstrap (`n_boot`) | Permutation (`n_perm`) |
|---|---|---|---|
| All pairs | U-statistic (Hoeffding) variance | patients resampled within arm | arm labels permuted (Finkelstein–Schoenfeld) |
| Stratified | stratified U-statistic | within arm and stratum | within strata |
| Matched pairs | multinomial, $1/W + 1/L$ | pairs resampled | arms swapped within pairs (sign-flip) |
| Weighted | U-statistic with fixed weights | patients (or clusters) with their weights | not available |
| `cluster_col` (all pairs) | cluster-robust U-statistic, Student's *t* with clusters − 1 df | clusters (optionally within arm) | whole clusters when each cluster belongs to one arm (cluster-level exposure); otherwise patients within clusters |
| `cluster_col` (matched pairs) | cluster-robust multinomial, *t* with clusters − 1 df | clusters of pairs | signs flipped for whole clusters of pairs |

Practical guidance:

- The analytic intervals are fast and accurate for moderate samples (they
  agree with BuyseTest and WINS; see [Validation](../validation.md)). The
  win ratio interval is built on the log scale.
- The bootstrap is useful with small samples, clusters, or when the design is
  re-estimated in each replicate (propensity refit). Replicates with wins but no
  losses give WR = ∞. pywinratio keeps them, so in very small samples the upper
  limit can be infinite, which is correct. Use at least 2000 replicates
  for a publication.
- The permutation test is exact under the null hypothesis of exchangeable
  arms and is the classical Finkelstein–Schoenfeld test for unmatched data. The
  permuted units follow the design: with a hospital-level exposure, whole
  hospitals are permuted, so the test keeps its size even with strong hospital
  effects (permuting patients would not).
- With few clusters, cluster-robust Wald intervals use Student's *t* with
  K − 1 degrees of freedom; with fewer than about 20 clusters prefer the cluster
  permutation test for the p-value. An arm with a single cluster (or a single
  patient) has no estimable variance: the intervals are then reported as
  undefined, with a warning.
- Report the win ratio together with the net benefit, whose interval is
  always finite.

```{code-cell} ipython3
result.inference["method"], result.bootstrap["bootstrap_type"], result.permutation["method"]
```

## Analytic results

```{code-cell} ipython3
pd.DataFrame(
    {
        "estimate": [result.wr, result.net_benefit, result.win_odds],
        "SE (log or natural scale)": [
            result.inference["se_log_wr"],
            result.inference["se_net_benefit"],
            result.inference["se_log_win_odds"],
        ],
        "CI": [result.inference["wr_ci"], result.inference["net_benefit_ci"], result.inference["win_odds_ci"]],
        "p": [result.inference["wr_p_value"], result.inference["net_benefit_p_value"], result.inference["win_odds_p_value"]],
    },
    index=["win ratio", "net benefit", "win odds"],
)
```

## Bootstrap results

```{code-cell} ipython3
{k: result.bootstrap[k] for k in ("ci", "net_benefit_ci", "win_odds_ci", "se_log_wr", "n_infinite", "n_undefined")}
```

```{code-cell} ipython3
ax = result.plot("bootstrap")
```

The bootstrap standard error of log WR is close to the analytic one:

```{code-cell} ipython3
result.bootstrap["se_log_wr"], result.inference["se_log_wr"]
```

## Permutation test

```{code-cell} ipython3
result.permutation
```

`statistic` is the number of wins minus losses of arm A (the Finkelstein–
Schoenfeld statistic). `p_value_asymptotic` uses the exact permutation variance
with a normal approximation.

## Lower-level functions

All of the above are also available as functions returning dictionaries:
{func}`~pywinratio.bootstrap_win_ratio`, {func}`~pywinratio.bootstrap_win_ratio_matched`,
{func}`~pywinratio.bootstrap_win_ratio_cluster`,
{func}`~pywinratio.bootstrap_win_ratio_cluster_within_arm`,
{func}`~pywinratio.bootstrap_weighted_gpc`, {func}`~pywinratio.permutation_test`,
{func}`~pywinratio.u_statistic_inference` and {func}`~pywinratio.matched_pair_inference`.

```{warning}
{func}`~pywinratio.logwr_wald_ci` (variance $1/W + 1/L$) is valid for
assigned pairs only. Applied to all-pairs counts it treats the
$n_A \times n_B$ overlapping comparisons as independent and gives intervals that
are far too narrow.
```
