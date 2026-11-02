---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Observational studies

Without randomization the arms differ in case mix. pywinratio offers two
propensity-score designs: 1:1 matching without replacement, analyzed with the
matched-pair win ratio, and weighting (overlap, ATT or ATE), analyzed with the
weighted all-pairs comparison. Both come with balance diagnostics and with
bootstraps that redo the design in every replicate.

```{code-cell} ipython3
import pandas as pd
from pywinratio import (
    Outcome, WinRatio, propensity_match, estimate_propensity_weights,
    propensity_overlap_coefficient, plot_balance,
    bootstrap_propensity_matched_win_ratio, bootstrap_propensity_weighted_gpc,
)
from pywinratio.datasets import make_surgical_cohort

pd.set_option("display.precision", 3)
cohort = make_surgical_cohort()
covariates = ["age", "sex", "bmi", "asa3", "malignant"]
hierarchy = [
    Outcome.binary("death90", name="Death", terminal=True),
    Outcome.ordinal("clavien", name="Clavien-Dindo"),
    Outcome.ordinal("popf", name="POPF"),
    Outcome.binary("readmission", name="Readmission"),
    Outcome.continuous("los", name="Length of stay", margin=1),
]
analysis = WinRatio(hierarchy, group="volume", treatment="high", control="low")
```

## Propensity-score matching

{func}`~pywinratio.propensity_match` fits a logistic model (unpenalized by
default), matches on the logit of the score and returns a
{class}`~pywinratio.MatchResult`.

```{code-cell} ipython3
matching = propensity_match(
    cohort,
    treat_col="volume",
    treated_value="high",
    covariates=covariates,      # must be listed explicitly: outcomes can never slip in
    exact_cols=["period"],      # exact matching within period
    method="optimal",           # maximum number of pairs, then minimum total distance
)
matching.summary()
```

Defaults and options:

- caliper: 0.2 × the pooled within-arm SD of the logit score (Austin, 2011);
  `caliper=float("inf")` disables it;
- `method="nearest"`: greedy nearest-available-neighbor matching in
  `order="largest"` (default), `"smallest"`, `"random"` or `"data"` order;
- `missing="drop"` (complete cases, default) or `"simple"` (explicit single
  imputation of covariates);
- `trim_common_support=True` drops patients outside the overlap of score ranges;
- `ps_col=` uses propensity scores you estimated yourself (e.g. with machine
  learning or after multiple imputation);
- `control_value=` chooses the comparator when the exposure has more than two levels.

### Balance

```{code-cell} ipython3
matching.balance_after.round(3)
```

```{code-cell} ipython3
ax = plot_balance(matching.balance_before, matching.balance_after)
```

```{code-cell} ipython3
propensity_overlap_coefficient(matching.full_df["_PS"], matching.full_df["volume"], treated="high")
```

### Win ratio of the matched pairs

```{code-cell} ipython3
matched = analysis.fit(matching.matched_df, pair_id="PAIR_ID", n_boot=2000, seed=3)
matched.to_frame()
```

This pair bootstrap holds the match fixed.

## Propensity-score weighting

```{code-cell} ipython3
weighting = estimate_propensity_weights(
    cohort, treatment="volume", treated="high",
    covariates=covariates + ["period"], estimand="overlap",
)
weighting.weight_summary
```

Overlap weights from an unpenalized logistic model balance every covariate mean
exactly:

```{code-cell} ipython3
weighting.balance_after[["term", "smd", "variance_ratio", "ks"]].round(4)
```

```{code-cell} ipython3
weighting.ess_by_arm
```

```{code-cell} ipython3
weighted = analysis.fit(weighting.weighted_df, weight_col="analysis_weight")
weighted.to_frame()
```

| Estimand | Target population | Comment |
|---|---|---|
| `"overlap"` | patients in clinical equipoise | bounded weights, exact balance, no trimming needed (default) |
| `"att"` | patients like those treated | comparable to matching |
| `"ate"` | the whole cohort | sensitive to extreme scores; trimmed to common support by default |

## Re-estimating the design in each bootstrap replicate

The pair bootstrap and fixed-weight intervals ignore the uncertainty of the
propensity model and of the matching. For a hospital-level exposure, resample
hospitals within arm, refit the score and rematch (or reweight) in each
replicate. Use `n_jobs=-1` to run replicates in parallel.

```{code-cell} ipython3
from pywinratio import WinRatioConfig

cfg = WinRatioConfig("volume", "high", "low", hierarchy)
refit_match = bootstrap_propensity_matched_win_ratio(
    cohort, cfg, covariates=covariates, exact_cols=["period"],
    cluster_col="hospital", n_boot=200, seed=11,
)
{k: refit_match[k] for k in ("wr", "ci", "net_benefit_ci", "n_successful", "n_failed")}
```

```{code-cell} ipython3
refit_weight = bootstrap_propensity_weighted_gpc(
    cohort, cfg, covariates=covariates + ["period"], cluster_col="hospital", n_boot=200, seed=11,
)
{k: refit_weight[k] for k in ("wr", "ci", "net_benefit_ci", "n_successful", "n_failed")}
```

(200 replicates keep this page fast; use 2000 or more for publication.)

Each replicate builds a new set of matched pairs, so the bootstrap distribution
describes the whole matching procedure rather than the particular pairs of the
original match. The percentile interval can therefore be asymmetric around the
point estimate; report it as is, together with the number of failed replicates.

## Multiple imputation

pywinratio never imputes silently. With missing covariates, create *m* imputed
data sets, run the design and the analysis in each, and pool the log win ratio
(or the net benefit) with Rubin's rules:

```python
import numpy as np

estimates, variances = [], []
for completed in imputed_datasets:          # e.g. from statsmodels MICE or sklearn IterativeImputer
    m = propensity_match(completed, "volume", "high", covariates=covariates)
    r = analysis.fit(m.matched_df, pair_id="PAIR_ID")
    estimates.append(np.log(r.wr))
    variances.append(r.inference["se_log_wr"] ** 2)
q = np.mean(estimates)
total_var = np.mean(variances) + (1 + 1 / len(estimates)) * np.var(estimates, ddof=1)
```

## Unmeasured confounding

{func}`~pywinratio.compute_e_value_for_ci` reports E-values (VanderWeele and Ding,
2017). The E-value is defined on the risk-ratio scale and a win ratio is not a
risk ratio, so treat it as a heuristic summary of robustness only.
