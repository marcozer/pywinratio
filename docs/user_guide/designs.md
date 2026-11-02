---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Designs

The same hierarchy can be analyzed under several designs. The design decides
which pairs are compared and therefore the estimand ([Concepts](concepts.md#estimands)).

```{code-cell} ipython3
import pandas as pd
from pywinratio import Outcome, WinRatio, propensity_match, estimate_propensity_weights
from pywinratio.datasets import make_surgical_cohort

pd.set_option("display.precision", 3)
cohort = make_surgical_cohort()
hierarchy = [
    Outcome.binary("death90", name="Death", terminal=True),
    Outcome.ordinal("clavien", name="Clavien-Dindo"),
    Outcome.ordinal("popf", name="POPF"),
    Outcome.binary("readmission", name="Readmission"),
    Outcome.continuous("los", name="Length of stay", margin=1),
]
analysis = WinRatio(hierarchy, group="volume", treatment="high", control="low")
```

## All pairs (randomized comparisons)

```{code-cell} ipython3
all_pairs = analysis.fit(cohort)
all_pairs.to_frame()
```

In a randomized trial this is the standard analysis. In this observational
cohort it is *unadjusted*: high-volume hospitals treat older, sicker patients.

## Stratified

Pass `strata` to compare patients only within strata, for example a design
factor of randomization or a period:

```{code-cell} ipython3
stratified = WinRatio(hierarchy, group="volume", treatment="high", control="low", strata=["period"]).fit(cohort)
stratified.strata()
```

The overall estimate sums the stratum counts (weights proportional to the number
of pairs per stratum, "Buyse" pooling in BuyseTest):

```{code-cell} ipython3
stratified.to_frame()
```

## Assigned (matched) pairs

After 1:1 matching, `pair_id` restricts comparisons to the two members of each
pair:

```{code-cell} ipython3
matching = propensity_match(
    cohort, "volume", "high",
    covariates=["age", "sex", "bmi", "asa3", "malignant"],
    exact_cols=["period"],
)
matched = analysis.fit(matching.matched_df, pair_id="PAIR_ID")
matched.to_frame()
```

Per-pair results show which tier decided each pair:

```{code-cell} ipython3
matched.pairs().head()
```

```{code-cell} ipython3
matched.pairs()["deciding_tier"].value_counts(dropna=False)
```

Any external pairing works (e.g. a matched-pair trial or a crossover with
pairs of periods): give each pair an identifier and one patient per arm. Pairs
without exactly one patient per arm are excluded with a warning.

## Weighted all pairs

With propensity-score weights, every cross-arm pair is weighted by the product of
the two patients' weights:

```{code-cell} ipython3
weighting = estimate_propensity_weights(
    cohort, treatment="volume", treated="high",
    covariates=["age", "sex", "bmi", "asa3", "malignant", "period"],
    estimand="overlap",
)
weighted = analysis.fit(weighting.weighted_df, weight_col="analysis_weight")
weighted.to_frame()
```

The analytic interval treats weights as fixed. To include propensity-model
uncertainty, use {func}`~pywinratio.bootstrap_propensity_weighted_gpc`
([Observational studies](observational.md)).

## Clustered data

Patients treated in the same hospital are correlated. `cluster_col` makes the
analytic variance cluster-robust and the bootstrap resample hospitals:

```{code-cell} ipython3
clustered = analysis.fit(cohort, cluster_col="hospital", cluster_within_arm=True, n_boot=1000, seed=1)
clustered.to_frame()
```

When the exposure itself is a hospital characteristic (hospital volume), use
`cluster_within_arm=True` so the number of hospitals per arm stays fixed in each
replicate, and consider a cluster permutation test (`n_perm=`), which permutes
whole hospitals when every hospital belongs to one arm.

## Side by side

```{code-cell} ipython3
designs = {
    "All pairs (unadjusted)": all_pairs,
    "Stratified by period": stratified,
    "Matched pairs": matched,
    "Overlap-weighted": weighted,
    "All pairs, cluster-robust": clustered,
}
pd.DataFrame(
    {
        name: {"WR": r.wr, "CI lower": r.ci[0], "CI upper": r.ci[1], "NB": r.net_benefit,
               "ties %": 100 * r.ties / r.n_pairs, "pairs": r.n_pairs}
        for name, r in designs.items()
    }
).T
```
