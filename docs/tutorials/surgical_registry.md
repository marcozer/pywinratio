---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# An observational surgical registry

Do patients operated on in high-volume hospitals do better? This tutorial
compares high- and low-volume hospitals in a synthetic pancreatic-surgery
registry with a five-tier postoperative hierarchy, adjusting for case mix by
matching and by weighting, and accounting for hospital clustering.

```{code-cell} ipython3
import pandas as pd
from pywinratio import (
    Outcome, WinRatio, WinRatioConfig, propensity_match, estimate_propensity_weights,
    bootstrap_propensity_matched_win_ratio, paired_risk_difference_bootstrap,
    compare_hierarchies, plot_balance, plot_forest,
)
from pywinratio.datasets import make_surgical_cohort

pd.set_option("display.precision", 3)
cohort = make_surgical_cohort(n=800, seed=2026)
cohort.groupby("volume")[["age", "asa3", "malignant", "death90", "readmission", "los"]].mean()
```

High-volume hospitals operate on older, sicker patients with more cancers: an
unadjusted comparison is confounded.

## The hierarchy

```{code-cell} ipython3
hierarchy = [
    Outcome.binary("death90", name="90-day death", terminal=True),
    Outcome.ordinal("clavien", name="Clavien-Dindo (none, III, IV)"),
    Outcome.ordinal("popf", name="Pancreatic fistula (none, B, C)"),
    Outcome.binary("readmission", name="Readmission"),
    Outcome.continuous("los", name="Length of stay (0-1 day ties)", margin=1),
]
analysis = WinRatio(hierarchy, group="volume", treatment="high", control="low")
unadjusted = analysis.fit(cohort, cluster_col="hospital")
unadjusted.to_frame()
```

## Design 1: propensity-score matching

```{code-cell} ipython3
covariates = ["age", "sex", "bmi", "asa3", "malignant"]
matching = propensity_match(cohort, "volume", "high", covariates=covariates, exact_cols=["period"])
matching.summary()
```

```{code-cell} ipython3
ax = plot_balance(matching.balance_before, matching.balance_after)
```

```{code-cell} ipython3
matched = analysis.fit(matching.matched_df, pair_id="PAIR_ID", n_boot=2000, n_perm=5000, seed=1)
print(matched)
```

Which tiers decided the matched pairs?

```{code-cell} ipython3
matched.pairs()["deciding_tier"].fillna("tie").value_counts()
```

### Comparison with binary composites

Binary composites such as "textbook outcome" discard the grading of
complications. With matched pairs they are compared with paired risk
differences:

```{code-cell} ipython3
matched_df = matching.matched_df.assign(
    any_event=lambda d: ((d.death90 == 1) | (d.clavien >= 3) | (d.popf >= 1) | (d.readmission == 1)).astype(int)
)
binary_cfg = WinRatioConfig("volume", "high", "low", [Outcome.binary("any_event", name="Any adverse event")],
                            pair_strategy="matched", pair_id="PAIR_ID")
paired_risk_difference_bootstrap(matched_df, binary_cfg, n_boot=2000)[
    ["outcome", "A_risk", "B_risk", "risk_difference_A_minus_B", "ci_lower", "ci_upper",
     "A_event_B_no_event", "A_no_event_B_event"]
]
```

Pairs concordant on the binary composite (both with or both without an event)
carry no information there; the hierarchy can still separate many of them by
complication grade or length of stay.

## Design 2: overlap weighting

```{code-cell} ipython3
weighting = estimate_propensity_weights(cohort, treatment="volume", treated="high",
                                        covariates=covariates + ["period"])
weighted = analysis.fit(weighting.weighted_df, weight_col="analysis_weight", cluster_col="hospital",
                        cluster_within_arm=True, n_boot=1000, seed=2)
weighted.to_frame()
```

## Hospital-level uncertainty

Volume is a hospital characteristic, so the effective sample size is closer to
the number of hospitals than to the number of patients. Resample hospitals
within arm, refit the score and rematch in every replicate (use `n_jobs=-1` and
2000+ replicates in practice):

```{code-cell} ipython3
cfg = WinRatioConfig("volume", "high", "low", hierarchy)
refit = bootstrap_propensity_matched_win_ratio(
    cohort, cfg, covariates=covariates, exact_cols=["period"], cluster_col="hospital", n_boot=200, seed=3,
)
{k: refit[k] for k in ("wr", "ci", "net_benefit_ci", "n_successful")}
```

A cluster permutation test answers the same question for the p-value:

```{code-cell} ipython3
analysis.fit(cohort, cluster_col="hospital", n_perm=5000, seed=4).permutation
```

## Summary of designs

```{code-cell} ipython3
summary = pd.DataFrame(
    [
        {"design": "Unadjusted, cluster-robust", "estimate": unadjusted.wr, "ci_lower": unadjusted.ci[0], "ci_upper": unadjusted.ci[1]},
        {"design": "Matched pairs, pair bootstrap", "estimate": matched.wr, "ci_lower": matched.ci[0], "ci_upper": matched.ci[1]},
        {"design": "Matched, hospital refit bootstrap", "estimate": refit["wr"], "ci_lower": refit["ci"][0], "ci_upper": refit["ci"][1]},
        {"design": "Overlap-weighted, hospital bootstrap", "estimate": weighted.wr, "ci_lower": weighted.ci[0], "ci_upper": weighted.ci[1]},
    ]
)
ax = plot_forest(summary, label="design")
summary
```

The interval that resamples hospitals and re-estimates the propensity score and
the match is the widest, because it reflects every source of uncertainty of the
matched design. Prespecify the interval that matches the question (see
[Inference](../user_guide/inference.md)).

## Sensitivity to the length-of-stay tier

```{code-cell} ipython3
death, clavien, popf, readm, los = hierarchy
variants = {
    "Primary (LOS 0-1 day ties)": hierarchy,
    "Exact LOS": [death, clavien, popf, readm, Outcome.continuous("los", name="LOS")],
    "LOS categories": [death, clavien, popf, readm, Outcome.continuous("los", cutpoints=[7, 13], name="LOS")],
    "Without LOS": [death, clavien, popf, readm],
}
compare_hierarchies(matching.matched_df, variants, group="volume", treatment="high", control="low",
                    pair_id="PAIR_ID")[["hierarchy", "wr", "ci_lower", "ci_upper", "net_benefit", "tie_share"]]
```
