---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A randomized heart-failure trial

This tutorial reproduces the classical win-ratio analysis of a cardiovascular
trial (Pocock et al., 2012): death first, then heart-failure hospitalization,
then a patient-reported outcome, with censored follow-up.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from pywinratio import Outcome, WinRatio, compare_hierarchies, plot_forest, summarize_component_outcomes
from pywinratio.datasets import make_heart_failure_trial

pd.set_option("display.precision", 3)
trial = make_heart_failure_trial(n=600, seed=2012)
trial.describe(include="all").T[["count", "mean", "unique", "top"]]
```

The data hold 600 randomized patients followed 12–36 months, with the time to
death, the time to first heart-failure hospitalization (censored at death or end
of follow-up) and the 12-month change in the Kansas City Cardiomyopathy
Questionnaire (KCCQ; missing for patients who died before 12 months).

## Conventional analysis of the components

```{code-cell} ipython3
hierarchy = [
    Outcome.time_to_event("time_death", "death", name="Death", terminal=True),
    Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization"),
    Outcome.continuous("kccq_change", name="KCCQ change", favorable="higher", margin=5),
]
analysis = WinRatio(hierarchy, group="arm", treatment="treatment", control="placebo")
components = summarize_component_outcomes(trial, analysis.config())
components[["outcome", "A_events", "A_rate", "B_events", "B_rate", "A_median", "B_median"]]
```

## The win ratio

```{code-cell} ipython3
result = analysis.fit(trial, n_boot=2000, n_perm=5000, seed=2012)
print(result)
```

How each tier contributed:

```{code-cell} ipython3
result.tiers()
```

```{code-cell} ipython3
ax = result.plot("tiers", arm_labels=("Treatment", "Placebo"))
```

Most pairs were separated by death or hospitalization within the common
follow-up; the KCCQ tier decided the pairs that were event-free in both
patients. Two deaths at exactly the same time would have stopped as a terminal
tie.

## Censoring and the follow-up horizon

Pairs in which the earlier follow-up ends by censoring cannot be ranked on that
tier. Truncating follow-up at a common horizon makes every comparison refer to
the same period:

```{code-cell} ipython3
horizons = {
    f"Horizon {h} months": [
        Outcome.time_to_event("time_death", "death", name="Death", terminal=True, horizon=h),
        Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization", horizon=h),
        hierarchy[2],
    ]
    for h in (12, 24)
}
horizons["Full follow-up (primary)"] = hierarchy
table = compare_hierarchies(trial, horizons, group="arm", treatment="treatment", control="placebo")
table[["hierarchy", "wr", "ci_lower", "ci_upper", "net_benefit", "tie_share"]]
```

## Margin of the patient-reported outcome

```{code-cell} ipython3
margins = {
    f"KCCQ margin {m}": hierarchy[:2] + [Outcome.continuous("kccq_change", name="KCCQ", favorable="higher", margin=m)]
    for m in (0, 5, 10)
}
margins["Events only"] = hierarchy[:2]
sens = compare_hierarchies(trial, margins, group="arm", treatment="treatment", control="placebo")
ax = plot_forest(sens, label="hierarchy")
sens[["hierarchy", "wr", "ci_lower", "ci_upper", "net_benefit", "tie_share"]]
```

Without the quality-of-life tier more than a third of the pairs are tied (both
patients event-free), so the win ratio is computed from fewer informative pairs
and its interval is wider, while the net benefit shrinks towards zero. Widening
the KCCQ margin moves pairs from the KCCQ tier to ties, with the same effects
on a smaller scale.

## Cluster-robust inference by site

Patients of the same site may resemble each other. `cluster_col` sums the
influence of each site's patients before computing the variance:

```{code-cell} ipython3
clustered = analysis.fit(trial, cluster_col="site")
clustered.to_frame()[["estimate", "ci_lower", "ci_upper", "ci_method"]]
```

The simulated sites have no site effect, so the cluster-robust interval is close
to the ordinary one; with real multicenter data it is usually wider. To compare
patients only within sites, use `strata=["site"]` instead (a stratified win
ratio).

## Planning the next trial

```{code-cell} ipython3
from pywinratio import sample_size_win_ratio

tie_share = result.ties / result.n_pairs
sample_size_win_ratio(round(result.wr, 2), p_tie=tie_share, power=0.9)
```
