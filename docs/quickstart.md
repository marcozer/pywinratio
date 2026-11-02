---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Quick start

This page analyzes a synthetic randomized heart-failure trial in four steps.

```{code-cell} ipython3
import pandas as pd
from pywinratio import Outcome, WinRatio
from pywinratio.datasets import make_heart_failure_trial

pd.set_option("display.precision", 3)
trial = make_heart_failure_trial()
trial.head()
```

## 1. Define the hierarchy

Outcomes are listed from most to least important. Each tier says which values
are better and when two values count as a tie.

```{code-cell} ipython3
outcomes = [
    # death first: later (or no) death is better; two deaths at the same time stop the comparison
    Outcome.time_to_event("time_death", "death", name="Death", terminal=True),
    # then the first heart-failure hospitalization
    Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization"),
    # then quality of life: higher is better, differences of 5 points or less tie
    Outcome.continuous("kccq_change", name="KCCQ change", favorable="higher", margin=5),
]
for tier in outcomes:
    print(tier.describe())
```

## 2. Fit

```{code-cell} ipython3
analysis = WinRatio(outcomes, group="arm", treatment="treatment", control="placebo")
result = analysis.fit(trial)
print(result)
```

Every treatment patient was compared with every placebo patient (300 x 300 =
90,000 pairs). The treatment arm won 52% of the comparisons, lost 41% and tied
7%: a win ratio of about 1.27 and a net benefit of 0.11, with asymptotic
(U-statistic) 95% confidence intervals.

## 3. Tables

```{code-cell} ipython3
result.to_frame()
```

```{code-cell} ipython3
result.tiers()
```

`tiers()` shows where comparisons were decided: the share of pairs separated by
death, then by hospitalization among pairs tied on death, and so on.

## 4. Bootstrap, permutation test and a figure

```{code-cell} ipython3
result = analysis.fit(trial, n_boot=2000, n_perm=2000, seed=1)
result.to_frame()
```

```{code-cell} ipython3
ax = result.plot("tiers")
```

## Next steps

- [Concepts](user_guide/concepts.md): what the win ratio, net benefit and win odds estimate.
- [Defining outcomes](user_guide/outcomes.md): every tier option.
- [Designs](user_guide/designs.md): stratified, matched, weighted and clustered analyses.
- [Inference](user_guide/inference.md): which interval and test to report.
