---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Reporting

## Checklist

Recommendations for reporting hierarchical composite outcomes (Redfors et al.,
2020; Pocock et al., 2024; CONSORT-Outcomes, Butcher et al., 2022) can be
summarized as:

1. Prespecify the hierarchy with its clinical rationale, fatal events first,
   and document any change.
2. Define each tier's win, loss and tie rule: direction, margins or thresholds,
   cut points, terminal events, follow-up horizon.
3. State the pairing and the estimand: all pairs, stratified (and stratum
   weights), matched pairs, or weighted.
4. Describe censoring and follow-up for time-to-event tiers.
5. Prespecify the missing-data rules.
6. Report the win ratio with its CI and p-value, plus the net benefit and the
   proportions of wins, losses and ties.
7. Report wins, losses and ties tier by tier.
8. Report conventional estimates for each component, as descriptive results.
9. Name the inference method, the software and its version (e.g. "pywinratio 0.3.0,
   U-statistic variance").
10. Describe the sample-size method and its assumptions.
11. Label subgroup and post hoc analyses as exploratory; compare subgroups by
    interaction on the log win ratio.

pywinratio produces the numbers for items 6–9 directly.

```{code-cell} ipython3
import pandas as pd
import pywinratio
from pywinratio import Outcome, WinRatio, summarize_component_outcomes
from pywinratio.datasets import make_heart_failure_trial

pd.set_option("display.precision", 3)
trial = make_heart_failure_trial()
analysis = WinRatio(
    [
        Outcome.time_to_event("time_death", "death", name="Death", terminal=True),
        Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization"),
        Outcome.continuous("kccq_change", name="KCCQ change", favorable="higher", margin=5),
    ],
    group="arm", treatment="treatment", control="placebo",
)
result = analysis.fit(trial, n_boot=2000, seed=2026)
```

## Main result (item 6)

```{code-cell} ipython3
result.to_frame()
```

```{code-cell} ipython3
shares = pd.Series({"wins": result.wins, "losses": result.losses, "ties": result.ties}) / result.n_pairs
shares.round(3)
```

## Tier table (item 7)

```{code-cell} ipython3
result.tiers()[["priority", "outcome", "entering", "wins", "losses", "terminal_ties", "conditional_resolution", "net_benefit_contribution"]]
```

```{code-cell} ipython3
ax = result.plot("tiers", arm_labels=("Treatment", "Placebo"))
```

## Components (item 8)

```{code-cell} ipython3
summarize_component_outcomes(trial, analysis.config())[["outcome", "kind", "A_n", "A_events", "A_rate", "B_events", "B_rate", "A_median", "B_median"]]
```

## Methods (item 9)

```{code-cell} ipython3
print(f"pywinratio {pywinratio.__version__}; CI: {result.to_frame()['ci_method'].iloc[0]}; "
      f"analytic SE: {result.inference['method']}")
```

## Exporting

```{code-cell} ipython3
payload = result.to_dict()          # JSON-serializable; result.to_json("result.json") writes a file
sorted(payload)
```

`result.to_frame()` and `result.tiers()` are pandas data frames: use
`.to_csv()`, `.to_excel()` or `.to_latex()` for manuscripts. In notebooks the
result displays as an HTML table.

## Suggested wording

> The win ratio was estimated by comparing every patient of the treatment arm with
> every patient of the placebo arm on the prespecified hierarchy (death, first
> heart-failure hospitalization, change in KCCQ score with differences of 5
> points or less considered ties), using the Gehan rule for censored times
> (Pocock et al., 2012). We report the win ratio, net benefit and win odds with
> 95% confidence intervals from 2000 bootstrap replicates and tier-specific
> wins, losses and ties. Analyses used pywinratio (version X.Y.Z; DOI ...).
