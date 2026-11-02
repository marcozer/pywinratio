---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Defining outcomes

A hierarchy is a list of `Outcome` objects (an alias of
{class}`~pywinratio.WinRatioOutcome`), highest priority first. Use the factory
methods; {func}`~pywinratio.compare_patients` shows how any two patients are
compared, which is the quickest way to check a definition.

```{code-cell} ipython3
from pywinratio import Outcome, compare_patients
```

## Binary outcomes

Binary columns must be coded 0/1 with 1 = event. Adverse events are the default
(`favorable="lower"`: no event beats event); use `favorable="higher"` for
desirable events such as textbook outcome.

```{code-cell} ipython3
death = Outcome.binary("death90", name="90-day mortality", terminal=True)
textbook = Outcome.binary("textbook", name="Textbook outcome", favorable="higher")
compare_patients({"death90": 0}, {"death90": 1}, [death])
```

Strings such as `"yes"`/`"no"` or codes such as 1/2 raise an error rather than
being silently treated as missing.

## Ordinal outcomes

Any ordered numeric coding works; only the order matters. `cutpoints=` collapses
grades before comparing.

```{code-cell} ipython3
clavien = Outcome.ordinal("clavien", name="Clavien-Dindo grade")          # 0 / 3 / 4
popf = Outcome.ordinal("popf", name="Pancreatic fistula (none/B/C)")     # 0 / 1 / 2
major_only = Outcome.ordinal("clavien", cutpoints=[2])                   # <=2 vs >=3
compare_patients({"clavien": 3}, {"clavien": 4}, [clavien])
```

## Continuous outcomes and clinical margins

`margin=` ties absolute differences up to the margin. For completed days of
length of stay, `margin=1` means a patient needs a stay at least 2 days shorter
to win.

```{code-cell} ipython3
los = Outcome.continuous("los", name="Length of stay", margin=1)
[compare_patients({"los": 6}, {"los": b}, [los])["result"] for b in (5, 6, 7, 8)]
```

`threshold=` uses the convention of the R package BuyseTest: a win needs a
difference of at least the threshold. On integer data `margin=m` equals
`threshold=m + 1`.

```{code-cell} ipython3
same_rule = Outcome.continuous("los", threshold=2)
[compare_patients({"los": 6}, {"los": b}, [same_rule])["result"] for b in (5, 6, 7, 8)]
```

For scores where higher is better use `favorable="higher"`. Margins are
protected against floating-point noise: 1.1 versus 1.0 ties at a margin of 0.1.

### Categorized continuous outcomes

```{code-cell} ipython3
los_categories = Outcome.continuous("los", cutpoints=[7, 13])   # <=7, 8-13, >13 days
[compare_patients({"los": 7}, {"los": b}, [los_categories])["result"] for b in (5, 8, 14)]
```

## Time-to-event outcomes with censoring

```{code-cell} ipython3
death_tte = Outcome.time_to_event("time_death", "death", name="Death", terminal=True)
hosp_tte = Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization", horizon=12)
```

`time` is the follow-up time and `event` the indicator (1 = event, 0 = censored).
The Gehan rule compares two patients within their common follow-up:

```{code-cell} ipython3
cases = {
    "B died at 3, A censored at 5": ({"t": 5, "e": 0}, {"t": 3, "e": 1}),
    "A died at 5, B censored at 3": ({"t": 5, "e": 1}, {"t": 3, "e": 0}),
    "both died, A later": ({"t": 6, "e": 1}, {"t": 2, "e": 1}),
    "A censored when B died": ({"t": 4, "e": 0}, {"t": 4, "e": 1}),
}
tier = [Outcome.time_to_event("t", "e")]
{case: compare_patients(a, b, tier)["result"] for case, (a, b) in cases.items()}
```

The second case is *indeterminate*: B may have died later than 5 or not, so the
next tier decides. A censoring time equal to the other patient's event time
counts as event-free at that time (Gehan, 1965), as in BuyseTest.

Options: `favorable="earlier"` for desirable events (discharge, recovery),
`margin=` / `threshold=` for minimal time differences, `horizon=` to truncate
follow-up.

## Terminal events

With `terminal=True`, two patients who both had the event and are not separated
by the tier are declared tied, and lower tiers are skipped:

```{code-cell} ipython3
hierarchy = [death, los]
compare_patients({"death90": 1, "los": 3}, {"death90": 1, "los": 40}, hierarchy)
```

Without `terminal=True` the two deaths would be ranked by their length of stay.
For ordinal tiers give the grade that is terminal, e.g.
`Outcome.ordinal("clavien", terminal=True, terminal_value=5)`.

## Missing values

Each tier has a missing-data policy:

| `missing_is` | Effect |
|---|---|
| `"tie"` (default) | the tier is uninformative; the next tier decides |
| `"loss"` | the patient with the missing value loses the tier (worst case) |
| `"win"` | the patient with the missing value wins the tier (best case) |

A pair with both values missing always ties. Policies are symmetric: swapping
the arms inverts the win ratio.

```{code-cell} ipython3
worst_case = [Outcome.binary("readmission", missing_is="loss")]
compare_patients({"readmission": None}, {"readmission": 0}, worst_case)
```

Because the default treats missing values as uninformative, it is unbiased only
if values are missing at random within arms. Report the extent of missing data
and check robustness with `missing_is="loss"` for each arm
([Sensitivity analyses](sensitivity.md)), or impute outcomes beforehand.

## Checking a hierarchy

```{code-cell} ipython3
for tier in [death, clavien, popf, Outcome.binary("readmission"), los]:
    print("-", tier.describe())
```
