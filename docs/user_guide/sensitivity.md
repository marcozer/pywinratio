---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Sensitivity analyses

A hierarchy embeds clinical judgements: which outcomes enter, in which order,
and which differences matter. Prespecify the primary hierarchy and show how the
conclusion depends on these choices (Redfors et al., 2020; Pocock et al., 2024).

```{code-cell} ipython3
import pandas as pd
from pywinratio import Outcome, compare_hierarchies, order_sensitivity, plot_forest
from pywinratio.datasets import make_surgical_cohort

pd.set_option("display.precision", 3)
cohort = make_surgical_cohort()
death = Outcome.binary("death90", name="Death", terminal=True)
clavien = Outcome.ordinal("clavien", name="Clavien-Dindo")
popf = Outcome.ordinal("popf", name="POPF")
readmission = Outcome.binary("readmission", name="Readmission")
los = Outcome.continuous("los", name="LOS (margin 1 day)", margin=1)
primary = [death, clavien, popf, readmission, los]
```

## Alternative hierarchies, margins and missing-data rules

```{code-cell} ipython3
variants = {
    "Primary": primary,
    "LOS exact days": [death, clavien, popf, readmission, Outcome.continuous("los", name="LOS")],
    "LOS margin 2 days": [death, clavien, popf, readmission, Outcome.continuous("los", margin=2, name="LOS")],
    "LOS categories <=7/8-13/>13": [death, clavien, popf, readmission, Outcome.continuous("los", cutpoints=[7, 13], name="LOS")],
    "Without LOS": [death, clavien, popf, readmission],
    "POPF before Clavien-Dindo": [death, popf, clavien, readmission, los],
    "Major complication as binary": [death, Outcome.ordinal("clavien", cutpoints=[0], name="Major"), popf, readmission, los],
}
table = compare_hierarchies(cohort, variants, group="volume", treatment="high", control="low")
table[["hierarchy", "wr", "ci_lower", "ci_upper", "net_benefit", "tie_share", "p_value"]]
```

```{code-cell} ipython3
ax = plot_forest(table, label="hierarchy")
```

The tie share explains much of the variation: removing or coarsening the last
tier leaves more pairs tied, which changes the win ratio far more than the net
benefit.

## Every ordering of the tiers

{func}`~pywinratio.order_sensitivity` refits the analysis under every permutation
of the tiers, optionally keeping the first tiers fixed (death usually stays first):

```{code-cell} ipython3
orders = order_sensitivity(cohort, primary, group="volume", treatment="high", control="low", fixed=1)
orders[["hierarchy", "wr", "net_benefit", "is_prespecified"]].sort_values("wr").head(8)
```

```{code-cell} ipython3
orders["wr"].describe()[["min", "50%", "max"]]
```

## Missing outcome data

Repeat the analysis with missing values counted as losses (worst case for the
patient with missing data) or as wins (best case). Only the tier rule changes:

```{code-cell} ipython3
with_missing = cohort.copy()
with_missing.loc[with_missing.sample(frac=0.08, random_state=1).index, "readmission"] = None
rules = {
    "missing = uninformative": [death, clavien, popf, Outcome.binary("readmission"), los],
    "missing = loss": [death, clavien, popf, Outcome.binary("readmission", missing_is="loss"), los],
    "missing = win": [death, clavien, popf, Outcome.binary("readmission", missing_is="win"), los],
}
compare_hierarchies(with_missing, rules, group="volume", treatment="high", control="low")[["hierarchy", "wr", "ci_lower", "ci_upper"]]
```

## Unmeasured confounding

```{code-cell} ipython3
from pywinratio import compute_e_value_for_ci

row = table.iloc[0]
compute_e_value_for_ci(row["wr"], row["ci_lower"], row["ci_upper"])["interpretation"]
```

E-values are defined for risk ratios; for a win ratio they are only a rough
guide (see [Observational studies](observational.md#unmeasured-confounding)).
