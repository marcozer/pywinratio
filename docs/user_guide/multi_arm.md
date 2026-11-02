---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Several arms

Win statistics compare two groups at a time. With three or more arms, compare
each arm with a reference (or every pair of arms) and adjust the p-values for
multiplicity.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from pywinratio import Outcome, pairwise_win_ratios, plot_forest
from pywinratio.datasets import make_heart_failure_trial

pd.set_option("display.precision", 3)
trial = make_heart_failure_trial(n=900, seed=5)
# split the treatment arm into two doses for the example
rng = np.random.default_rng(0)
trial["dose"] = np.where(trial["arm"] == "placebo", "placebo", rng.choice(["low dose", "high dose"], len(trial)))
hierarchy = [
    Outcome.time_to_event("time_death", "death", name="Death", terminal=True),
    Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization"),
]
table = pairwise_win_ratios(trial, hierarchy, group="dose", reference="placebo", adjust="holm")
table[["comparison", "wr", "ci_lower", "ci_upper", "net_benefit", "p_value", "p_value_adjusted"]]
```

`adjust` accepts `"holm"` (default), `"bonferroni"` and `"fdr_bh"`
(Benjamini–Hochberg). Without `reference` all pairs of arms are compared.

```{code-cell} ipython3
ax = plot_forest(table, label="comparison")
```

Any argument of {meth}`WinRatio.fit() <pywinratio.WinRatioAnalysis.fit>` can be
passed (e.g. `n_boot=2000`, `n_perm=5000`, `cluster_col="site"`). The
dictionary-based functions {func}`~pywinratio.compute_win_ratio_multi_arm` and
{func}`~pywinratio.bootstrap_win_ratio_multi_arm` with a
{class}`~pywinratio.MultiArmWinRatioConfig` are also available.

A win ratio is not transitive: A may beat B and B beat C while C beats A. Do
not rank more than two arms from pairwise win ratios alone.
