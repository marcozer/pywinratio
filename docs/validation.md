---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Validation

pywinratio is checked against the two reference implementations in R:

- BuyseTest 3.3.9 (Ozenne and Péron; the generalized pairwise comparison
  package from the group that introduced the method), and
- WINS 1.5.1 (Cui and Huang; win statistics with the variance estimators of
  Dong et al.).

The comparison is fully scripted in the repository's `validation/` folder:

```bash
python validation/generate_data.py      # synthetic data sets (CSV)
Rscript validation/reference_r.R        # R reference values (JSON, committed)
python validation/compare.py            # agreement table; exit code 1 on disagreement
```

The committed reference values are re-checked by the test suite on every change
(`tests/test_validation_r.py`), without needing R.

## Scenarios

| Scenario | Data | Hierarchy | Reference |
|---|---|---|---|
| S1 | randomized trial, 600 patients | death (censored, Gehan) > HF hospitalization (censored) > KCCQ (threshold 5 points) | BuyseTest |
| S2 | observational cohort, 800 patients | death > Clavien-Dindo grade > fistula grade > readmission > LOS (margin 1 day) | BuyseTest |
| S3 | as S2, stratified by period | idem, strata pooled by number of pairs | BuyseTest |
| S4 | 344 propensity-matched pairs | idem, assigned pairs | BuyseTest (one stratum per pair) |
| S5 | trial of S1 | death > HF hospitalization > KCCQ (no threshold) | WINS (WR, NB, WO and CIs) |
| S6 | 6 patients | censoring time equal to an event time | BuyseTest (Gehan convention) |

## Agreement

```{code-cell} ipython3
import pandas as pd
from pathlib import Path

root = Path.cwd()
while not (root / "validation").exists() and root != root.parent:
    root = root.parent
table = pd.read_csv(root / "validation" / "results" / "agreement.csv")
pd.set_option("display.float_format", lambda v: f"{v:.6g}")
table[["scenario", "reference", "statistic", "r_value", "pywinratio", "abs_diff", "agree"]]
```

```{code-cell} ipython3
point = table[table["statistic"].isin(["wins", "losses", "wr", "net_benefit", "win_odds"])]
print(f"largest absolute difference in point estimates: {point['abs_diff'].max():.1e}")
```

Point estimates (wins, losses, win ratio, net benefit, win odds) agree to
machine precision in every scenario. Standard errors agree to about $10^{-4}$:
pywinratio applies the usual $n/(n-1)$ small-sample factor to the U-statistic
variance (as DeLong et al. do), BuyseTest does not, and WINS uses a slightly
different variance estimator; the differences vanish as $n$ grows.

## Conventions that differ between packages

| Topic | pywinratio | BuyseTest | WINS |
|---|---|---|---|
| Continuous tie rule | `margin=m`: differences $\le m$ tie; `threshold=t`: differences $< t$ tie | differences $< t$ tie (`threshold`) | `tau` |
| Censoring time equal to the other patient's event time | censored patient wins (Gehan, 1965) | same | tie |
| Floating-point margins | relative tolerance $10^{-9}$ (1.1 − 1.0 ties at 0.1) | raw comparison | raw comparison |
| Event exactly at the `horizon` vs a patient event-free at the horizon | event counts (the event-free patient wins) | `restriction`: neutral | not checked |
| Two deaths, same time, terminal tier | tie, lower tiers skipped (`terminal=True`) | neutral, next endpoint | next endpoint |
| Stratum weights | number of pairs ("Buyse") | several options | several options |

## Internal checks

The test suite also verifies, among others:

- equality of the vectorized engine with a brute-force pair-by-pair
  implementation on random hierarchies of every tier type, with censoring,
  missing values and terminal events;
- symmetry (swapping the arms inverts the win ratio and the sign of the net
  benefit), row-order invariance, stratified totals equal to sums of strata,
  matched results equal to the diagonal of the all-pairs comparisons;
- the U-statistic variance against an explicit DeLong computation and the
  coverage of the intervals by simulation;
- the permutation test against exact enumeration, and the sample-size formula
  against simulated power.
