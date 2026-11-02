---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Sample size and power

## Closed-form sample size

{func}`~pywinratio.sample_size_win_ratio` implements the formula of Yu and Ganju
(2022) for a two-sided test of WR = 1 with an all-pairs (unmatched) analysis:

$$
N = \frac{4\,(1 + p_T)\,(z_{1-\alpha/2} + z_{1-\beta})^2}{3\,k(1-k)\,(1 - p_T)\,(\log \mathrm{WR})^2},
$$

where $p_T$ is the anticipated proportion of tied pairs and $k$ the allocation
fraction to arm A.

```{code-cell} ipython3
import pandas as pd
from pywinratio import sample_size_win_ratio

pd.DataFrame(
    [sample_size_win_ratio(wr, p_tie=p_tie) for wr in (1.2, 1.3, 1.5) for p_tie in (0.0, 0.2, 0.4)]
)[["wr", "p_tie", "n_total", "n_a", "n_b"]]
```

The formula approximates the null variance of the win ratio by that of an
ordinal outcome with equally likely categories. Ties inflate the required
sample size quickly; anticipate $p_T$ from pilot data or a previous trial, and
confirm by simulation.

## Simulation-based power

{func}`~pywinratio.simulate_power` runs any analysis on simulated trials. The
generator receives a NumPy random generator and returns a data frame.

```{code-cell} ipython3
from pywinratio import Outcome, WinRatio, simulate_power
from pywinratio.datasets import make_heart_failure_trial

analysis = WinRatio(
    [
        Outcome.time_to_event("time_death", "death", name="Death", terminal=True),
        Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization"),
        Outcome.continuous("kccq_change", name="KCCQ", favorable="higher", margin=5),
    ],
    group="arm", treatment="treatment", control="placebo",
)

def one_trial(rng, n=800):
    return make_heart_failure_trial(n=n, seed=int(rng.integers(2**31)))

simulate_power(one_trial, analysis, n_sim=100, seed=1)
```

Use a few hundred to a few thousand simulated trials for a final design, and
vary the effect sizes, follow-up and censoring in the generator.
