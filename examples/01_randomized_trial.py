"""Win ratio of a randomized trial with censored outcomes.

Run: python examples/01_randomized_trial.py
"""

from __future__ import annotations

from pywinratio import Outcome, WinRatio
from pywinratio.datasets import make_heart_failure_trial

trial = make_heart_failure_trial(n=600, seed=2012)

analysis = WinRatio(
    [
        Outcome.time_to_event("time_death", "death", name="Death", terminal=True),
        Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization"),
        Outcome.continuous("kccq_change", name="KCCQ change", favorable="higher", margin=5),
    ],
    group="arm",
    treatment="treatment",
    control="placebo",
)

# analytic (U-statistic) inference, bootstrap intervals and a permutation p-value
result = analysis.fit(trial, n_boot=2000, n_perm=5000, seed=1)
print(result)
print()
print(result.to_frame().round(4).to_string())
print()
print(result.tiers().round(3).to_string(index=False))
