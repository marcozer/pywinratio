# pywinratio

pywinratio computes the win ratio, the net benefit and the win odds for
hierarchical composite outcomes in Python.

Patients are compared two by two on a list of outcomes ranked by importance.
The most important outcome is looked at first, for example death. When it does
not separate the two patients, the next one decides, and so on. Wins, losses and
ties counted over all pairs give the win ratio (Pocock et al., 2012), the net
benefit (Buyse, 2010) and the win odds (Brunner et al., 2021).

```python
from pywinratio import Outcome, win_ratio
from pywinratio.datasets import make_heart_failure_trial

trial = make_heart_failure_trial()
result = win_ratio(
    trial,
    [
        Outcome.time_to_event("time_death", "death", name="Death", terminal=True),
        Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization"),
        Outcome.continuous("kccq_change", name="KCCQ", favorable="higher", margin=5),
    ],
    group="arm", treatment="treatment", control="placebo",
)
print(result)
```

The package handles binary, ordinal, continuous and censored time-to-event
outcomes. The design can be a randomized trial, a stratified analysis, matched
pairs or a propensity-score weighted cohort, with or without clustering by
hospital. Confidence intervals come from U-statistic theory, the bootstrap or
permutation. The results agree with the R packages BuyseTest and WINS (see
[Validation](validation.md)).

If you are new to win statistics, start with [Concepts](user_guide/concepts.md).
The [Quick start](quickstart.md) gets you to a result in a few minutes, and the
two tutorials walk through a [randomized trial](tutorials/heart_failure_trial.md)
and an [observational registry](tutorials/surgical_registry.md). Every function
is described in the [API reference](api/index.md).

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
user_guide/index
tutorials/index
validation
comparison
api/index
faq
references
citing
changelog
contributing
```
