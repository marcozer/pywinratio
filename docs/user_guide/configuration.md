---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Configuration files

A prespecified analysis can live in a version-controlled YAML (or JSON) file.
{func}`~pywinratio.config_from_dict` turns a mapping into a
{class}`~pywinratio.WinRatioConfig`.

```{code-cell} ipython3
import yaml
from pywinratio import config_from_dict, WinRatio
from pywinratio.datasets import make_heart_failure_trial

spec = yaml.safe_load("""
group_col: arm
arm_a: treatment
arm_b: placebo
pair_strategy: all_pairs
outcomes:
  - name: Death
    column: time_death
    event_column: death
    kind: time_to_event
    direction: higher        # later death is better
    terminal: true
  - name: HF hospitalization
    column: time_hosp
    event_column: hosp
    kind: time_to_event
    direction: higher
    horizon: 24              # months
  - name: KCCQ change
    column: kccq_change
    kind: continuous
    direction: higher
    tie_tol: 5               # differences <= 5 points tie
    missing_is: tie
""")
cfg = config_from_dict(spec)
for outcome in cfg.outcomes:
    print(outcome.describe())
```

```{code-cell} ipython3
analysis = WinRatio(cfg.outcomes, group=cfg.group_col, treatment=cfg.arm_a, control=cfg.arm_b, strata=cfg.strata)
analysis.fit(make_heart_failure_trial()).to_frame()
```

## Keys

Analysis keys:

| Key | Meaning |
|---|---|
| `group_col` | arm column |
| `arm_a`, `arm_b` | values of the treatment and control arms (WR > 1 favors `arm_a`) |
| `outcomes` | list of tiers, highest priority first |
| `strata` | list of stratification columns (optional) |
| `pair_strategy` | `all_pairs` (default) or `matched` |
| `pair_id` | matched-pair identifier for `pair_strategy: matched` |

Outcome keys:

| Key | Values | Default |
|---|---|---|
| `name` | label | the column name |
| `column` | outcome (or time) column | required |
| `kind` | `binary`, `ordinal`, `continuous`, `time_to_event` | required |
| `direction` | `lower` or `higher` values (later times) are better | required |
| `tie_tol` | tie margin (differences `<=` margin tie) | 0 |
| `tie_inclusive` | `false` switches to the threshold convention (differences `<` tie) | `true` |
| `cutpoints` | ascending category boundaries | none |
| `terminal`, `terminal_value` | stop when both patients have the terminal value | `false`, 1 |
| `missing_is` | `tie`, `loss`, `win` | `tie` |
| `event_column` | event indicator of a time-to-event tier | required for `time_to_event` |
| `horizon` | truncation time of a time-to-event tier | none |
| `los_tolerance_mode`, `los_threshold` | legacy options of WinRatioPy 0.2 | `exact` |

Unknown keys raise an error, so typos are caught early.
{func}`~pywinratio.multi_arm_config_from_dict` builds a
{class}`~pywinratio.MultiArmWinRatioConfig` with `arms`, `arm_labels`,
`reference_arm` and `comparison_pairs`.
