---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Case study: a configuration-driven registry analysis

pywinratio was developed for a national registry study of minimally invasive
distal pancreatectomy comparing higher- and lower-volume centers on a
hierarchical postoperative outcome. The repository ships the analysis pipeline
of that study, run on synthetic records, as a template for a complete,
prespecified and reproducible application. It lives in
`studies/distal_pancreatectomy/` (not installed with the package).

## What the pipeline does

1. reads the prespecified analysis from `config/primary_analysis.yaml`
   (exposure, hierarchy, matching, weighting, sensitivity analyses);
2. compares the arms on all pairs with a center-within-arm bootstrap
   (unadjusted);
3. matches 1:1 on the propensity score (complete cases, optimal matching,
   exact strata) and estimates the assigned-pair win ratio, with a
   center-resampling, propensity-refit, rematch bootstrap;
4. reports balance (SMD, variance ratio, KS), common support, retention and
   propensity-score overlap;
5. runs the hierarchy sensitivities (exact-day, 3-day and categorical length of
   stay, no length of stay, two order permutations);
6. estimates matched-sample all-pairs GPC and overlap-weighted GPC as
   efficiency sensitivities;
7. compares with binary composites (ideal outcome, textbook outcome) by paired
   risk differences;
8. writes tables, a JSON summary, a reproducibility manifest and a figure.

## Running it

```bash
python -m pip install "pywinratio[study]"   # or a development install
python -m studies.distal_pancreatectomy.export_public_data
python -m studies.distal_pancreatectomy.run_public_analysis --n-boot 400
```

Outputs are written to `results/public/`.

## Prespecified configuration

```{code-cell} ipython3
from pathlib import Path

root = Path.cwd()
while not (root / "studies").exists() and root != root.parent:
    root = root.parent
print((root / "studies/distal_pancreatectomy/config/primary_analysis.yaml").read_text(encoding="utf-8"))
```

## Committed results

```{code-cell} ipython3
import json
import pandas as pd

pd.set_option("display.precision", 3)
results = root / "results" / "public"
summary = json.loads((results / "win_ratio_summary.json").read_text(encoding="utf-8"))
rows = []
for key in ("unadjusted_all_pairs", "primary_assigned_pairs", "matched_sample_all_pair_gpc", "overlap_weighted_gpc"):
    block = summary[key]
    point = block.get("point", block)
    boot = block.get("bootstrap", {})
    rows.append({"analysis": key, "wr": point.get("wr"), "ci": boot.get("ci"), "bootstrap": boot.get("bootstrap_type")})
pd.DataFrame(rows)
```

```{code-cell} ipython3
pd.read_csv(results / "hierarchy_sensitivities.csv")
```

```{code-cell} ipython3
pd.read_csv(results / "matching_support.csv")
```

## Data dictionary (synthetic records)

| Column | Meaning |
|---|---|
| `site_id`, `year_band`, `hospital_type` | neutral center code, two-year period, center type |
| `treatment_group` | `HIGH_VOLUME` (mean annual minimally invasive pancreatic resections > 10) or `OTHER` |
| `age_years`, `sex`, `bmi`, `asa_ge3`, `cci`, ... | baseline covariates used for matching |
| `mort90` | death within 90 days |
| `clavien_grade` | 0, III or IV complication severity (grade V is represented by `mort90`) |
| `popf_grade` | none, B or C postoperative pancreatic fistula |
| `readmission`, `reoperation`, `postpancreatectomy_hemorrhage`, `bile_leak` | binary events |
| `los_days` | completed postoperative days |
| `ideal_outcome`, `textbook_outcome` | binary composites rebuilt from the components |

The committed data set is entirely synthetic: its estimates illustrate the
software and are not results of the registry study.
