# pywinratio

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![CI](https://github.com/marcozer/pywinratio/actions/workflows/ci.yml/badge.svg)](https://github.com/marcozer/pywinratio/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/pywinratio/badge/?version=latest)](https://pywinratio.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/marcozer/pywinratio/blob/main/LICENSE)

pywinratio computes the win ratio, the net benefit and the win odds for
hierarchical composite outcomes in Python.

With a hierarchical outcome, two patients are compared on the most important
outcome first (death, say). If that does not separate them, they are compared on
the next one (hospitalization, then quality of life, and so on). Counting the
wins, losses and ties over all pairs of patients gives the win ratio of Pocock et
al. (2012), the net benefit of Buyse (2010) and the win odds of Brunner et al.
(2021).

We wrote the package for registry studies in pancreatic surgery, where the arms
are not randomized and patients are clustered in hospitals. It therefore covers
propensity-score matching and weighting and cluster-level inference, as well as
the usual analysis of a randomized trial.

## Installation

```bash
pip install "git+https://github.com/marcozer/pywinratio.git"
```

Python 3.10 or later. The dependencies are NumPy, pandas, SciPy and scikit-learn;
install `pywinratio[plot]` to get the figures (matplotlib). Each release is
also archived on Zenodo with its wheel.

## Example

```python
from pywinratio import Outcome, win_ratio
from pywinratio.datasets import make_heart_failure_trial

trial = make_heart_failure_trial()   # simulated trial, 600 patients

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

```text
Win-ratio analysis
========================================================================
Arm A: treatment (n=300)   vs   Arm B: placebo (n=300)
Design: all cross-arm pairs
Pairs: 90000  |  A wins 46845  |  B wins 36969  |  ties 6186

                Estimate                  95% CI   p-value
Win ratio          1.267          [1.046, 1.535]     0.016
Net benefit        0.110          [0.022, 0.198]     0.015
Win odds           1.247          [1.043, 1.490]     0.016
CI: u_statistic;  p: Wald (u_statistic)

Hierarchy (tier: wins / losses, share of pairs resolved)
  1. Death: 14501 / 11198 (28.6%), 2 stopped as terminal ties
  2. HF hospitalization: 18052 / 13909 (35.5%)
  3. KCCQ: 14292 / 11862 (29.1%)
```

The result can be turned into tables (`result.to_frame()`, `result.tiers()`,
`result.pairs()`), saved as JSON or plotted. Pass `n_boot=2000` to get bootstrap
intervals and `n_perm=5000` to get a Finkelstein–Schoenfeld permutation test.

## What it does

Outcomes can be binary, ordinal, continuous (with a clinical margin, a threshold
or cut points) or censored times to an event, compared with the Gehan rule. Death
can be declared terminal, so that two patients who both died are not ranked on
their later complications. Each outcome has its own rule for missing values.

The comparison can use every pair of patients, only pairs within strata, only
matched pairs, or weighted pairs. Standard errors come from U-statistic theory,
with a cluster-robust version when patients are grouped in hospitals. Patients,
pairs or hospitals can also be resampled in a bootstrap, and arm labels can be
permuted for a test.

For observational data there is 1:1 propensity-score matching (optimal or
greedy) and overlap, ATT and ATE weighting, with balance diagnostics. A bootstrap
can refit the propensity model and redo the matching in every replicate.

There are also helpers for sensitivity analyses (other hierarchies, margins,
orders of the outcomes), for comparing several arms, for sample size and power,
and for the figures usually needed in a paper.

## Documentation

The documentation is at https://pywinratio.readthedocs.io. It covers the concepts,
how to define outcomes, the designs, the choice of inference method, observational
studies and reporting, and it has three worked examples: a heart-failure trial, a
surgical registry and a configuration-driven registry analysis. The pages are
also in the [`docs/`](docs/) folder.

## Validation

We compared the results with the R packages BuyseTest (3.3.9) and WINS (1.5.1) in
six scenarios. These covered censored, ordinal and continuous outcomes, and
stratified and matched analyses. The point estimates were identical (differences
below 10⁻¹²) and the standard errors agreed to about 10⁻⁴. The scripts and the R
reference values are in [`validation/`](validation/), and the test suite checks
them on every change.

## Citation

If you use pywinratio, please cite it (see [`CITATION.cff`](CITATION.cff)):

> Chouillard M-A, Husson T. pywinratio: win ratio, net benefit and win odds for hierarchical composite outcomes in Python (version 0.3.0). Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

Please cite the methods as well, for example Pocock et al. (2012) for the win
ratio and Buyse (2010) for generalized pairwise comparisons
([references](docs/references.md)).

## Authors

Marc-Anthony Chouillard and Thomas Husson, Department of HPB Surgery and Liver
Transplantation, Beaujon Hospital, AP-HP, Université Paris Cité. The package grew
out of hierarchical outcome analyses in pancreatic surgery with Clément Pastier,
Sébastien Gaujoux and the Association Française de Chirurgie study group. The
repository includes a synthetic version of that registry analysis as a case
study; it contains no patient data.

Questions and bug reports are welcome on the
[issue tracker](https://github.com/marcozer/pywinratio/issues). See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT.
