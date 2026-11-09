# Installation

pywinratio supports Python 3.10 and later on Linux, macOS and Windows.

## From the Zenodo or GitHub release

Download the wheel (`pywinratio-<version>-py3-none-any.whl`) from the
[GitHub release](https://github.com/marcozer/pywinratio/releases) and install it
(every release is also archived on [Zenodo](https://doi.org/10.5281/zenodo.24075609)):

```bash
python -m pip install pywinratio-0.3.0-py3-none-any.whl
```

or install directly from the repository:

```bash
python -m pip install "git+https://github.com/marcozer/pywinratio.git"
```

Once published on PyPI:

```bash
python -m pip install pywinratio
```

## Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `plot` | matplotlib | {func}`~pywinratio.plot_tiers`, {func}`~pywinratio.plot_forest`, {func}`~pywinratio.plot_balance`, {func}`~pywinratio.plot_bootstrap` |
| `study` | matplotlib, PyYAML | the distal pancreatectomy [case study](tutorials/case_study.md) |
| `docs` | Sphinx, MyST-NB, ... | building this documentation |
| `dev` | pytest, ruff, build, twine | development |

```bash
python -m pip install "pywinratio[plot]"
```

## Dependencies

NumPy, pandas, SciPy and scikit-learn (for propensity-score models). All
computations are vectorized NumPy; no compiler is needed.

## Development install

```bash
git clone https://github.com/marcozer/pywinratio.git
cd pywinratio
python -m pip install -e ".[dev,docs,study]"
pytest
```

## Upgrading from WinRatioPy

The package was distributed as `WinRatioPy` (imports `winratio` and
`winratiopy`) up to version 0.2.2. Both old import names still work in 0.3 and
emit a `DeprecationWarning`; replace them with `import pywinratio`. See the
[changelog](changelog.md) for behaviour changes.
