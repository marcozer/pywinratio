# Contributing

Bug reports, questions and pull requests are welcome.

## Reporting a problem

Open an issue at https://github.com/marcozer/pywinratio/issues. Please include
the versions you use (`python -c "import pywinratio; print(pywinratio.__version__)"`),
a small example that shows the problem (the simulated data in
`pywinratio.datasets` are handy for this), and what you expected.

Please never post patient data, even de-identified.

## Setting up

```bash
git clone https://github.com/marcozer/pywinratio.git
cd pywinratio
python -m pip install -e ".[dev,docs,study]"
```

## Before opening a pull request

`make check` runs everything: tests, doctests, lint, the comparison with the R
reference values, the documentation build and the package build. The pieces can
also be run one by one:

```bash
pytest
pytest --doctest-modules src/pywinratio
ruff check src tests studies examples validation
python -m sphinx -W -b html docs docs/_build/html
```

A few conventions:

- The comparison rules live in one place, `src/pywinratio/_engine.py`. Please do
  not add a second implementation elsewhere.
- Public functions need a NumPy-style docstring with an example, and tests. For
  a statistical method, test it against something independent: a brute-force
  computation, exact enumeration, a simulation or an R reference value.
- If a change can alter results, say so in `CHANGELOG.md`.
- Study-specific code goes in `studies/` or `examples/`, not in the library.
- Synthetic data only.

Releases are described in `RELEASING.md`.
