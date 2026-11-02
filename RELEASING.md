# Making a release

## Before the release

1. Set the new version in `src/pywinratio/_version.py`, and the version and date
   in `CITATION.cff` and `.zenodo.json`. `tests/test_metadata.py` fails if they
   disagree.
2. Add a dated section to `CHANGELOG.md`.
3. Run `make check`.
4. If the comparison rules changed, regenerate the R reference values (this needs
   R with BuyseTest and WINS) and commit the new JSON file:

   ```bash
   python validation/generate_data.py
   Rscript validation/reference_r.R
   python validation/compare.py
   ```

5. Commit and tag, for example `git tag -a v0.3.0 -m "pywinratio 0.3.0"`.

## Zenodo

The simplest route is the GitHub integration. Log in to zenodo.org with GitHub,
go to Account > GitHub and switch the repository on. Then push the tag and
publish a GitHub release from it. Zenodo archives the release, reads the metadata
from `.zenodo.json`, and gives two DOIs: one for this version and a "concept" DOI
that always points to the latest version.

Without the integration, run `make zenodo`. It puts a source zip, the wheel and
the sdist in `dist/`. Upload them at zenodo.org/uploads/new and copy the metadata
from `.zenodo.json`. You can reserve the DOI before publishing if you want to add
it to the files first.

Once the DOI exists, replace `XXXXXXX` in `README.md`, `docs/citing.md` and
`CITATION.cff` with the concept DOI.

## PyPI (optional)

Create the project on pypi.org with a trusted publisher pointing to this
repository and the `release.yml` workflow (environment `pypi`). Publishing a
GitHub release then uploads the packages.

## Documentation

Import the repository on readthedocs.org. `.readthedocs.yaml` holds the
configuration; the build runs every tutorial.
