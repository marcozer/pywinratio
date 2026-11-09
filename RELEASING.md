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

The repository is connected to Zenodo through the GitHub integration. Push the
tag and publish a GitHub release from it: Zenodo archives the release within a
minute or two and reads the metadata from `.zenodo.json`.

Each release gets its own DOI. The concept DOI, 10.5281/zenodo.24075609, covers
all versions and does not change, so the README badge and the citation text stay
as they are. After a release, add the new version DOI to `identifiers` in
`CITATION.cff` and to the example in `docs/citing.md`.

If the integration ever stops working, run `make zenodo` and upload the files
from `dist/` with "New version" on the existing Zenodo record, so the concept DOI
stays the same.

## PyPI (optional)

Create the project on pypi.org with a trusted publisher pointing to this
repository and the `release.yml` workflow (environment `pypi`). Then add the
repository variable `PUBLISH_TO_PYPI` with the value `true`, under Settings,
Secrets and variables, Actions, Variables. From then on, publishing a GitHub
release also uploads the packages to PyPI.

## Documentation

Import the repository on readthedocs.org. `.readthedocs.yaml` holds the
configuration; the build runs every tutorial.
