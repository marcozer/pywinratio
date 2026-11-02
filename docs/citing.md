# Citing pywinratio

If pywinratio contributes to a publication, please cite the software release you
used. Its DOI is minted by Zenodo for every version; the *concept DOI* always
resolves to the latest version.

> Chouillard M-A, Husson T. *pywinratio: win ratio, net benefit and win odds for
> hierarchical composite outcomes in Python* (version 0.3.0) [Computer software].
> Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

```bibtex
@software{chouillard_pywinratio,
  author    = {Chouillard, Marc-Anthony and Husson, Thomas},
  title     = {pywinratio: win ratio, net benefit and win odds for hierarchical
               composite outcomes in Python},
  version   = {0.3.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/marcozer/pywinratio}
}
```

The repository's `CITATION.cff` holds the same metadata (GitHub shows it as
"Cite this repository").

Please also cite the methods you used, for example Pocock et al. (2012) for the
win ratio, Buyse (2010) for generalized pairwise comparisons and the net benefit,
Dong et al. (2020) or Brunner et al. (2021) for the win odds, and Finkelstein and
Schoenfeld (1999) for the permutation test ([References](references.md)).

In the methods section, state the version and the inference method, e.g.
"Win statistics were estimated with pywinratio 0.3.0 (U-statistic variance;
2000 bootstrap replicates resampling hospitals within arm)."
