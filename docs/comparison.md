# Comparison with other software

Win statistics are well supported in R; pywinratio brings them to Python and
adds tools for observational data. Feature coverage as of November 2026:

| Feature | pywinratio | BuyseTest (R) | WINS (R) | WinRatio (R) |
|---|---|---|---|---|
| Binary, ordinal, continuous tiers with margins | ✓ | ✓ | ✓ | ✓ |
| Censored time-to-event tiers | ✓ Gehan | ✓ Gehan, Péron, Efron | ✓ incl. IPCW | ✓ |
| Terminal events stopping lower tiers | ✓ | – | – | – |
| Explicit missing-data rules per tier | ✓ | partial | – | – |
| Win ratio / net benefit / win odds | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | ✓ / – / – |
| Matched (assigned) pairs | ✓ | via strata | – | – |
| Stratified analysis | ✓ (pair-count weights) | ✓ (5 weightings) | ✓ (4 weightings) | – |
| U-statistic variance | ✓ | ✓ | ✓ | ✓ |
| Cluster-robust variance | ✓ | – | – | – |
| Bootstrap | ✓ subject, pair, cluster, cluster-within-arm | ✓ | – | – |
| Permutation (Finkelstein–Schoenfeld) test | ✓ incl. stratified, cluster, sign-flip | ✓ | – | – |
| Propensity matching integrated | ✓ optimal / greedy, exact strata | – | – | – |
| Propensity weighting | ✓ overlap, ATT, ATE | – | ✓ IPTW | – |
| Bootstrap re-estimating the propensity design | ✓ | – | – | – |
| Balance diagnostics (SMD, KS, variance ratio, ESS) | ✓ | – | – | – |
| Several arms with multiplicity adjustment | ✓ | partial | – | – |
| Sample size / power | ✓ formula + simulation | simulation | simulation | – |
| Tier decomposition, per-pair explanations | ✓ | ✓ | partial | – |
| Hierarchy-order and margin sensitivity helpers | ✓ | partial | – | – |

Not yet available in pywinratio (see the roadmap in the changelog): Péron
(Kaplan–Meier) and IPCW scoring for censored tiers, recurrent-event tiers,
stratum weights other than the number of pairs, regression models for win
statistics (Mao and Wang, 2021), and Maraca plots.

Single-endpoint statistics such as the probabilistic index are also available in
statsmodels (`rank_compare_2indep`); they do not handle hierarchies.

## Migrating from R

| BuyseTest | pywinratio |
|---|---|
| `BuyseTest(arm ~ tte(time, status) + cont(y, threshold = 5), data)` | `win_ratio(data, [Outcome.time_to_event("time", "status"), Outcome.continuous("y", threshold=5, favorable="higher")], group="arm", treatment=..., control=...)` |
| `operator = "<0"` | `favorable="lower"` (default for binary, ordinal, continuous) |
| `bin(x)` | `Outcome.binary("x")` |
| `strata(s)` with `pool.strata = "Buyse"` | `strata=["s"]` |
| `method.inference = "u-statistic"` | default analytic inference |
| `method.inference = "permutation"` | `n_perm=` |
| `method.inference = "bootstrap"` | `n_boot=` |

The treatment arm is the *second* factor level in BuyseTest; pywinratio names it
explicitly (`treatment=`).
