# Frequently asked questions

## Is a win ratio of 1.5 a "50% improvement"?

No. It means arm A won 1.5 times as many comparisons as it lost, among pairs that
were not tied. Report the net benefit (e.g. 0.12: arm A wins 12 percentage points
more comparisons than it loses) and the share of ties to convey magnitude.

## Why does removing a tier change the win ratio so much?

Lower tiers mostly resolve pairs that would otherwise tie. With fewer tiers, the
win ratio is computed from fewer informative pairs and can move away from or
towards 1; the net benefit is more stable. See [Sensitivity analyses](user_guide/sensitivity.md).

## Which design should be primary in an observational study, matched pairs or all pairs?

They estimate different quantities. Matched pairs give the win ratio among
matched (treated-like) patients and are easy to explain; weighted all-pairs GPC
uses every patient. Prespecify one and report the other as a sensitivity
analysis. Do not report the all-pairs win ratio of a matched sample as if it
were the assigned-pair estimand.

## Can I use the 1/W + 1/L formula?

Only for assigned (matched) pairs. For all-pairs comparisons each patient
appears in many pairs; use the default U-statistic variance, a bootstrap or a
permutation test.

## The bootstrap upper limit is infinite. Is that a bug?

No. Some resamples had wins and no losses, so the win ratio is infinite in those
replicates; discarding them would make the interval artificially narrow. Report
the net benefit, whose interval is finite, or increase the sample size.

## How are two deaths compared?

With a binary death tier and `terminal=True`, two deaths are a tie and lower
tiers are skipped. With a time-to-event death tier, the later death wins; only
deaths at the same time (within the margin) stop as ties.

## How are censored times handled?

With the Gehan rule: a pair is resolved only when the earlier time is an
observed event. See [Defining outcomes](user_guide/outcomes.md#time-to-event-outcomes-with-censoring).

## My binary outcome is coded "yes"/"no".

Recode it to 1/0 (1 = event). pywinratio raises an error rather than guessing.

## How large can the data be?

All-pairs comparisons are vectorized and evaluated in memory-bounded blocks. On a
laptop (Apple M-series), a five-tier analysis of 20,000 patients (100 million
pairs) with its U-statistic variance takes about 5 seconds; with 4,000 patients,
1,000 bootstrap replicates or 5,000 permutations take under a second. Point
estimates, analytic variance and permutation tests use bounded memory;
bootstraps hold the win and loss matrices of each stratum in memory (16 bytes
per pair), so very large bootstraps should be stratified or run on a larger
machine.

## Does pywinratio handle multiple imputation?

Run the analysis in each imputed data set and pool the log win ratio or net
benefit with Rubin's rules; see [Observational studies](user_guide/observational.md#multiple-imputation).

## I used WinRatioPy 0.2. Will my results change?

For the same data and the same pairs, point estimates without missing outcome
values are identical (all-pairs, stratified, matched and weighted analyses were
checked). Results change where 0.2 was wrong or less accurate: `missing_is="win"/"loss"` (now about the patient
with the missing value), percentile intervals (infinite replicates are kept),
greedy and optimal matching (more pairs), the propensity model (unpenalized),
and stratified cluster bootstraps (clusters resampled jointly across strata).
See the [changelog](changelog.md).
