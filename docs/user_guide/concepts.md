# Concepts

## Prioritized comparisons

A hierarchical (prioritized) composite outcome orders several outcomes by clinical
importance, for example

1. death,
2. heart-failure hospitalization,
3. change in quality of life.

Two patients are compared on outcome 1 first. If one did better, that patient
wins the comparison and the lower outcomes are not looked at. If they are
tied on outcome 1 (neither died, or censoring makes the comparison impossible),
they are compared on outcome 2, and so on. A pair tied on every outcome is a
tie.

This is the generalized pairwise comparison (GPC) framework of Buyse (2010). The
win ratio of Pocock et al. (2012) is one summary of it. Compared with a
conventional time-to-first-event composite, the hierarchy

- lets the most serious outcome dominate (a death is never "cancelled" by a
  hospitalization in the other patient);
- uses outcomes of any type, including continuous scores;
- counts recurrent or late events that a time-to-first-event analysis ignores.

## Win statistics

Let $p_W$, $p_L$ and $p_T = 1 - p_W - p_L$ be the probabilities that a randomly
chosen patient of arm A wins, loses or ties against a randomly chosen patient of
arm B. pywinratio reports three measures, all from arm A's perspective.

| Measure | Definition | No difference | Interpretation |
|---|---|---|---|
| Win ratio (WR) | $p_W / p_L$ | 1 | wins per loss, ignoring ties |
| Net benefit (NB) | $p_W - p_L$ | 0 | Buyse's net treatment benefit; also called win difference. Its inverse is a "number needed to treat" for one more win |
| Win odds (WO) | $\dfrac{p_W + p_T/2}{p_L + p_T/2} = \dfrac{1+\text{NB}}{1-\text{NB}}$ | 1 | odds of winning when ties count as half a win and half a loss |

The *probabilistic index* (Mann–Whitney probability) $p_W + p_T/2$ is also
reported as `win_probability`.

The three measures test the same null hypothesis but are not interchangeable.
The win ratio ignores ties: a WR of 2 can come from 2% versus 1% of pairs (98%
ties) or from 60% versus 30%. This is why the share of ties and the net benefit
should always be reported with the win ratio (Pocock et al., 2024; Dong et al.,
2023).

## Estimands

The design determines which pairs are compared and therefore what is estimated.

All pairs ({func}`~pywinratio.win_ratio` with no `pair_id`)
: every arm-A patient is compared with every arm-B patient. In a randomized trial
  this estimates the population win statistics above. Proportions are two-sample
  U-statistics.

Stratified (`strata=[...]`)
: patients are compared only within strata (e.g. center, region, baseline risk).
  Stratum totals are summed, i.e. strata are weighted by their number of pairs
  ($n_{A,s} n_{B,s}$), which matches the default "Buyse" pooling of the R package
  BuyseTest.

Assigned (matched) pairs (`pair_id=...`)
: only the two patients of each matched pair are compared. After 1:1 propensity
  matching this targets the matched (treated-like) population; pair results are
  independent, which simplifies inference.

Weighted all pairs (`weight_col=...`)
: each cross-arm pair is weighted by the product of the patients' weights (e.g.
  propensity-score overlap weights), targeting the corresponding weighted
  population (Li, Morgan and Zaslavsky, 2018).

All-pairs and matched-pair win ratios computed on the same matched sample are
different estimands; name the primary one in advance.

## What the tier table means

{meth}`WinRatioResult.tiers() <pywinratio.WinRatioResult.tiers>` reports, for each
tier, how many pairs it separated *given that all higher tiers tied*. These
counts describe how the hierarchy worked; they are not treatment effects on the
individual components. Conventional per-component analyses (event rates, hazard
ratios) should be reported alongside ({func}`~pywinratio.summarize_component_outcomes`).

## Ties, margins and terminal events

- Clinical margins. Small differences in continuous outcomes are usually not
  meaningful. `Outcome.continuous("los", margin=1)` ties lengths of stay that
  differ by 0 or 1 day. The BuyseTest convention ("a win needs a difference of at
  least $\tau$") is available as `threshold=`.
- Terminal events. When both patients died, comparing their later
  complications or length of stay is meaningless. A `terminal=True` tier stops the
  comparison as a tie when both patients had the event and the tier cannot
  separate them (for a time-to-event tier the later death still wins).
- Missing values. By default a missing value makes the tier uninformative and
  the next tier is used. `missing_is="loss"` (worst case) or `"win"` assign the
  tier to the patient with available data.

## Censoring

For a right-censored time-to-event tier, two patients can only be ranked within
their common follow-up. pywinratio uses the Gehan scoring rule (Gehan, 1965;
Pocock et al., 2012): patient A beats patient B when B had the event and A was
still event-free (with or without a later event or censoring) at that time. When
the earlier time is a censoring time, the comparison is indeterminate and passes
to the next tier. `horizon=` truncates follow-up (e.g. at 1 year).

With Gehan scoring, the estimand depends on the follow-up: heavier or longer
censoring leaves more pairs indeterminate, and those pairs are passed to lower
tiers or tied. Win statistics from trials with different follow-up are therefore
not directly comparable, and early censoring tends to move the win ratio towards
the lower tiers' result or towards 1. Truncating follow-up at a common `horizon`
makes the estimand explicit. Inverse-probability-of-censoring weighting, which
targets the censoring-free win statistics (Dong, Mao et al., 2020), is not yet
implemented. Report the follow-up, the horizon and the share of ties.

## Formulas

For unmatched data with $n_A$ and $n_B$ patients, let $W_{ij}$ ($L_{ij}$) be 1 when
patient $i$ of arm A wins (loses) against patient $j$ of arm B. Then

$$
\hat p_W = \frac{1}{n_A n_B}\sum_{i,j} W_{ij},\qquad
\hat p_L = \frac{1}{n_A n_B}\sum_{i,j} L_{ij}.
$$

Their variance is estimated from the Hoeffding projection with the structural
components $w_i = \frac{1}{n_B}\sum_j W_{ij}$ and $w_j = \frac{1}{n_A}\sum_i W_{ij}$
(and likewise for losses):

$$
\widehat{\operatorname{Var}}(\hat p_W) = \frac{S^2(w_{i\cdot})}{n_A} + \frac{S^2(w_{\cdot j})}{n_B},
$$

with the covariance of wins and losses estimated the same way (DeLong et al.,
1988; Bebu and Lachin, 2016; Dong et al., 2016). The delta method then gives

$$
\operatorname{Var}(\log \widehat{WR}) \approx \frac{V_W}{p_W^2} + \frac{V_L}{p_L^2} - \frac{2\,C_{WL}}{p_W p_L},\qquad
\operatorname{Var}(\widehat{NB}) = V_W + V_L - 2 C_{WL},
$$

and $\operatorname{Var}(\log \widehat{WO}) = \left(\frac{2}{1-NB^2}\right)^2 \operatorname{Var}(\widehat{NB})$.
Stratified estimates sum the stratum influence functions; the cluster-robust
version sums them within clusters before squaring (Obuchowski, 1997). For $m$
independent matched pairs with $W$ wins and $L$ losses,
$\operatorname{Var}(\log \widehat{WR}) = 1/W + 1/L$.

See [Inference](inference.md) for when to use each method and
[References](../references.md) for the sources.
