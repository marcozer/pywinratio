"""Adjusted win ratios in an observational cohort: matching and overlap weighting.

Run: python examples/02_observational_matching_weighting.py
"""

from __future__ import annotations

from pywinratio import Outcome, WinRatio, estimate_propensity_weights, propensity_match
from pywinratio.datasets import make_surgical_cohort

cohort = make_surgical_cohort(n=800, seed=2026)
covariates = ["age", "sex", "bmi", "asa3", "malignant"]
analysis = WinRatio(
    [
        Outcome.binary("death90", name="90-day death", terminal=True),
        Outcome.ordinal("clavien", name="Clavien-Dindo grade"),
        Outcome.ordinal("popf", name="Pancreatic fistula grade"),
        Outcome.binary("readmission", name="Readmission"),
        Outcome.continuous("los", name="Length of stay", margin=1),
    ],
    group="volume",
    treatment="high",
    control="low",
)

# 1:1 optimal propensity-score matching, exact on period
matching = propensity_match(cohort, "volume", "high", covariates=covariates, exact_cols=["period"])
print(matching.summary().to_string(index=False))
print("max |SMD| after matching:", round(matching.balance_after["abs_smd"].max(), 3))
matched = analysis.fit(matching.matched_df, pair_id="PAIR_ID", n_boot=2000, seed=1)
print(matched)

# overlap weighting with hospital-level bootstrap
weighting = estimate_propensity_weights(cohort, treatment="volume", treated="high", covariates=covariates + ["period"])
weighted = analysis.fit(
    weighting.weighted_df,
    weight_col="analysis_weight",
    cluster_col="hospital",
    cluster_within_arm=True,
    n_boot=1000,
    seed=2,
)
print(weighted)
