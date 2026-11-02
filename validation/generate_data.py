"""Write the synthetic data sets used to validate pywinratio against R.

Run from the repository root::

    python validation/generate_data.py
"""

from __future__ import annotations

from pathlib import Path

from pywinratio import propensity_match
from pywinratio.datasets import make_heart_failure_trial, make_surgical_cohort

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    # full-precision times: no exact event/censoring coincidences, on which
    # BuyseTest (Gehan 1965 convention, as pywinratio) and WINS differ
    trial = make_heart_failure_trial(n=600, seed=2012, time_decimals=None)
    # R packages differ in missing-data handling; use a complete-case copy.
    # Whole KCCQ points make every difference exact, so the threshold of 5
    # points is not blurred by binary floating-point rounding.
    trial["kccq_change"] = trial["kccq_change"].fillna(-60.0).round()
    trial.to_csv(DATA / "heart_failure_trial.csv", index=False)

    cohort = make_surgical_cohort(n=800, seed=2026)
    cohort.to_csv(DATA / "surgical_cohort.csv", index=False)

    matched = propensity_match(
        cohort,
        treat_col="volume",
        treated_value="high",
        covariates=["age", "sex", "bmi", "asa3", "malignant"],
        exact_cols=["period"],
    ).matched_df
    matched.drop(columns=["_T", "_PS", "_LPS"]).to_csv(DATA / "surgical_matched.csv", index=False)
    print(f"wrote {len(trial)} trial rows, {len(cohort)} cohort rows, {len(matched) // 2} matched pairs")


if __name__ == "__main__":
    main()
