"""Synthetic example data sets used throughout the documentation.

Both generators are deterministic for a given seed and contain no real
patient data.

* :func:`make_surgical_cohort`: an observational surgical registry in
  which hospital volume (a hospital-level exposure) is confounded by case
  mix; outcomes are 90-day death, graded complications, graded pancreatic
  fistula, readmission and length of stay.
* :func:`make_heart_failure_trial`: a randomized trial with censored time
  to death and to first heart-failure hospitalization and a
  quality-of-life score, the classic win-ratio setting (Pocock et al., 2012).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_surgical_cohort(n: int = 800, seed: int = 2026, n_hospitals: int = 24) -> pd.DataFrame:
    """Synthetic observational cohort comparing high- and low-volume hospitals.

    Parameters
    ----------
    n : int, default 800
        Number of patients.
    seed : int, default 2026
        Random seed.
    n_hospitals : int, default 24
        Number of hospitals (one third are high volume).

    Returns
    -------
    pandas.DataFrame
        One row per patient:

        ``patient_id``, ``hospital``, ``volume`` (``"high"``/``"low"``),
        ``period`` (``"early"``/``"late"``), ``age``, ``sex``, ``bmi``,
        ``asa3`` (ASA >= 3), ``malignant``, ``death90`` (0/1),
        ``clavien`` (0 = none or minor, 3 = grade III, 4 = grade IV),
        ``popf`` (0 = none, 1 = grade B, 2 = grade C), ``readmission``
        (0/1) and ``los`` (length of stay, completed days).

    Examples
    --------
    >>> cohort = make_surgical_cohort(n=200, seed=1)
    >>> cohort["volume"].value_counts().sort_index().index.tolist()
    ['high', 'low']
    """

    rng = np.random.default_rng(seed)
    hospitals = np.array([f"H{h + 1:02d}" for h in range(n_hospitals)])
    high = np.arange(n_hospitals) < max(1, n_hospitals // 3)
    hospital_effect = rng.normal(0.0, 0.25, n_hospitals)
    # high-volume hospitals receive more patients
    attraction = np.where(high, 2.2, 1.0)
    index = rng.choice(n_hospitals, size=n, p=attraction / attraction.sum())
    is_high = high[index]

    age = np.round(rng.normal(62 + 2.5 * is_high, 11, n)).clip(25, 90)
    sex = np.where(rng.random(n) < 0.48, "F", "M")
    bmi = np.round(rng.normal(26.5, 4.5, n).clip(16, 48), 1)
    asa3 = (rng.random(n) < _expit(-1.1 + 0.035 * (age - 62) + 0.45 * is_high)).astype(int)
    malignant = (rng.random(n) < _expit(0.1 + 0.6 * is_high)).astype(int)
    period = np.where(rng.random(n) < 0.5, "early", "late")

    frailty = rng.normal(0.0, 0.6, n)
    risk = (
        0.03 * (age - 62)
        + 0.55 * asa3
        + 0.05 * (bmi - 26.5)
        + 0.25 * malignant
        + hospital_effect[index]
        + frailty
        - 0.45 * is_high
    )
    death90 = (rng.random(n) < _expit(-4.4 + 1.1 * risk)).astype(int)
    major = rng.random(n) < _expit(-1.7 + risk)
    grade4 = rng.random(n) < 0.3
    clavien = np.where(major, np.where(grade4, 4, 3), 0)
    clavien = np.where(death90 == 1, np.maximum(clavien, 4), clavien)
    popf_any = rng.random(n) < _expit(-1.9 + 0.7 * risk + 0.04 * (bmi - 26.5))
    popf_c = popf_any & (rng.random(n) < _expit(-1.6 + 0.6 * risk))
    popf = np.where(popf_c, 2, np.where(popf_any, 1, 0))
    readmission = (rng.random(n) < _expit(-2.3 + 0.5 * risk + 0.4 * (popf > 0))).astype(int)
    los = np.round(
        5
        + rng.gamma(2.0, 1.5, n)
        + 4.0 * (clavien == 3)
        + 9.0 * (clavien == 4)
        + 3.0 * (popf == 1)
        + 8.0 * (popf == 2)
        + 1.2 * risk.clip(-1, None)
    ).clip(3, 90)

    return pd.DataFrame(
        {
            "patient_id": [f"P{i + 1:05d}" for i in range(n)],
            "hospital": hospitals[index],
            "volume": np.where(is_high, "high", "low"),
            "period": period,
            "age": age.astype(int),
            "sex": sex,
            "bmi": bmi,
            "asa3": asa3,
            "malignant": malignant,
            "death90": death90,
            "clavien": clavien.astype(int),
            "popf": popf.astype(int),
            "readmission": readmission,
            "los": los.astype(int),
        }
    )


def make_heart_failure_trial(
    n: int = 600,
    seed: int = 2012,
    hr_death: float = 0.8,
    hr_hospitalization: float = 0.7,
    kccq_effect: float = 3.0,
    n_sites: int = 30,
    time_decimals: int | None = 2,
) -> pd.DataFrame:
    """Synthetic 1:1 randomized heart-failure trial with censored follow-up.

    Parameters
    ----------
    n : int, default 600
        Number of patients.
    seed : int, default 2012
        Random seed.
    hr_death, hr_hospitalization : float
        True hazard ratios (treatment vs placebo).
    kccq_effect : float, default 3.0
        True mean difference in KCCQ change (points; higher is better).
    n_sites : int, default 30
        Number of recruiting sites.
    time_decimals : int or None, default 2
        Rounding of times (months); ``None`` keeps full precision.

    Returns
    -------
    pandas.DataFrame
        ``patient_id``, ``site``, ``arm`` (``"treatment"``/``"placebo"``),
        ``followup`` (months), ``time_death`` and ``death`` (event
        indicator), ``time_hosp`` and ``hosp`` (first heart-failure
        hospitalization; censored at death or end of follow-up) and
        ``kccq_change`` (change in Kansas City Cardiomyopathy Questionnaire
        score at 12 months, missing after death).

    Examples
    --------
    >>> trial = make_heart_failure_trial(n=100, seed=1)
    >>> sorted(trial["arm"].unique())
    ['placebo', 'treatment']
    """

    rng = np.random.default_rng(seed)
    arm = rng.permutation(np.repeat(["treatment", "placebo"], [n // 2, n - n // 2]))
    treated = arm == "treatment"
    site = np.array([f"S{s + 1:02d}" for s in range(n_sites)])[rng.integers(0, n_sites, n)]
    followup = rng.uniform(12.0, 36.0, n)
    frailty = rng.gamma(4.0, 0.25, n)
    death_rate = 0.10 / 12 * frailty * np.where(treated, hr_death, 1.0)
    hosp_rate = 0.30 / 12 * frailty * np.where(treated, hr_hospitalization, 1.0)
    t_death = rng.exponential(1.0 / death_rate)
    t_hosp = rng.exponential(1.0 / hosp_rate)

    death = (t_death <= followup).astype(int)
    time_death = np.minimum(t_death, followup)
    hosp = (t_hosp < np.minimum(t_death, followup)).astype(int)
    time_hosp = np.minimum(t_hosp, np.minimum(t_death, followup))
    kccq = rng.normal(2.0 + kccq_effect * treated - 4.0 * (frailty - 1.0), 14.0, n)
    kccq = np.where((death == 1) & (time_death < 12.0), np.nan, np.round(kccq, 1))

    def rounded(x: np.ndarray) -> np.ndarray:
        return x if time_decimals is None else np.round(x, time_decimals)

    return pd.DataFrame(
        {
            "patient_id": [f"T{i + 1:05d}" for i in range(n)],
            "site": site,
            "arm": arm,
            "followup": rounded(followup),
            "time_death": rounded(time_death),
            "death": death,
            "time_hosp": rounded(time_hosp),
            "hosp": hosp,
            "kccq_change": kccq,
        }
    )
