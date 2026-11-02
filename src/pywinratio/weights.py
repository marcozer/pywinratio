"""Propensity-score weights: overlap, ATT and ATE.

Weights feed :func:`pywinratio.compute_weighted_gpc` (or
``WinRatio(...).fit(df, weight_col="analysis_weight")``).

==========  ======================  =====================  ======================
Estimand    Treated weight          Control weight         Target population
==========  ======================  =====================  ======================
overlap     ``1 - e(x)``            ``e(x)``               clinical equipoise
att         ``1``                   ``e(x) / (1 - e(x))``  the treated
ate         ``1 / e(x)``            ``1 / (1 - e(x))``     the whole cohort
==========  ======================  =====================  ======================

Overlap weights (Li, Morgan and Zaslavsky, 2018) are bounded, avoid extreme
weights, and give exact mean balance of every model covariate when the
propensity model is the unpenalized logistic MLE (the default here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd

from ._propensity import check_covariates, common_support_bounds, fit_propensity, treatment_indicator
from .diagnostics import balance_diagnostics, effective_sample_size

WeightEstimand = Literal["overlap", "att", "ate"]
MissingStrategy = Literal["drop", "simple"]


@dataclass
class WeightingResult:
    """Output of :func:`estimate_propensity_weights`.

    Attributes
    ----------
    weighted_df : pandas.DataFrame
        Analyzed patients with the weight (``weight_col``) and propensity
        (``propensity_col``) columns added.
    propensity_score, weights : pandas.Series
        Scores and weights aligned with ``weighted_df``.
    balance_before : pandas.DataFrame
        Unweighted balance of the complete-case sample before trimming.
    balance_after : pandas.DataFrame
        Weighted balance of the analyzed sample.
    ess_by_arm : dict
        Kish effective sample size by arm.
    weight_summary : pandas.DataFrame
        Minimum, median, mean, maximum and total weight by arm.
    rows_dropped_missing, rows_trimmed : int
        Patients removed for missing covariates and by trimming.
    """

    weighted_df: pd.DataFrame
    propensity_score: pd.Series
    weights: pd.Series
    balance_before: pd.DataFrame
    balance_after: pd.DataFrame
    ess_by_arm: dict[Any, float]
    weight_summary: pd.DataFrame
    covariates: list[str]
    estimand: WeightEstimand
    missing: MissingStrategy
    rows_dropped_missing: int
    model: Any
    rows_trimmed: int = 0
    weight_col: str = "analysis_weight"


def estimate_propensity_weights(
    df: pd.DataFrame,
    *,
    treatment: str,
    treated: Any,
    covariates: Iterable[str],
    estimand: WeightEstimand = "overlap",
    missing: MissingStrategy = "drop",
    trim_common_support: Optional[bool] = None,
    weight_col: str = "analysis_weight",
    propensity_col: str = "propensity_score",
    seed: int = 42,
    control: Any = None,
    penalty: Optional[float] = None,
    ps_col: Optional[str] = None,
    normalize: bool = False,
) -> WeightingResult:
    """Estimate overlap, ATT or ATE propensity-score weights.

    Parameters
    ----------
    df : pandas.DataFrame
        Cohort, one row per patient.
    treatment : str
        Treatment (exposure) column.
    treated : any
        Value of ``treatment`` identifying treated patients.
    covariates : iterable of str
        Propensity-model covariates (numeric columns standardized, others
        one-hot encoded). Also used for the balance tables.
    estimand : {"overlap", "att", "ate"}, default "overlap"
        Target population (see the module table).
    missing : {"drop", "simple"}, default "drop"
        Complete cases, or explicit single (median / mode) imputation.
        Multiply imputed data should be analyzed per imputation.
    trim_common_support : bool, optional
        Drop patients outside the overlap of arm-specific score ranges.
        Defaults to ``False`` for overlap weights (which already down-weight
        patients without counterparts, and lose their exact-balance property
        when the sample is trimmed) and ``True`` for ATT and ATE weights.
    weight_col, propensity_col : str
        Names of the columns added to ``weighted_df``.
    seed : int, default 42
        Random state of the model.
    control : any, optional
        Comparator value; required when ``treatment`` has more than two levels.
    penalty : float, optional
        Inverse L2 regularization strength ``C``; ``None`` (default) fits the
        unpenalized logistic model.
    ps_col : str, optional
        Use precomputed propensity scores from this column.
    normalize : bool, default False
        Rescale weights to sum to the number of patients in each arm (does
        not change win statistics).

    Returns
    -------
    WeightingResult

    Examples
    --------
    >>> from pywinratio.datasets import make_surgical_cohort
    >>> cohort = make_surgical_cohort(n=400, seed=1)
    >>> w = estimate_propensity_weights(cohort, treatment="volume", treated="high",
    ...                                 covariates=["age", "asa3", "bmi"])
    >>> bool(w.balance_after["abs_smd"].max() < 1e-6)
    True
    """

    covariates = list(covariates)
    if estimand not in {"overlap", "att", "ate"}:
        raise ValueError("estimand must be 'overlap', 'att', or 'ate'")
    if missing not in {"drop", "simple"}:
        raise ValueError("missing must be 'drop' or 'simple'")
    data, indicator = treatment_indicator(df, treatment, treated, control)
    if ps_col is None:
        covariates = check_covariates(data, covariates)
    elif ps_col not in data.columns:
        raise ValueError(f"ps_col {ps_col!r} not found")

    original_rows = len(data)
    needed = covariates + ([ps_col] if ps_col else [])
    if missing == "drop" or ps_col is not None:
        keep = data[needed].notna().all(axis=1) if needed else pd.Series(True, index=data.index)
        data = data[keep].copy()
        indicator = indicator[keep]
    rows_dropped_missing = original_rows - len(data)
    if indicator.nunique() != 2:
        raise ValueError("both treatment arms are required after missing-data handling")

    if ps_col is None:
        model, propensity = fit_propensity(data, covariates, indicator, missing=missing, penalty=penalty, seed=seed)
    else:
        model = None
        propensity = pd.to_numeric(data[ps_col], errors="coerce")
        if not propensity.between(0, 1).all():
            raise ValueError("propensity scores must lie in [0, 1]")
    propensity = propensity.clip(1e-6, 1 - 1e-6).rename(propensity_col)
    data[propensity_col] = propensity

    before = (
        balance_diagnostics(data, treatment=treatment, treated=treated, covariates=covariates)
        if covariates
        else pd.DataFrame()
    )
    rows_trimmed = 0
    if trim_common_support is None:
        trim_common_support = estimand != "overlap"
    if trim_common_support:
        low, high = common_support_bounds(propensity, indicator)
        inside = propensity.between(low, high)
        rows_trimmed = int((~inside).sum())
        data = data[inside].copy()
        indicator = indicator[inside]
        propensity = data[propensity_col]

    t = indicator.to_numpy(dtype=float)
    ps = propensity.to_numpy(dtype=float)
    if estimand == "overlap":
        values = np.where(t == 1, 1 - ps, ps)
    elif estimand == "att":
        values = np.where(t == 1, 1.0, ps / (1 - ps))
    else:
        values = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    if normalize:
        for arm in (0, 1):
            in_arm = t == arm
            values[in_arm] = values[in_arm] * in_arm.sum() / values[in_arm].sum()
    data[weight_col] = values

    after = (
        balance_diagnostics(data, treatment=treatment, treated=treated, covariates=covariates, weights=weight_col)
        if covariates
        else pd.DataFrame()
    )
    ess_by_arm = {arm: effective_sample_size(group[weight_col]) for arm, group in data.groupby(treatment, sort=True)}
    weight_summary = (
        data.groupby(treatment)[weight_col]
        .agg(n="size", minimum="min", median="median", mean="mean", maximum="max", total="sum")
        .reset_index()
    )
    return WeightingResult(
        weighted_df=data,
        propensity_score=data[propensity_col].copy(),
        weights=data[weight_col].copy(),
        balance_before=before,
        balance_after=after,
        ess_by_arm=ess_by_arm,
        weight_summary=weight_summary,
        covariates=covariates,
        estimand=estimand,
        missing=missing,
        rows_dropped_missing=rows_dropped_missing,
        model=model,
        rows_trimmed=rows_trimmed,
        weight_col=weight_col,
    )
