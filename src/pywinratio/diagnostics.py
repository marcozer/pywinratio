from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


def effective_sample_size(weights: Iterable[float]) -> float:
    """Kish's effective sample size ``(sum w)^2 / sum w^2``.

    Parameters
    ----------
    weights : iterable of float
        Nonnegative analysis weights (non-finite and negative values ignored).

    Returns
    -------
    float
        Effective sample size (NaN when all weights are zero).

    Examples
    --------
    >>> effective_sample_size([1, 1, 1, 1])
    4.0
    >>> round(effective_sample_size([1, 1, 1, 3]), 2)
    3.0
    """

    values = np.asarray(list(weights), dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0 or values.sum() == 0:
        return float("nan")
    return float(values.sum() ** 2 / np.square(values).sum())


def propensity_overlap_coefficient(
    propensity: Iterable[float],
    treatment: Iterable[Any],
    *,
    treated: Any,
    bins: int = 40,
) -> float:
    """Overlap of the arm-specific propensity-score distributions.

    Estimated as the area under the pointwise minimum of the two histogram
    densities on ``bins`` equal-width bins of [0, 1]. The coefficient ranges
    from 0 (no overlap) to 1 (identical distributions). It is a descriptive
    diagnostic that depends modestly on ``bins``.

    Parameters
    ----------
    propensity : iterable of float
        Propensity scores in [0, 1].
    treatment : iterable
        Arm of each patient (at most two observed values).
    treated : any
        Value identifying the treated arm.
    bins : int, default 40
        Number of histogram bins.

    Returns
    -------
    float
        Overlap coefficient (NaN when an arm is empty).

    Examples
    --------
    >>> propensity_overlap_coefficient([0.1, 0.9, 0.1, 0.9], ["a", "a", "b", "b"], treated="a", bins=10)
    1.0
    """

    scores = np.asarray(list(propensity), dtype=float)
    groups = np.asarray(list(treatment), dtype=object)
    if scores.size != groups.size:
        raise ValueError("propensity and treatment must have the same length")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    valid = np.isfinite(scores) & (scores >= 0) & (scores <= 1) & ~pd.isna(groups)
    scores = scores[valid]
    groups = groups[valid]
    if pd.Series(groups).nunique() > 2:
        raise ValueError("treatment must contain at most 2 observed groups")
    first = scores[groups == treated]
    second = scores[groups != treated]
    if not len(first) or not len(second):
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    first_density, _ = np.histogram(first, bins=edges, density=True)
    second_density, _ = np.histogram(second, bins=edges, density=True)
    return float(np.minimum(first_density, second_density).sum() * np.diff(edges)[0])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not valid.any() or weights[valid].sum() == 0:
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    """Unbiased variance with reliability weights (the sample variance for unit weights)."""

    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not valid.any() or weights[valid].sum() == 0:
        return float("nan")
    x = values[valid]
    w = weights[valid]
    mean = np.average(x, weights=w)
    v1 = w.sum()
    denominator = v1 - np.square(w).sum() / v1
    if denominator <= 0:
        return float(np.average(np.square(x - mean), weights=w))
    return float(np.sum(w * np.square(x - mean)) / denominator)


def _weighted_ks(
    treated: np.ndarray,
    control: np.ndarray,
    treated_weights: np.ndarray,
    control_weights: np.ndarray,
) -> float:
    valid_t = np.isfinite(treated) & np.isfinite(treated_weights) & (treated_weights >= 0)
    valid_c = np.isfinite(control) & np.isfinite(control_weights) & (control_weights >= 0)
    if not valid_t.any() or not valid_c.any():
        return float("nan")
    x_t, w_t = treated[valid_t], treated_weights[valid_t]
    x_c, w_c = control[valid_c], control_weights[valid_c]
    if w_t.sum() == 0 or w_c.sum() == 0:
        return float("nan")
    grid = np.unique(np.concatenate([x_t, x_c]))
    cdf_t = np.array([w_t[x_t <= point].sum() / w_t.sum() for point in grid])
    cdf_c = np.array([w_c[x_c <= point].sum() / w_c.sum() for point in grid])
    return float(np.max(np.abs(cdf_t - cdf_c)))


def balance_diagnostics(
    df: pd.DataFrame,
    *,
    treatment: str,
    treated: Any,
    covariates: Iterable[str],
    weights: Optional[str | pd.Series] = None,
) -> pd.DataFrame:
    """Covariate balance between two arms: SMD, variance ratio and KS statistic.

    Categorical covariates are expanded into one indicator per observed level.
    SMDs use the square root of the mean of the two arm-specific variances in
    the denominator; with ``weights`` all statistics are weighted.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing the treatment and covariate columns.
    treatment : str
        Treatment column.
    treated : any
        Value of ``treatment`` identifying treated patients; all other rows are
        the comparison group.
    covariates : iterable of str
        Covariates to describe.
    weights : str or pandas.Series, optional
        Weight column name or weights aligned with ``df``.

    Returns
    -------
    pandas.DataFrame
        One row per term with ``covariate``, ``term``, ``kind``,
        ``mean_treated``, ``mean_control``, ``smd``, ``abs_smd``,
        ``variance_ratio`` and ``ks``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"g": [1, 1, 0, 0], "x": [1.0, 2.0, 2.0, 3.0]})
    >>> round(float(balance_diagnostics(df, treatment="g", treated=1, covariates=["x"])["smd"].iloc[0]), 3)
    -1.414
    """

    if treatment not in df.columns:
        raise ValueError(f"treatment column {treatment!r} not found")
    covariates = list(covariates)
    missing = [column for column in covariates if column not in df.columns]
    if missing:
        raise ValueError(f"covariates not found: {missing}")

    if weights is None:
        weight_values = pd.Series(1.0, index=df.index)
    elif isinstance(weights, str):
        if weights not in df.columns:
            raise ValueError(f"weight column {weights!r} not found")
        weight_values = pd.to_numeric(df[weights], errors="coerce")
    else:
        weight_values = pd.Series(weights, index=df.index, dtype=float)

    treated_mask = df[treatment] == treated
    rows: list[dict[str, Any]] = []

    for covariate in covariates:
        source = df[covariate]
        if pd.api.types.is_numeric_dtype(source):
            encoded = {covariate: pd.to_numeric(source, errors="coerce")}
            kind = "numeric"
        else:
            levels = sorted(source.dropna().astype(str).unique())
            encoded = {
                f"{covariate}={level}": (source.astype("string") == level).fillna(False).astype(float)
                for level in levels
            }
            kind = "categorical"

        for term, values in encoded.items():
            x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            w = weight_values.to_numpy(dtype=float)
            x_t, x_c = x[treated_mask.to_numpy()], x[~treated_mask.to_numpy()]
            w_t, w_c = w[treated_mask.to_numpy()], w[~treated_mask.to_numpy()]
            mean_t = _weighted_mean(x_t, w_t)
            mean_c = _weighted_mean(x_c, w_c)
            var_t = _weighted_variance(x_t, w_t)
            var_c = _weighted_variance(x_c, w_c)
            pooled = np.sqrt((var_t + var_c) / 2.0) if np.isfinite(var_t + var_c) else np.nan
            if not np.isfinite(pooled) or pooled == 0:
                smd = 0.0 if mean_t == mean_c else float("nan")
            else:
                smd = float((mean_t - mean_c) / pooled)
            variance_ratio = float(var_t / var_c) if np.isfinite(var_c) and var_c > 0 else np.nan
            rows.append(
                {
                    "covariate": covariate,
                    "term": term,
                    "kind": kind,
                    "mean_treated": mean_t,
                    "mean_control": mean_c,
                    "smd": smd,
                    "abs_smd": abs(smd) if np.isfinite(smd) else np.nan,
                    "variance_ratio": variance_ratio,
                    "ks": _weighted_ks(x_t, x_c, w_t, w_c),
                }
            )

    return pd.DataFrame(rows)
