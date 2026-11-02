"""1:1 propensity-score matching without replacement.

The matched data set feeds an assigned-pair win ratio
(``WinRatio(...).fit(matched_df, pair_id="PAIR_ID")``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd

from ._propensity import check_covariates, common_support_bounds, fit_propensity, treatment_indicator
from .diagnostics import balance_diagnostics


@dataclass
class MatchResult:
    """Output of :func:`propensity_match`.

    Attributes
    ----------
    matched_df : pandas.DataFrame
        Matched patients (two rows per pair) with the pair identifier
        (``pair_col``, default ``"PAIR_ID"``), the propensity score ``_PS``,
        its logit ``_LPS`` and the treatment indicator ``_T``.
    pairs : int
        Number of matched pairs.
    dropped_rows : int
        Eligible patients (after missing-data handling and trimming) left
        unmatched.
    ps : pandas.Series
        Propensity scores of the eligible patients.
    balance_before, balance_after : pandas.DataFrame
        :func:`pywinratio.balance_diagnostics` tables (SMD, variance ratio,
        KS) before matching (complete cases, before trimming) and in the
        matched sample.
    full_df : pandas.DataFrame
        Eligible patients after trimming, with ``_PS``, ``_LPS`` and ``_T``.
    caliper : float
        Caliper used, on the logit-propensity scale.
    rows_dropped_missing : int
        Patients removed because of missing covariates (``missing="drop"``).
    rows_trimmed : int
        Patients removed by common-support trimming.
    model : sklearn.pipeline.Pipeline or None
        Fitted propensity model (``None`` when scores were supplied).
    """

    matched_df: pd.DataFrame
    pairs: int
    dropped_rows: int
    ps: pd.Series
    balance_before: pd.DataFrame
    balance_after: pd.DataFrame
    full_df: Optional[pd.DataFrame] = None
    covariates: Optional[List[str]] = None
    exact_cols: Optional[List[str]] = None
    caliper: Optional[float] = None
    method: str = "optimal"
    treat_col: Optional[str] = None
    treated_value: Optional[object] = None
    missing: str = "drop"
    rows_dropped_missing: int = 0
    model: Optional[Any] = None
    control_value: Optional[object] = None
    rows_trimmed: int = 0
    pair_col: str = "PAIR_ID"
    order: str = "largest"

    def summary(self) -> pd.DataFrame:
        """Counts of eligible, matched and unmatched patients by arm, and balance."""

        full = self.full_df if self.full_df is not None else self.matched_df
        rows = []
        for label, value in (("treated", 1), ("control", 0)):
            eligible = int((full["_T"] == value).sum()) if "_T" in full else np.nan
            matched = int((self.matched_df["_T"] == value).sum()) if "_T" in self.matched_df else np.nan
            rows.append(
                {
                    "arm": label,
                    "eligible": eligible,
                    "matched": matched,
                    "unmatched": eligible - matched,
                    "retention": matched / eligible if eligible else np.nan,
                }
            )
        table = pd.DataFrame(rows)
        table.attrs["max_abs_smd_before"] = float(self.balance_before["abs_smd"].max())
        table.attrs["max_abs_smd_after"] = float(self.balance_after["abs_smd"].max())
        return table


def _greedy(
    t_scores: np.ndarray,
    c_scores: np.ndarray,
    caliper: float,
    order: str,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Greedy nearest-available-neighbor matching (Austin, 2014)."""

    if order == "largest":
        sequence = np.argsort(-t_scores, kind="mergesort")
    elif order == "smallest":
        sequence = np.argsort(t_scores, kind="mergesort")
    elif order == "random":
        sequence = rng.permutation(t_scores.size)
    elif order == "data":
        sequence = np.arange(t_scores.size)
    else:
        raise ValueError("order must be 'largest', 'smallest', 'random', or 'data'")
    available = np.ones(c_scores.size, dtype=bool)
    pairs = []
    for t in sequence:
        if not available.any():
            break
        distance = np.where(available, np.abs(c_scores - t_scores[t]), np.inf)
        j = int(np.argmin(distance))
        if distance[j] <= caliper:
            available[j] = False
            pairs.append((int(t), j))
    return pairs


def _optimal(t_scores: np.ndarray, c_scores: np.ndarray, caliper: float) -> list[tuple[int, int]]:
    """Optimal pair matching: maximum number of pairs, then minimum total distance."""

    from scipy.optimize import linear_sum_assignment

    distance = np.abs(t_scores[:, None] - c_scores[None, :])
    feasible = distance <= caliper
    if not feasible.any():
        return []
    d_max = float(distance[feasible].max())
    penalty = (t_scores.size + 1) * (d_max + 1.0)
    cost = np.where(feasible, distance, penalty)
    cost = np.hstack([cost, np.full((t_scores.size, t_scores.size), penalty)])
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if c < c_scores.size and feasible[r, c]]


def propensity_match(
    df: pd.DataFrame,
    treat_col: str,
    treated_value: object,
    covariates: Optional[List[str]] = None,
    additional_covariates: Optional[List[str]] = None,
    exact_cols: Optional[List[str]] = None,
    caliper: Optional[float] = None,
    method: Literal["nearest", "optimal"] = "optimal",
    trim_common_support: bool = True,
    missing: Literal["drop", "simple"] = "drop",
    random_state: int = 42,
    *,
    control_value: object = None,
    order: Literal["largest", "smallest", "random", "data"] = "largest",
    penalty: Optional[float] = None,
    ps_col: Optional[str] = None,
    pair_col: str = "PAIR_ID",
) -> MatchResult:
    """1:1 propensity-score matching without replacement.

    A logistic propensity model is fitted on ``covariates`` (or scores are
    taken from ``ps_col``), patients are matched on the logit of the score
    within an optional caliper and within exact-matching strata.

    Parameters
    ----------
    df : pandas.DataFrame
        Cohort, one row per patient.
    treat_col : str
        Treatment (exposure) column.
    treated_value : any
        Value of ``treat_col`` identifying treated patients.
    covariates : list of str
        Propensity-model covariates. Required (unless ``ps_col`` is given) so
        outcome columns can never enter the model by accident. Numeric
        columns are standardized; other columns are one-hot encoded.
    additional_covariates : list of str, optional
        Extra covariates appended to ``covariates`` when present in ``df``.
    exact_cols : list of str, optional
        Columns on which patients must match exactly (e.g. period, hospital
        type). They are removed from the propensity model.
    caliper : float, optional
        Maximum absolute difference in logit propensity score. Defaults to
        0.2 x the pooled within-arm SD of the logit score (Austin, 2011).
        Use ``float("inf")`` for no caliper.
    method : {"optimal", "nearest"}, default "optimal"
        ``"optimal"`` maximizes the number of pairs and then minimizes the
        total distance (Hungarian algorithm). ``"nearest"`` is greedy
        nearest-available-neighbor matching in ``order``.
    trim_common_support : bool, default True
        Drop patients outside the overlap of the arm-specific score ranges.
    missing : {"drop", "simple"}, default "drop"
        Complete-case analysis, or explicit single (median / mode) imputation
        of covariates. Multiple imputation should be done outside.
    random_state : int, default 42
        Seed for ``order="random"`` and the model.
    control_value : any, optional
        Comparator value of ``treat_col``; required when it has more than
        two levels.
    order : {"largest", "smallest", "random", "data"}, default "largest"
        Order in which treated patients choose controls (greedy only).
    penalty : float, optional
        Inverse L2 regularization strength ``C``; ``None`` fits the
        unpenalized logistic model.
    ps_col : str, optional
        Use precomputed propensity scores from this column.
    pair_col : str, default "PAIR_ID"
        Name of the pair-identifier column in ``matched_df``.

    Returns
    -------
    MatchResult

    Examples
    --------
    >>> from pywinratio.datasets import make_surgical_cohort
    >>> cohort = make_surgical_cohort(n=400, seed=1)
    >>> m = propensity_match(cohort, "volume", "high", covariates=["age", "asa3", "bmi"])
    >>> bool(m.matched_df.groupby("PAIR_ID").size().eq(2).all())
    True
    """

    if method not in {"nearest", "optimal"}:
        raise ValueError("method must be 'nearest' or 'optimal'")
    if missing not in {"drop", "simple"}:
        raise ValueError("missing must be 'drop' or 'simple'")
    data, indicator = treatment_indicator(df, treat_col, treated_value, control_value)
    control_value = data.loc[indicator == 0, treat_col].iloc[0] if (indicator == 0).any() else control_value
    exact_cols = list(exact_cols or [])

    if ps_col is None:
        if covariates is None:
            raise ValueError("covariates must be specified explicitly to prevent accidental outcome adjustment")
        covariates = list(covariates)
        for column in additional_covariates or []:
            if column in data.columns and column not in covariates:
                covariates.append(column)
        covariates = [c for c in covariates if c not in set(exact_cols)]
        if not covariates:
            raise ValueError("propensity_match requires at least one covariate.")
        covariates = check_covariates(data, covariates)
    else:
        if ps_col not in data.columns:
            raise ValueError(f"ps_col {ps_col!r} not found")
        covariates = list(covariates or [])

    needed = list(covariates) + exact_cols + ([ps_col] if ps_col else [])
    rows_before = len(data)
    if missing == "drop" or ps_col is not None:
        keep = data[needed].notna().all(axis=1)
    else:
        keep = data[exact_cols].notna().all(axis=1) if exact_cols else pd.Series(True, index=data.index)
    data = data[keep].copy()
    indicator = indicator[keep]
    rows_dropped_missing = rows_before - len(data)
    if indicator.nunique() < 2:
        raise ValueError("Matching requires both treatment arms after missing-data handling.")

    if ps_col is None:
        model, scores = fit_propensity(data, covariates, indicator, missing=missing, penalty=penalty, seed=random_state)
    else:
        model = None
        scores = pd.to_numeric(data[ps_col], errors="coerce")
        if not scores.between(0, 1).all():
            raise ValueError("propensity scores must lie in [0, 1]")
    data["_T"] = indicator.astype(int)
    data["_PS"] = scores
    clipped = scores.clip(1e-6, 1 - 1e-6)
    data["_LPS"] = np.log(clipped / (1 - clipped))

    balance_columns = list(dict.fromkeys(list(covariates) + exact_cols))
    balance_before = (
        balance_diagnostics(data, treatment=treat_col, treated=treated_value, covariates=balance_columns)
        if balance_columns
        else pd.DataFrame()
    )

    rows_trimmed = 0
    if trim_common_support:
        low, high = common_support_bounds(data["_PS"], data["_T"])
        inside = data["_PS"].between(low, high)
        rows_trimmed = int((~inside).sum())
        data = data[inside].copy()
    if data["_T"].nunique() < 2:
        raise ValueError("Matching requires both treatment arms after filtering.")

    if caliper is None:
        sd_t = data.loc[data["_T"] == 1, "_LPS"].std(ddof=1)
        sd_c = data.loc[data["_T"] == 0, "_LPS"].std(ddof=1)
        pooled = float(np.sqrt(np.nansum([sd_t**2, sd_c**2]) / 2.0))
        caliper = 0.2 * pooled if pooled > 0 else float("inf")
    caliper = float(caliper)
    if caliper < 0:
        raise ValueError("caliper must be nonnegative")

    rng = np.random.default_rng(random_state)
    if exact_cols:
        groups = [sub for _, sub in data.groupby(exact_cols, sort=True, dropna=True)]
    else:
        groups = [data]
    pieces = []
    pair_id = 0
    for sub in groups:
        treated = sub[sub["_T"] == 1]
        control = sub[sub["_T"] == 0]
        if treated.empty or control.empty:
            continue
        t_scores = treated["_LPS"].to_numpy(dtype=float)
        c_scores = control["_LPS"].to_numpy(dtype=float)
        if method == "nearest":
            matches = _greedy(t_scores, c_scores, caliper, order, rng)
        else:
            matches = _optimal(t_scores, c_scores, caliper)
        for t, c in sorted(matches):
            pair_id += 1
            pair = pd.concat([treated.iloc[[t]], control.iloc[[c]]])
            pair[pair_col] = pair_id
            pieces.append(pair)

    matched_df = pd.concat(pieces, axis=0) if pieces else data.iloc[0:0].assign(**{pair_col: []})
    balance_after = (
        balance_diagnostics(matched_df, treatment=treat_col, treated=treated_value, covariates=balance_columns)
        if balance_columns and pair_id
        else balance_before.iloc[0:0].copy()
    )
    return MatchResult(
        matched_df=matched_df,
        pairs=pair_id,
        dropped_rows=int(len(data) - len(matched_df)),
        ps=data["_PS"].copy(),
        balance_before=balance_before,
        balance_after=balance_after,
        full_df=data,
        covariates=list(covariates),
        exact_cols=exact_cols,
        caliper=caliper,
        method=method,
        treat_col=treat_col,
        treated_value=treated_value,
        missing=missing,
        rows_dropped_missing=rows_dropped_missing,
        model=model,
        control_value=control_value,
        rows_trimmed=rows_trimmed,
        pair_col=pair_col,
        order=order,
    )
