"""Vectorised hierarchical comparison engine.

This private module holds the only implementation of the prioritized
comparison rule. Every public estimator (assigned pairs, all pairs,
stratified, weighted, bootstrap, permutation) calls into it, so the
hierarchy is evaluated identically everywhere.

Shapes are broadcast-agnostic: passing column vectors ``(n_a, 1)`` and row
vectors ``(1, n_b)`` evaluates every cross-arm pair, while passing two
aligned 1-D arrays ``(m,)`` evaluates assigned pairs element by element.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .config import WinRatioOutcome

#: Default maximum number of pair cells evaluated in one block.
MAX_CELLS = 16_000_000

#: Level code for a pair tied on every tier.
LEVEL_TIE = -1


def terminal_stop_code(tier_index: int) -> int:
    """Level code for a pair stopped as a tie by a shared terminal event."""

    return -(tier_index + 2)


# ---------------------------------------------------------------------------
# Data extraction and validation
# ---------------------------------------------------------------------------


def column_as_float(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Return ``frame[column]`` as a float array; non-numeric values raise."""

    if column not in frame.columns:
        raise ValueError(f"column {column!r} not found")
    series = frame[column]
    if series.dtype.kind in "iufb":
        return series.to_numpy(dtype=float, copy=False)
    converted = pd.to_numeric(series, errors="coerce")
    bad = series.notna() & converted.isna()
    if bad.any():
        examples = sorted({str(value) for value in series[bad].unique()[:5]})
        raise ValueError(
            f"column {column!r} contains non-numeric values {examples}; "
            "recode outcomes as numbers (binary outcomes as 0/1)"
        )
    return converted.to_numpy(dtype=float, copy=False)


def required_columns(outcomes: Sequence[WinRatioOutcome]) -> list[str]:
    """Columns needed to evaluate a hierarchy."""

    columns: list[str] = []
    for outcome in outcomes:
        columns.append(outcome.column)
        if outcome.event_column is not None:
            columns.append(outcome.event_column)
    return columns


def validate_outcome_data(frame: pd.DataFrame, outcomes: Sequence[WinRatioOutcome]) -> None:
    """Check that outcome columns exist and hold values valid for their kind."""

    missing = sorted({column for column in required_columns(outcomes) if column not in frame.columns})
    if missing:
        raise ValueError(f"required columns not found: {missing}")
    for outcome in outcomes:
        values = column_as_float(frame, outcome.column)
        observed = values[~np.isnan(values)]
        if outcome.kind == "binary":
            invalid = np.setdiff1d(np.unique(observed), [0.0, 1.0])
            if invalid.size:
                raise ValueError(
                    f"binary outcome {outcome.name!r} ({outcome.column!r}) must be coded 0/1; "
                    f"found {invalid[:5].tolist()}"
                )
        elif outcome.kind == "time_to_event":
            if observed.size and observed.min() < 0:
                raise ValueError(f"time-to-event outcome {outcome.name!r} has negative times")
            events = column_as_float(frame, outcome.event_column)  # type: ignore[arg-type]
            invalid = np.setdiff1d(np.unique(events[~np.isnan(events)]), [0.0, 1.0])
            if invalid.size:
                raise ValueError(
                    f"event indicator {outcome.event_column!r} of {outcome.name!r} must be coded "
                    f"1 = event, 0 = censored; found {invalid[:5].tolist()}"
                )


@dataclass(frozen=True)
class TierValues:
    """Numeric values (and event indicators for time-to-event tiers) of one tier."""

    values: np.ndarray
    events: Optional[np.ndarray] = None

    def take(self, index: np.ndarray | slice) -> TierValues:
        return TierValues(
            self.values[index],
            None if self.events is None else self.events[index],
        )

    def reshape(self, shape: tuple[int, ...]) -> TierValues:
        return TierValues(
            self.values.reshape(shape),
            None if self.events is None else self.events.reshape(shape),
        )


def extract_tiers(frame: pd.DataFrame, outcomes: Sequence[WinRatioOutcome]) -> list[TierValues]:
    """Extract the arrays needed to evaluate each tier of a hierarchy."""

    tiers = []
    for outcome in outcomes:
        values = column_as_float(frame, outcome.column)
        events = column_as_float(frame, outcome.event_column) if outcome.event_column else None
        tiers.append(TierValues(np.asarray(values, dtype=float), events))
    return tiers


# ---------------------------------------------------------------------------
# One tier
# ---------------------------------------------------------------------------


def _apply_missing_policy(
    win: np.ndarray,
    loss: np.ndarray,
    a_missing: np.ndarray,
    b_missing: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve comparisons involving a missing value.

    ``tie`` passes the pair to the next tier; ``loss`` makes the patient whose
    value is missing lose the tier; ``win`` makes that patient win it. When
    both values are missing the tier is always tied.
    """

    either = a_missing | b_missing
    win = win & ~either
    loss = loss & ~either
    if policy == "tie":
        return win, loss
    only_a = a_missing & ~b_missing
    only_b = b_missing & ~a_missing
    if policy == "loss":
        return win | only_b, loss | only_a
    return win | only_a, loss | only_b


def _separated(diff: np.ndarray, outcome: WinRatioOutcome, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """``(diff beyond +tolerance, diff beyond -tolerance)`` under the tie convention.

    A relative epsilon protects margins from floating-point noise
    (``1.1 - 1.0`` is not larger than a 0.1 margin).
    """

    eps = 1e-9 * tolerance
    if outcome.tie_inclusive:
        return diff > tolerance + eps, diff < -(tolerance + eps)
    return diff >= tolerance - eps, diff <= -(tolerance - eps)


def tier_masks(
    a: TierValues,
    b: TierValues,
    outcome: WinRatioOutcome,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return broadcast ``(win, loss, stop)`` masks for one tier.

    ``win`` and ``loss`` are from the arm-A patient's perspective. ``stop``
    flags tied pairs in which both patients had the terminal event, so lower
    tiers must not be evaluated.
    """

    if outcome.kind == "time_to_event":
        return _time_to_event_masks(a, b, outcome)

    a_raw = a.values
    b_raw = b.values
    a_missing = np.isnan(a_raw)
    b_missing = np.isnan(b_raw)

    if outcome.kind == "binary":
        if outcome.direction == "lower":
            win = (a_raw == 0.0) & (b_raw == 1.0)
            loss = (a_raw == 1.0) & (b_raw == 0.0)
        else:
            win = (a_raw == 1.0) & (b_raw == 0.0)
            loss = (a_raw == 0.0) & (b_raw == 1.0)
    else:
        a_num = a_raw
        b_num = b_raw
        if outcome.cutpoints is not None:
            cuts = np.asarray(outcome.cutpoints, dtype=float)
            a_num = np.where(a_missing, np.nan, np.searchsorted(cuts, a_raw, side="left"))
            b_num = np.where(b_missing, np.nan, np.searchsorted(cuts, b_raw, side="left"))
        mode = outcome.los_tolerance_mode
        with np.errstate(invalid="ignore"):
            if mode == "threshold_based" and outcome.los_threshold is not None:
                a_meets = a_num <= outcome.los_threshold
                b_meets = b_num <= outcome.los_threshold
                if outcome.direction == "lower":
                    win = a_meets & ~b_meets
                    loss = ~a_meets & b_meets
                else:
                    win = ~a_meets & b_meets
                    loss = a_meets & ~b_meets
            else:
                tolerance = 1.0 if mode == "tolerance_1day" else float(outcome.tie_tol or 0.0)
                above, below = _separated(a_num - b_num, outcome, tolerance)
                win, loss = (below, above) if outcome.direction == "lower" else (above, below)

    win, loss = _apply_missing_policy(win, loss, a_missing, b_missing, outcome.missing_is)
    if outcome.terminal:
        stop = (a_raw == outcome.terminal_value) & (b_raw == outcome.terminal_value) & ~win & ~loss
    else:
        stop = np.zeros(np.broadcast_shapes(a_raw.shape, b_raw.shape), dtype=bool)
    return win, loss, stop


def _time_to_event_masks(
    a: TierValues,
    b: TierValues,
    outcome: WinRatioOutcome,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gehan scoring rule for a right-censored time-to-event tier.

    A pair is resolved only when the earlier of the two times is an observed
    event, i.e. within the patients' common follow-up. With an adverse event
    (``direction="higher"``: later is better) arm A wins when B had the event
    and A was still event-free more than ``tie_tol`` time units later.

    Without a margin, a censoring time equal to the other patient's event
    time counts as event-free at that time (Gehan, 1965), as in BuyseTest.
    """

    t_a = a.values
    t_b = b.values
    e_a = a.events
    e_b = b.events
    assert e_a is not None and e_b is not None
    a_missing = np.isnan(t_a) | np.isnan(e_a)
    b_missing = np.isnan(t_b) | np.isnan(e_b)
    event_a = e_a == 1.0
    event_b = e_b == 1.0
    if outcome.horizon is not None:
        tau = float(outcome.horizon)
        event_a = event_a & (t_a <= tau)
        event_b = event_b & (t_b <= tau)
        t_a = np.minimum(t_a, tau)
        t_b = np.minimum(t_b, tau)
    margin = float(outcome.tie_tol or 0.0)
    with np.errstate(invalid="ignore"):
        a_later, a_earlier = _separated(t_a - t_b, outcome, margin)
        if outcome.direction == "higher":
            win = event_b & a_later
            loss = event_a & a_earlier
        else:
            win = event_a & a_earlier
            loss = event_b & a_later
        if margin == 0 and outcome.tie_inclusive:
            same = t_a == t_b
            a_only = same & event_a & ~event_b
            b_only = same & event_b & ~event_a
            if outcome.direction == "higher":
                win = win | b_only
                loss = loss | a_only
            else:
                win = win | a_only
                loss = loss | b_only
    win, loss = _apply_missing_policy(win, loss, a_missing, b_missing, outcome.missing_is)
    if outcome.terminal:
        stop = event_a & event_b & ~win & ~loss & ~a_missing & ~b_missing
    else:
        stop = np.zeros(np.broadcast_shapes(t_a.shape, t_b.shape), dtype=bool)
    return win, loss, stop


# ---------------------------------------------------------------------------
# Whole hierarchy
# ---------------------------------------------------------------------------


def resolve(
    a_tiers: Sequence[TierValues],
    b_tiers: Sequence[TierValues],
    outcomes: Sequence[WinRatioOutcome],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the hierarchy on broadcast-compatible tier values.

    Returns
    -------
    result : int8 array
        ``1`` when the arm-A patient wins, ``-1`` when it loses, ``0`` for a tie.
    level : int16 array
        Index of the deciding tier for wins and losses, :data:`LEVEL_TIE` for
        pairs tied on every tier, and :func:`terminal_stop_code` for pairs
        stopped by a shared terminal event.
    """

    shape = np.broadcast_shapes(a_tiers[0].values.shape, b_tiers[0].values.shape)
    result = np.zeros(shape, dtype=np.int8)
    level = np.full(shape, LEVEL_TIE, dtype=np.int16)
    unresolved = np.ones(shape, dtype=bool)
    for index, (outcome, a, b) in enumerate(zip(outcomes, a_tiers, b_tiers)):
        win, loss, stop = tier_masks(a, b, outcome)
        new_win = unresolved & win
        new_loss = unresolved & loss
        new_stop = unresolved & stop
        result[new_win] = 1
        result[new_loss] = -1
        level[new_win | new_loss] = index
        level[new_stop] = terminal_stop_code(index)
        unresolved &= ~(win | loss | stop)
        if not unresolved.any():
            break
    return result, level


def resolve_all_pairs(
    a_tiers: Sequence[TierValues],
    b_tiers: Sequence[TierValues],
    outcomes: Sequence[WinRatioOutcome],
    rows: slice | np.ndarray = slice(None),
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate selected arm-A patients against every arm-B patient."""

    a_block = [tier.take(rows) for tier in a_tiers]
    a_block = [tier.reshape((-1, 1)) for tier in a_block]
    b_block = [tier.reshape((1, -1)) for tier in b_tiers]
    return resolve(a_block, b_block, outcomes)


def resolve_matched(
    a_tiers: Sequence[TierValues],
    b_tiers: Sequence[TierValues],
    outcomes: Sequence[WinRatioOutcome],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate aligned assigned pairs element by element."""

    return resolve(a_tiers, b_tiers, outcomes)


# ---------------------------------------------------------------------------
# Sufficient statistics for one comparison block
# ---------------------------------------------------------------------------


@dataclass
class PairSummary:
    """Weighted win/loss totals of one stratum, with per-patient margins.

    The per-patient margins (``a_wins`` = weighted number of arm-B patients
    beaten by each arm-A patient, etc.) are the structural components used by
    the U-statistic variance.
    """

    n_tiers: int
    a_labels: np.ndarray
    b_labels: np.ndarray
    a_weights: np.ndarray
    b_weights: np.ndarray
    wins: float = 0.0
    losses: float = 0.0
    total: float = 0.0
    tier_wins: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tier_losses: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tier_stops: np.ndarray = field(default_factory=lambda: np.zeros(0))
    a_wins: np.ndarray = field(default_factory=lambda: np.zeros(0))
    a_losses: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b_wins: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b_losses: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def ties(self) -> float:
        return self.total - self.wins - self.losses

    @property
    def n_a(self) -> int:
        return int(self.a_labels.size)

    @property
    def n_b(self) -> int:
        return int(self.b_labels.size)


def summarize_all_pairs(
    a_frame: pd.DataFrame,
    b_frame: pd.DataFrame,
    outcomes: Sequence[WinRatioOutcome],
    *,
    a_weights: Optional[np.ndarray] = None,
    b_weights: Optional[np.ndarray] = None,
    max_cells: int = MAX_CELLS,
) -> PairSummary:
    """Compare every arm-A patient with every arm-B patient, in row blocks."""

    n_a, n_b = len(a_frame), len(b_frame)
    k = len(outcomes)
    w_a = np.ones(n_a) if a_weights is None else np.asarray(a_weights, dtype=float)
    w_b = np.ones(n_b) if b_weights is None else np.asarray(b_weights, dtype=float)
    summary = PairSummary(
        n_tiers=k,
        a_labels=a_frame.index.to_numpy(),
        b_labels=b_frame.index.to_numpy(),
        a_weights=w_a,
        b_weights=w_b,
        tier_wins=np.zeros(k),
        tier_losses=np.zeros(k),
        tier_stops=np.zeros(k),
        a_wins=np.zeros(n_a),
        a_losses=np.zeros(n_a),
        b_wins=np.zeros(n_b),
        b_losses=np.zeros(n_b),
    )
    summary.total = float(w_a.sum() * w_b.sum())
    if n_a == 0 or n_b == 0:
        return summary

    a_tiers = extract_tiers(a_frame, outcomes)
    b_tiers = extract_tiers(b_frame, outcomes)
    block = max(1, int(max_cells // max(n_b, 1)))
    for start in range(0, n_a, block):
        rows = slice(start, min(start + block, n_a))
        result, level = resolve_all_pairs(a_tiers, b_tiers, outcomes, rows)
        win = (result == 1).astype(float)
        loss = (result == -1).astype(float)
        wa = w_a[rows]
        # weighted margins: sum over the other arm of pair indicators x weights
        summary.a_wins[rows] = win @ w_b
        summary.a_losses[rows] = loss @ w_b
        summary.b_wins += wa @ win
        summary.b_losses += wa @ loss
        pair_w = wa[:, None] * w_b[None, :]
        for index in range(k):
            at_level = level == index
            summary.tier_wins[index] += float(pair_w[at_level & (result == 1)].sum())
            summary.tier_losses[index] += float(pair_w[at_level & (result == -1)].sum())
            summary.tier_stops[index] += float(pair_w[level == terminal_stop_code(index)].sum())
    summary.wins = float(summary.tier_wins.sum())
    summary.losses = float(summary.tier_losses.sum())
    return summary


def pair_result_matrices(
    a_frame: pd.DataFrame,
    b_frame: pd.DataFrame,
    outcomes: Sequence[WinRatioOutcome],
    *,
    max_cells: int = MAX_CELLS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 ``(wins, losses)`` indicator matrices of shape ``(n_a, n_b)``."""

    n_a, n_b = len(a_frame), len(b_frame)
    wins = np.zeros((n_a, n_b), dtype=np.uint8)
    losses = np.zeros((n_a, n_b), dtype=np.uint8)
    if n_a == 0 or n_b == 0:
        return wins, losses
    a_tiers = extract_tiers(a_frame, outcomes)
    b_tiers = extract_tiers(b_frame, outcomes)
    block = max(1, int(max_cells // max(n_b, 1)))
    for start in range(0, n_a, block):
        rows = slice(start, min(start + block, n_a))
        result, _ = resolve_all_pairs(a_tiers, b_tiers, outcomes, rows)
        wins[rows] = result == 1
        losses[rows] = result == -1
    return wins, losses
