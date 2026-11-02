"""Brute-force, loop-based reference implementation used to test the engine.

It follows the documented comparison rules one pair at a time with plain
Python scalars, independently of the vectorized implementation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from pywinratio import WinRatioOutcome


def _nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _category(x: float, cuts: Sequence[float]) -> int:
    return sum(1 for c in cuts if x > c)


def _beyond(diff: float, oc: WinRatioOutcome, tol: float) -> Tuple[bool, bool]:
    eps = 1e-9 * tol
    if oc.tie_inclusive:
        return diff > tol + eps, diff < -(tol + eps)
    return diff >= tol - eps, diff <= -(tol - eps)


def compare_tier(a: Dict[str, Any], b: Dict[str, Any], oc: WinRatioOutcome) -> Tuple[int, bool]:
    """Return (result for A, stop) on one tier."""

    if oc.kind == "time_to_event":
        ta, ea, tb, eb = a[oc.column], a[oc.event_column], b[oc.column], b[oc.event_column]
        a_missing, b_missing = _nan(ta) or _nan(ea), _nan(tb) or _nan(eb)
    else:
        a_missing, b_missing = _nan(a[oc.column]), _nan(b[oc.column])
    if a_missing or b_missing:
        if a_missing and b_missing or oc.missing_is == "tie":
            return 0, False
        if oc.missing_is == "loss":
            return (-1 if a_missing else 1), False
        return (1 if a_missing else -1), False

    if oc.kind == "time_to_event":
        ea, eb = ea == 1, eb == 1
        if oc.horizon is not None:
            ea, eb = ea and ta <= oc.horizon, eb and tb <= oc.horizon
            ta, tb = min(ta, oc.horizon), min(tb, oc.horizon)
        later, earlier = _beyond(ta - tb, oc, oc.tie_tol)
        if oc.direction == "higher":
            win, loss = eb and later, ea and earlier
        else:
            win, loss = ea and earlier, eb and later
        if oc.tie_tol == 0 and oc.tie_inclusive and ta == tb:
            a_only, b_only = ea and not eb, eb and not ea
            if oc.direction == "higher":
                win, loss = win or b_only, loss or a_only
            else:
                win, loss = win or a_only, loss or b_only
        result = 1 if win else (-1 if loss else 0)
        return result, bool(oc.terminal and ea and eb and result == 0)

    x, y = float(a[oc.column]), float(b[oc.column])
    if oc.kind == "binary":
        if oc.direction == "lower":
            result = 1 if (x == 0 and y == 1) else (-1 if (x == 1 and y == 0) else 0)
        else:
            result = 1 if (x == 1 and y == 0) else (-1 if (x == 0 and y == 1) else 0)
    else:
        if oc.cutpoints is not None:
            x, y = float(_category(x, oc.cutpoints)), float(_category(y, oc.cutpoints))
        above, below = _beyond(x - y, oc, oc.tie_tol)
        better, worse = (below, above) if oc.direction == "lower" else (above, below)
        result = 1 if better else (-1 if worse else 0)
    tv = oc.terminal_value
    stop = bool(oc.terminal and result == 0 and a[oc.column] == tv and b[oc.column] == tv)
    return result, stop


def compare_pair(a: Dict[str, Any], b: Dict[str, Any], outcomes: List[WinRatioOutcome]) -> Tuple[int, int]:
    """Return (result, level) with level as in the engine."""

    for index, oc in enumerate(outcomes):
        result, stop = compare_tier(a, b, oc)
        if result != 0:
            return result, index
        if stop:
            return 0, -(index + 2)
    return 0, -1


def all_pairs_counts(df: pd.DataFrame, group: str, arm_a: Any, arm_b: Any, outcomes: List[WinRatioOutcome]):
    rows_a = df[df[group] == arm_a].to_dict("records")
    rows_b = df[df[group] == arm_b].to_dict("records")
    wins = losses = ties = 0
    tier_wins = [0] * len(outcomes)
    tier_losses = [0] * len(outcomes)
    for a in rows_a:
        for b in rows_b:
            result, level = compare_pair(a, b, outcomes)
            if result > 0:
                wins += 1
                tier_wins[level] += 1
            elif result < 0:
                losses += 1
                tier_losses[level] += 1
            else:
                ties += 1
    return wins, losses, ties, tier_wins, tier_losses
