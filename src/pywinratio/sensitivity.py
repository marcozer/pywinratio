"""Sensitivity analyses for hierarchical endpoints.

A hierarchy embeds clinical judgements: which outcomes enter, in which
order, and with which tie margins. Reporting guidance asks for these choices
to be prespecified and their influence shown (Redfors et al., 2020;
Pocock et al., 2024).
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from .analysis import WinRatioAnalysis, WinRatioResult
from .config import WinRatioOutcome
from .inference import adjust_pvalues


def _row(label: str, result: WinRatioResult) -> Dict[str, Any]:
    frame = result.to_frame()
    return {
        "hierarchy": label,
        "wr": result.wr,
        "ci_lower": frame.loc["Win ratio", "ci_lower"],
        "ci_upper": frame.loc["Win ratio", "ci_upper"],
        "net_benefit": result.net_benefit,
        "net_benefit_ci_lower": frame.loc["Net benefit", "ci_lower"],
        "net_benefit_ci_upper": frame.loc["Net benefit", "ci_upper"],
        "win_odds": result.win_odds,
        "p_value": result.p_value,
        "wins": result.wins,
        "losses": result.losses,
        "ties": result.ties,
        "tie_share": result.ties / result.n_pairs if result.n_pairs else float("nan"),
    }


def compare_hierarchies(
    data: pd.DataFrame,
    hierarchies: Mapping[str, Sequence[WinRatioOutcome]],
    *,
    group: str,
    treatment: Any,
    control: Any,
    strata: Optional[Iterable[str]] = None,
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """Fit several hierarchies on the same data and tabulate the results.

    Parameters
    ----------
    data : pandas.DataFrame
        Analysis data.
    hierarchies : mapping of str to list of WinRatioOutcome
        Named hierarchies, e.g. primary, alternative margins, alternative
        orders, hierarchy without length of stay.
    group, treatment, control, strata
        As in :class:`pywinratio.WinRatioAnalysis`.
    **fit_kwargs
        Passed to :meth:`pywinratio.WinRatioAnalysis.fit` (``pair_id``,
        ``weight_col``, ``n_boot``, ...).

    Returns
    -------
    pandas.DataFrame
        One row per hierarchy with WR, net benefit, win odds, intervals,
        p-value, wins, losses and ties (ready for :func:`pywinratio.plot_forest`).

    Examples
    --------
    >>> from pywinratio import Outcome, compare_hierarchies
    >>> from pywinratio.datasets import make_surgical_cohort
    >>> cohort = make_surgical_cohort(n=300, seed=4)
    >>> table = compare_hierarchies(cohort, {
    ...     "exact LOS": [Outcome.binary("death90"), Outcome.continuous("los")],
    ...     "LOS margin 1 day": [Outcome.binary("death90"), Outcome.continuous("los", margin=1)],
    ... }, group="volume", treatment="high", control="low")
    >>> table["hierarchy"].tolist()
    ['exact LOS', 'LOS margin 1 day']
    """

    rows = []
    for label, outcomes in hierarchies.items():
        analysis = WinRatioAnalysis(outcomes, group=group, treatment=treatment, control=control, strata=strata)
        rows.append(_row(label, analysis.fit(data, **fit_kwargs)))
    return pd.DataFrame(rows)


def order_sensitivity(
    data: pd.DataFrame,
    outcomes: Sequence[WinRatioOutcome],
    *,
    group: str,
    treatment: Any,
    control: Any,
    fixed: int = 0,
    max_orders: int = 720,
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """Win ratio under every ordering of the hierarchy's tiers.

    Parameters
    ----------
    outcomes : sequence of WinRatioOutcome
        Prespecified hierarchy.
    fixed : int, default 0
        Number of leading tiers kept in place (e.g. ``fixed=1`` keeps death
        first and permutes the rest).
    max_orders : int, default 720
        Safety limit on the number of orderings.
    group, treatment, control, **fit_kwargs
        As in :func:`compare_hierarchies`.

    Returns
    -------
    pandas.DataFrame
        One row per ordering; ``is_prespecified`` flags the original order.
    """

    head = list(outcomes[:fixed])
    tail = list(outcomes[fixed:])
    orders = list(permutations(range(len(tail))))
    if len(orders) > max_orders:
        raise ValueError(f"{len(orders)} orderings exceed max_orders={max_orders}; fix more leading tiers")
    rows = []
    for order in orders:
        tiers = head + [tail[i] for i in order]
        label = " > ".join(t.name for t in tiers)
        analysis = WinRatioAnalysis(tiers, group=group, treatment=treatment, control=control)
        row = _row(label, analysis.fit(data, **fit_kwargs))
        row["is_prespecified"] = list(order) == sorted(order)
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_win_ratios(
    data: pd.DataFrame,
    outcomes: Sequence[WinRatioOutcome],
    *,
    group: str,
    arms: Optional[Sequence[Any]] = None,
    reference: Any = None,
    adjust: str = "holm",
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """Pairwise win statistics across several arms with multiplicity adjustment.

    Parameters
    ----------
    group : str
        Arm column.
    arms : sequence, optional
        Arms to compare (default: all observed, sorted).
    reference : any, optional
        Compare every other arm with this reference arm; otherwise all pairs.
    adjust : {"holm", "bonferroni", "fdr_bh"}, default "holm"
        Multiplicity adjustment of the p-values.
    **fit_kwargs
        Passed to :meth:`pywinratio.WinRatioAnalysis.fit`.

    Returns
    -------
    pandas.DataFrame
        One row per comparison with ``p_value_adjusted``.
    """

    arms = list(arms) if arms is not None else sorted(data[group].dropna().unique().tolist())
    if reference is not None:
        comparisons = [(arm, reference) for arm in arms if arm != reference]
    else:
        comparisons = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1 :]]
    rows = []
    for arm_a, arm_b in comparisons:
        analysis = WinRatioAnalysis(outcomes, group=group, treatment=arm_a, control=arm_b)
        row = _row(f"{arm_a} vs {arm_b}", analysis.fit(data, **fit_kwargs))
        row.update({"arm_a": arm_a, "arm_b": arm_b})
        rows.append(row)
    table = pd.DataFrame(rows).rename(columns={"hierarchy": "comparison"})
    table["p_value_adjusted"] = adjust_pvalues(table["p_value"].tolist(), adjust)
    return table
