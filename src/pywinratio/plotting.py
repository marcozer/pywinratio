"""Matplotlib figures for win-ratio analyses (``pip install pywinratio[plot]``).

* :func:`plot_tiers`: where pairs were won, lost or tied, tier by tier;
* :func:`plot_forest`: estimates with intervals (sensitivity analyses,
  subgroups, multi-arm comparisons);
* :func:`plot_balance`: Love plot of absolute standardized differences;
* :func:`plot_bootstrap`: bootstrap distribution of the log win ratio.

All functions accept an optional ``ax`` and return it.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

#: Colors (validated categorical slots 1-2 and neutral tones).
COLORS = {
    "arm_a": "#2a78d6",
    "arm_b": "#eb6834",
    "tie": "#c9c8c2",
    "terminal": "#8c8b85",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e6e5e0",
}


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("plotting requires matplotlib: pip install 'pywinratio[plot]'") from exc
    return plt


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"], labelcolor=COLORS["text"])
    ax.set_axisbelow(True)


def plot_tiers(
    result: Any,
    *,
    ax: Any = None,
    arm_labels: Optional[Sequence[str]] = None,
    min_label: float = 0.03,
) -> Any:
    """Stacked bars of wins, losses and terminal ties per tier (share of all pairs).

    Parameters
    ----------
    result : WinRatioResult
        A fitted result.
    ax : matplotlib Axes, optional
        Axes to draw on.
    arm_labels : (str, str), optional
        Labels of arm A and arm B (default: the arm values).
    min_label : float, default 0.03
        Segments smaller than this share are not labelled.

    Returns
    -------
    matplotlib Axes
    """

    plt = _plt()
    tiers = result.tiers()
    total = float(result.wins + result.losses + result.ties)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 0.55 * (len(tiers) + 1) + 1.4))
    a_label, b_label = arm_labels or (str(result.config.arm_a), str(result.config.arm_b))
    names = list(tiers["outcome"]) + ["Tied on all tiers"]
    win = list(tiers["wins"] / total) + [0.0]
    loss = list(tiers["losses"] / total) + [0.0]
    stop = list(tiers["terminal_ties"] / total) + [0.0]
    final_ties = (result.ties - tiers["terminal_ties"].sum()) / total
    tie = [0.0] * len(tiers) + [final_ties]
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    segments = [
        (win, COLORS["arm_a"], f"{a_label} wins", None),
        (loss, COLORS["arm_b"], f"{b_label} wins", None),
        (stop, COLORS["terminal"], "Tie stopped by terminal event", "//"),
        (tie, COLORS["tie"], "Tie", None),
    ]
    for values, color, label, hatch in segments:
        values = np.asarray(values)
        if not values.any():
            continue
        ax.barh(
            y, values, left=left, color=color, height=0.62, label=label, edgecolor="white", linewidth=2, hatch=hatch
        )
        for yi, xi, v in zip(y, left, values):
            if v >= min_label:
                ink = "white" if color in (COLORS["arm_a"], COLORS["arm_b"], COLORS["terminal"]) else COLORS["text"]
                ax.text(xi + v / 2, yi, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8.5, color=ink)
        left = left + values
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(1e-9, left.max()) * 1.02)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("Share of all compared pairs", color=COLORS["muted"])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    _style(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=8.5)
    lo, hi = result.ci
    ax.set_title(
        f"Win ratio {result.wr:.2f} [{lo:.2f}, {hi:.2f}]   Net benefit {result.net_benefit:.3f}",
        loc="left",
        fontsize=10,
        color=COLORS["text"],
        pad=40,
    )
    return ax


def plot_forest(
    table: pd.DataFrame,
    *,
    label: Optional[str] = None,
    estimate: Optional[str] = None,
    lower: str = "ci_lower",
    upper: str = "ci_upper",
    reference: float = 1.0,
    log: bool = True,
    ax: Any = None,
    xlabel: str = "Win ratio",
    annotate: bool = True,
) -> Any:
    """Forest plot of estimates and intervals.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per estimate, e.g. from :func:`pywinratio.compare_hierarchies`.
    label : str, optional
        Column with row labels (default: the index).
    estimate, lower, upper : str
        Columns of the point estimate and interval limits (``estimate``
        defaults to ``"estimate"`` or, if absent, ``"wr"``).
    reference : float, default 1.0
        Null value drawn as a vertical line (use 0 for net benefit).
    log : bool, default True
        Log-scale x axis (ratios).
    annotate : bool, default True
        Print "estimate [lower, upper]" to the right of each row.

    Returns
    -------
    matplotlib Axes
    """

    plt = _plt()
    if estimate is None:
        estimate = "estimate" if "estimate" in table.columns else "wr"
    labels = table[label].astype(str).tolist() if label else table.index.astype(str).tolist()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 0.45 * len(labels) + 1.2))
    y = np.arange(len(labels))[::-1]
    est = table[estimate].to_numpy(dtype=float)
    lo = table[lower].to_numpy(dtype=float)
    hi = table[upper].to_numpy(dtype=float)
    finite_hi = np.where(np.isfinite(hi), hi, np.nan)
    ax.hlines(y, lo, finite_hi, color=COLORS["arm_a"], linewidth=2)
    ax.plot(est, y, "o", color=COLORS["arm_a"], markersize=7, markeredgecolor="white", markeredgewidth=1.5)
    ax.axvline(reference, color=COLORS["muted"], linewidth=1, linestyle="--")
    if log:
        from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

        ax.set_xscale("log")
        values = np.concatenate([lo, finite_hi, est, [reference]])
        values = values[np.isfinite(values) & (values > 0)]
        span = (values.min() / 1.1, values.max() * 1.1)
        candidates = np.array([0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4, 5, 10])
        ticks = candidates[(candidates >= span[0]) & (candidates <= span[1])]
        ax.set_xlim(*span)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel, color=COLORS["muted"])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    _style(ax)
    if annotate:
        for yi, e, a, b in zip(y, est, lo, hi):
            upper = "inf" if not np.isfinite(b) else f"{b:.2f}"
            ax.annotate(
                f"{e:.2f} [{a:.2f}, {upper}]",
                xy=(1.0, yi),
                xycoords=("axes fraction", "data"),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=COLORS["text"],
            )
    return ax


def plot_balance(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    threshold: float = 0.1,
    term: str = "term",
    value: str = "abs_smd",
    labels: Sequence[str] = ("Before", "After"),
    ax: Any = None,
) -> Any:
    """Love plot of absolute standardized mean differences before and after adjustment.

    Parameters
    ----------
    before, after : pandas.DataFrame
        Tables from :func:`pywinratio.balance_diagnostics` (e.g.
        ``MatchResult.balance_before`` and ``balance_after``).
    threshold : float, default 0.1
        Conventional balance threshold drawn as a dashed line.

    Returns
    -------
    matplotlib Axes
    """

    plt = _plt()
    merged = before[[term, value]].merge(after[[term, value]], on=term, how="outer", suffixes=("_before", "_after"))
    merged = merged.sort_values(f"{value}_before", na_position="first")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 0.32 * len(merged) + 1.3))
    y = np.arange(len(merged))
    ax.hlines(y, merged[f"{value}_after"], merged[f"{value}_before"], color=COLORS["grid"], linewidth=2)
    ax.plot(merged[f"{value}_before"], y, "o", color=COLORS["arm_b"], label=labels[0], markersize=7)
    ax.plot(merged[f"{value}_after"], y, "o", color=COLORS["arm_a"], label=labels[1], markersize=7)
    ax.axvline(threshold, color=COLORS["muted"], linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(merged[term])
    ax.set_xlabel("Absolute standardized mean difference", color=COLORS["muted"])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    _style(ax)
    ax.legend(frameon=False, loc="lower right")
    return ax


def plot_bootstrap(result: Any, *, ax: Any = None, bins: int = 40) -> Any:
    """Histogram of bootstrap log win ratios with the percentile interval.

    Parameters
    ----------
    result : WinRatioResult
        A result fitted with ``n_boot > 0``.
    ax : matplotlib Axes, optional
        Axes to draw on.
    bins : int, default 40
        Number of histogram bins.

    Returns
    -------
    matplotlib Axes
    """

    plt = _plt()
    if result.bootstrap is None:
        raise ValueError("fit with n_boot > 0 to plot the bootstrap distribution")
    samples = np.asarray(result.bootstrap.get("wr_samples", []), dtype=float)
    finite = samples[np.isfinite(samples) & (samples > 0)]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(np.log(finite), bins=bins, color=COLORS["arm_a"], edgecolor="white", linewidth=1)
    lo, hi = result.bootstrap["ci"]
    for limit in (lo, hi):
        if np.isfinite(limit) and limit > 0:
            ax.axvline(np.log(limit), color=COLORS["text"], linestyle="--", linewidth=1)
    if np.isfinite(result.wr) and result.wr > 0:
        ax.axvline(np.log(result.wr), color=COLORS["arm_b"], linewidth=2)
    ax.axvline(0.0, color=COLORS["muted"], linewidth=1)
    ax.set_xlabel("log win ratio (bootstrap replicates)", color=COLORS["muted"])
    ax.set_ylabel("Replicates", color=COLORS["muted"])
    n_inf = result.bootstrap.get("n_infinite", 0)
    if n_inf:
        ax.set_title(f"{n_inf} replicates with WR = inf not shown", loc="left", fontsize=9, color=COLORS["muted"])
    _style(ax)
    return ax
