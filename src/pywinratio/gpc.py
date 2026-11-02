"""Weighted all-pairs generalized pairwise comparisons (GPC).

Each cross-arm pair receives the product of the two patients' analysis
weights (for example overlap, ATT or ATE propensity-score weights), so the
win, loss and tie *proportions* estimate the corresponding quantities in the
weighted (pseudo-)population. With unit weights this is the ordinary
all-pairs win ratio.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from . import _bootstrap as bs
from .config import WinRatioConfig
from .wr import _blocks, _check_inputs, _check_n_boot, _result_from_summaries, _stratum_labels, _summaries


def _join(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.array([f"{a}|{b}" for a, b in zip(left, right)], dtype=object)


def _weight_series(frame: pd.DataFrame, weight_col: Optional[str]) -> Optional[pd.Series]:
    if weight_col is None:
        return None
    if weight_col not in frame.columns:
        raise ValueError(f"weight column {weight_col!r} not found")
    weights = pd.to_numeric(frame[weight_col], errors="coerce")
    if (~np.isfinite(weights.to_numpy(dtype=float))).any() or (weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    return weights.astype(float)


def compute_weighted_gpc(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    *,
    weight_col: Optional[str] = None,
) -> dict[str, Any]:
    """Weighted all-pairs win ratio, net benefit and win odds.

    Pair weights are the cross-product of patient weights. Comparisons stay
    within strata when ``cfg.strata`` is set.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis data.
    cfg : WinRatioConfig
        All-pairs configuration (``pair_strategy="all_pairs"``).
    weight_col : str, optional
        Nonnegative analysis-weight column; unit weights when omitted.

    Returns
    -------
    dict
        Weighted ``wins``, ``losses``, ``ties``, ``wr``, ``net_benefit``,
        ``win_odds``, ``win_probability`` and the legacy aliases
        ``favorable_pair_probability`` (= ``win_probability``),
        ``win_difference`` (= ``net_benefit``), ``tie_adjusted_odds``
        (= ``win_odds``) and ``information_rate`` (share of pairs resolved),
        plus ``details`` with effective sample sizes by arm.

    Examples
    --------
    >>> import pandas as pd
    >>> from pywinratio import Outcome, WinRatioConfig, compute_weighted_gpc
    >>> df = pd.DataFrame({"arm": ["A", "A", "B", "B"], "event": [0, 1, 1, 1], "w": [1.0, 0.5, 1.0, 1.0]})
    >>> cfg = WinRatioConfig("arm", "A", "B", [Outcome.binary("event")])
    >>> compute_weighted_gpc(df, cfg, weight_col="w")["wins"]
    2.0
    """

    if cfg.pair_strategy != "all_pairs":
        raise ValueError("weighted GPC uses all cross-arm pairs; pair_strategy must be 'all_pairs'")
    frame = _check_inputs(df, cfg, extra_columns=[weight_col] if weight_col else [])
    weights = _weight_series(frame, weight_col)
    summaries = [s for _, s in _summaries(frame, cfg, weights=weights)]
    result = _result_from_summaries(summaries, cfg, integer=False, extra_details={"weight_col": weight_col})
    from .diagnostics import effective_sample_size

    ess_a = [effective_sample_size(s.a_weights) for s in summaries if s.n_a]
    ess_b = [effective_sample_size(s.b_weights) for s in summaries if s.n_b]
    total = float(sum(s.total for s in summaries))
    result.update(
        {
            "favorable_pair_probability": result["win_probability"],
            "win_difference": result["net_benefit"],
            "tie_adjusted_odds": result["win_odds"],
            "information_rate": (result["wins"] + result["losses"]) / total if total > 0 else float("nan"),
        }
    )
    result["details"].update(
        {
            "total_pair_weight": total,
            "ess_a": float(np.nansum(ess_a)),
            "ess_b": float(np.nansum(ess_b)),
            "n_strata": len(summaries),
        }
    )
    return result


def bootstrap_weighted_gpc(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    *,
    weight_col: Optional[str] = None,
    cluster_col: Optional[str] = None,
    within_arm: bool = True,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap a weighted GPC by patients or clusters, holding weights fixed.

    The supplied weights are carried along with the resampled patients; the
    propensity model is **not** refitted. To propagate propensity-score
    estimation uncertainty use
    :func:`pywinratio.bootstrap_propensity_weighted_gpc`.

    Parameters
    ----------
    df, cfg, weight_col
        As in :func:`compute_weighted_gpc`.
    cluster_col : str, optional
        Resample clusters instead of patients.
    within_arm : bool, default True
        Resample within each arm (fixing the number of patients or clusters
        per arm), as appropriate for an arm-level or cluster-level exposure.
        ``False`` resamples from the pooled sample.
    n_boot : int, default 1000
        Number of replicates.
    seed : int, default 42
        Random seed.
    alpha : float, default 0.05
        Two-sided level of the percentile intervals.

    Returns
    -------
    dict
        ``wr``, ``ci``, ``net_benefit_ci``, ``win_odds_ci``, bootstrap
        standard errors, replicate samples and resampling metadata.
    """

    _check_n_boot(n_boot)
    extra = [c for c in (weight_col, cluster_col) if c]
    frame = _check_inputs(df, cfg, extra_columns=extra)
    if not frame.index.is_unique:
        frame = frame.reset_index(drop=True)
    frame = frame.sort_index(kind="mergesort")
    weights = _weight_series(frame, weight_col)
    point = compute_weighted_gpc(frame, cfg, weight_col=weight_col)
    blocks = _blocks(frame, cfg, weights=weights)
    arms = frame[cfg.group_col].astype(str).to_numpy()
    strata = _stratum_labels(frame, cfg).astype(str)
    if cluster_col is not None:
        if frame[cluster_col].isna().any():
            raise ValueError(f"cluster column {cluster_col!r} has missing values")
        clusters = frame[cluster_col].astype(str).to_numpy()
        units = _join(arms, clusters) if within_arm else clusters
        cells = arms if within_arm else np.zeros(len(frame), dtype=int)
    else:
        units = np.arange(len(frame))
        cells = _join(strata, arms) if within_arm else strata
    design = bs.design_from_labels(units, cells)
    rng = np.random.default_rng(seed)
    wins, losses, totals = bs.replicate_totals(blocks, design, n_boot, rng)
    replicate = bs.summarize_replicates(wins, losses, totals, alpha=alpha)
    kind = ("cluster" if cluster_col else "subject") + ("_within_arm" if within_arm else "")
    return bs.bootstrap_payload(
        point["wr"],
        replicate,
        n_boot=n_boot,
        seed=seed,
        bootstrap_type=f"weighted_{kind}",
        extra={"cluster_col": cluster_col, "within_arm": bool(within_arm), "weight_col": weight_col},
    )
