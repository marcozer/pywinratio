"""Win-ratio estimators: all pairs, stratified, assigned pairs, multi-arm.

These functions return plain dictionaries and are the stable low-level API.
Most analyses are easier through :class:`pywinratio.WinRatio`, which wraps
them and returns a :class:`pywinratio.WinRatioResult`.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import _bootstrap as bs
from ._engine import (
    LEVEL_TIE,
    MAX_CELLS,
    PairSummary,
    extract_tiers,
    pair_result_matrices,
    required_columns,
    resolve_matched,
    summarize_all_pairs,
    validate_outcome_data,
)
from .config import MultiArmWinRatioConfig, WinRatioConfig, WinRatioOutcome
from .inference import adjust_pvalues_holm, win_statistics

__all__ = [
    "add_pvalues_to_multi_arm_results",
    "adjust_pvalues_holm",
    "bootstrap_win_ratio",
    "bootstrap_win_ratio_cluster",
    "bootstrap_win_ratio_cluster_within_arm",
    "bootstrap_win_ratio_matched",
    "bootstrap_win_ratio_multi_arm",
    "compare_patients",
    "compute_pvalue_from_bootstrap",
    "compute_win_ratio",
    "compute_win_ratio_all_pairs",
    "compute_win_ratio_matched",
    "compute_win_ratio_multi_arm",
    "pair_results",
]


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _check_inputs(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Validate a two-arm analysis and return the rows of the two arms."""

    if not cfg.outcomes:
        raise ValueError("No outcomes configured for win ratio.")
    needed = [cfg.group_col, *required_columns(cfg.outcomes), *(cfg.strata or []), *extra_columns]
    if cfg.pair_strategy == "matched" and cfg.pair_id:
        needed.append(cfg.pair_id)
    missing_columns = sorted({column for column in needed if column not in df.columns})
    if missing_columns:
        raise ValueError(f"required columns not found: {missing_columns}")
    observed_arms = set(df[cfg.group_col].dropna().unique())
    if cfg.arm_a not in observed_arms or cfg.arm_b not in observed_arms:
        raise ValueError(
            f"both configured arms must be present in {cfg.group_col!r}: "
            f"arm_a={cfg.arm_a!r}, arm_b={cfg.arm_b!r}, observed={sorted(map(str, observed_arms))}"
        )
    frame = df[df[cfg.group_col].isin([cfg.arm_a, cfg.arm_b])]
    if not frame.index.is_unique:
        # row labels identify patients in variance and cluster computations
        frame = frame.reset_index(drop=True)
    validate_outcome_data(frame, cfg.outcomes)
    if cfg.strata and frame[cfg.strata].isna().any().any():
        raise ValueError("strata columns contain missing values; drop or recode them before the analysis")
    return frame


def _strata_groups(frame: pd.DataFrame, cfg: WinRatioConfig) -> List[Tuple[Optional[tuple], pd.DataFrame]]:
    if not cfg.strata:
        return [(None, frame)]
    groups = []
    for key, sub in frame.groupby(cfg.strata, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        groups.append((tuple(key), sub))
    return groups


def _arms(frame: pd.DataFrame, cfg: WinRatioConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return frame[frame[cfg.group_col] == cfg.arm_a], frame[frame[cfg.group_col] == cfg.arm_b]


def _summaries(
    frame: pd.DataFrame,
    cfg: WinRatioConfig,
    *,
    weights: Optional[pd.Series] = None,
    max_cells: int = MAX_CELLS,
) -> List[Tuple[Optional[tuple], PairSummary]]:
    out = []
    for key, sub in _strata_groups(frame, cfg):
        a, b = _arms(sub, cfg)
        w_a = None if weights is None else weights.loc[a.index].to_numpy(dtype=float)
        w_b = None if weights is None else weights.loc[b.index].to_numpy(dtype=float)
        out.append((key, summarize_all_pairs(a, b, cfg.outcomes, a_weights=w_a, b_weights=w_b, max_cells=max_cells)))
    return out


def _wr(wins: float, losses: float) -> float:
    if losses > 0:
        return wins / losses
    if wins > 0:
        return float(np.inf)
    return float("nan")


def _as_number(value: float, integer: bool) -> float | int:
    return int(round(value)) if integer else float(value)


def _result_from_summaries(
    summaries: Sequence[PairSummary],
    cfg: WinRatioConfig,
    *,
    integer: bool = True,
    extra_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    k = len(cfg.outcomes)
    wins = float(sum(s.wins for s in summaries))
    losses = float(sum(s.losses for s in summaries))
    total = float(sum(s.total for s in summaries))
    tier_wins = np.sum([s.tier_wins for s in summaries], axis=0) if summaries else np.zeros(k)
    tier_losses = np.sum([s.tier_losses for s in summaries], axis=0) if summaries else np.zeros(k)
    tier_stops = np.sum([s.tier_stops for s in summaries], axis=0) if summaries else np.zeros(k)
    ties = total - wins - losses
    stats = win_statistics(wins, losses, total)
    details: Dict[str, Any] = {
        "n_a": int(sum(s.n_a for s in summaries)),
        "n_b": int(sum(s.n_b for s in summaries)),
        "n_pairs": _as_number(total, integer),
        "group_col": cfg.group_col,
        "arm_a": cfg.arm_a,
        "arm_b": cfg.arm_b,
        "pair_strategy": cfg.pair_strategy,
        "tier_names": [outcome.name for outcome in cfg.outcomes],
        "tier_wins": [_as_number(v, integer) for v in tier_wins],
        "tier_losses": [_as_number(v, integer) for v in tier_losses],
        "tier_terminal_ties": [_as_number(v, integer) for v in tier_stops],
        "tier_ties": _as_number(ties, integer),
    }
    if extra_details:
        details.update(extra_details)
    result = {
        "wins": _as_number(wins, integer),
        "losses": _as_number(losses, integer),
        "ties": _as_number(ties, integer),
        **stats,
        "details": details,
    }
    result["wr"] = float(stats["wr"]) if np.isfinite(stats["wr"]) else stats["wr"]
    return result


# ---------------------------------------------------------------------------
# All pairs
# ---------------------------------------------------------------------------


def compute_win_ratio_all_pairs(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    strata_values: Optional[Tuple] = None,
    chunk_size: int = 512,
) -> Dict[str, Any]:
    """Unstratified all-pairs win ratio (optionally within one stratum).

    Every arm-A patient is compared with every arm-B patient on the
    hierarchy. Large comparisons are evaluated in vectorized row blocks.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis data, one row per patient.
    cfg : WinRatioConfig
        Analysis configuration.
    strata_values : tuple, optional
        Restrict to the stratum with these values of ``cfg.strata``.
    chunk_size : int
        Unused; kept for backward compatibility (blocking is automatic).

    Returns
    -------
    dict
        ``wins``, ``losses``, ``ties``, ``wr``, ``net_benefit``,
        ``win_odds`` and other proportions from
        :func:`pywinratio.win_statistics`, plus ``details`` with
        arm sizes and tier counts (``tier_wins``, ``tier_losses``,
        ``tier_terminal_ties``).
    """

    del chunk_size
    if not cfg.outcomes:
        raise ValueError("No outcomes configured for win ratio.")
    subset = df
    if strata_values and cfg.strata:
        for col, val in zip(cfg.strata, strata_values):
            subset = subset[subset[col] == val]
    subset = subset[subset[cfg.group_col].isin([cfg.arm_a, cfg.arm_b])]
    validate_outcome_data(subset, cfg.outcomes)
    a, b = _arms(subset, cfg)
    summary = summarize_all_pairs(a, b, cfg.outcomes)
    return _result_from_summaries([summary], cfg, extra_details={"strata": strata_values})


def compute_win_ratio(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
) -> Dict[str, Any]:
    """Win ratio for an all-pairs, stratified, or assigned-pair design.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis data, one row per patient.
    cfg : WinRatioConfig
        With ``pair_strategy="matched"`` and ``pair_id`` only patients of the
        same assigned pair are compared. With ``strata``, patients are only
        compared within strata and the stratum totals are summed.

    Returns
    -------
    dict
        ``{"overall": ...}``; stratified analyses also return
        ``"by_strata"`` (one result per stratum) and
        ``"overall_unstratified"`` (all cross-arm pairs, for reference).

    Examples
    --------
    >>> import pandas as pd
    >>> from pywinratio import Outcome, WinRatioConfig, compute_win_ratio
    >>> df = pd.DataFrame({"arm": ["A", "A", "B", "B"], "death": [0, 0, 1, 0]})
    >>> cfg = WinRatioConfig("arm", "A", "B", [Outcome.binary("death")])
    >>> compute_win_ratio(df, cfg)["overall"]["wins"]
    2
    """

    frame = _check_inputs(df, cfg)
    if cfg.pair_strategy == "matched" and cfg.pair_id:
        return {"overall": compute_win_ratio_matched(frame, cfg)}

    summaries = _summaries(frame, cfg)
    if not cfg.strata:
        return {"overall": _result_from_summaries([summaries[0][1]], cfg, extra_details={"strata": None})}

    per = []
    for key, summary in summaries:
        res = _result_from_summaries([summary], cfg, extra_details={"strata": key})
        res["strata"] = dict(zip(cfg.strata, key))
        per.append(res)
    overall = _result_from_summaries(
        [s for _, s in summaries],
        cfg,
        extra_details={"strata": "stratified", "strata_cols": cfg.strata, "n_strata": len(summaries)},
    )
    unstratified_cfg = WinRatioConfig(cfg.group_col, cfg.arm_a, cfg.arm_b, cfg.outcomes)
    overall_unstratified = _result_from_summaries(_summaries(frame, unstratified_cfg)[0][1:], cfg)
    return {"overall": overall, "overall_unstratified": overall_unstratified, "by_strata": per}


# ---------------------------------------------------------------------------
# Assigned pairs
# ---------------------------------------------------------------------------


def _aligned_pairs(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return aligned arm-A and arm-B rows of valid pairs and the number excluded."""

    pid = cfg.pair_id
    if not pid:
        raise ValueError("pair_id must be set for matched strategy")
    frame = df[df[cfg.group_col].isin([cfg.arm_a, cfg.arm_b]) & df[pid].notna()]
    n_without_id = int((df[cfg.group_col].isin([cfg.arm_a, cfg.arm_b]) & df[pid].isna()).sum())
    if n_without_id:
        warnings.warn(f"{n_without_id} row(s) without a pair identifier were excluded", UserWarning, stacklevel=3)
    is_a = frame[cfg.group_col] == cfg.arm_a
    counts = pd.DataFrame({"pair": frame[pid], "a": is_a.astype(int), "b": (~is_a).astype(int)})
    per_pair = counts.groupby("pair", sort=True)[["a", "b"]].sum()
    valid = per_pair.index[(per_pair["a"] == 1) & (per_pair["b"] == 1)]
    n_excluded = int(per_pair.shape[0] - valid.size)
    keep = frame[pid].isin(valid)
    a = frame[keep & is_a].set_index(pid, drop=False).loc[valid]
    b = frame[keep & ~is_a].set_index(pid, drop=False).loc[valid]
    if n_excluded:
        warnings.warn(
            f"{n_excluded} pair(s) without exactly one patient per arm were excluded",
            UserWarning,
            stacklevel=3,
        )
    return a, b, n_excluded


def _matched_results(
    df: pd.DataFrame, cfg: WinRatioConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    a, b, n_excluded = _aligned_pairs(df, cfg)
    if len(a) == 0:
        return a, b, np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.int16), n_excluded
    result, level = resolve_matched(extract_tiers(a, cfg.outcomes), extract_tiers(b, cfg.outcomes), cfg.outcomes)
    return a, b, result, level, n_excluded


def compute_win_ratio_matched(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
) -> Dict[str, Any]:
    """Win ratio of assigned (matched) pairs identified by ``cfg.pair_id``.

    Each pair must contain exactly one patient of each arm; other pairs are
    excluded with a warning and counted in ``details["pairs_excluded"]``.

    Parameters
    ----------
    df : pandas.DataFrame
        Matched data, two rows per pair.
    cfg : WinRatioConfig
        Configuration with ``pair_id`` set.

    Returns
    -------
    dict
        Same keys as :func:`compute_win_ratio_all_pairs`; ``details`` also has
        ``pairs`` (number of analyzed pairs) and ``pairs_excluded``.
    """

    validate_outcome_data(df[df[cfg.group_col].isin([cfg.arm_a, cfg.arm_b])], cfg.outcomes)
    a, b, result, level, n_excluded = _matched_results(df, cfg)
    k = len(cfg.outcomes)
    summary = PairSummary(
        n_tiers=k,
        a_labels=a.index.to_numpy(),
        b_labels=b.index.to_numpy(),
        a_weights=np.ones(len(a)),
        b_weights=np.ones(len(b)),
    )
    summary.wins = float((result == 1).sum())
    summary.losses = float((result == -1).sum())
    summary.total = float(result.size)
    summary.tier_wins = np.array([float(((result == 1) & (level == i)).sum()) for i in range(k)])
    summary.tier_losses = np.array([float(((result == -1) & (level == i)).sum()) for i in range(k)])
    summary.tier_stops = np.array([float((level == -(i + 2)).sum()) for i in range(k)])
    n_a = int((df[cfg.group_col] == cfg.arm_a).sum())
    n_b = int((df[cfg.group_col] == cfg.arm_b).sum())
    out = _result_from_summaries(
        [summary],
        cfg,
        extra_details={"pairs": int(result.size), "pairs_excluded": n_excluded, "pair_id": cfg.pair_id},
    )
    out["details"]["n_a"] = n_a
    out["details"]["n_b"] = n_b
    return out


def pair_results(df: pd.DataFrame, cfg: WinRatioConfig) -> pd.DataFrame:
    """Per-pair outcome of an assigned-pair analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Matched data, two rows per pair.
    cfg : WinRatioConfig
        Configuration with ``pair_strategy="matched"`` and ``pair_id``.

    Returns
    -------
    pandas.DataFrame
        One row per analyzed pair with the pair identifier, ``result``
        (``"win"``, ``"loss"`` or ``"tie"`` for arm A), ``deciding_tier``
        (name of the tier that resolved the pair) and ``stopped_by``
        (terminal tier that stopped a tied comparison, if any).
    """

    if cfg.pair_strategy != "matched" or not cfg.pair_id:
        raise ValueError("pair_results requires pair_strategy='matched' and pair_id")
    _check_inputs(df, cfg)
    a, _, result, level, _ = _matched_results(df, cfg)
    names = [outcome.name for outcome in cfg.outcomes]
    labels = np.array(["tie", "win", "loss"], dtype=object)[np.where(result == -1, 2, result)]
    deciding = [names[lv] if lv >= 0 else None for lv in level]
    stopped = [names[-lv - 2] if lv < LEVEL_TIE else None for lv in level]
    return pd.DataFrame(
        {
            cfg.pair_id: a.index.to_numpy(),
            "result": labels,
            "deciding_tier": deciding,
            "stopped_by": stopped,
        }
    )


def compare_patients(
    patient_a: Mapping[str, Any],
    patient_b: Mapping[str, Any],
    outcomes: Sequence[WinRatioOutcome],
) -> Dict[str, Any]:
    """Explain the comparison of two patients on a hierarchy.

    Parameters
    ----------
    patient_a, patient_b : mapping
        Outcome values by column name (e.g. a dict or a DataFrame row).
    outcomes : sequence of WinRatioOutcome
        The hierarchy.

    Returns
    -------
    dict
        ``result`` (``"win"``, ``"loss"`` or ``"tie"`` for patient A),
        ``deciding_tier`` and ``stopped_by``.

    Examples
    --------
    >>> from pywinratio import Outcome, compare_patients
    >>> tiers = [Outcome.binary("death", terminal=True), Outcome.continuous("los", margin=1)]
    >>> compare_patients({"death": 0, "los": 12}, {"death": 0, "los": 6}, tiers)["result"]
    'loss'
    """

    columns = required_columns(outcomes)
    a = pd.DataFrame([{c: patient_a.get(c, np.nan) for c in columns}])
    b = pd.DataFrame([{c: patient_b.get(c, np.nan) for c in columns}])
    result, level = resolve_matched(extract_tiers(a, outcomes), extract_tiers(b, outcomes), outcomes)
    names = [outcome.name for outcome in outcomes]
    lv = int(level[0])
    return {
        "result": {1: "win", -1: "loss", 0: "tie"}[int(result[0])],
        "deciding_tier": names[lv] if lv >= 0 else None,
        "stopped_by": names[-lv - 2] if lv < LEVEL_TIE else None,
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _blocks(frame: pd.DataFrame, cfg: WinRatioConfig, weights: Optional[pd.Series] = None) -> List[bs.Block]:
    position = pd.Series(np.arange(len(frame)), index=frame.index)
    blocks = []
    for _, sub in _strata_groups(frame, cfg):
        a, b = _arms(sub, cfg)
        if a.empty or b.empty:
            continue
        wins, losses = pair_result_matrices(a, b, cfg.outcomes)
        blocks.append(
            bs.Block(
                wins=wins.astype(float),
                losses=losses.astype(float),
                a_pos=position.loc[a.index].to_numpy(),
                b_pos=position.loc[b.index].to_numpy(),
                a_weights=np.ones(len(a)) if weights is None else weights.loc[a.index].to_numpy(dtype=float),
                b_weights=np.ones(len(b)) if weights is None else weights.loc[b.index].to_numpy(dtype=float),
            )
        )
    if not blocks:
        raise ValueError("One or both arms are empty after filtering by group_col and arms.")
    return blocks


def _stratum_labels(frame: pd.DataFrame, cfg: WinRatioConfig) -> np.ndarray:
    if not cfg.strata:
        return np.zeros(len(frame), dtype=int)
    return frame[cfg.strata].astype(str).agg("|".join, axis=1).to_numpy()


def _check_unique_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Unique, sorted row labels so that resampling does not depend on row order."""

    if not frame.index.is_unique:
        return frame.reset_index(drop=True)
    return frame.sort_index(kind="mergesort")


def bootstrap_win_ratio(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Patient-level bootstrap of an all-pairs (optionally stratified) WR.

    Patients are resampled with replacement within each arm (and stratum),
    preserving arm and stratum sizes. Percentile intervals are returned for
    the win ratio (``ci``), net benefit and win odds.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis data.
    cfg : WinRatioConfig
        All-pairs configuration.
    n_boot : int, default 1000
        Number of bootstrap replicates.
    seed : int, default 42
        Seed of the random generator.
    alpha : float, default 0.05
        Two-sided level of the percentile intervals.

    Returns
    -------
    dict
        ``wr`` (point estimate), ``ci``, ``net_benefit_ci``,
        ``win_odds_ci``, bootstrap standard errors, ``wr_samples`` (defined
        replicates, possibly ``inf``), ``n_infinite``, ``n_undefined``,
        ``n_boot``, ``seed`` and ``bootstrap_type``.
    """

    _check_n_boot(n_boot)
    frame = _check_unique_index(_check_inputs(df, cfg))
    point = compute_win_ratio(frame, cfg)["overall"]["wr"]
    blocks = _blocks(frame, cfg)
    cells = [f"{s}|{arm}" for s, arm in zip(_stratum_labels(frame, cfg), frame[cfg.group_col].astype(str))]
    design = bs.design_from_labels(np.arange(len(frame)), cells)
    rng = np.random.default_rng(seed)
    wins, losses, totals = bs.replicate_totals(blocks, design, n_boot, rng)
    replicate = bs.summarize_replicates(wins, losses, totals, alpha=alpha)
    return bs.bootstrap_payload(
        point,
        replicate,
        n_boot=n_boot,
        seed=seed,
        bootstrap_type="subject_within_strata" if cfg.strata else "subject",
    )


def _cluster_bootstrap(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    cluster_col: str,
    n_boot: int,
    seed: int,
    alpha: float,
    within_arm: bool,
    weights: Optional[pd.Series] = None,
    point: Optional[float] = None,
) -> Dict[str, Any]:
    _check_n_boot(n_boot)
    if cluster_col not in df.columns:
        raise ValueError(f"cluster_col {cluster_col!r} not found in dataframe")
    frame = _check_unique_index(_check_inputs(df, cfg, extra_columns=[cluster_col]))
    if frame[cluster_col].isna().any():
        raise ValueError(f"cluster column {cluster_col!r} has missing values")
    if point is None:
        point = compute_win_ratio(frame, cfg)["overall"]["wr"]
    blocks = _blocks(frame, cfg, weights=None if weights is None else weights.loc[frame.index])
    clusters = frame[cluster_col].astype(str).to_numpy()
    arms = frame[cfg.group_col].astype(str).to_numpy()
    if within_arm:
        units = np.array([f"{arm}|{cluster}" for arm, cluster in zip(arms, clusters)], dtype=object)
        cells = arms
    else:
        units = clusters
        cells = np.zeros(len(frame), dtype=int)
    design = bs.design_from_labels(units, cells)
    rng = np.random.default_rng(seed)
    wins, losses, totals = bs.replicate_totals(blocks, design, n_boot, rng)
    replicate = bs.summarize_replicates(wins, losses, totals, alpha=alpha)
    extra: Dict[str, Any] = {"cluster_col": cluster_col, "stratified_analysis": bool(cfg.strata)}
    if within_arm:
        extra["n_clusters_a"] = int(np.unique(clusters[arms == str(cfg.arm_a)]).size)
        extra["n_clusters_b"] = int(np.unique(clusters[arms == str(cfg.arm_b)]).size)
    else:
        extra["n_clusters"] = int(np.unique(clusters).size)
    return bs.bootstrap_payload(
        point,
        replicate,
        n_boot=n_boot,
        seed=seed,
        bootstrap_type="cluster_within_arm" if within_arm else "cluster",
        extra=extra,
    )


def bootstrap_win_ratio_cluster(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    cluster_col: str,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Cluster (e.g. hospital) bootstrap of an all-pairs WR.

    Whole clusters are resampled with replacement, keeping all their patients
    (of both arms and all strata) with the cluster's multiplicity. Use it when
    outcomes are correlated within clusters and the exposure varies within
    clusters. For a cluster-level exposure prefer
    :func:`bootstrap_win_ratio_cluster_within_arm`.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis data.
    cfg : WinRatioConfig
        All-pairs configuration (optionally stratified).
    cluster_col : str
        Cluster column (no missing values).
    n_boot : int, default 1000
        Number of replicates.
    seed : int, default 42
        Random seed.
    alpha : float, default 0.05
        Two-sided level of the percentile intervals.

    Returns
    -------
    dict
        As :func:`bootstrap_win_ratio`, plus ``cluster_col`` and ``n_clusters``.
    """

    if cfg.pair_strategy != "all_pairs":
        raise ValueError("bootstrap_win_ratio_cluster supports all_pairs only")
    return _cluster_bootstrap(df, cfg, cluster_col, n_boot, seed, alpha, within_arm=False)


def bootstrap_win_ratio_cluster_within_arm(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    cluster_col: str,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Cluster bootstrap resampling clusters separately within each arm.

    The number of clusters per arm is fixed across replicates, which suits a
    cluster-level exposure such as hospital volume. A cluster contributing
    patients to both arms is treated as two arm-specific units.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis data.
    cfg : WinRatioConfig
        All-pairs configuration (optionally stratified).
    cluster_col : str
        Cluster column (no missing values).
    n_boot : int, default 1000
        Number of replicates.
    seed : int, default 42
        Random seed.
    alpha : float, default 0.05
        Two-sided level of the percentile intervals.

    Returns
    -------
    dict
        As :func:`bootstrap_win_ratio`, plus ``cluster_col``, ``n_clusters_a``
        and ``n_clusters_b``.
    """

    if cfg.pair_strategy != "all_pairs":
        raise ValueError("bootstrap_win_ratio_cluster_within_arm supports all_pairs only")
    return _cluster_bootstrap(df, cfg, cluster_col, n_boot, seed, alpha, within_arm=True)


def bootstrap_win_ratio_matched(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    cluster_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Pair bootstrap of an assigned-pair WR.

    Assigned pairs are resampled with replacement, so the match itself is held
    fixed. ``cluster_col`` resamples whole clusters of pairs instead, using the
    cluster of the arm-A patient of each pair.

    Parameters
    ----------
    df : pandas.DataFrame
        Matched data, two rows per pair.
    cfg : WinRatioConfig
        Configuration with ``pair_strategy="matched"`` and ``pair_id``.
    n_boot : int, default 1000
        Number of replicates.
    seed : int, default 42
        Random seed.
    alpha : float, default 0.05
        Two-sided level of the percentile intervals.
    cluster_col : str, optional
        Resample clusters of pairs.

    Returns
    -------
    dict
        As :func:`bootstrap_win_ratio`.
    """

    _check_n_boot(n_boot)
    if cfg.pair_strategy != "matched" or not cfg.pair_id:
        raise ValueError("bootstrap_win_ratio_matched requires matched strategy and pair_id")
    frame = _check_inputs(df, cfg, extra_columns=[cluster_col] if cluster_col else [])
    a, _, result, _, _ = _matched_results(frame, cfg)
    point = compute_win_ratio_matched(frame, cfg)["wr"]
    if result.size == 0:
        return {"wr": np.nan, "ci": (np.nan, np.nan), "n_boot": n_boot, "wr_samples": []}
    cells = None if cluster_col is None else a[cluster_col].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    wins, losses, totals = bs.pair_replicate_totals(result, cells, n_boot, rng)
    replicate = bs.summarize_replicates(wins, losses, totals, alpha=alpha)
    return bs.bootstrap_payload(
        point,
        replicate,
        n_boot=n_boot,
        seed=seed,
        bootstrap_type="pair_cluster" if cluster_col else "pair",
        extra={"cluster_col": cluster_col} if cluster_col else None,
    )


def _check_n_boot(n_boot: int) -> None:
    if int(n_boot) <= 0:
        raise ValueError("n_boot must be positive")


# ---------------------------------------------------------------------------
# Multi-arm
# ---------------------------------------------------------------------------


def _multi_arm_to_pairwise_config(
    multi_cfg: MultiArmWinRatioConfig,
    arm_a: Any,
    arm_b: Any,
) -> WinRatioConfig:
    return WinRatioConfig(
        group_col=multi_cfg.group_col,
        arm_a=arm_a,
        arm_b=arm_b,
        outcomes=multi_cfg.outcomes,
        strata=multi_cfg.strata,
        id_col=multi_cfg.id_col,
        pair_strategy=multi_cfg.pair_strategy,
        pair_id=multi_cfg.pair_id,
    )


def _comparison_pairs(cfg: MultiArmWinRatioConfig) -> List[Tuple[Any, Any]]:
    if cfg.comparison_pairs:
        return [tuple(pair) for pair in cfg.comparison_pairs]  # type: ignore[misc]
    if cfg.reference_arm is not None:
        return [(arm, cfg.reference_arm) for arm in cfg.arms if arm != cfg.reference_arm]
    return list(combinations(cfg.arms, 2))


def compute_win_ratio_multi_arm(
    df: pd.DataFrame,
    cfg: MultiArmWinRatioConfig,
) -> Dict[str, Any]:
    """Pairwise win ratios across several arms.

    Comparisons are ``cfg.comparison_pairs`` if given, otherwise each arm
    versus ``cfg.reference_arm`` if given, otherwise all pairwise
    combinations. :func:`pywinratio.pairwise_win_ratios` offers the same with
    intervals and multiplicity adjustment in a data frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis data.
    cfg : MultiArmWinRatioConfig
        Multi-arm configuration.

    Returns
    -------
    dict
        ``comparisons`` (one result dict per comparison, with arm labels) and
        ``summary``.
    """

    results = []
    for arm_a, arm_b in _comparison_pairs(cfg):
        pairwise_cfg = _multi_arm_to_pairwise_config(cfg, arm_a, arm_b)
        wr_result = compute_win_ratio(df, pairwise_cfg)
        results.append(
            {
                "arm_a": arm_a,
                "arm_b": arm_b,
                "arm_a_label": cfg.arm_labels.get(arm_a, arm_a),
                "arm_b_label": cfg.arm_labels.get(arm_b, arm_b),
                **wr_result["overall"],
            }
        )
    return {
        "comparisons": results,
        "summary": {
            "n_arms": len(cfg.arms),
            "arms": cfg.arms,
            "arm_labels": cfg.arm_labels,
            "n_comparisons": len(results),
            "reference_arm": cfg.reference_arm,
        },
    }


def bootstrap_win_ratio_multi_arm(
    df: pd.DataFrame,
    cfg: MultiArmWinRatioConfig,
    n_boot: int = 2000,
    cluster_col: Optional[str] = None,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Bootstrap intervals for every pairwise comparison of a multi-arm design.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis data.
    cfg : MultiArmWinRatioConfig
        Multi-arm configuration.
    n_boot : int, default 2000
        Replicates per comparison.
    cluster_col : str, optional
        Cluster bootstrap (:func:`bootstrap_win_ratio_cluster`) instead of the
        patient (or pair) bootstrap.
    seed : int, default 42
        Seed; comparison ``i`` uses ``seed + i``.
    alpha : float, default 0.05
        Two-sided level.

    Returns
    -------
    dict
        ``comparisons`` (``wr``, ``ci_lower``, ``ci_upper``, ``wr_samples`` and
        arm labels for each comparison) and ``summary``.
    """

    results = []
    for i, (arm_a, arm_b) in enumerate(_comparison_pairs(cfg)):
        pairwise_cfg = _multi_arm_to_pairwise_config(cfg, arm_a, arm_b)
        pair_seed = seed + i
        if cluster_col:
            boot = bootstrap_win_ratio_cluster(df, pairwise_cfg, cluster_col, n_boot, pair_seed, alpha)
        elif pairwise_cfg.pair_strategy == "matched" and pairwise_cfg.pair_id:
            boot = bootstrap_win_ratio_matched(df, pairwise_cfg, n_boot, pair_seed, alpha)
        else:
            boot = bootstrap_win_ratio(df, pairwise_cfg, n_boot, pair_seed, alpha)
        results.append(
            {
                "arm_a": arm_a,
                "arm_b": arm_b,
                "arm_a_label": cfg.arm_labels.get(arm_a, arm_a),
                "arm_b_label": cfg.arm_labels.get(arm_b, arm_b),
                "wr": boot["wr"],
                "ci_lower": boot["ci"][0],
                "ci_upper": boot["ci"][1],
                "n_boot": boot["n_boot"],
                "wr_samples": boot.get("wr_samples", []),
            }
        )
    return {
        "comparisons": results,
        "summary": {
            "n_arms": len(cfg.arms),
            "arms": cfg.arms,
            "n_comparisons": len(results),
            "n_boot": n_boot,
            "cluster_col": cluster_col,
            "ci_alpha": alpha,
        },
    }


def compute_pvalue_from_bootstrap(
    wr: float,
    ci_lower: float,
    ci_upper: float,
    null_wr: float = 1.0,
    alpha: float = 0.05,
) -> float:
    """Approximate p-value backed out of a symmetric-on-log-scale interval.

    Assumes the ``1 - alpha`` interval is ``log WR +/- z * SE``. Prefer the
    analytic p-values of :class:`pywinratio.WinRatioResult` or a permutation
    test when available.

    Parameters
    ----------
    wr : float
        Win-ratio estimate.
    ci_lower, ci_upper : float
        Confidence limits.
    null_wr : float, default 1.0
        Null value.
    alpha : float, default 0.05
        Level of the interval.

    Returns
    -------
    float
        Two-sided p-value (NaN for non-finite or non-positive inputs).
    """

    if not np.isfinite(wr) or wr <= 0:
        return np.nan
    if not np.isfinite(ci_lower) or not np.isfinite(ci_upper) or ci_lower <= 0 or ci_upper <= 0:
        return np.nan
    from scipy.stats import norm

    z_crit = float(norm.ppf(1 - alpha / 2))
    log_se = (np.log(ci_upper) - np.log(ci_lower)) / (2 * z_crit)
    if log_se <= 0:
        return np.nan
    z = abs(np.log(wr) - np.log(null_wr)) / log_se
    return float(2 * norm.sf(z))


def add_pvalues_to_multi_arm_results(
    results: Dict[str, Any],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Add CI-derived and Holm-adjusted p-values to multi-arm bootstrap results.

    Parameters
    ----------
    results : dict
        Output of :func:`pywinratio.bootstrap_win_ratio_multi_arm`; modified in place.
    alpha : float, default 0.05
        Family-wise error rate for the ``significant_holm`` flags.

    Returns
    -------
    dict
        ``results`` with ``p_value``, ``p_value_holm`` and ``significant_holm``
        added to every comparison and counts of significant comparisons in
        ``summary``.
    """

    comparisons = results.get("comparisons", [])
    ci_alpha = results.get("summary", {}).get("ci_alpha", 0.05)
    p_values = []
    for comp in comparisons:
        p = compute_pvalue_from_bootstrap(
            comp.get("wr", np.nan),
            comp.get("ci_lower", np.nan),
            comp.get("ci_upper", np.nan),
            alpha=ci_alpha,
        )
        comp["p_value"] = p
        p_values.append(p)
    adjusted_p, significant = adjust_pvalues_holm(p_values, alpha)
    for comp, adj_p, sig in zip(comparisons, adjusted_p, significant):
        comp["p_value_holm"] = adj_p
        comp["significant_holm"] = sig
    results["summary"]["alpha"] = alpha
    results["summary"]["n_significant_nominal"] = sum(1 for p in p_values if np.isfinite(p) and p < alpha)
    results["summary"]["n_significant_holm"] = sum(significant)
    return results
