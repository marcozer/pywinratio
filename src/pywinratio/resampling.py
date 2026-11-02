"""Bootstraps that re-estimate the design in every replicate.

For a cluster-level exposure (e.g. hospital volume), the uncertainty of an
adjusted win ratio comes from the clusters, from the propensity model and
from the matching itself. These functions resample clusters within arm,
refit the propensity score and rematch (or reweight) in every replicate.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from ._bootstrap import summarize_replicates
from .config import WinRatioConfig
from .gpc import compute_weighted_gpc
from .match import MatchResult, propensity_match
from .weights import estimate_propensity_weights
from .wr import compute_win_ratio


def _resample_clusters_within_arm(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    cluster_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pieces = []
    for arm in (cfg.arm_a, cfg.arm_b):
        arm_data = df[df[cfg.group_col] == arm]
        clusters = np.sort(arm_data[cluster_col].dropna().unique().astype(str))
        if len(clusters) == 0:
            continue
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        by_cluster = {key: frame for key, frame in arm_data.groupby(arm_data[cluster_col].astype(str))}
        for draw, cluster in enumerate(sampled):
            piece = by_cluster[cluster].copy()
            piece[cluster_col] = f"BOOT_{arm}_{draw:04d}"
            pieces.append(piece)
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def _percentile_interval(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    from ._bootstrap import percentile_interval

    return percentile_interval(np.asarray(samples, dtype=float), alpha)


def _run_replicates(fit_one, n_boot: int, seed: int, n_jobs: int) -> list:
    """Run ``fit_one(iteration, child_seed)`` for every replicate, optionally in parallel."""

    children = np.random.SeedSequence(seed).spawn(n_boot)
    if n_jobs == 1:
        return [fit_one(i, child) for i, child in enumerate(children)]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=n_jobs)(delayed(fit_one)(i, child) for i, child in enumerate(children))


def bootstrap_propensity_matched_win_ratio(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    *,
    covariates: Iterable[str],
    cluster_col: str,
    exact_cols: Optional[Iterable[str]] = None,
    caliper: Optional[float] = None,
    method: str = "optimal",
    trim_common_support: bool = True,
    missing: str = "drop",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    n_jobs: int = 1,
    penalty: Optional[float] = None,
) -> dict[str, Any]:
    """Resample clusters, refit the score, rematch and recompute the matched WR.

    Clusters are resampled separately within each arm, which is appropriate
    for a cluster-level exposure. Each replicate refits the propensity model
    and rematches, so the interval reflects cluster sampling, score
    estimation and matching. Failed replicates (e.g. one arm without
    matches) are counted in ``n_failed`` and not silently replaced.

    Parameters
    ----------
    df : pandas.DataFrame
        Full cohort.
    cfg : WinRatioConfig
        Hierarchy and arms; ``arm_a`` is the treated value.
    covariates, exact_cols, caliper, method, trim_common_support, missing, penalty
        Passed to :func:`pywinratio.propensity_match`.
    cluster_col : str
        Cluster (e.g. hospital) column.
    n_boot : int, default 1000
        Number of replicates.
    seed : int, default 42
        Seed; replicates use independent child seeds.
    alpha : float, default 0.05
        Two-sided level.
    n_jobs : int, default 1
        Parallel workers (``-1`` = all cores).

    Returns
    -------
    dict
        ``wr``, ``ci`` (percentile, infinite replicates kept), ``net_benefit_ci``,
        ``win_odds_ci``, ``n_successful``, ``n_failed``, ``pair_samples`` and
        ``original_matching`` (the :class:`MatchResult` of the full data).
    """

    if cluster_col not in df.columns:
        raise ValueError(f"cluster column {cluster_col!r} not found")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    covariates = list(covariates)
    exact_cols = list(exact_cols or [])

    def fit(source: pd.DataFrame, fit_seed: int) -> tuple[MatchResult, dict[str, Any]]:
        matching = propensity_match(
            source,
            treat_col=cfg.group_col,
            treated_value=cfg.arm_a,
            covariates=covariates,
            exact_cols=exact_cols,
            caliper=caliper,
            method=method,  # type: ignore[arg-type]
            trim_common_support=trim_common_support,
            missing=missing,  # type: ignore[arg-type]
            random_state=fit_seed,
            control_value=cfg.arm_b,
            penalty=penalty,
        )
        matched_cfg = WinRatioConfig(
            group_col=cfg.group_col,
            arm_a=cfg.arm_a,
            arm_b=cfg.arm_b,
            outcomes=cfg.outcomes,
            pair_strategy="matched",
            pair_id=matching.pair_col,
        )
        return matching, compute_win_ratio(matching.matched_df, matched_cfg)["overall"]

    original_matching, point = fit(df, seed)

    def one(iteration: int, child: np.random.SeedSequence):
        rng = np.random.default_rng(child)
        sampled = _resample_clusters_within_arm(df, cfg, cluster_col, rng)
        try:
            matching, estimate = fit(sampled, int(child.generate_state(1)[0] % (2**31)))
        except (ValueError, np.linalg.LinAlgError):
            return None
        return estimate["wins"], estimate["losses"], estimate["details"]["n_pairs"], matching.pairs

    outputs = _run_replicates(one, n_boot, seed, n_jobs)
    ok = [o for o in outputs if o is not None]
    wins = np.array([o[0] for o in ok], dtype=float)
    losses = np.array([o[1] for o in ok], dtype=float)
    totals = np.array([o[2] for o in ok], dtype=float)
    replicate = summarize_replicates(wins, losses, totals, alpha=alpha)
    return {
        "wr": float(point["wr"]) if np.isfinite(point["wr"]) else point["wr"],
        **replicate,
        "n_boot": int(n_boot),
        "n_successful": len(ok),
        "n_failed": int(n_boot - len(ok)),
        "n_finite": int(np.isfinite(np.asarray(replicate["wr_samples"], dtype=float)).sum()),
        "seed": int(seed),
        "bootstrap_type": "cluster_within_arm_propensity_refit_rematch",
        "cluster_col": cluster_col,
        "pair_samples": [int(o[3]) for o in ok],
        "original_matching": original_matching,
    }


def bootstrap_propensity_weighted_gpc(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    *,
    covariates: Iterable[str],
    cluster_col: str,
    estimand: str = "overlap",
    missing: str = "drop",
    trim_common_support: Optional[bool] = None,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    n_jobs: int = 1,
    penalty: Optional[float] = None,
) -> dict[str, Any]:
    """Resample clusters within arm, refit weights and recompute weighted GPC.

    Parameters are as in :func:`bootstrap_propensity_matched_win_ratio`,
    with ``estimand`` passed to :func:`pywinratio.estimate_propensity_weights`.

    Returns
    -------
    dict
        ``wr``, ``ci``, ``net_benefit_ci``, ``win_odds_ci``, counts of
        successful and failed replicates and ``original_weighting``.
    """

    if cluster_col not in df.columns:
        raise ValueError(f"cluster column {cluster_col!r} not found")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    covariates = list(covariates)

    def fit(source: pd.DataFrame, fit_seed: int) -> tuple[Any, dict[str, Any]]:
        weighting = estimate_propensity_weights(
            source,
            treatment=cfg.group_col,
            treated=cfg.arm_a,
            control=cfg.arm_b,
            covariates=covariates,
            estimand=estimand,  # type: ignore[arg-type]
            missing=missing,  # type: ignore[arg-type]
            trim_common_support=trim_common_support,
            seed=fit_seed,
            penalty=penalty,
        )
        all_pairs = WinRatioConfig(cfg.group_col, cfg.arm_a, cfg.arm_b, cfg.outcomes, strata=cfg.strata)
        return weighting, compute_weighted_gpc(weighting.weighted_df, all_pairs, weight_col=weighting.weight_col)

    original_weighting, point = fit(df, seed)

    def one(iteration: int, child: np.random.SeedSequence):
        rng = np.random.default_rng(child)
        sampled = _resample_clusters_within_arm(df, cfg, cluster_col, rng)
        try:
            _, estimate = fit(sampled, int(child.generate_state(1)[0] % (2**31)))
        except (ValueError, np.linalg.LinAlgError):
            return None
        return estimate["wins"], estimate["losses"], estimate["details"]["total_pair_weight"]

    outputs = _run_replicates(one, n_boot, seed, n_jobs)
    ok = [o for o in outputs if o is not None]
    replicate = summarize_replicates(
        np.array([o[0] for o in ok], dtype=float),
        np.array([o[1] for o in ok], dtype=float),
        np.array([o[2] for o in ok], dtype=float),
        alpha=alpha,
    )
    return {
        "wr": point["wr"],
        **replicate,
        "n_boot": int(n_boot),
        "n_successful": len(ok),
        "n_failed": int(n_boot - len(ok)),
        "n_finite": int(np.isfinite(np.asarray(replicate["wr_samples"], dtype=float)).sum()),
        "seed": int(seed),
        "bootstrap_type": "cluster_within_arm_propensity_refit_weighted_gpc",
        "cluster_col": cluster_col,
        "original_weighting": original_weighting,
    }
