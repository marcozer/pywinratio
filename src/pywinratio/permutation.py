"""Permutation (Finkelstein-Schoenfeld) tests of no difference between arms.

Each patient receives a score ``U_i`` = number of patients (of either arm,
within the same stratum) they beat on the hierarchy minus the number they
lose to. The total wins minus losses of arm A equals the sum of the scores
of arm A, whatever the arm labels, so its permutation distribution is that
of a random subset sum (Finkelstein and Schoenfeld, 1999). Comparisons are
computed once; each permutation then costs O(n).

The units that are permuted must match how the exposure was assigned:
patients (randomized trials), patients within strata (stratified
randomization), whole clusters (cluster-level exposure such as hospital
volume), or patients within clusters (patient-level exposure in a
multicenter study).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._engine import MAX_CELLS, extract_tiers, resolve_all_pairs
from .config import WinRatioConfig
from .wr import _check_inputs, _matched_results, _strata_groups, _stratum_labels

_BATCH_CELLS = 20_000_000


def _scores(frame: pd.DataFrame, cfg: WinRatioConfig, max_cells: int = MAX_CELLS) -> np.ndarray:
    """Finkelstein-Schoenfeld scores of the patients of one stratum."""

    tiers = extract_tiers(frame, cfg.outcomes)
    n = len(frame)
    scores = np.zeros(n)
    block = max(1, int(max_cells // max(n, 1)))
    for start in range(0, n, block):
        rows = slice(start, min(start + block, n))
        result, _ = resolve_all_pairs(tiers, tiers, cfg.outcomes, rows)
        scores[rows] = result.sum(axis=1, dtype=np.int64)
    return scores


def _permutation_null(
    scores: np.ndarray,
    is_a: np.ndarray,
    blocks: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Null distribution of ``sum(scores[A])`` when labels are permuted within blocks.

    Returns the simulated statistics and the exact permutation variance.
    """

    null = np.zeros(n_perm)
    variance = 0.0
    for block in np.unique(blocks):
        members = np.flatnonzero(blocks == block)
        u = scores[members]
        a = is_a[members]
        n, k = members.size, int(a.sum())
        if k == 0 or k == n:
            null += float(u[a].sum())
            continue
        variance += k * (n - k) / (n * (n - 1)) * float(np.sum((u - u.mean()) ** 2))
        batch = max(1, min(n_perm, _BATCH_CELLS // n))
        for start in range(0, n_perm, batch):
            size = min(batch, n_perm - start)
            perm = rng.permuted(np.tile(a, (size, 1)), axis=1)
            null[start : start + size] += perm.astype(float) @ u
    return null, variance


def _p_values(observed: float, null: np.ndarray, variance: float) -> Dict[str, float]:
    n_perm = null.size
    extreme = int(np.sum(np.abs(null) >= abs(observed) - 1e-9))
    z = observed / np.sqrt(variance) if variance > 0 else float("nan")
    return {
        "p_value": (1 + extreme) / (1 + n_perm),
        "z": float(z),
        "p_value_asymptotic": float(2 * norm.sf(abs(z))) if np.isfinite(z) else float("nan"),
    }


def clusters_nested_in_arms(frame: pd.DataFrame, group_col: str, cluster_col: str) -> bool:
    """Whether every cluster contains patients of a single arm (cluster-level exposure)."""

    return bool((frame.groupby(cluster_col)[group_col].nunique() == 1).all())


def permutation_test(
    df: pd.DataFrame,
    cfg: WinRatioConfig,
    *,
    n_perm: int = 5000,
    seed: int = 42,
    cluster_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Permutation test of the null hypothesis that the arms do not differ.

    The permutation scheme follows the design:

    ==============================  =============================================
    Design                          Permuted units
    ==============================  =============================================
    all pairs                       patients (Finkelstein-Schoenfeld test)
    stratified                      patients within strata
    ``cluster_col``, each cluster   whole clusters (within strata when every
    in one arm                      cluster lies in one stratum)
    ``cluster_col``, clusters with  patients within clusters (and strata)
    both arms
    assigned pairs                  arm labels swapped within pairs (sign flip)
    assigned pairs + clusters       signs flipped for whole clusters of pairs
    ==============================  =============================================

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis data.
    cfg : WinRatioConfig
        Configuration (all pairs, stratified or matched).
    n_perm : int, default 5000
        Number of random permutations.
    seed : int, default 42
        Random seed.
    cluster_col : str, optional
        Cluster (e.g. hospital) column. For assigned pairs the cluster of the
        arm-A patient of each pair is used.

    Returns
    -------
    dict
        ``statistic`` (wins minus losses of arm A), ``net_benefit``,
        ``p_value`` (Monte Carlo, two-sided, ``(1 + #extreme) / (1 + n_perm)``),
        ``z`` and ``p_value_asymptotic`` (normal approximation with the exact
        permutation variance), ``permuted_units``, ``n_perm`` and ``method``.

    Examples
    --------
    >>> from pywinratio.datasets import make_heart_failure_trial
    >>> from pywinratio import WinRatioConfig, Outcome, permutation_test
    >>> trial = make_heart_failure_trial(n=200, seed=3)
    >>> cfg = WinRatioConfig("arm", "treatment", "placebo",
    ...     [Outcome.time_to_event("time_death", "death", terminal=True)])
    >>> 0 < permutation_test(trial, cfg, n_perm=200)["p_value"] <= 1
    True
    """

    if n_perm <= 0:
        raise ValueError("n_perm must be positive")
    rng = np.random.default_rng(seed)
    frame = _check_inputs(df, cfg, extra_columns=[cluster_col] if cluster_col else [])
    if cluster_col is not None and frame[cluster_col].isna().any():
        raise ValueError(f"cluster column {cluster_col!r} has missing values")

    if cfg.pair_strategy == "matched" and cfg.pair_id:
        a, _, result, _, _ = _matched_results(frame, cfg)
        r = result.astype(float)
        if cluster_col is None:
            units, method = r, "sign_flip_matched_pairs"
        else:
            codes, inverse = np.unique(a[cluster_col].astype(str).to_numpy(), return_inverse=True)
            units, method = np.bincount(inverse, weights=r, minlength=codes.size), "sign_flip_cluster_of_pairs"
        observed = float(r.sum())
        null = np.zeros(n_perm)
        batch = max(1, min(n_perm, _BATCH_CELLS // max(units.size, 1)))
        for start in range(0, n_perm, batch):
            size = min(batch, n_perm - start)
            null[start : start + size] = rng.choice([-1.0, 1.0], size=(size, units.size)) @ units
        out = _p_values(observed, null, float(np.sum(units**2)))
        out.update(
            {
                "statistic": observed,
                "net_benefit": observed / r.size if r.size else float("nan"),
                "permuted_units": "clusters of pairs" if cluster_col else "pairs",
                "n_units": int(units.size),
                "n_perm": int(n_perm),
                "seed": int(seed),
                "method": method,
            }
        )
        return out

    position = pd.Series(np.arange(len(frame)), index=frame.index)
    scores = np.zeros(len(frame))
    total_pairs = 0.0
    for _, sub in _strata_groups(frame, cfg):
        n_a = int((sub[cfg.group_col] == cfg.arm_a).sum())
        total_pairs += n_a * (len(sub) - n_a)
        scores[position.loc[sub.index].to_numpy()] = _scores(sub, cfg)
    is_a = (frame[cfg.group_col] == cfg.arm_a).to_numpy()
    observed = float(scores[is_a].sum())
    strata = _stratum_labels(frame, cfg).astype(str)

    if cluster_col is None:
        units, unit_is_a, blocks = scores, is_a, strata
        permuted, method = "patients", "finkelstein_schoenfeld_stratified" if cfg.strata else "finkelstein_schoenfeld"
    else:
        clusters = frame[cluster_col].astype(str).to_numpy()
        if clusters_nested_in_arms(frame, cfg.group_col, cluster_col):
            codes, inverse = np.unique(clusters, return_inverse=True)
            units = np.bincount(inverse, weights=scores, minlength=codes.size)
            unit_is_a = np.zeros(codes.size, dtype=bool)
            unit_is_a[inverse] = is_a
            cluster_strata = pd.Series(strata).groupby(inverse).agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
            within = bool(cfg.strata) and bool(cluster_strata.notna().all())
            blocks = cluster_strata.to_numpy(dtype=object).astype(str) if within else np.zeros(codes.size, dtype=int)
            permuted, method = "clusters", "cluster_permutation"
        else:
            units, unit_is_a = scores, is_a
            blocks = np.array([f"{s}|{c}" for s, c in zip(strata, clusters)], dtype=object)
            permuted, method = "patients within clusters", "finkelstein_schoenfeld_within_clusters"

    null, variance = _permutation_null(units, unit_is_a, blocks, n_perm, rng)
    out = _p_values(observed, null, variance)
    out.update(
        {
            "statistic": observed,
            "net_benefit": observed / total_pairs if total_pairs else float("nan"),
            "permuted_units": permuted,
            "n_units": int(np.asarray(units).size),
            "n_perm": int(n_perm),
            "seed": int(seed),
            "method": method,
        }
    )
    return out
