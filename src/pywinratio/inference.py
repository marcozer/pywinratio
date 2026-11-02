"""Analytic inference for win statistics.

Three effect measures are estimated from the same pairwise comparisons
(Buyse, 2010; Pocock et al., 2012; Dong et al., 2020):

* win ratio ``WR = P(win) / P(loss)``;
* net benefit ``NB = P(win) - P(loss)`` (also called win difference or
  Buyse's net treatment benefit);
* win odds ``WO = (P(win) + P(tie)/2) / (P(loss) + P(tie)/2) = (1 + NB) / (1 - NB)``.

For all-pairs (unmatched) designs the proportions are two-sample
U-statistics. Their variance is estimated from the Hoeffding projection
(structural components; DeLong et al., 1988; Bebu and Lachin, 2016; Dong et
al., 2016), optionally cluster-robust (Obuchowski, 1997), and transferred to
``log WR``, ``NB`` and ``log WO`` with the delta method. For assigned
(matched) pairs the pair results are independent multinomial draws, giving
``Var(log WR) = 1/W + 1/L``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from ._engine import PairSummary

STATISTICS = ("win_ratio", "net_benefit", "win_odds")


# ---------------------------------------------------------------------------
# Point estimates
# ---------------------------------------------------------------------------


def win_statistics(wins: float, losses: float, total: float) -> Dict[str, float]:
    """Win statistics from (possibly weighted) win, loss and pair totals.

    Parameters
    ----------
    wins, losses : float
        Number (or weight) of pairs won and lost by arm A.
    total : float
        Number (or weight) of all compared pairs, ties included.

    Returns
    -------
    dict
        ``wr`` (win ratio), ``net_benefit``, ``win_odds``,
        ``win_proportion``, ``loss_proportion``, ``tie_proportion``,
        ``win_probability`` (probabilistic index ``P(win) + P(tie)/2``) and
        ``win_probability_among_informative`` (``W / (W + L)``).

    Examples
    --------
    >>> stats = win_statistics(wins=30, losses=20, total=100)
    >>> round(stats["wr"], 2), round(stats["net_benefit"], 2), round(stats["win_odds"], 3)
    (1.5, 0.1, 1.222)
    """

    wins = float(wins)
    losses = float(losses)
    total = float(total)
    ties = total - wins - losses
    if losses > 0:
        wr = wins / losses
    elif wins > 0:
        wr = float("inf")
    else:
        wr = float("nan")
    if total > 0:
        p_win, p_loss, p_tie = wins / total, losses / total, ties / total
        net_benefit = p_win - p_loss
        win_probability = p_win + 0.5 * p_tie
        win_odds = win_probability / (1.0 - win_probability) if win_probability < 1 else float("inf")
    else:
        p_win = p_loss = p_tie = net_benefit = win_probability = win_odds = float("nan")
    informative = wins + losses
    return {
        "wr": wr,
        "net_benefit": net_benefit,
        "win_odds": win_odds,
        "win_proportion": p_win,
        "loss_proportion": p_loss,
        "tie_proportion": p_tie,
        "win_probability": win_probability,
        "win_probability_among_informative": wins / informative if informative > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Variance machinery
# ---------------------------------------------------------------------------


def _z(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return float(norm.ppf(1.0 - alpha / 2.0))


def _two_sided_p(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(2.0 * norm.sf(abs(z)))


def _intervals_from_variances(
    p_win: float,
    p_loss: float,
    var_w: float,
    var_l: float,
    cov_wl: float,
    alpha: float,
    df: Optional[int] = None,
) -> Dict[str, Any]:
    """Delta-method SEs, Wald CIs and p-values for WR, NB and WO.

    With ``df`` the reference distribution is Student's t with ``df`` degrees
    of freedom (used for cluster-robust variances, ``df = clusters - 1``).
    Undefined variances, and standard errors of zero, give NaN intervals.
    """

    if df is not None and df >= 1:
        z = float(student_t.ppf(1.0 - alpha / 2.0, df))

        def p_of(stat: float) -> float:
            return float(2.0 * student_t.sf(abs(stat), df)) if np.isfinite(stat) else float("nan")
    else:
        z = _z(alpha)
        p_of = _two_sided_p
    out: Dict[str, Any] = {"alpha": float(alpha), "df": df}
    nan2 = (float("nan"), float("nan"))
    if not all(np.isfinite([var_w, var_l, cov_wl])):
        for key in ("se_log_wr", "se_net_benefit", "se_log_win_odds"):
            out[key] = float("nan")
        for key in ("wr", "net_benefit", "win_odds"):
            out[f"{key}_ci"] = nan2
            out[f"{key}_p_value"] = float("nan")
        return out

    # win ratio on the log scale
    if p_win > 0 and p_loss > 0:
        var_log_wr = var_w / p_win**2 + var_l / p_loss**2 - 2.0 * cov_wl / (p_win * p_loss)
        se = float(np.sqrt(max(var_log_wr, 0.0)))
        log_wr = float(np.log(p_win / p_loss))
        out["se_log_wr"] = se
        out["wr_ci"] = (float(np.exp(log_wr - z * se)), float(np.exp(log_wr + z * se))) if se > 0 else nan2
        out["wr_p_value"] = p_of(log_wr / se) if se > 0 else float("nan")
    else:
        out["se_log_wr"] = float("nan")
        out["wr_ci"] = nan2
        out["wr_p_value"] = float("nan")

    # net benefit on the natural scale
    nb = p_win - p_loss
    var_nb = var_w + var_l - 2.0 * cov_wl
    se_nb = float(np.sqrt(max(var_nb, 0.0)))
    out["se_net_benefit"] = se_nb
    out["net_benefit_ci"] = (float(max(-1.0, nb - z * se_nb)), float(min(1.0, nb + z * se_nb))) if se_nb > 0 else nan2
    out["net_benefit_p_value"] = p_of(nb / se_nb) if se_nb > 0 else float("nan")

    # win odds on the log scale: WO = (1 + NB) / (1 - NB)
    if abs(nb) < 1:
        log_wo = float(np.log((1.0 + nb) / (1.0 - nb)))
        se_log_wo = float(2.0 / (1.0 - nb**2) * se_nb)
        out["se_log_win_odds"] = se_log_wo
        out["win_odds_ci"] = (
            (float(np.exp(log_wo - z * se_log_wo)), float(np.exp(log_wo + z * se_log_wo))) if se_log_wo > 0 else nan2
        )
        out["win_odds_p_value"] = p_of(log_wo / se_log_wo) if se_log_wo > 0 else float("nan")
    else:
        out["se_log_win_odds"] = float("nan")
        out["win_odds_ci"] = nan2
        out["win_odds_p_value"] = float("nan")
    return out


def _influence_components(
    summaries: Sequence[PairSummary],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Per-patient influence values of the overall win and loss proportions.

    Returns ``(psi_win, psi_loss, labels, group, p_win, p_loss)`` where
    ``group`` identifies the (stratum, arm) cell of each patient.
    """

    total = float(sum(s.total for s in summaries))
    if total <= 0:
        raise ValueError("no comparable pairs")
    wins = float(sum(s.wins for s in summaries))
    losses = float(sum(s.losses for s in summaries))
    psi_w: List[np.ndarray] = []
    psi_l: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    for index, s in enumerate(summaries):
        if s.total <= 0 or s.n_a == 0 or s.n_b == 0:
            continue
        sum_a = float(s.a_weights.sum())
        sum_b = float(s.b_weights.sum())
        share = s.total / total
        p_ws = s.wins / s.total
        p_ls = s.losses / s.total
        psi_w.append(share * s.a_weights * (s.a_wins / sum_b - p_ws) / sum_a)
        psi_l.append(share * s.a_weights * (s.a_losses / sum_b - p_ls) / sum_a)
        psi_w.append(share * s.b_weights * (s.b_wins / sum_a - p_ws) / sum_b)
        psi_l.append(share * s.b_weights * (s.b_losses / sum_a - p_ls) / sum_b)
        labels.extend([s.a_labels, s.b_labels])
        groups.extend([np.full(s.n_a, 2 * index), np.full(s.n_b, 2 * index + 1)])
    return (
        np.concatenate(psi_w),
        np.concatenate(psi_l),
        np.concatenate(labels),
        np.concatenate(groups),
        wins / total,
        losses / total,
    )


def _covariance(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clusters: Optional[np.ndarray],
) -> float:
    """Variance/covariance of a sum of influence values."""

    if clusters is None:
        cov = 0.0
        for g in np.unique(groups):
            in_g = groups == g
            n_g = int(in_g.sum())
            factor = n_g / (n_g - 1) if n_g > 1 else 1.0
            cov += factor * float(np.sum(x[in_g] * y[in_g]))
        return cov
    codes, inverse = np.unique(clusters, return_inverse=True)
    k = codes.size
    if k < 2:
        return float("nan")
    sx = np.bincount(inverse, weights=x, minlength=k)
    sy = np.bincount(inverse, weights=y, minlength=k)
    return float(k / (k - 1) * np.sum(sx * sy))


def u_statistic_inference(
    summaries: Sequence[PairSummary],
    *,
    alpha: float = 0.05,
    clusters: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    """Asymptotic inference for all-pairs win statistics.

    Parameters
    ----------
    summaries : sequence of PairSummary
        One summary per stratum, as produced by the comparison engine.
    alpha : float, default 0.05
        Two-sided level of the confidence intervals.
    clusters : mapping, optional
        Map from row label to cluster (e.g. hospital). When given, the
        variance is cluster-robust (influence values are summed within
        clusters before squaring) and intervals and p-values use Student's t
        with ``clusters - 1`` degrees of freedom. At least two clusters per
        arm are required.

    Returns
    -------
    dict
        Standard errors (``se_log_wr``, ``se_net_benefit``,
        ``se_log_win_odds``), Wald intervals (``wr_ci``, ``net_benefit_ci``,
        ``win_odds_ci``) and two-sided p-values for the null of no difference.
    """

    psi_w, psi_l, labels, groups, p_win, p_loss = _influence_components(summaries)
    cluster_values = None
    df = None
    estimable = True
    if clusters is not None:
        cluster_values = np.array([clusters[label] for label in labels], dtype=object)
        if any(value is None or value != value for value in cluster_values):
            raise ValueError("cluster labels are missing for some patients")
        cluster_values = cluster_values.astype(str)
        n_clusters = int(np.unique(cluster_values).size)
        per_arm = [np.unique(cluster_values[groups % 2 == arm]).size for arm in (0, 1)]
        if min(per_arm) < 2:
            estimable = False
            warnings.warn(
                "an arm has fewer than 2 clusters: the cluster-robust variance cannot be estimated",
                UserWarning,
                stacklevel=2,
            )
        df = n_clusters - 1
    else:
        sizes = np.bincount(groups)
        sizes = sizes[sizes > 0]
        if len(summaries) == 1 and sizes.min() < 2:
            estimable = False
            warnings.warn("an arm has a single patient: the variance cannot be estimated", UserWarning, stacklevel=2)
        elif sizes.min() < 2:
            warnings.warn(
                "some strata have a single patient in an arm; their variance contribution cannot be "
                "estimated and the standard error may be underestimated",
                UserWarning,
                stacklevel=2,
            )
    if estimable:
        var_w = _covariance(psi_w, psi_w, groups, cluster_values)
        var_l = _covariance(psi_l, psi_l, groups, cluster_values)
        cov_wl = _covariance(psi_w, psi_l, groups, cluster_values)
    else:
        var_w = var_l = cov_wl = float("nan")
    out = _intervals_from_variances(p_win, p_loss, var_w, var_l, cov_wl, alpha, df=df)
    out["method"] = "u_statistic_cluster" if clusters is not None else "u_statistic"
    if cluster_values is not None:
        out["n_clusters"] = int(np.unique(cluster_values).size)
    return out


def matched_pair_inference(
    results: Iterable[int],
    *,
    alpha: float = 0.05,
    clusters: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Asymptotic inference for assigned (matched) pairs.

    Pair outcomes are independent multinomial draws (win, loss, tie), so
    ``Var(log WR) = 1/W + 1/L`` and ``Var(NB) = (p_w + p_l - NB^2) / m``.

    Parameters
    ----------
    results : iterable of int
        Pair results from arm A's perspective: 1 win, -1 loss, 0 tie.
    alpha : float, default 0.05
        Two-sided level.
    clusters : sequence, optional
        Cluster of each pair for a cluster-robust variance; intervals and
        p-values then use Student's t with ``clusters - 1`` degrees of freedom.

    Returns
    -------
    dict
        Standard errors, Wald intervals and p-values for the win ratio, net
        benefit and win odds (keys as in :func:`u_statistic_inference`).
    """

    r = np.asarray(list(results), dtype=int)
    m = r.size
    if m == 0:
        raise ValueError("no assigned pairs")
    win = (r == 1).astype(float)
    loss = (r == -1).astype(float)
    p_win, p_loss = float(win.mean()), float(loss.mean())
    psi_w = (win - p_win) / m
    psi_l = (loss - p_loss) / m
    if clusters is None:
        var_w = float(np.sum(psi_w**2))
        var_l = float(np.sum(psi_l**2))
        cov_wl = float(np.sum(psi_w * psi_l))
        df = None
    else:
        cluster_values = np.asarray(clusters).astype(str)
        if cluster_values.size != m:
            raise ValueError("clusters must have one value per pair")
        groups = np.zeros(m, dtype=int)
        var_w = _covariance(psi_w, psi_w, groups, cluster_values)
        var_l = _covariance(psi_l, psi_l, groups, cluster_values)
        cov_wl = _covariance(psi_w, psi_l, groups, cluster_values)
        df = int(np.unique(cluster_values).size) - 1
    out = _intervals_from_variances(p_win, p_loss, var_w, var_l, cov_wl, alpha, df=df)
    out["method"] = "matched_pairs_cluster" if clusters is not None else "matched_pairs"
    return out


# ---------------------------------------------------------------------------
# Simple helpers kept for backward compatibility
# ---------------------------------------------------------------------------


def logwr_wald_ci(
    wins: int,
    losses: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Wald interval for the win ratio of **assigned (matched) pairs**.

    Uses ``Var(log WR) = 1/W + 1/L``, which is correct when the ``W + L``
    informative pairs are independent, as with assigned pairs. It is **not**
    valid for all-pairs comparisons, where every patient enters many pairs;
    use :func:`u_statistic_inference` (or ``WinRatio(...).fit``) instead.

    Parameters
    ----------
    wins, losses : int
        Numbers of pairs won and lost by arm A.
    alpha : float, default 0.05
        Two-sided level.

    Returns
    -------
    dict
        ``wr``, ``ci`` (lower, upper), ``se_logwr`` and ``method``.

    Examples
    --------
    >>> round(logwr_wald_ci(60, 40)["ci"][0], 3)
    1.005
    """

    wins = int(wins)
    losses = int(losses)
    if wins <= 0 or losses <= 0:
        return {"wr": np.nan, "ci": (np.nan, np.nan), "se_logwr": np.nan, "method": "wald_logwr"}
    wr = wins / losses
    se = float(np.sqrt(1.0 / wins + 1.0 / losses))
    z = _z(alpha)
    lo = float(np.exp(np.log(wr) - z * se))
    hi = float(np.exp(np.log(wr) + z * se))
    return {"wr": float(wr), "ci": (lo, hi), "se_logwr": se, "method": "wald_logwr"}


def logwr_wald_p_value(wins: int, losses: int, null_wr: float = 1.0) -> float:
    """Two-sided Wald p-value for ``H0: WR = null_wr`` with assigned pairs.

    See :func:`logwr_wald_ci` for the validity conditions.

    Parameters
    ----------
    wins, losses : int
        Numbers of pairs won and lost by arm A.
    null_wr : float, default 1.0
        Null value.

    Returns
    -------
    float
        Two-sided p-value (NaN without wins or losses).
    """

    wins = int(wins)
    losses = int(losses)
    if wins <= 0 or losses <= 0 or null_wr <= 0:
        return float("nan")
    se = float(np.sqrt(1.0 / wins + 1.0 / losses))
    return _two_sided_p((np.log(wins / losses) - np.log(null_wr)) / se)


def adjust_pvalues(p_values: Sequence[float], method: str = "holm") -> List[float]:
    """Adjust p-values for multiple comparisons.

    Parameters
    ----------
    p_values : sequence of float
        Unadjusted p-values; NaN values are left as NaN and ignored.
    method : {"holm", "bonferroni", "fdr_bh"}, default "holm"
        Holm step-down, Bonferroni, or Benjamini-Hochberg false discovery rate.

    Returns
    -------
    list of float
        Adjusted p-values in the input order.

    Examples
    --------
    >>> [round(p, 3) for p in adjust_pvalues([0.01, 0.04, 0.03], "holm")]
    [0.03, 0.06, 0.06]
    """

    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    m = valid.size
    if m == 0:
        return adjusted.tolist()
    pv = p[valid]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    if method == "bonferroni":
        adj = np.minimum(ranked * m, 1.0)
    elif method == "holm":
        adj = np.minimum(np.maximum.accumulate(ranked * (m - np.arange(m))), 1.0)
    elif method == "fdr_bh":
        adj = ranked * m / np.arange(1, m + 1)
        adj = np.minimum(np.minimum.accumulate(adj[::-1])[::-1], 1.0)
    else:
        raise ValueError("method must be 'holm', 'bonferroni', or 'fdr_bh'")
    out = np.empty(m)
    out[order] = adj
    adjusted[valid] = out
    return adjusted.tolist()


def adjust_pvalues_holm(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """Holm-Bonferroni adjustment with significance flags.

    Parameters
    ----------
    p_values : list of float
        Unadjusted p-values (NaN values are ignored).
    alpha : float, default 0.05
        Family-wise error rate.

    Returns
    -------
    adjusted : list of float
        Holm-adjusted p-values.
    significant : list of bool
        Whether each adjusted p-value is ``<= alpha``.
    """

    adjusted = adjust_pvalues(p_values, "holm")
    significant = [bool(np.isfinite(value) and value <= alpha) for value in adjusted]
    return adjusted, significant


# ---------------------------------------------------------------------------
# E-values
# ---------------------------------------------------------------------------


def compute_e_value(rr: float) -> Dict[str, Any]:
    """E-value of a ratio estimate (VanderWeele and Ding, 2017).

    ``E = RR + sqrt(RR * (RR - 1))`` (estimates below 1 are inverted).

    The E-value is defined on the risk-ratio scale. A win ratio is not a risk
    ratio, so treating it as one gives only a heuristic summary of robustness
    to unmeasured confounding; report it as such.

    Parameters
    ----------
    rr : float
        Ratio estimate (e.g. a win ratio).

    Returns
    -------
    dict
        ``e_value``, ``rr_used`` (the estimate or its inverse) and a text
        ``interpretation``.

    Examples
    --------
    >>> round(compute_e_value(2.0)["e_value"], 3)
    3.414
    """

    if rr is None or not np.isfinite(rr) or rr <= 0:
        return {"e_value": np.nan, "rr_used": np.nan, "interpretation": "Cannot compute E-value for invalid RR"}
    rr_calc = rr if rr >= 1 else 1 / rr
    e_value = float(rr_calc + np.sqrt(rr_calc * (rr_calc - 1)))
    if e_value < 1.5:
        strength = "weak"
    elif e_value < 2.0:
        strength = "moderate"
    elif e_value < 3.0:
        strength = "moderately strong"
    else:
        strength = "strong"
    interpretation = (
        f"An unmeasured confounder would need to have an RR of {e_value:.2f} "
        f"with both the exposure and outcome to fully explain away the observed effect. "
        f"This represents a {strength} level of robustness to unmeasured confounding."
    )
    return {"e_value": e_value, "rr_used": float(rr_calc), "interpretation": interpretation}


def compute_e_value_for_ci(rr: float, ci_lower: float, ci_upper: float) -> Dict[str, Any]:
    """E-values for a ratio estimate and the confidence limit closest to 1.

    See :func:`compute_e_value` for the caveat about win ratios.

    Parameters
    ----------
    rr : float
        Point estimate.
    ci_lower, ci_upper : float
        Confidence limits.

    Returns
    -------
    dict
        ``e_value_point``, ``e_value_ci`` (1 when the interval contains 1),
        ``ci_limit_used``, ``rr_point``, ``rr_ci_limit`` and ``interpretation``.
    """

    e_point = compute_e_value(rr)
    if rr >= 1:
        ci_limit, limit_used = ci_lower, "lower"
    else:
        ci_limit, limit_used = ci_upper, "upper"
    e_ci = compute_e_value(ci_limit) if ci_limit is not None and np.isfinite(ci_limit) else {"e_value": np.nan}
    if ci_lower is not None and ci_upper is not None and ci_lower <= 1.0 <= ci_upper:
        e_ci_value = 1.0
        interpretation = (
            f"E-value for point estimate: {e_point['e_value']:.2f}. "
            "However, the CI includes the null, so E-value for CI = 1.0 "
            "(any unmeasured confounder could explain away the effect)."
        )
    else:
        e_ci_value = e_ci.get("e_value", np.nan)
        interpretation = (
            f"E-value for point estimate: {e_point['e_value']:.2f}. "
            f"E-value for CI {limit_used} limit: {e_ci_value:.2f}. "
            f"An unmeasured confounder would need an RR >= {e_ci_value:.2f} with both "
            "exposure and outcome to move the CI to include the null."
        )
    return {
        "e_value_point": e_point["e_value"],
        "e_value_ci": float(e_ci_value),
        "ci_limit_used": limit_used,
        "rr_point": float(rr),
        "rr_ci_limit": float(ci_limit) if ci_limit is not None else np.nan,
        "interpretation": interpretation,
    }
