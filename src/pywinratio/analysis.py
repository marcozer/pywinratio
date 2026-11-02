"""High-level interface: :class:`WinRatio`, :func:`win_ratio` and :class:`WinRatioResult`."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import WinRatioConfig, WinRatioOutcome

_STAT_LABELS = {
    "win_ratio": "Win ratio",
    "net_benefit": "Net benefit",
    "win_odds": "Win odds",
}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "NA"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def _fmt_count(value: Any) -> str:
    value = float(value)
    return f"{value:.0f}" if value.is_integer() else f"{value:.1f}"


def _fmt_p(p: Any) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else (None if math.isnan(value) else str(value))
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


@dataclass
class WinRatioResult:
    """Fitted win statistics with reporting helpers.

    Returned by :meth:`pywinratio.WinRatioAnalysis.fit` and :func:`pywinratio.win_ratio`. The point
    estimates come from the chosen design (assigned pairs, all pairs,
    stratified, or weighted all pairs).

    Attributes
    ----------
    estimate : dict
        Point estimates and tier counts (see :func:`pywinratio.compute_win_ratio`).
    config : WinRatioConfig
        The analysis configuration.
    bootstrap : dict or None
        Bootstrap output when ``n_boot > 0``.
    method : str
        ``"matched"``, ``"all_pairs"``, ``"stratified"`` or ``"weighted_gpc"``.
    inference : dict or None
        Analytic standard errors, Wald intervals and p-values.
    permutation : dict or None
        Permutation-test output when ``n_perm > 0``.
    alpha : float
        Two-sided level of all intervals.
    pair_table : pandas.DataFrame or None
        Per-pair results of an assigned-pair analysis.
    strata_results : list or None
        Per-stratum estimates of a stratified analysis.
    """

    estimate: dict[str, Any]
    config: WinRatioConfig
    bootstrap: Optional[dict[str, Any]] = None
    method: str = "all_pairs"
    inference: Optional[dict[str, Any]] = None
    permutation: Optional[dict[str, Any]] = None
    alpha: float = 0.05
    pair_table: Optional[pd.DataFrame] = None
    strata_results: Optional[List[dict[str, Any]]] = None
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Point estimates
    # ------------------------------------------------------------------

    @property
    def wr(self) -> float:
        """Win ratio (arm A wins / arm B wins)."""
        return self.estimate["wr"]

    win_ratio = wr

    @property
    def net_benefit(self) -> float:
        """Net benefit ``P(win) - P(loss)`` (Buyse's net treatment benefit)."""
        return self.estimate["net_benefit"]

    @property
    def win_odds(self) -> float:
        """Win odds ``(P(win) + P(tie)/2) / (P(loss) + P(tie)/2)``."""
        return self.estimate["win_odds"]

    @property
    def wins(self) -> float:
        return self.estimate["wins"]

    @property
    def losses(self) -> float:
        return self.estimate["losses"]

    @property
    def ties(self) -> float:
        return self.estimate["ties"]

    @property
    def n_pairs(self) -> float:
        """Number (or total weight) of compared pairs."""
        return self.estimate["details"].get("n_pairs", self.wins + self.losses + self.ties)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _ci(self, statistic: str) -> tuple[float, float, str]:
        key = {"win_ratio": "ci", "net_benefit": "net_benefit_ci", "win_odds": "win_odds_ci"}[statistic]
        if self.bootstrap is not None and key in self.bootstrap:
            lo, hi = self.bootstrap[key]
            return lo, hi, f"bootstrap ({self.bootstrap.get('bootstrap_type', 'percentile')})"
        if self.inference is not None:
            lo, hi = self.inference[{"win_ratio": "wr_ci"}.get(statistic, key)]
            return lo, hi, self.inference["method"]
        return float("nan"), float("nan"), "none"

    def _p(self, statistic: str) -> tuple[float, str]:
        if self.permutation is not None:
            return self.permutation["p_value"], f"permutation ({self.permutation['method']})"
        if self.inference is not None:
            key = {"win_ratio": "wr_p_value"}.get(statistic, f"{statistic}_p_value")
            return self.inference[key], f"Wald ({self.inference['method']})"
        if self.bootstrap is not None and statistic == "win_ratio":
            from .summary import bootstrap_p_value_from_samples

            return bootstrap_p_value_from_samples(self.bootstrap.get("wr_samples", [])), "bootstrap"
        return float("nan"), "none"

    @property
    def ci(self) -> tuple[float, float]:
        """Confidence interval of the win ratio (bootstrap if run, else analytic)."""
        lo, hi, _ = self._ci("win_ratio")
        return lo, hi

    @property
    def p_value(self) -> float:
        """Two-sided p-value for no difference (permutation if run, else Wald)."""
        return self._p("win_ratio")[0]

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Flat dictionary of estimates, intervals and p-values.

        Keeps the keys of earlier versions (``wins``, ``losses``, ``ties``,
        ``wr``, ``ci_lower``, ``ci_upper``, ``n_boot``) and adds
        ``net_benefit``, ``win_odds`` with their intervals and p-values.
        """

        result = dict(self.estimate)
        result["method"] = self.method
        lo, hi, ci_method = self._ci("win_ratio")
        result["ci_lower"], result["ci_upper"] = lo, hi
        result["ci_method"] = ci_method
        result["p_value"], result["p_value_method"] = self._p("win_ratio")
        for statistic in ("net_benefit", "win_odds"):
            lo, hi, _ = self._ci(statistic)
            result[f"{statistic}_ci_lower"], result[f"{statistic}_ci_upper"] = lo, hi
            result[f"{statistic}_p_value"] = self._p(statistic)[0]
        if self.bootstrap is not None:
            result["n_boot"] = self.bootstrap["n_boot"]
        result["alpha"] = self.alpha
        return result

    def to_frame(self) -> pd.DataFrame:
        """One row per effect measure: estimate, interval, p-value and methods.

        Examples
        --------
        >>> from pywinratio import Outcome, win_ratio
        >>> from pywinratio.datasets import make_heart_failure_trial
        >>> trial = make_heart_failure_trial(n=300, seed=5)
        >>> res = win_ratio(trial, [Outcome.time_to_event("time_death", "death")],
        ...                 group="arm", treatment="treatment", control="placebo")
        >>> res.to_frame().index.tolist()
        ['Win ratio', 'Net benefit', 'Win odds']
        """

        rows = []
        for statistic, label in _STAT_LABELS.items():
            estimate = {"win_ratio": self.wr, "net_benefit": self.net_benefit, "win_odds": self.win_odds}[statistic]
            lo, hi, ci_method = self._ci(statistic)
            p, p_method = self._p(statistic)
            rows.append(
                {
                    "statistic": label,
                    "estimate": estimate,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "p_value": p,
                    "ci_method": ci_method,
                    "p_method": p_method,
                }
            )
        return pd.DataFrame(rows).set_index("statistic")

    table = to_frame

    def tiers(self) -> pd.DataFrame:
        """Tier-by-tier decomposition of the comparisons.

        Columns: ``priority``, ``outcome``, ``entering`` (pairs still tied
        when the tier is reached), ``wins``, ``losses``, ``resolved``,
        ``terminal_ties`` (pairs stopped by a shared terminal event),
        ``conditional_resolution`` (resolved / entering),
        ``overall_resolution`` (resolved / all pairs), ``tier_wr``,
        ``net_benefit_contribution`` ((wins - losses) / all pairs) and
        ``cumulative_wr`` (WR using tiers 1..k only).

        The tier counts describe where pairs were first separated, given
        that all higher tiers tied; they are not separate treatment effects
        on each component.
        """

        details = self.estimate.get("details", {}) or {}
        wins = list(details.get("tier_wins", []))
        losses = list(details.get("tier_losses", []))
        stops = list(details.get("tier_terminal_ties", [0] * len(wins)))
        total = float(self.estimate.get("wins", 0) + self.estimate.get("losses", 0) + self.estimate.get("ties", 0))
        rows = []
        entering = total
        cum_w = cum_l = 0.0
        for index, outcome in enumerate(self.config.outcomes):
            w = float(wins[index]) if index < len(wins) else 0.0
            lo = float(losses[index]) if index < len(losses) else 0.0
            st = float(stops[index]) if index < len(stops) else 0.0
            resolved = w + lo
            cum_w += w
            cum_l += lo
            rows.append(
                {
                    "priority": index + 1,
                    "outcome": outcome.name,
                    "entering": entering,
                    "wins": w,
                    "losses": lo,
                    "resolved": resolved,
                    "terminal_ties": st,
                    "conditional_resolution": resolved / entering if entering > 0 else np.nan,
                    "overall_resolution": resolved / total if total > 0 else np.nan,
                    "tier_wr": w / lo if lo > 0 else (np.inf if w > 0 else np.nan),
                    "net_benefit_contribution": (w - lo) / total if total > 0 else np.nan,
                    "cumulative_wr": cum_w / cum_l if cum_l > 0 else (np.inf if cum_w > 0 else np.nan),
                }
            )
            entering -= resolved + st
        return pd.DataFrame(rows)

    def pairs(self) -> pd.DataFrame:
        """Per-pair results of an assigned-pair analysis (result and deciding tier)."""

        if self.pair_table is None:
            raise ValueError("per-pair results are only available for assigned (matched) pairs")
        return self.pair_table.copy()

    def strata(self) -> pd.DataFrame:
        """Per-stratum wins, losses, ties and win statistics of a stratified analysis."""

        if not self.strata_results:
            raise ValueError("the analysis was not stratified")
        rows = []
        for res in self.strata_results:
            rows.append(
                {
                    **res.get("strata", {}),
                    "n_a": res["details"]["n_a"],
                    "n_b": res["details"]["n_b"],
                    "wins": res["wins"],
                    "losses": res["losses"],
                    "ties": res["ties"],
                    "wr": res["wr"],
                    "net_benefit": res["net_benefit"],
                    "win_odds": res["win_odds"],
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Text, HTML, JSON
    # ------------------------------------------------------------------

    def report(self, digits: int = 3) -> str:
        """Plain-text report of the analysis."""

        cfg = self.config
        details = self.estimate.get("details", {})
        level = int(round(100 * (1 - self.alpha)))
        design = {
            "matched": "assigned (matched) pairs",
            "all_pairs": "all cross-arm pairs",
            "stratified": f"all pairs within {details.get('n_strata', '?')} strata",
            "weighted_gpc": f"weighted all pairs (weights: {details.get('weight_col')})",
        }.get(self.method, self.method)
        lines = [
            "Win-ratio analysis",
            "=" * 72,
            f"Arm A: {cfg.arm_a!s} (n={details.get('n_a', '?')})   vs   Arm B: {cfg.arm_b!s} "
            f"(n={details.get('n_b', '?')})",
            f"Design: {design}",
            f"Pairs: {_fmt_count(self.n_pairs)}  |  A wins {_fmt_count(self.wins)}  |  B wins "
            f"{_fmt_count(self.losses)}  |  ties {_fmt_count(self.ties)}",
            "",
            f"{'':<14}{'Estimate':>10}{f'{level}% CI':>24}{'p-value':>10}",
        ]
        frame = self.to_frame()
        for label, row in frame.iterrows():
            ci = f"[{_fmt(row['ci_lower'], digits)}, {_fmt(row['ci_upper'], digits)}]"
            lines.append(f"{label:<14}{_fmt(row['estimate'], digits):>10}{ci:>24}{_fmt_p(row['p_value']):>10}")
        lines.append(f"CI: {frame['ci_method'].iloc[0]};  p: {frame['p_method'].iloc[0]}")
        lines.append("")
        lines.append("Hierarchy (tier: wins / losses, share of pairs resolved)")
        tiers = self.tiers()
        for _, row in tiers.iterrows():
            stop = f", {int(row['terminal_ties'])} stopped as terminal ties" if row["terminal_ties"] else ""
            lines.append(
                f"  {int(row['priority'])}. {row['outcome']}: {_fmt(row['wins'], 0)} / {_fmt(row['losses'], 0)}"
                f" ({100 * row['overall_resolution']:.1f}%){stop}"
            )
        for note in self.notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()

    def __repr__(self) -> str:
        lo, hi = self.ci
        return (
            f"<WinRatioResult method={self.method!r} WR={_fmt(self.wr)} "
            f"CI=[{_fmt(lo)}, {_fmt(hi)}] NB={_fmt(self.net_benefit)} WO={_fmt(self.win_odds)}>"
        )

    def _repr_html_(self) -> str:
        frame = self.to_frame()
        return (
            f"<p><b>Win-ratio analysis</b> &mdash; {self.config.arm_a} vs {self.config.arm_b} ({self.method})</p>"
            + frame.to_html(float_format=lambda v: _fmt(v))
            + self.tiers().to_html(index=False, float_format=lambda v: _fmt(v))
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dictionary (replicate samples omitted)."""

        boot = None
        if self.bootstrap is not None:
            boot = {
                k: v for k, v in self.bootstrap.items() if not k.endswith("samples") and not k.startswith("original")
            }
        return _jsonable(
            {
                "method": self.method,
                "alpha": self.alpha,
                "arm_a": self.config.arm_a,
                "arm_b": self.config.arm_b,
                "outcomes": [o.describe() for o in self.config.outcomes],
                "estimate": self.estimate,
                "statistics": self.to_frame().reset_index().to_dict(orient="records"),
                "tiers": self.tiers().to_dict(orient="records"),
                "inference": self.inference,
                "bootstrap": boot,
                "permutation": self.permutation,
                "notes": self.notes,
            }
        )

    def to_json(self, path: Optional[str] = None, **kwargs: Any) -> str:
        """Serialize :meth:`to_dict` to JSON, optionally writing it to ``path``."""

        text = json.dumps(self.to_dict(), indent=kwargs.pop("indent", 2), **kwargs)
        if path is not None:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        return text

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot(self, kind: str = "tiers", **kwargs: Any):
        """Plot the result: ``kind="tiers"`` (tier decomposition) or ``"bootstrap"``."""

        from . import plotting

        if kind == "tiers":
            return plotting.plot_tiers(self, **kwargs)
        if kind == "bootstrap":
            return plotting.plot_bootstrap(self, **kwargs)
        raise ValueError("kind must be 'tiers' or 'bootstrap'")


class WinRatioAnalysis:
    """Two-arm win-ratio analysis.

    Parameters
    ----------
    outcomes : iterable of WinRatioOutcome
        Hierarchy, highest priority first.
    group : str
        Arm column.
    treatment, control : any
        Values of ``group`` for arm A (treatment) and arm B (control); win
        statistics favor ``treatment`` when WR > 1. ``arm_a`` and ``arm_b``
        are accepted as synonyms.
    strata : iterable of str, optional
        Compare patients only within strata (stratified win ratio).

    Examples
    --------
    >>> from pywinratio import Outcome, WinRatio
    >>> from pywinratio.datasets import make_heart_failure_trial
    >>> trial = make_heart_failure_trial(n=400, seed=7)
    >>> analysis = WinRatio(
    ...     [Outcome.time_to_event("time_death", "death", name="Death", terminal=True),
    ...      Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization"),
    ...      Outcome.continuous("kccq_change", name="KCCQ", favorable="higher", margin=5)],
    ...     group="arm", treatment="treatment", control="placebo")
    >>> result = analysis.fit(trial)
    >>> round(result.wr, 3) > 0
    True
    """

    def __init__(
        self,
        outcomes: Optional[Iterable[WinRatioOutcome]] = None,
        *,
        group: str,
        treatment: Any = None,
        control: Any = None,
        arm_a: Any = None,
        arm_b: Any = None,
        strata: Optional[Iterable[str]] = None,
    ) -> None:
        if treatment is not None and arm_a is not None and treatment != arm_a:
            raise ValueError("pass either treatment or arm_a, not both")
        if control is not None and arm_b is not None and control != arm_b:
            raise ValueError("pass either control or arm_b, not both")
        self.group = group
        self.arm_a = treatment if treatment is not None else arm_a
        self.arm_b = control if control is not None else arm_b
        if self.arm_a is None or self.arm_b is None:
            raise ValueError("treatment (arm_a) and control (arm_b) are required")
        self.outcomes = list(outcomes or [])
        if not self.outcomes:
            raise ValueError("at least one outcome tier is required")
        if isinstance(strata, str):
            strata = [strata]
        self.strata = list(strata) if strata is not None else None

    def config(self, *, pair_id: Optional[str] = None) -> WinRatioConfig:
        """The :class:`WinRatioConfig` used for a given design."""

        return WinRatioConfig(
            group_col=self.group,
            arm_a=self.arm_a,
            arm_b=self.arm_b,
            outcomes=self.outcomes,
            strata=None if pair_id is not None else self.strata,
            pair_strategy="matched" if pair_id is not None else "all_pairs",
            pair_id=pair_id,
        )

    def fit(
        self,
        df: pd.DataFrame,
        *,
        pair_id: Optional[str] = None,
        weight_col: Optional[str] = None,
        cluster_col: Optional[str] = None,
        cluster_within_arm: bool = False,
        n_boot: int = 0,
        n_perm: int = 0,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> WinRatioResult:
        """Estimate win statistics with analytic, bootstrap and permutation inference.

        Parameters
        ----------
        df : pandas.DataFrame
            One row per patient.
        pair_id : str, optional
            Matched-pair identifier: compare only the two patients of each
            assigned pair (e.g. after :func:`pywinratio.propensity_match`).
        weight_col : str, optional
            Analysis-weight column for a weighted all-pairs GPC (e.g.
            overlap weights from :func:`pywinratio.estimate_propensity_weights`).
        cluster_col : str, optional
            Cluster (e.g. hospital) column. Makes the analytic variance
            cluster-robust (Student's t with clusters - 1 degrees of
            freedom), the bootstrap resample clusters, and the permutation
            test permute whole clusters (when every cluster belongs to one
            arm) or patients within clusters (otherwise); see
            :func:`pywinratio.permutation_test`.
        cluster_within_arm : bool, default False
            Resample clusters separately within each arm (cluster-level
            exposure). All-pairs and weighted designs only.
        n_boot : int, default 0
            Bootstrap replicates (0 = analytic inference only).
        n_perm : int, default 0
            Permutation replicates for a p-value (0 = none).
        alpha : float, default 0.05
            Two-sided level of the intervals.
        seed : int, default 42
            Random seed for bootstrap and permutation.

        Returns
        -------
        WinRatioResult
        """

        from .gpc import bootstrap_weighted_gpc, compute_weighted_gpc
        from .inference import matched_pair_inference, u_statistic_inference
        from .permutation import permutation_test
        from .wr import (
            _check_inputs,
            _matched_results,
            _summaries,
            bootstrap_win_ratio,
            bootstrap_win_ratio_cluster,
            bootstrap_win_ratio_cluster_within_arm,
            bootstrap_win_ratio_matched,
            compute_win_ratio,
            pair_results,
        )

        if pair_id is not None and weight_col is not None:
            raise ValueError("choose assigned-pair WR (pair_id) or weighted all-pair GPC (weight_col), not both")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if cluster_within_arm and cluster_col is None:
            raise ValueError("cluster_within_arm requires cluster_col")
        if cluster_within_arm and pair_id is not None:
            raise ValueError(
                "cluster_within_arm does not apply to assigned pairs: pairs are resampled by the cluster "
                "of their arm-A patient"
            )
        if pair_id is not None and self.strata:
            warnings.warn(
                "strata are ignored for assigned pairs (pairs are already matched)", UserWarning, stacklevel=2
            )
        cfg = self.config(pair_id=pair_id)
        extra = [c for c in (cluster_col, weight_col) if c]
        frame = _check_inputs(df, cfg, extra_columns=extra)
        if cluster_col is not None and frame[cluster_col].isna().any():
            raise ValueError(f"cluster column {cluster_col!r} has missing values")
        notes: List[str] = []
        pair_table = None
        strata_results = None

        if weight_col is not None:
            method = "weighted_gpc"
            estimate = compute_weighted_gpc(frame, cfg, weight_col=weight_col)
            weights = frame[weight_col].astype(float)
            summaries = [s for _, s in _summaries(frame, cfg, weights=weights)]
            clusters = frame[cluster_col].to_dict() if cluster_col else None
            inference = u_statistic_inference(summaries, alpha=alpha, clusters=clusters)
            inference["method"] += "_fixed_weights"
            notes.append(
                "Analytic intervals treat the weights as fixed; bootstrap_propensity_weighted_gpc "
                "also propagates propensity-score estimation."
            )
            bootstrap = (
                bootstrap_weighted_gpc(
                    frame,
                    cfg,
                    weight_col=weight_col,
                    cluster_col=cluster_col,
                    within_arm=cluster_within_arm if cluster_col else True,
                    n_boot=n_boot,
                    seed=seed,
                    alpha=alpha,
                )
                if n_boot
                else None
            )
            if n_perm:
                raise ValueError("permutation tests are not available for weighted analyses")
            permutation = None
        elif pair_id is not None:
            method = "matched"
            estimate = compute_win_ratio(frame, cfg)["overall"]
            a, _, results, _, _ = _matched_results(frame, cfg)
            pair_clusters = a[cluster_col].astype(str).to_numpy() if cluster_col else None
            inference = matched_pair_inference(results, alpha=alpha, clusters=pair_clusters) if results.size else None
            bootstrap = (
                bootstrap_win_ratio_matched(frame, cfg, n_boot=n_boot, seed=seed, alpha=alpha, cluster_col=cluster_col)
                if n_boot
                else None
            )
            permutation = (
                permutation_test(frame, cfg, n_perm=n_perm, seed=seed, cluster_col=cluster_col) if n_perm else None
            )
            pair_table = pair_results(frame, cfg)
        else:
            full = compute_win_ratio(frame, cfg)
            estimate = full["overall"]
            method = "stratified" if cfg.strata else "all_pairs"
            strata_results = full.get("by_strata")
            summaries = [s for _, s in _summaries(frame, cfg)]
            clusters = frame[cluster_col].to_dict() if cluster_col else None
            inference = u_statistic_inference(summaries, alpha=alpha, clusters=clusters)
            if n_boot:
                if cluster_col is None:
                    bootstrap = bootstrap_win_ratio(frame, cfg, n_boot=n_boot, seed=seed, alpha=alpha)
                elif cluster_within_arm:
                    bootstrap = bootstrap_win_ratio_cluster_within_arm(
                        frame, cfg, cluster_col=cluster_col, n_boot=n_boot, seed=seed, alpha=alpha
                    )
                else:
                    bootstrap = bootstrap_win_ratio_cluster(
                        frame, cfg, cluster_col=cluster_col, n_boot=n_boot, seed=seed, alpha=alpha
                    )
            else:
                bootstrap = None
            permutation = (
                permutation_test(frame, cfg, n_perm=n_perm, seed=seed, cluster_col=cluster_col) if n_perm else None
            )

        if permutation is not None:
            notes.append(f"Permutation test: {permutation['permuted_units']} permuted ({permutation['method']}).")
        if estimate["wins"] == 0 or estimate["losses"] == 0:
            notes.append(
                "One arm never lost: Wald intervals are unreliable at this boundary; prefer a permutation "
                "test (n_perm) and report the counts."
            )
        if bootstrap is not None and bootstrap.get("n_undefined"):
            notes.append(f"{bootstrap['n_undefined']} bootstrap replicates had neither wins nor losses (undefined).")
        if bootstrap is not None and bootstrap.get("n_infinite"):
            notes.append(
                f"{bootstrap['n_infinite']} bootstrap replicates had no losses (WR = inf); "
                "they are kept in the percentile interval."
            )
        return WinRatioResult(
            estimate=estimate,
            config=cfg,
            bootstrap=bootstrap,
            method=method,
            inference=inference,
            permutation=permutation,
            alpha=alpha,
            pair_table=pair_table,
            strata_results=strata_results,
            notes=notes,
        )


def win_ratio(
    data: pd.DataFrame,
    outcomes: Sequence[WinRatioOutcome],
    *,
    group: str,
    treatment: Any,
    control: Any,
    strata: Optional[Iterable[str]] = None,
    pair_id: Optional[str] = None,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    cluster_within_arm: bool = False,
    n_boot: int = 0,
    n_perm: int = 0,
    alpha: float = 0.05,
    seed: int = 42,
) -> WinRatioResult:
    """Estimate win statistics in one call.

    Equivalent to ``WinRatio(outcomes, group=..., treatment=..., control=...,
    strata=...).fit(data, pair_id=..., weight_col=weights, cluster_col=cluster, ...)``.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per patient.
    outcomes : sequence of WinRatioOutcome
        Hierarchy, highest priority first.
    group, treatment, control
        Arm column and the values of the two compared arms.
    strata, pair_id, weights, cluster, cluster_within_arm, n_boot, n_perm, alpha, seed
        See :meth:`pywinratio.WinRatioAnalysis.fit`.

    Returns
    -------
    WinRatioResult

    Examples
    --------
    >>> from pywinratio import Outcome, win_ratio
    >>> from pywinratio.datasets import make_heart_failure_trial
    >>> trial = make_heart_failure_trial(seed=2012)
    >>> res = win_ratio(
    ...     trial,
    ...     [Outcome.time_to_event("time_death", "death", name="Death", terminal=True),
    ...      Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization")],
    ...     group="arm", treatment="treatment", control="placebo")
    >>> res.to_frame().columns.tolist()[:3]
    ['estimate', 'ci_lower', 'ci_upper']
    """

    analysis = WinRatioAnalysis(outcomes, group=group, treatment=treatment, control=control, strata=strata)
    return analysis.fit(
        data,
        pair_id=pair_id,
        weight_col=weights,
        cluster_col=cluster,
        cluster_within_arm=cluster_within_arm,
        n_boot=n_boot,
        n_perm=n_perm,
        alpha=alpha,
        seed=seed,
    )


WinRatio = WinRatioAnalysis
Outcome = WinRatioOutcome
