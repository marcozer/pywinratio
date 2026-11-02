"""Outcome tiers and analysis configurations.

A hierarchy is an ordered list of :class:`WinRatioOutcome` objects (aliased
``Outcome``). Each tier says which column to read, which values are
favorable, how close two values must be to tie, how missing values are
handled, and whether a shared event is terminal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

Direction = Literal["higher", "lower"]
OutcomeKind = Literal["binary", "ordinal", "continuous", "time_to_event"]
LostToleranceMode = Literal["exact", "tolerance_1day", "threshold_based"]
MissingValuePolicy = Literal["tie", "loss", "win"]
PairStrategy = Literal["all_pairs", "matched"]

_KINDS = {"binary", "ordinal", "continuous", "time_to_event"}
_TIME_DIRECTIONS = {"later": "higher", "earlier": "lower", "higher": "higher", "lower": "lower"}


@dataclass
class WinRatioOutcome:
    """One prioritized outcome (tier) of a hierarchical comparison.

    Most users build tiers with the factory methods :meth:`binary`,
    :meth:`ordinal`, :meth:`continuous` and :meth:`time_to_event` rather than
    calling the constructor directly.

    Parameters
    ----------
    name : str
        Label used in tables and plots.
    column : str
        Data column holding the outcome (the time column for a
        time-to-event tier).
    kind : {"binary", "ordinal", "continuous", "time_to_event"}
        Outcome type. Binary outcomes must be coded 0/1 with 1 = event.
    direction : {"higher", "lower"}
        Which values are favorable. For binary adverse events use
        ``"lower"`` (no event beats event). For a time-to-event tier,
        ``"higher"`` means a later event is better (adverse event) and
        ``"lower"`` means an earlier event is better (e.g. recovery).
    tie_tol : float, default 0
        Clinical tie margin: absolute differences ``<= tie_tol`` are ties.
        For completed-day length of stay, ``tie_tol=1`` ties differences of
        0 or 1 day, so a win needs at least a 2-day difference.
    missing_is : {"tie", "loss", "win"}, default "tie"
        Missing-value policy for this tier. ``"tie"`` passes the pair to the
        next tier; ``"loss"`` makes the patient whose value is missing lose
        the tier (worst case); ``"win"`` makes that patient win it. A pair
        with both values missing always ties.
    los_tolerance_mode : {"exact", "tolerance_1day", "threshold_based"}
        Legacy length-of-stay modes kept for backward compatibility. Prefer
        ``tie_tol`` (``tolerance_1day`` equals ``tie_tol=1``) and
        ``cutpoints`` (``threshold_based`` equals one cut point).
    los_threshold : float, optional
        Threshold for the legacy ``threshold_based`` mode.
    cutpoints : sequence of float, optional
        Ascending boundaries that categorize values before comparison.
        Values ``<= c1`` form the first category, ``(c1, c2]`` the second,
        and so on. ``cutpoints=[7, 13]`` gives ``<=7``, ``8-13`` and ``>13``
        days for integer stays.
    terminal : bool, default False
        If True and both patients experienced the terminal event without the
        tier separating them, the pair is a tie and lower tiers are **not**
        evaluated (two deaths are not ranked by later complications).
    terminal_value : any, default 1
        Value that denotes the terminal event for non time-to-event tiers.
    event_column : str, optional
        Event indicator (1 = event, 0 = censored) for time-to-event tiers.
    horizon : float, optional
        Truncation time for time-to-event tiers: events after ``horizon``
        are ignored and follow-up is censored there.
    tie_inclusive : bool, default True
        ``True``: differences ``<= tie_tol`` tie (margin convention).
        ``False``: differences ``< tie_tol`` tie, i.e. a win needs a
        difference of at least ``tie_tol`` (threshold convention of the R
        package BuyseTest). Set through the ``threshold`` argument of the
        factories.

    Examples
    --------
    >>> from pywinratio import Outcome
    >>> Outcome.binary("death90", name="90-day mortality", terminal=True).describe()
    '90-day mortality: binary, no event is better, terminal'
    """

    name: str
    column: str
    kind: OutcomeKind
    direction: Direction
    tie_tol: float = 0.0
    missing_is: MissingValuePolicy = "tie"
    los_tolerance_mode: LostToleranceMode = "exact"
    los_threshold: Optional[float] = None
    cutpoints: Optional[Tuple[float, ...]] = None
    terminal: bool = False
    terminal_value: Any = 1
    event_column: Optional[str] = None
    horizon: Optional[float] = None
    tie_inclusive: bool = True

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError("kind must be 'binary', 'ordinal', 'continuous', or 'time_to_event'")
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'")
        if self.missing_is not in {"tie", "loss", "win"}:
            raise ValueError("missing_is must be 'tie', 'loss', or 'win'")
        if self.los_tolerance_mode not in {"exact", "tolerance_1day", "threshold_based"}:
            raise ValueError("los_tolerance_mode must be 'exact', 'tolerance_1day', or 'threshold_based'")
        if self.tie_tol is None or not math.isfinite(float(self.tie_tol)) or self.tie_tol < 0:
            raise ValueError("tie_tol must be a finite nonnegative number")
        self.tie_tol = float(self.tie_tol)
        if not self.tie_inclusive and self.tie_tol <= 0:
            raise ValueError("a threshold (tie_inclusive=False) must be positive")
        if self.cutpoints is not None:
            self.cutpoints = tuple(float(value) for value in self.cutpoints)
            if tuple(sorted(self.cutpoints)) != self.cutpoints:
                raise ValueError("cutpoints must be sorted in ascending order")
            if len(set(self.cutpoints)) != len(self.cutpoints):
                raise ValueError("cutpoints must be unique")
            if self.kind in {"binary", "time_to_event"}:
                raise ValueError("cutpoints are only valid for ordinal or continuous outcomes")
        if self.kind == "binary" and self.tie_tol:
            raise ValueError("tie_tol has no meaning for a binary outcome")
        if self.kind == "time_to_event":
            if not self.event_column:
                raise ValueError("time_to_event outcomes require event_column")
            if self.horizon is not None and not float(self.horizon) > 0:
                raise ValueError("horizon must be positive")
        elif self.event_column is not None or self.horizon is not None:
            raise ValueError("event_column and horizon are only valid for time_to_event outcomes")

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def binary(
        cls,
        column: str,
        *,
        name: Optional[str] = None,
        favorable: Direction = "lower",
        terminal: bool = False,
        terminal_value: Any = 1,
        missing_is: MissingValuePolicy = "tie",
    ) -> WinRatioOutcome:
        """Binary tier coded 0/1 with 1 = event.

        Parameters
        ----------
        column : str
            0/1 column.
        name : str, optional
            Display label (defaults to ``column``).
        favorable : {"lower", "higher"}, default "lower"
            ``"lower"`` for adverse events (0 beats 1); ``"higher"`` for
            desirable events such as textbook outcome (1 beats 0).
        terminal : bool, default False
            Stop the comparison as a tie when both patients have the event.
        terminal_value : any, default 1
            Value denoting the terminal event.
        missing_is : {"tie", "loss", "win"}, default "tie"
            Missing-value policy; see :class:`WinRatioOutcome`.
        """

        return cls(
            name=name or column,
            column=column,
            kind="binary",
            direction=favorable,
            missing_is=missing_is,
            terminal=terminal,
            terminal_value=terminal_value,
        )

    @classmethod
    def ordinal(
        cls,
        column: str,
        *,
        name: Optional[str] = None,
        favorable: Direction = "lower",
        cutpoints: Optional[Sequence[float]] = None,
        terminal: bool = False,
        terminal_value: Any = None,
        missing_is: MissingValuePolicy = "tie",
    ) -> WinRatioOutcome:
        """Ordered categorical tier such as a complication grade.

        Any strictly ordered numeric coding works (e.g. Clavien-Dindo
        0/3/4, fistula none/B/C coded 0/1/2): only the order matters.

        Parameters
        ----------
        column : str
            Numeric ordinal column.
        name : str, optional
            Display label.
        favorable : {"lower", "higher"}, default "lower"
            Direction of the better grades.
        cutpoints : sequence of float, optional
            Collapse grades into categories before comparison.
        terminal : bool, default False
            Stop as a tie when both patients have ``terminal_value``.
        terminal_value : any, optional
            Grade denoting a terminal event (e.g. Clavien-Dindo grade V).
        missing_is : {"tie", "loss", "win"}, default "tie"
            Missing-value policy.
        """

        if terminal and terminal_value is None:
            raise ValueError("terminal ordinal tiers need terminal_value")
        return cls(
            name=name or column,
            column=column,
            kind="ordinal",
            direction=favorable,
            cutpoints=tuple(cutpoints) if cutpoints is not None else None,
            terminal=terminal,
            terminal_value=terminal_value if terminal_value is not None else 1,
            missing_is=missing_is,
        )

    @classmethod
    def continuous(
        cls,
        column: str,
        *,
        name: Optional[str] = None,
        favorable: Direction = "lower",
        margin: float = 0.0,
        threshold: Optional[float] = None,
        cutpoints: Optional[Sequence[float]] = None,
        missing_is: MissingValuePolicy = "tie",
    ) -> WinRatioOutcome:
        """Continuous tier with an optional clinical tie margin.

        Parameters
        ----------
        column : str
            Numeric column (e.g. length of stay, quality-of-life score).
        name : str, optional
            Display label.
        favorable : {"lower", "higher"}, default "lower"
            ``"lower"`` for stays or costs, ``"higher"`` for scores.
        margin : float, default 0
            Absolute differences ``<= margin`` are ties: with completed days,
            ``margin=1`` needs a difference of at least 2 days to win.
        threshold : float, optional
            Alternative, BuyseTest-compatible convention: a win needs an
            absolute difference ``>= threshold`` (differences ``< threshold``
            tie). Use either ``margin`` or ``threshold``.
        cutpoints : sequence of float, optional
            Categorize values before comparison (the margin then applies to
            category indices).
        missing_is : {"tie", "loss", "win"}, default "tie"
            Missing-value policy.
        """

        tol, inclusive = _margin_or_threshold(margin, threshold)
        return cls(
            name=name or column,
            column=column,
            kind="continuous",
            direction=favorable,
            tie_tol=tol,
            tie_inclusive=inclusive,
            cutpoints=tuple(cutpoints) if cutpoints is not None else None,
            missing_is=missing_is,
        )

    @classmethod
    def time_to_event(
        cls,
        time: str,
        event: str,
        *,
        name: Optional[str] = None,
        favorable: str = "later",
        margin: float = 0.0,
        threshold: Optional[float] = None,
        horizon: Optional[float] = None,
        terminal: bool = False,
        missing_is: MissingValuePolicy = "tie",
    ) -> WinRatioOutcome:
        """Right-censored time-to-event tier (Gehan scoring rule).

        A pair is resolved only when the earlier time is an observed event,
        that is when the comparison can be made within both patients'
        follow-up (Pocock et al., 2012). Otherwise the tier is tied and the
        next tier is evaluated.

        Parameters
        ----------
        time : str
            Follow-up time column (time to event or to censoring).
        event : str
            Event indicator column, 1 = event, 0 = censored.
        name : str, optional
            Display label.
        favorable : {"later", "earlier"}, default "later"
            ``"later"`` for adverse events such as death or hospitalization;
            ``"earlier"`` for desirable events such as recovery or discharge.
        margin : float, default 0
            Event times closer than or equal to ``margin`` are ties.
        threshold : float, optional
            BuyseTest-compatible alternative: a win needs a time difference
            ``>= threshold``.
        horizon : float, optional
            Truncate follow-up at this time.
        terminal : bool, default False
            When both patients had the (terminal) event without being
            separated, stop as a tie and skip lower tiers.
        missing_is : {"tie", "loss", "win"}, default "tie"
            Missing-value policy.
        """

        if favorable not in _TIME_DIRECTIONS:
            raise ValueError("favorable must be 'later' or 'earlier'")
        tol, inclusive = _margin_or_threshold(margin, threshold)
        return cls(
            name=name or time,
            column=time,
            kind="time_to_event",
            direction=_TIME_DIRECTIONS[favorable],  # type: ignore[arg-type]
            tie_tol=tol,
            tie_inclusive=inclusive,
            event_column=event,
            horizon=horizon,
            terminal=terminal,
            missing_is=missing_is,
        )

    def describe(self) -> str:
        """One-line human-readable description of the tier rule."""

        if self.kind == "binary":
            better = "no event" if self.direction == "lower" else "event"
            text = f"{self.name}: binary, {better} is better"
        elif self.kind == "time_to_event":
            better = "later" if self.direction == "higher" else "earlier"
            text = f"{self.name}: time to event ({self.event_column}), {better} is better (Gehan rule)"
            if self.horizon is not None:
                text += f", truncated at {self.horizon:g}"
        else:
            better = "lower" if self.direction == "lower" else "higher"
            text = f"{self.name}: {self.kind}, {better} is better"
            if self.cutpoints is not None:
                text += f", categorized at {list(self.cutpoints)}"
        if self.tie_tol:
            sign = "<=" if self.tie_inclusive else "<"
            text += f", differences {sign} {self.tie_tol:g} tie"
        if self.los_tolerance_mode != "exact":
            text += f", legacy mode {self.los_tolerance_mode}"
        if self.terminal:
            text += ", terminal"
        if self.missing_is != "tie":
            text += f", missing value = {self.missing_is}"
        return text


def _margin_or_threshold(margin: float, threshold: Optional[float]) -> Tuple[float, bool]:
    if threshold is None:
        return float(margin), True
    if margin:
        raise ValueError("pass either margin or threshold, not both")
    if not threshold > 0:
        raise ValueError("threshold must be positive")
    return float(threshold), False


@dataclass
class WinRatioConfig:
    """Configuration of a two-arm win-ratio analysis.

    Parameters
    ----------
    group_col : str
        Column holding the arm of each patient.
    arm_a, arm_b : any
        Values of ``group_col`` for the two arms. Win statistics are reported
        from arm A's perspective: WR > 1 favors arm A.
    outcomes : list of WinRatioOutcome
        Tiers in decreasing priority.
    strata : list of str, optional
        Columns defining strata; patients are only compared within strata
        and stratum-level counts are summed (stratified win ratio).
    id_col : str, optional
        Patient identifier, carried for reporting.
    pair_strategy : {"all_pairs", "matched"}, default "all_pairs"
        Compare every cross-arm pair, or only patients sharing ``pair_id``.
    pair_id : str, optional
        Matched-pair identifier for ``pair_strategy="matched"``.
    """

    group_col: str
    arm_a: Any
    arm_b: Any
    outcomes: List[WinRatioOutcome] = field(default_factory=list)
    strata: Optional[List[str]] = None
    id_col: Optional[str] = None
    pair_strategy: PairStrategy = "all_pairs"
    pair_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.pair_strategy not in {"all_pairs", "matched"}:
            raise ValueError("pair_strategy must be 'all_pairs' or 'matched'")
        if self.arm_a == self.arm_b:
            raise ValueError("arm_a and arm_b must differ")
        if self.strata is not None:
            self.strata = list(self.strata)


@dataclass
class MultiArmWinRatioConfig:
    """Configuration for pairwise win-ratio comparisons across several arms.

    Parameters
    ----------
    group_col : str
        Arm column.
    arms : list
        Arm values to compare.
    arm_labels : dict, optional
        Display labels by arm value.
    reference_arm : any, optional
        Reference arm. When given and ``comparison_pairs`` is not, each other
        arm is compared with the reference (other arm as arm A).
    outcomes, strata, id_col, pair_strategy, pair_id
        As in :class:`WinRatioConfig`.
    comparison_pairs : list of (arm_a, arm_b), optional
        Explicit comparisons; defaults to all pairwise combinations.
    """

    group_col: str
    arms: List[Any]
    arm_labels: Dict[Any, str] = field(default_factory=dict)
    reference_arm: Optional[Any] = None
    outcomes: List[WinRatioOutcome] = field(default_factory=list)
    strata: Optional[List[str]] = None
    id_col: Optional[str] = None
    pair_strategy: PairStrategy = "all_pairs"
    pair_id: Optional[str] = None
    comparison_pairs: Optional[List[Tuple[Any, Any]]] = None


_OUTCOME_KEYS = {
    "name",
    "column",
    "kind",
    "direction",
    "tie_tol",
    "missing_is",
    "los_tolerance_mode",
    "los_threshold",
    "cutpoints",
    "terminal",
    "terminal_value",
    "event_column",
    "horizon",
    "tie_inclusive",
}


def _parse_outcome(values: Dict[str, Any]) -> WinRatioOutcome:
    unknown = set(values) - _OUTCOME_KEYS
    if unknown:
        raise ValueError(f"unknown outcome keys: {sorted(unknown)}")
    return WinRatioOutcome(
        name=values.get("name", values["column"]),
        column=values["column"],
        kind=values["kind"],
        direction=values["direction"],
        tie_tol=values.get("tie_tol", 0.0),
        missing_is=values.get("missing_is", "tie"),
        los_tolerance_mode=values.get("los_tolerance_mode", "exact"),
        los_threshold=values.get("los_threshold"),
        cutpoints=tuple(values["cutpoints"]) if values.get("cutpoints") is not None else None,
        terminal=bool(values.get("terminal", False)),
        terminal_value=values.get("terminal_value", 1),
        event_column=values.get("event_column"),
        horizon=values.get("horizon"),
        tie_inclusive=bool(values.get("tie_inclusive", True)),
    )


def config_from_dict(values: Dict[str, Any]) -> WinRatioConfig:
    """Create a :class:`WinRatioConfig` from a mapping (e.g. parsed YAML).

    Parameters
    ----------
    values : dict
        Keys ``group_col``, ``arm_a``, ``arm_b``, ``outcomes`` (list of
        mappings with the fields of :class:`WinRatioOutcome`) and optionally
        ``strata``, ``id_col``, ``pair_strategy``, ``pair_id``. Unknown outcome
        keys raise an error.

    Returns
    -------
    WinRatioConfig

    Examples
    --------
    >>> cfg = config_from_dict({
    ...     "group_col": "arm", "arm_a": "A", "arm_b": "B",
    ...     "outcomes": [{"name": "Death", "column": "death", "kind": "binary", "direction": "lower"}],
    ... })
    >>> cfg.outcomes[0].name
    'Death'
    """

    return WinRatioConfig(
        group_col=values["group_col"],
        arm_a=values["arm_a"],
        arm_b=values["arm_b"],
        outcomes=[_parse_outcome(item) for item in values.get("outcomes", [])],
        strata=values.get("strata"),
        id_col=values.get("id_col"),
        pair_strategy=values.get("pair_strategy", "all_pairs"),
        pair_id=values.get("pair_id"),
    )


def multi_arm_config_from_dict(values: Dict[str, Any]) -> MultiArmWinRatioConfig:
    """Create a :class:`MultiArmWinRatioConfig` from a mapping.

    Parameters
    ----------
    values : dict
        Keys ``group_col``, ``arms``, ``outcomes`` and optionally
        ``arm_labels``, ``reference_arm``, ``comparison_pairs``, ``strata``,
        ``id_col``, ``pair_strategy`` and ``pair_id``.

    Returns
    -------
    MultiArmWinRatioConfig
    """

    comparison_pairs = None
    if values.get("comparison_pairs"):
        comparison_pairs = [tuple(item) for item in values["comparison_pairs"]]

    return MultiArmWinRatioConfig(
        group_col=values["group_col"],
        arms=list(values["arms"]),
        arm_labels=dict(values.get("arm_labels", {})),
        reference_arm=values.get("reference_arm"),
        outcomes=[_parse_outcome(item) for item in values.get("outcomes", [])],
        strata=values.get("strata"),
        id_col=values.get("id_col"),
        pair_strategy=values.get("pair_strategy", "all_pairs"),
        pair_id=values.get("pair_id"),
        comparison_pairs=comparison_pairs,
    )
