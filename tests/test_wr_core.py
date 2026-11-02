from __future__ import annotations

import pandas as pd

from pywinratio import (
    MultiArmWinRatioConfig,
    Outcome,
    WinRatio,
    WinRatioConfig,
    WinRatioOutcome,
    compute_weighted_gpc,
    compute_win_ratio,
    compute_win_ratio_multi_arm,
)


def test_hierarchical_resolution_prefers_higher_priority_outcome() -> None:
    df = pd.DataFrame(
        {
            "group": ["A", "B"],
            "mort90": [0, 0],
            "major_comp": [0, 1],
            "los_days": [12, 8],
        }
    )
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        outcomes=[
            WinRatioOutcome("Mortality", "mort90", "binary", "lower"),
            WinRatioOutcome("Major complication", "major_comp", "binary", "lower"),
            WinRatioOutcome("Length of stay", "los_days", "continuous", "lower"),
        ],
    )
    overall = compute_win_ratio(df, cfg)["overall"]
    assert overall["wins"] == 1
    assert overall["losses"] == 0


def test_missing_value_policy_can_force_loss() -> None:
    df = pd.DataFrame({"group": ["A", "B"], "endpoint": [None, 0]})
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        outcomes=[WinRatioOutcome("Endpoint", "endpoint", "binary", "lower", missing_is="loss")],
    )
    overall = compute_win_ratio(df, cfg)["overall"]
    assert overall["losses"] == 1


def test_los_tolerance_mode_respects_one_day_ties() -> None:
    df = pd.DataFrame({"group": ["A", "B"], "los": [8, 7.4]})
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        outcomes=[
            WinRatioOutcome(
                "LOS",
                "los",
                "continuous",
                "lower",
                los_tolerance_mode="tolerance_1day",
            )
        ],
    )
    overall = compute_win_ratio(df, cfg)["overall"]
    assert overall["ties"] == 1


def test_multi_arm_returns_all_pairwise_comparisons() -> None:
    df = pd.DataFrame(
        {
            "arm": ["A", "B", "C"],
            "mort90": [0, 1, 0],
        }
    )
    cfg = MultiArmWinRatioConfig(
        group_col="arm",
        arms=["A", "B", "C"],
        outcomes=[WinRatioOutcome("Mortality", "mort90", "binary", "lower")],
    )
    result = compute_win_ratio_multi_arm(df, cfg)
    assert len(result["comparisons"]) == 3


def test_shared_terminal_death_stops_lower_tiers() -> None:
    df = pd.DataFrame(
        {
            "group": ["A", "B"],
            "mort90": [1, 1],
            "major": [0, 1],
            "los": [4, 30],
        }
    )
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        outcomes=[
            Outcome.binary("mort90", name="Mortality", terminal=True),
            Outcome.binary("major", name="Major complications"),
            Outcome.continuous("los", name="LOS"),
        ],
    )
    overall = compute_win_ratio(df, cfg)["overall"]
    assert overall["wins"] == 0
    assert overall["losses"] == 0
    assert overall["ties"] == 1


def test_ordinal_severity_and_los_margin() -> None:
    df = pd.DataFrame({"group": ["A", "B"], "grade": [3, 4], "los": [7, 8]})
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        outcomes=[
            Outcome.ordinal("grade", name="Clavien severity"),
            Outcome.continuous("los", name="LOS", margin=1),
        ],
    )
    overall = compute_win_ratio(df, cfg)["overall"]
    assert overall["wins"] == 1
    assert overall["details"]["tier_wins"] == [1, 0]


def test_categorical_los_uses_seven_and_thirteen_day_boundaries() -> None:
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        outcomes=[Outcome.ordinal("los", cutpoints=[7, 13])],
    )
    first = compute_win_ratio(pd.DataFrame({"group": ["A", "B"], "los": [7, 8]}), cfg)["overall"]
    middle = compute_win_ratio(pd.DataFrame({"group": ["A", "B"], "los": [8, 13]}), cfg)["overall"]
    last = compute_win_ratio(pd.DataFrame({"group": ["A", "B"], "los": [13, 14]}), cfg)["overall"]
    assert first["wins"] == 1
    assert middle["ties"] == 1
    assert last["wins"] == 1


def test_unit_weighted_gpc_matches_unweighted_all_pairs() -> None:
    df = pd.DataFrame({"group": ["A", "A", "B", "B"], "event": [0, 1, 1, 1], "weight": 1.0})
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        outcomes=[Outcome.binary("event")],
    )
    ordinary = compute_win_ratio(df, cfg)["overall"]
    weighted = compute_weighted_gpc(df, cfg, weight_col="weight")
    assert weighted["wins"] == ordinary["wins"]
    assert weighted["losses"] == ordinary["losses"]
    assert weighted["ties"] == ordinary["ties"]


def test_high_level_api_returns_tier_table() -> None:
    df = pd.DataFrame({"group": ["A", "B"], "event": [0, 1]})
    result = WinRatio(
        group="group",
        arm_a="A",
        arm_b="B",
        outcomes=[Outcome.binary("event", name="Event")],
    ).fit(df)
    assert result.summary()["wr"] == float("inf")
    assert result.tiers().iloc[0]["outcome"] == "Event"
