from __future__ import annotations

import pandas as pd

from pywinratio import (
    WinRatioConfig,
    WinRatioOutcome,
    bootstrap_win_ratio,
    bootstrap_win_ratio_cluster,
    bootstrap_win_ratio_matched,
    paired_risk_difference_bootstrap,
)


def test_subject_bootstrap_returns_finite_ci() -> None:
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "site_id": ["S1", "S1", "S2", "S1", "S2", "S2"],
            "mort90": [0, 1, 0, 1, 0, 0],
        }
    )
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        outcomes=[WinRatioOutcome("Mortality", "mort90", "binary", "lower")],
    )
    result = bootstrap_win_ratio(df, cfg, n_boot=100, seed=1)
    assert result["n_boot"] == 100
    assert len(result["wr_samples"]) > 0


def test_cluster_bootstrap_uses_cluster_column() -> None:
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "site_id": ["S1", "S2", "S1", "S2"],
            "mort90": [0, 0, 1, 0],
        }
    )
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        outcomes=[WinRatioOutcome("Mortality", "mort90", "binary", "lower")],
    )
    result = bootstrap_win_ratio_cluster(df, cfg, cluster_col="site_id", n_boot=50, seed=2)
    assert result["cluster_col"] == "site_id"


def test_matched_bootstrap_runs_on_pair_identifier() -> None:
    df = pd.DataFrame(
        {
            "group": ["A", "B", "A", "B"],
            "PAIR_ID": [1, 1, 2, 2],
            "mort90": [0, 1, 0, 0],
        }
    )
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        pair_strategy="matched",
        pair_id="PAIR_ID",
        outcomes=[WinRatioOutcome("Mortality", "mort90", "binary", "lower")],
    )
    result = bootstrap_win_ratio_matched(df, cfg, n_boot=50, seed=3)
    assert result["n_boot"] == 50


def test_paired_risk_difference_bootstrap_includes_concordant_pairs() -> None:
    df = pd.DataFrame(
        {
            "group": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "PAIR_ID": [1, 1, 2, 2, 3, 3, 4, 4],
            "mort90": [1, 0, 0, 1, 1, 1, 0, 0],
        }
    )
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        pair_strategy="matched",
        pair_id="PAIR_ID",
        outcomes=[WinRatioOutcome("Mortality", "mort90", "binary", "lower")],
    )
    result = paired_risk_difference_bootstrap(df, cfg, n_boot=50, seed=4)
    row = result.iloc[0]
    assert row["n_pairs_complete"] == 4
    assert row["both_event"] == 1
    assert row["neither_event"] == 1
    assert row["A_event_B_no_event"] == 1
    assert row["A_no_event_B_event"] == 1
    assert row["risk_difference_A_minus_B"] == 0.0
