from __future__ import annotations

import pandas as pd

from pywinratio import (
    Outcome,
    WinRatioConfig,
    bootstrap_propensity_matched_win_ratio,
    bootstrap_propensity_weighted_gpc,
)


def _clustered_data() -> pd.DataFrame:
    rows = []
    for arm, prefix in [("A", "TA"), ("B", "CB")]:
        for center in range(4):
            for patient in range(4):
                rows.append(
                    {
                        "group": arm,
                        "site": f"{prefix}{center}",
                        "age": 45 + 4 * center + patient + (1 if arm == "A" else 0),
                        "sex": "F" if patient % 2 else "M",
                        "year": "late" if center % 2 else "early",
                        "event": int((center + patient + (arm == "B")) % 5 == 0),
                    }
                )
    return pd.DataFrame(rows)


def test_refit_rematch_cluster_bootstrap_runs() -> None:
    df = _clustered_data()
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        outcomes=[Outcome.binary("event")],
    )
    result = bootstrap_propensity_matched_win_ratio(
        df,
        cfg,
        covariates=["age", "sex"],
        exact_cols=["year"],
        cluster_col="site",
        caliper=2.0,
        n_boot=8,
        seed=8,
    )
    assert result["bootstrap_type"].endswith("refit_rematch")
    assert result["original_matching"].pairs > 0
    assert result["n_successful"] > 0


def test_refit_overlap_weight_cluster_bootstrap_runs() -> None:
    df = _clustered_data()
    cfg = WinRatioConfig(
        group_col="group",
        arm_a="A",
        arm_b="B",
        outcomes=[Outcome.binary("event")],
    )
    result = bootstrap_propensity_weighted_gpc(
        df,
        cfg,
        covariates=["age", "sex", "year"],
        cluster_col="site",
        n_boot=8,
        seed=9,
    )
    assert result["bootstrap_type"].endswith("weighted_gpc")
    assert result["n_successful"] > 0


def test_parallel_refit_bootstrap_reproduces_serial_results() -> None:
    df = _clustered_data()
    cfg = WinRatioConfig(group_col="group", arm_a="A", arm_b="B", outcomes=[Outcome.binary("event")])
    kwargs = dict(covariates=["age", "sex"], cluster_col="site", caliper=2.0, n_boot=6, seed=3)
    serial = bootstrap_propensity_matched_win_ratio(df, cfg, n_jobs=1, **kwargs)
    parallel = bootstrap_propensity_matched_win_ratio(df, cfg, n_jobs=2, **kwargs)
    assert serial["wr_samples"] == parallel["wr_samples"]
