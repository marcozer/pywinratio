"""Comparison engine: equality with a brute-force reference and invariances."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from reference import all_pairs_counts

from pywinratio import (
    Outcome,
    WinRatioConfig,
    WinRatioOutcome,
    compare_patients,
    compute_weighted_gpc,
    compute_win_ratio,
    pair_results,
)
from pywinratio._engine import summarize_all_pairs


def _random_data(seed: int, n: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size = 2 * n
    data = pd.DataFrame(
        {
            "arm": np.repeat(["A", "B"], n),
            "death": rng.binomial(1, 0.3, size).astype(float),
            "t_death": rng.integers(1, 12, size).astype(float),
            "e_death": rng.binomial(1, 0.5, size).astype(float),
            "grade": rng.choice([0, 1, 2, 3], size).astype(float),
            "los": rng.integers(3, 20, size).astype(float),
            "score": np.round(rng.normal(0, 1, size), 1),
            "pair": np.tile(np.arange(n), 2),
            "stratum": rng.choice(["s1", "s2"], size),
        }
    )
    for column in ("death", "grade", "los", "score", "t_death"):
        data.loc[rng.random(size) < 0.12, column] = np.nan
    return data


def _hierarchies() -> list[list[WinRatioOutcome]]:
    return [
        [Outcome.binary("death", terminal=True), Outcome.ordinal("grade"), Outcome.continuous("los", margin=1)],
        [
            Outcome.time_to_event("t_death", "e_death", terminal=True),
            Outcome.continuous("los", cutpoints=[7, 13]),
            Outcome.continuous("score", favorable="higher", threshold=0.3),
        ],
        [
            Outcome.binary("death", missing_is="loss"),
            Outcome.ordinal("grade", missing_is="win", terminal=True, terminal_value=3),
            Outcome.time_to_event("t_death", "e_death", margin=2, horizon=9, favorable="earlier"),
            Outcome.continuous("score", margin=0.2, missing_is="loss"),
        ],
    ]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("which", range(3))
def test_engine_matches_brute_force(seed: int, which: int) -> None:
    data = _random_data(seed)
    outcomes = _hierarchies()[which]
    cfg = WinRatioConfig("arm", "A", "B", outcomes)
    overall = compute_win_ratio(data, cfg)["overall"]
    wins, losses, ties, tier_wins, tier_losses = all_pairs_counts(data, "arm", "A", "B", outcomes)
    assert (overall["wins"], overall["losses"], overall["ties"]) == (wins, losses, ties)
    assert overall["details"]["tier_wins"] == tier_wins
    assert overall["details"]["tier_losses"] == tier_losses


@pytest.mark.parametrize("seed", range(6))
def test_swapping_arms_inverts_statistics(seed: int) -> None:
    data = _random_data(seed, n=20)
    outcomes = _hierarchies()[2]
    ab = compute_win_ratio(data, WinRatioConfig("arm", "A", "B", outcomes))["overall"]
    ba = compute_win_ratio(data, WinRatioConfig("arm", "B", "A", outcomes))["overall"]
    assert ab["wins"] == ba["losses"] and ab["losses"] == ba["wins"]
    assert ab["wr"] * ba["wr"] == pytest.approx(1.0)
    assert ab["net_benefit"] == pytest.approx(-ba["net_benefit"])


def test_row_order_does_not_change_estimates() -> None:
    data = _random_data(3, n=25)
    cfg = WinRatioConfig("arm", "A", "B", _hierarchies()[0])
    shuffled = data.sample(frac=1.0, random_state=1)
    assert compute_win_ratio(data, cfg)["overall"]["wins"] == compute_win_ratio(shuffled, cfg)["overall"]["wins"]


def test_matched_equals_diagonal_of_all_pairs() -> None:
    data = _random_data(5, n=18)
    outcomes = _hierarchies()[1]
    matched = compute_win_ratio(
        data, WinRatioConfig("arm", "A", "B", outcomes, pair_strategy="matched", pair_id="pair")
    )
    a = data[data["arm"] == "A"].sort_values("pair")
    b = data[data["arm"] == "B"].sort_values("pair")
    expected = [
        compare_patients(ra, rb, outcomes)["result"] for ra, rb in zip(a.to_dict("records"), b.to_dict("records"))
    ]
    assert matched["overall"]["wins"] == expected.count("win")
    assert matched["overall"]["losses"] == expected.count("loss")


def test_stratified_totals_are_sums_of_strata() -> None:
    data = _random_data(8, n=30)
    cfg = WinRatioConfig("arm", "A", "B", _hierarchies()[0], strata=["stratum"])
    result = compute_win_ratio(data, cfg)
    assert result["overall"]["wins"] == sum(r["wins"] for r in result["by_strata"])
    assert result["overall"]["losses"] == sum(r["losses"] for r in result["by_strata"])


def test_unit_weights_reproduce_unweighted_counts() -> None:
    data = _random_data(9, n=20).assign(w=1.0)
    cfg = WinRatioConfig("arm", "A", "B", _hierarchies()[1])
    plain = compute_win_ratio(data, cfg)["overall"]
    weighted = compute_weighted_gpc(data, cfg, weight_col="w")
    assert weighted["wins"] == plain["wins"] and weighted["losses"] == plain["losses"]


def test_blocked_evaluation_equals_single_block() -> None:
    data = _random_data(4, n=40)
    outcomes = _hierarchies()[2]
    a, b = data[data["arm"] == "A"], data[data["arm"] == "B"]
    small = summarize_all_pairs(a, b, outcomes, max_cells=7)
    large = summarize_all_pairs(a, b, outcomes)
    assert small.wins == large.wins and small.losses == large.losses
    np.testing.assert_array_equal(small.a_wins, large.a_wins)
    np.testing.assert_array_equal(small.b_losses, large.b_losses)


# --- documented semantics ---------------------------------------------------


def test_missing_value_policy_is_about_the_patient_with_missing_data() -> None:
    tiers = [Outcome.binary("x", missing_is="loss")]
    assert compare_patients({"x": np.nan}, {"x": 1}, tiers)["result"] == "loss"
    assert compare_patients({"x": 1}, {"x": np.nan}, tiers)["result"] == "win"
    assert compare_patients({"x": np.nan}, {"x": np.nan}, tiers)["result"] == "tie"
    tiers = [Outcome.binary("x", missing_is="win")]
    assert compare_patients({"x": 0}, {"x": np.nan}, tiers)["result"] == "loss"


def test_margin_and_threshold_conventions() -> None:
    margin = [Outcome.continuous("los", margin=1)]
    threshold = [Outcome.continuous("los", threshold=2)]
    for tiers in (margin, threshold):
        assert compare_patients({"los": 6}, {"los": 7}, tiers)["result"] == "tie"
        assert compare_patients({"los": 5}, {"los": 7}, tiers)["result"] == "win"
    # float noise does not create spurious wins at the margin
    assert (
        compare_patients({"s": 1.1}, {"s": 1.0}, [Outcome.continuous("s", favorable="higher", margin=0.1)])["result"]
        == "tie"
    )


def test_gehan_rule_with_censoring() -> None:
    tiers = [Outcome.time_to_event("t", "e")]
    # B died at 3, A censored at 5: A known to survive longer
    assert compare_patients({"t": 5, "e": 0}, {"t": 3, "e": 1}, tiers)["result"] == "win"
    # A died at 5, B censored at 3: cannot tell
    assert compare_patients({"t": 5, "e": 1}, {"t": 3, "e": 0}, tiers)["result"] == "tie"
    # censoring at the event time counts as event-free (Gehan 1965)
    assert compare_patients({"t": 4, "e": 0}, {"t": 4, "e": 1}, tiers)["result"] == "win"
    # horizon: events after the horizon are ignored
    tiers = [Outcome.time_to_event("t", "e", horizon=4)]
    assert compare_patients({"t": 6, "e": 1}, {"t": 9, "e": 1}, tiers)["result"] == "tie"


def test_terminal_events_stop_the_comparison() -> None:
    tiers = [Outcome.binary("death", terminal=True), Outcome.continuous("los")]
    explained = compare_patients({"death": 1, "los": 3}, {"death": 1, "los": 30}, tiers)
    assert explained == {"result": "tie", "deciding_tier": None, "stopped_by": "death"}
    tte = [Outcome.time_to_event("t", "e", terminal=True), Outcome.continuous("los")]
    # both died: the later death wins, no need to look at LOS
    assert compare_patients({"t": 5, "e": 1, "los": 30}, {"t": 2, "e": 1, "los": 3}, tte)["result"] == "win"
    assert compare_patients({"t": 2, "e": 1, "los": 3}, {"t": 2, "e": 1, "los": 30}, tte)["stopped_by"] == "t"


def test_pair_results_table() -> None:
    data = _random_data(1, n=10)
    cfg = WinRatioConfig("arm", "A", "B", _hierarchies()[0], pair_strategy="matched", pair_id="pair")
    table = pair_results(data, cfg)
    assert len(table) == 10
    assert set(table["result"]) <= {"win", "loss", "tie"}
    decided = table[table["result"] != "tie"]
    assert decided["deciding_tier"].notna().all()


# --- validation -------------------------------------------------------------


def test_invalid_inputs_raise_clear_errors() -> None:
    df = pd.DataFrame({"arm": ["A", "B"], "x": [0, 2]})
    with pytest.raises(ValueError, match="coded 0/1"):
        compute_win_ratio(df, WinRatioConfig("arm", "A", "B", [Outcome.binary("x")]))
    df = pd.DataFrame({"arm": ["A", "B"], "x": ["yes", "no"]})
    with pytest.raises(ValueError, match="non-numeric"):
        compute_win_ratio(df, WinRatioConfig("arm", "A", "B", [Outcome.binary("x")]))
    with pytest.raises(ValueError, match="must differ"):
        WinRatioConfig("arm", "A", "A", [Outcome.binary("x")])
    with pytest.raises(ValueError, match="pair_strategy"):
        WinRatioConfig("arm", "A", "B", [Outcome.binary("x")], pair_strategy="match")
    with pytest.raises(ValueError, match="tie_tol"):
        WinRatioOutcome("x", "x", "binary", "lower", tie_tol=1)
    with pytest.raises(ValueError, match="event_column"):
        WinRatioOutcome("t", "t", "time_to_event", "higher")
    df = pd.DataFrame({"arm": ["A", "B"], "x": [0, 1], "s": [None, "a"]})
    with pytest.raises(ValueError, match="strata"):
        compute_win_ratio(df, WinRatioConfig("arm", "A", "B", [Outcome.binary("x")], strata=["s"]))
    with pytest.raises(ValueError, match="both configured arms"):
        compute_win_ratio(df, WinRatioConfig("arm", "A", "C", [Outcome.binary("x")]))


def test_incomplete_pairs_are_reported() -> None:
    df = pd.DataFrame({"arm": ["A", "B", "A", "A"], "pair": [1, 1, 2, 2], "x": [0, 1, 0, 1]})
    cfg = WinRatioConfig("arm", "A", "B", [Outcome.binary("x")], pair_strategy="matched", pair_id="pair")
    with pytest.warns(UserWarning, match="excluded"):
        overall = compute_win_ratio(df, cfg)["overall"]
    assert overall["details"]["pairs"] == 1
    assert overall["details"]["pairs_excluded"] == 1
