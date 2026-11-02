from __future__ import annotations

import math

import pytest

from pywinratio import propensity_overlap_coefficient


def test_propensity_overlap_is_one_for_identical_empirical_distributions() -> None:
    scores = [0.1, 0.3, 0.7, 0.9, 0.1, 0.3, 0.7, 0.9]
    groups = ["A"] * 4 + ["B"] * 4
    assert propensity_overlap_coefficient(
        scores,
        groups,
        treated="A",
        bins=10,
    ) == pytest.approx(1.0)


def test_propensity_overlap_is_zero_for_separated_distributions() -> None:
    scores = [0.05, 0.10, 0.90, 0.95]
    groups = ["A", "A", "B", "B"]
    assert propensity_overlap_coefficient(
        scores,
        groups,
        treated="A",
        bins=10,
    ) == pytest.approx(0.0)


def test_propensity_overlap_requires_both_arms() -> None:
    result = propensity_overlap_coefficient(
        [0.2, 0.4],
        ["A", "A"],
        treated="A",
    )
    assert math.isnan(result)


def test_propensity_overlap_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        propensity_overlap_coefficient(
            [0.2, 0.4],
            ["A"],
            treated="A",
        )


def test_propensity_overlap_rejects_nonbinary_treatment() -> None:
    with pytest.raises(ValueError, match="at most 2"):
        propensity_overlap_coefficient(
            [0.2, 0.4, 0.6],
            ["A", "B", "C"],
            treated="A",
        )


def test_propensity_overlap_excludes_missing_group_labels() -> None:
    result = propensity_overlap_coefficient(
        [0.2, 0.2, 0.8],
        ["A", "B", None],
        treated="A",
        bins=10,
    )
    assert result == pytest.approx(1.0)
