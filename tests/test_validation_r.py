"""Agreement with the R packages BuyseTest and WINS (reference values committed).

The reference values in ``validation/reference/r_reference.json`` were
produced by ``validation/reference_r.R``; R is not needed to run this test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"

pytestmark = pytest.mark.skipif(
    not (VALIDATION / "reference" / "r_reference.json").exists(),
    reason="validation reference files not available (e.g. installed from a wheel)",
)


def test_pywinratio_agrees_with_buysetest_and_wins() -> None:
    sys.path.insert(0, str(VALIDATION))
    try:
        import compare
    finally:
        sys.path.pop(0)
    assert compare.main() == 0
