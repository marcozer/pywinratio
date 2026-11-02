"""Release metadata stays consistent across files."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import pywinratio

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(not (ROOT / "CITATION.cff").exists(), reason="repository metadata not available")


def test_versions_and_dates_agree() -> None:
    version = pywinratio.__version__
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"^version: {re.escape(version)}$", citation, flags=re.M)
    assert zenodo["version"] == version
    date = re.search(r'^date-released: "(\d{4}-\d{2}-\d{2})"$', citation, flags=re.M).group(1)
    assert zenodo["publication_date"] == date
    assert re.search(rf"^## {re.escape(version)} \({date}\)$", changelog, flags=re.M)


def test_public_api_is_documented() -> None:
    api = (ROOT / "docs" / "api" / "index.md").read_text(encoding="utf-8")
    aliases = {"WinRatio", "Outcome", "__version__"}
    missing = [name for name in pywinratio.__all__ if name not in aliases and f"   {name}\n" not in api]
    assert not missing, f"add to docs/api/index.md: {missing}"
