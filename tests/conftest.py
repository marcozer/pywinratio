from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = Path(__file__).resolve().parent

for path in (TESTS, SRC, ROOT):
    as_text = str(path)
    if as_text not in sys.path:
        sys.path.insert(0, as_text)

collect_ignore = []
if not (ROOT / "studies").exists():
    collect_ignore.append("test_public_study.py")
