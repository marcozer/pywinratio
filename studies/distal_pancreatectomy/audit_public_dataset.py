from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .preprocess import audit_public_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a public dataset against disclosure rules.")
    parser.add_argument("csv", type=Path, help="CSV file to audit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    is_safe, violations = audit_public_dataframe(df)
    if not is_safe:
        raise SystemExit("\n".join(violations))
    print(f"{args.csv}: audit passed")


if __name__ == "__main__":
    main()
