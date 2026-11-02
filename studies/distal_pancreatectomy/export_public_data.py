"""Write the synthetic distal pancreatectomy demonstration data set.

The case study ships only synthetic records. Run from the repository root::

    python -m studies.distal_pancreatectomy.export_public_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preprocess import audit_public_dataframe, create_synthetic_public_dataset, write_public_dataset


def build_public_dataset(output_path: Path, seed: int = 42, n: int = 800) -> dict:
    """Generate the synthetic cohort, audit its schema and write it to ``output_path``."""

    public_df = create_synthetic_public_dataset(n=n, seed=seed)
    is_safe, violations = audit_public_dataframe(public_df)
    write_public_dataset(public_df, output_path)
    return {
        "dataset_path": str(output_path),
        "source_type": "synthetic",
        "seed": int(seed),
        "rows": int(len(public_df)),
        "columns": list(public_df.columns),
        "audit_passed": bool(is_safe),
        "audit_violations": violations,
        "notes": ["Fully synthetic records generated for the pywinratio case study."],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the synthetic distal pancreatectomy data set.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/public/distal_pancreatectomy_public.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/public/distal_pancreatectomy_manifest.json"),
        help="Output manifest path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n", type=int, default=800, help="Number of synthetic patients.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_public_dataset(args.output, args.seed, args.n)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
