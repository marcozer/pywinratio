from __future__ import annotations

import json
from pathlib import Path

import yaml

from pywinratio import config_from_dict
from studies.distal_pancreatectomy.export_public_data import build_public_dataset
from studies.distal_pancreatectomy.preprocess import audit_public_dataframe, create_synthetic_public_dataset
from studies.distal_pancreatectomy.run_public_analysis import (
    _sensitivity_config,
)
from studies.distal_pancreatectomy.run_public_analysis import (
    main as run_public_analysis_main,
)

ROOT = Path(__file__).resolve().parents[1]


def test_public_dataset_audit_rejects_free_text_column() -> None:
    df = create_synthetic_public_dataset(n=10, seed=1)
    df["operative_comment"] = "too much detail for public release"
    is_safe, violations = audit_public_dataframe(df)
    assert not is_safe
    assert any("forbidden column name" in item or "possible free-text" in item for item in violations)


def test_build_public_dataset_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "public.csv"
    manifest = build_public_dataset(output_path=output, seed=5)
    assert output.exists()
    assert manifest["source_type"] == "synthetic"
    assert manifest["audit_passed"] is True


def test_public_analysis_script_creates_outputs(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "public.csv"
    manifest = build_public_dataset(output_path=dataset_path, seed=7)
    assert manifest["audit_passed"] is True

    config_path = ROOT / "studies/distal_pancreatectomy/config/primary_analysis.yaml"
    output_dir = tmp_path / "results"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_public_analysis",
            "--dataset",
            str(dataset_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--n-boot",
            "50",
        ],
    )
    run_public_analysis_main()
    assert (output_dir / "win_ratio_summary.json").exists()
    assert (output_dir / "reproducibility_manifest.json").exists()
    assert (output_dir / "matched_risk_differences.csv").exists()
    assert (output_dir / "hierarchy_sensitivities.csv").exists()
    assert (output_dir / "binary_benchmarks.csv").exists()
    assert (output_dir / "matching_support.csv").exists()
    assert (output_dir / "win_ratio_overview.svg").exists()
    summary = json.loads((output_dir / "win_ratio_summary.json").read_text(encoding="utf-8"))
    assert summary["analysis_version"] == "1.0"
    assert "primary_assigned_pairs" in summary
    assert "matched_sample_all_pair_gpc" in summary
    assert "overlap_weighted_gpc" in summary


def test_public_configuration_locks_exposure_and_primary_hierarchy() -> None:
    config = yaml.safe_load(
        (ROOT / "studies/distal_pancreatectomy/config/primary_analysis.yaml").read_text(encoding="utf-8")
    )
    volume = config["volume_definition"]
    assert config["analysis_version"] == "1.0"
    assert volume["numerator"] == "all minimally invasive pancreatic resections"
    assert "published" in volume["source"]
    assert "MIDP-only" in volume["reconstruction_policy"]
    assert "complete contributed years" not in str(volume)

    primary = config_from_dict(config["primary"])
    assert [outcome.column for outcome in primary.outcomes] == [
        "mort90",
        "clavien_grade",
        "popf_grade",
        "readmission",
        "los_days",
    ]
    assert primary.outcomes[1].kind == "ordinal"
    assert primary.outcomes[2].kind == "ordinal"
    assert primary.outcomes[-1].tie_tol == 1
    assert "popf_before_clavien" in config["sensitivities"]
    assert "readmission_before_popf" in config["sensitivities"]


def test_order_sensitivities_preserve_the_main_los_margin() -> None:
    config = yaml.safe_load(
        (ROOT / "studies/distal_pancreatectomy/config/primary_analysis.yaml").read_text(encoding="utf-8")
    )
    primary = config["primary"]
    for key in ("popf_before_clavien", "readmission_before_popf"):
        sensitivity = _sensitivity_config(primary, config["sensitivities"][key])
        parsed = config_from_dict(sensitivity)
        assert parsed.outcomes[-1].tie_tol == 1
        assert "differences of 0 or 1 day tie" in parsed.outcomes[-1].name
