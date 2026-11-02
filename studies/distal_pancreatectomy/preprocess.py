from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

FORBIDDEN_COLUMN_PATTERNS = (
    "code",
    "centre",
    "center",
    "date",
    "texte",
    "detail",
    "remarque",
    "comment",
    "narrative",
    "free_text",
    "medication",
)

PUBLIC_COLUMNS = [
    "site_id",
    "year_band",
    "hospital_type",
    "age_years",
    "age_band",
    "sex",
    "bmi",
    "bmi_band",
    "asa_group",
    "asa_ge3",
    "cci",
    "malignant_case",
    "neoadjuvant",
    "pathology_group",
    "tumor_size_mm",
    "functional_impairment",
    "prior_abdominal_surgery",
    "chronic_pancreatitis",
    "kidney_disease",
    "cardiac_history",
    "treatment_group",
    "spleen_management",
    "mort90",
    "clavien_grade",
    "clavien_major",
    "popf_grade",
    "popf_BC",
    "postpancreatectomy_hemorrhage",
    "bile_leak",
    "reoperation",
    "readmission",
    "los_days",
    "ideal_outcome",
    "textbook_outcome",
]

MIP_VOLUME_COLUMNS = (
    "MIP_VOLUME_ANNUAL",
    "VOLUME_MIP_ANNUEL",
    "MIP_PER_YEAR",
    "VOL_MIP",
    "N_MIP_AN",
)


def _to_float(value: object) -> float:
    if value in ("", None) or pd.isna(value):
        return float("nan")
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return float("nan")


def _derive_binary_benchmarks(out: pd.DataFrame) -> pd.DataFrame:
    no_mortality = out["mort90"] == 0
    no_major = out["clavien_major"] == 0
    no_popf = out["popf_BC"] == 0
    no_readmission = out["readmission"] == 0
    out["ideal_outcome"] = (
        no_mortality & no_major & no_popf & no_readmission & (out["reoperation"] == 0) & (out["los_days"] <= 13)
    ).astype(int)
    out["textbook_outcome"] = (
        no_mortality
        & no_major
        & no_popf
        & no_readmission
        & (out["postpancreatectomy_hemorrhage"] == 0)
        & (out["bile_leak"] == 0)
    ).astype(int)
    return out


def audit_public_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check the public schema against fixed disclosure rules."""

    violations: List[str] = []
    lower_columns = [column.lower() for column in df.columns]
    for column in lower_columns:
        if any(pattern in column for pattern in FORBIDDEN_COLUMN_PATTERNS):
            violations.append(f"forbidden column name: {column}")
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            non_missing = series.dropna().astype(str)
            max_len = non_missing.str.len().max() if not non_missing.empty else 0
            if max_len and max_len > 32:
                violations.append(f"possible free-text column: {column}")
    for column in PUBLIC_COLUMNS:
        if column not in df.columns:
            violations.append(f"missing required public column: {column}")
    return (len(violations) == 0, violations)


def create_synthetic_public_dataset(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic, correlated MIDP dataset for public demonstrations."""

    rng = np.random.default_rng(seed)
    sites = [f"SITE_{index:02d}" for index in range(1, 21)]
    high_volume_sites = set(sites[:5])
    site_type = {
        site: ("ACADEMIC" if index % 3 == 0 else ("PUBLIC" if index % 3 == 1 else "PRIVATE"))
        for index, site in enumerate(sites)
    }
    rows = []

    for _ in range(n):
        site_id = rng.choice(sites, p=np.array([2.0] * 5 + [1.0] * 15) / 25.0)
        high_volume = site_id in high_volume_sites
        age = float(np.clip(rng.normal(60, 15), 18, 90))
        bmi = float(np.clip(rng.normal(26.2, 5.2), 15, 48))
        sex = rng.choice(["MALE", "FEMALE"], p=[0.43, 0.57])
        asa = int(rng.choice([1, 2, 3, 4], p=[0.08, 0.49, 0.38, 0.05]))
        cci = int(np.clip(rng.poisson(1.2 + 0.03 * max(age - 60, 0)), 0, 9))
        pathology = rng.choice(
            ["DUCTAL_ADENOCARCINOMA", "NEUROENDOCRINE", "IPMN", "SOLID_PSEUDOPAPILLARY", "OTHER"],
            p=[0.30, 0.22, 0.17, 0.06, 0.25],
        )
        malignant = int(pathology == "DUCTAL_ADENOCARCINOMA" or rng.random() < 0.10)
        neoadjuvant = int(pathology == "DUCTAL_ADENOCARCINOMA" and rng.random() < 0.35)
        tumor_size = float(np.clip(rng.lognormal(np.log(28), 0.55), 4, 130))
        functional = int(rng.random() < 0.03 + 0.003 * max(age - 65, 0))
        prior_surgery = int(rng.random() < 0.28)
        chronic_pancreatitis = int(rng.random() < 0.07)
        kidney = int(rng.random() < 0.04 + 0.003 * max(age - 65, 0))
        cardiac = int(rng.random() < 0.08 + 0.005 * max(age - 60, 0))

        baseline = (
            0.35 * (asa >= 3)
            + 0.22 * (bmi >= 30)
            + 0.18 * (pathology == "NEUROENDOCRINE")
            + 0.12 * kidney
            - 0.30 * high_volume
        )
        popf_probability = 1 / (1 + np.exp(-(-1.75 + baseline)))
        popf_bc = int(rng.random() < popf_probability)
        popf_grade = 2 if popf_bc and rng.random() < 0.12 else (1 if popf_bc else 0)
        major_probability = 1 / (1 + np.exp(-(-2.15 + baseline + 1.0 * popf_bc)))
        major = int(rng.random() < major_probability)
        clavien_grade = 4 if major and rng.random() < 0.18 else (3 if major else 0)
        mortality = int(rng.random() < (0.003 + 0.045 * (clavien_grade == 4)))
        readmission = int(rng.random() < (0.075 + 0.16 * popf_bc + 0.04 * major))
        reoperation = int(rng.random() < (0.025 + 0.16 * (clavien_grade == 4) + 0.10 * (popf_grade == 2)))
        hemorrhage = int(rng.random() < (0.035 + 0.08 * major))
        bile_leak = int(rng.random() < 0.015)
        los = int(
            max(
                3,
                round(rng.normal(8.2 if high_volume else 9.7, 2.5) + 3.0 * popf_bc + 4.5 * major + 1.5 * readmission),
            )
        )
        spleen = rng.choice(
            ["PRESERVE_KIMURA", "PRESERVE_WARSHAW", "PLANNED_SPLENECTOMY", "UNPLANNED_SPLENECTOMY"],
            p=[0.38, 0.16, 0.40, 0.06],
        )
        rows.append(
            {
                "site_id": site_id,
                "year_band": rng.choice(["2016-2017", "2018-2019", "2020-2021"], p=[0.28, 0.36, 0.36]),
                "hospital_type": site_type[site_id],
                "age_years": round(age, 1),
                "age_band": pd.cut(
                    [age],
                    [0, 49, 59, 69, 79, 120],
                    labels=["<50", "50-59", "60-69", "70-79", "80+"],
                )[0],
                "sex": sex,
                "bmi": round(bmi, 1),
                "bmi_band": pd.cut(
                    [bmi],
                    [0, 18.5, 25, 30, 100],
                    labels=["UNDER_18_5", "NORMAL", "OVERWEIGHT", "OBESE"],
                )[0],
                "asa_group": f"ASA{asa}" if asa < 4 else "ASA4PLUS",
                "asa_ge3": int(asa >= 3),
                "cci": cci,
                "malignant_case": malignant,
                "neoadjuvant": neoadjuvant,
                "pathology_group": pathology,
                "tumor_size_mm": round(tumor_size, 1),
                "functional_impairment": functional,
                "prior_abdominal_surgery": prior_surgery,
                "chronic_pancreatitis": chronic_pancreatitis,
                "kidney_disease": kidney,
                "cardiac_history": cardiac,
                "treatment_group": "HIGH_VOLUME" if high_volume else "OTHER",
                "spleen_management": spleen,
                "mort90": mortality,
                "clavien_grade": clavien_grade,
                "clavien_major": major,
                "popf_grade": popf_grade,
                "popf_BC": popf_bc,
                "postpancreatectomy_hemorrhage": hemorrhage,
                "bile_leak": bile_leak,
                "reoperation": reoperation,
                "readmission": readmission,
                "los_days": los,
            }
        )

    public = pd.DataFrame(rows)
    public = _derive_binary_benchmarks(public)
    return public[PUBLIC_COLUMNS].copy()


def write_public_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
