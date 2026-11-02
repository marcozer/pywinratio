"""Compare pywinratio with the R reference results in validation/reference.

Run from the repository root::

    python validation/compare.py

Writes ``validation/results/agreement.csv`` and ``agreement.md`` and exits
with a non-zero status when any comparison exceeds its tolerance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pywinratio import Outcome, WinRatio, win_ratio

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

TRIAL_TIERS = [
    Outcome.time_to_event("time_death", "death", name="Death"),
    Outcome.time_to_event("time_hosp", "hosp", name="HF hospitalization"),
]
COHORT_TIERS = [
    Outcome.binary("death90", name="Death"),
    Outcome.ordinal("clavien", name="Clavien-Dindo"),
    Outcome.ordinal("popf", name="POPF"),
    Outcome.binary("readmission", name="Readmission"),
    Outcome.continuous("los", name="LOS", margin=1),
]


def pywinratio_results() -> dict[str, dict[str, float]]:
    trial = pd.read_csv(DATA / "heart_failure_trial.csv")
    cohort = pd.read_csv(DATA / "surgical_cohort.csv")
    matched = pd.read_csv(DATA / "surgical_matched.csv")
    out: dict[str, dict[str, float]] = {}

    def record(name: str, result) -> None:
        inf = result.inference or {}
        out[name] = {
            "wins": result.wins,
            "losses": result.losses,
            "wr": result.wr,
            "net_benefit": result.net_benefit,
            "win_odds": result.win_odds,
            "se_log_wr": inf.get("se_log_wr", np.nan),
            "se_net_benefit": inf.get("se_net_benefit", np.nan),
            "wr_ci_lower": inf.get("wr_ci", (np.nan, np.nan))[0],
            "wr_ci_upper": inf.get("wr_ci", (np.nan, np.nan))[1],
            "net_benefit_ci_lower": inf.get("net_benefit_ci", (np.nan, np.nan))[0],
            "net_benefit_ci_upper": inf.get("net_benefit_ci", (np.nan, np.nan))[1],
            "win_odds_ci_lower": inf.get("win_odds_ci", (np.nan, np.nan))[0],
            "win_odds_ci_upper": inf.get("win_odds_ci", (np.nan, np.nan))[1],
            "p_value_net_benefit": inf.get("net_benefit_p_value", np.nan),
        }

    s1 = TRIAL_TIERS + [Outcome.continuous("kccq_change", name="KCCQ", favorable="higher", threshold=5)]
    record("S1_trial_gehan", win_ratio(trial, s1, group="arm", treatment="treatment", control="placebo"))
    record("S2_cohort_all_pairs", win_ratio(cohort, COHORT_TIERS, group="volume", treatment="high", control="low"))
    record(
        "S3_cohort_stratified",
        win_ratio(cohort, COHORT_TIERS, group="volume", treatment="high", control="low", strata=["period"]),
    )
    record(
        "S4_matched_pairs",
        WinRatio(COHORT_TIERS, group="volume", treatment="high", control="low").fit(matched, pair_id="PAIR_ID"),
    )
    ties = pd.DataFrame(
        {"arm": ["C", "C", "C", "T", "T", "T"], "time": [5, 2, 7, 5, 3, 7], "status": [1, 0, 0, 0, 1, 1]}
    )
    record(
        "S6_event_censoring_ties",
        win_ratio(ties, [Outcome.time_to_event("time", "status")], group="arm", treatment="T", control="C"),
    )
    s5 = TRIAL_TIERS + [Outcome.continuous("kccq_change", name="KCCQ", favorable="higher")]
    record("S5_trial_wins", win_ratio(trial, s5, group="arm", treatment="treatment", control="placebo"))
    return out


# (statistic in reference, statistic in pywinratio, absolute tolerance)
CHECKS = [
    ("wins", "wins", 0),
    ("losses", "losses", 0),
    ("wr", "wr", 1e-9),
    ("net_benefit", "net_benefit", 1e-9),
    ("win_odds", "win_odds", 1e-6),
    ("se_net_benefit", "se_net_benefit", 2e-3),
    ("se_log_wr", "se_log_wr", 2e-3),
    ("p_value_net_benefit", "p_value_net_benefit", 2e-3),
]
CI_CHECKS = [
    ("wr_ci", "wr_ci", 5e-3),
    ("net_benefit_ci", "net_benefit_ci", 2e-3),
    ("win_odds_ci", "win_odds_ci", 5e-3),
]


def main() -> int:
    reference = json.loads((HERE / "reference" / "r_reference.json").read_text(encoding="utf-8"))
    ours = pywinratio_results()
    rows = []
    for ref in reference["results"]:
        scenario = ref["scenario"]
        mine = ours[scenario]
        for ref_key, my_key, tol in CHECKS:
            if ref_key not in ref:
                continue
            rows.append((scenario, ref["package"], ref_key, ref[ref_key], mine[my_key], tol))
        for ref_key, my_key, tol in CI_CHECKS:
            if ref_key in ref:
                for side, index in (("lower", 0), ("upper", 1)):
                    rows.append(
                        (
                            scenario,
                            ref["package"],
                            f"{ref_key}_{side}",
                            ref[ref_key][index],
                            mine[f"{my_key}_{side}"],
                            tol,
                        )
                    )
    table = pd.DataFrame(rows, columns=["scenario", "reference", "statistic", "r_value", "pywinratio", "tolerance"])
    table["abs_diff"] = (table["r_value"] - table["pywinratio"]).abs()
    table["agree"] = table["abs_diff"] <= table["tolerance"]
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "agreement.csv", index=False)
    md = ["| Scenario | Reference | Statistic | R | pywinratio | abs. diff | Agree |", "|---|---|---|---|---|---|---|"]
    for row in table.itertuples():
        md.append(
            f"| {row.scenario} | {row.reference} | {row.statistic} | {row.r_value:.6f} | {row.pywinratio:.6f} "
            f"| {row.abs_diff:.1e} | {'yes' if row.agree else 'NO'} |"
        )
    (out / "agreement.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    return 0 if table["agree"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
