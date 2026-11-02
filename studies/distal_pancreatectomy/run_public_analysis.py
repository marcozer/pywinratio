from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from pywinratio import (
    WinRatioConfig,
    balance_diagnostics,
    bootstrap_propensity_matched_win_ratio,
    bootstrap_propensity_weighted_gpc,
    bootstrap_weighted_gpc,
    bootstrap_win_ratio_cluster_within_arm,
    bootstrap_win_ratio_matched,
    compute_weighted_gpc,
    compute_win_ratio,
    config_from_dict,
    paired_risk_difference_bootstrap,
    propensity_overlap_coefficient,
    summarize_component_outcomes,
    summarize_wr_metrics_from_overall,
)


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _matched_config(config_dict: Dict[str, Any]) -> WinRatioConfig:
    parsed = config_from_dict(config_dict)
    return WinRatioConfig(
        group_col=parsed.group_col,
        arm_a=parsed.arm_a,
        arm_b=parsed.arm_b,
        outcomes=parsed.outcomes,
        pair_strategy="matched",
        pair_id="PAIR_ID",
    )


def _all_pair_config(config_dict: Dict[str, Any]) -> WinRatioConfig:
    parsed = config_from_dict(config_dict)
    return WinRatioConfig(
        group_col=parsed.group_col,
        arm_a=parsed.arm_a,
        arm_b=parsed.arm_b,
        outcomes=parsed.outcomes,
        pair_strategy="all_pairs",
    )


def _sensitivity_config(primary: Dict[str, Any], specification: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(primary)
    outcomes = result["outcomes"]
    if specification.get("graded"):
        outcomes[1].update(
            {
                "name": "Clavien-Dindo severity (none, III, IV)",
                "column": "clavien_grade",
                "kind": "ordinal",
            }
        )
        outcomes[2].update(
            {
                "name": "Pancreatic fistula severity (none, B, C)",
                "column": "popf_grade",
                "kind": "ordinal",
            }
        )
    if specification.get("omit_los"):
        result["outcomes"] = outcomes[:-1]
    elif specification.get("los_cutpoints") is not None:
        outcomes[-1].update(
            {
                "name": "Length of stay category (<=7, 8-13, >13 days)",
                "kind": "ordinal",
                "cutpoints": specification["los_cutpoints"],
                "tie_tol": 0,
            }
        )
    elif "los_margin" in specification or "outcome_name" in specification:
        outcomes[-1]["tie_tol"] = float(specification.get("los_margin", 0))
        outcomes[-1]["name"] = specification.get(
            "outcome_name",
            f"Length of stay ({int(outcomes[-1]['tie_tol'])}-day margin)",
        )
    if specification.get("outcome_order") is not None:
        order = [int(index) for index in specification["outcome_order"]]
        if sorted(order) != list(range(len(outcomes))):
            raise ValueError("outcome_order must permute every outcome index")
        result["outcomes"] = [outcomes[index] for index in order]
    return result


def _support_diagnostics(results: Dict[str, Any], primary: Dict[str, Any]) -> pd.DataFrame:
    matching = results["matching"]
    full = matching.full_df.copy()
    matched = matching.matched_df.copy()
    treatment = primary["group_col"]
    treated = primary["arm_a"]
    rows = []
    for population, frame in (("eligible_common_support", full), ("retained_matched", matched)):
        counts = frame[treatment].value_counts()
        first_n = int(counts.get(treated, 0))
        comparator_n = int(len(frame) - first_n)
        overlap = propensity_overlap_coefficient(
            frame["_PS"],
            frame[treatment],
            treated=treated,
        )
        rows.append(
            {
                "population": population,
                "first_listed_group": treated,
                "first_listed_n": first_n,
                "comparator_n": comparator_n,
                "propensity_histogram_overlap": overlap,
                "rows_dropped_for_missing_covariates": matching.rows_dropped_missing,
                "matched_pairs": matching.pairs,
            }
        )
    eligible = rows[0]
    retained = rows[1]
    retained["first_listed_retention_pct"] = (
        100 * retained["first_listed_n"] / eligible["first_listed_n"] if eligible["first_listed_n"] else None
    )
    retained["comparator_retention_pct"] = (
        100 * retained["comparator_n"] / eligible["comparator_n"] if eligible["comparator_n"] else None
    )
    return pd.DataFrame(rows)


def _tier_table(point: Dict[str, Any], cfg: WinRatioConfig, analysis: str) -> pd.DataFrame:
    total = point["wins"] + point["losses"] + point["ties"]
    remaining = total
    rows = []
    wins = point["details"].get("tier_wins", [])
    losses = point["details"].get("tier_losses", [])
    for index, outcome in enumerate(cfg.outcomes):
        tier_wins = wins[index] if index < len(wins) else 0
        tier_losses = losses[index] if index < len(losses) else 0
        resolved = tier_wins + tier_losses
        rows.append(
            {
                "analysis": analysis,
                "priority": index + 1,
                "outcome": outcome.name,
                "wins": tier_wins,
                "losses": tier_losses,
                "resolved": resolved,
                "conditional_resolution": resolved / remaining if remaining else None,
                "overall_resolution": resolved / total if total else None,
            }
        )
        remaining -= resolved
    return pd.DataFrame(rows)


def run_public_analysis(
    df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    primary_dict = config["primary"]
    primary_cfg = _all_pair_config(primary_dict)
    unadjusted_point = compute_win_ratio(df, primary_cfg)["overall"]
    unadjusted_boot = bootstrap_win_ratio_cluster_within_arm(
        df,
        primary_cfg,
        cluster_col="site_id",
        n_boot=n_boot,
        seed=seed,
    )

    matching_dict = config["matching"]
    matched_boot = bootstrap_propensity_matched_win_ratio(
        df,
        primary_cfg,
        covariates=matching_dict["covariates"],
        cluster_col="site_id",
        exact_cols=matching_dict.get("exact_cols"),
        caliper=matching_dict.get("caliper"),
        method=matching_dict.get("method", "optimal"),
        trim_common_support=matching_dict.get("trim_common_support", True),
        missing=matching_dict.get("missing", "drop"),
        n_boot=n_boot,
        seed=seed,
    )
    matching = matched_boot["original_matching"]
    matched_cfg = _matched_config(primary_dict)
    matched_point = compute_win_ratio(matching.matched_df, matched_cfg)["overall"]
    primary_risk_differences = paired_risk_difference_bootstrap(
        matching.matched_df,
        matched_cfg,
        n_boot=n_boot,
        seed=seed,
    )

    covariates = matching_dict["covariates"] + matching_dict.get("exact_cols", [])
    before_balance = balance_diagnostics(
        matching.full_df,
        treatment=primary_dict["group_col"],
        treated=primary_dict["arm_a"],
        covariates=covariates,
    ).assign(stage="before_matching")
    after_balance = balance_diagnostics(
        matching.matched_df,
        treatment=primary_dict["group_col"],
        treated=primary_dict["arm_a"],
        covariates=covariates,
    ).assign(stage="after_matching")
    matching_balance = pd.concat([before_balance, after_balance], ignore_index=True)

    sensitivity_rows = []
    primary_key = primary_dict.get("analysis_key", "primary")
    primary_label = primary_dict.get("label", "Primary hierarchy")
    primary_plot_label = primary_dict.get("plot_label", primary_label)
    tier_tables = [_tier_table(matched_point, matched_cfg, primary_key)]
    for name, specification in config["sensitivities"].items():
        sensitivity_dict = _sensitivity_config(primary_dict, specification)
        sensitivity_cfg = _matched_config(sensitivity_dict)
        point = compute_win_ratio(matching.matched_df, sensitivity_cfg)["overall"]
        boot = bootstrap_win_ratio_matched(
            matching.matched_df,
            sensitivity_cfg,
            n_boot=n_boot,
            seed=seed + len(sensitivity_rows) + 1,
        )
        sensitivity_rows.append(
            {
                "analysis": name,
                "label": specification["label"],
                "plot_label": specification.get("plot_label", specification["label"]),
                "wins": point["wins"],
                "losses": point["losses"],
                "ties": point["ties"],
                "wr": point["wr"],
                "ci_lower": boot["ci"][0],
                "ci_upper": boot["ci"][1],
                "information_rate": (point["wins"] + point["losses"]) / matching.pairs if matching.pairs else None,
            }
        )
        tier_tables.append(_tier_table(point, sensitivity_cfg, name))

    all_pair_matched_cfg = _all_pair_config(primary_dict)
    matched_gpc = compute_weighted_gpc(matching.matched_df, all_pair_matched_cfg)
    matched_gpc_boot = bootstrap_weighted_gpc(
        matching.matched_df,
        all_pair_matched_cfg,
        cluster_col="site_id",
        within_arm=True,
        n_boot=n_boot,
        seed=seed + 20,
    )

    weighted_gpc_boot = bootstrap_propensity_weighted_gpc(
        df,
        all_pair_matched_cfg,
        covariates=matching_dict["covariates"] + matching_dict.get("exact_cols", []),
        cluster_col="site_id",
        estimand=config["weighting"].get("estimand", "overlap"),
        missing=config["weighting"].get("missing", "drop"),
        trim_common_support=config["weighting"].get("trim_common_support", True),
        n_boot=n_boot,
        seed=seed + 30,
    )
    weighting = weighted_gpc_boot["original_weighting"]
    weighted_gpc = compute_weighted_gpc(
        weighting.weighted_df,
        all_pair_matched_cfg,
        weight_col="analysis_weight",
    )

    benchmark_dict = {
        **primary_dict,
        "outcomes": config["binary_benchmarks"]["outcomes"],
    }
    benchmark_cfg = _matched_config(benchmark_dict)
    benchmark_table = paired_risk_difference_bootstrap(
        matching.matched_df,
        benchmark_cfg,
        n_boot=n_boot,
        seed=seed + 40,
    )
    benchmark_table["discordant_pairs"] = benchmark_table["A_event_B_no_event"] + benchmark_table["A_no_event_B_event"]
    benchmark_table["information_rate"] = benchmark_table["discordant_pairs"] / benchmark_table["n_pairs_complete"]

    return {
        "unadjusted_point": unadjusted_point,
        "unadjusted_boot": unadjusted_boot,
        "matching": matching,
        "matched_cfg": matched_cfg,
        "matched_point": matched_point,
        "matched_boot": matched_boot,
        "primary_label": primary_label,
        "primary_plot_label": primary_plot_label,
        "primary_risk_differences": primary_risk_differences,
        "matching_balance": matching_balance,
        "sensitivities": pd.DataFrame(sensitivity_rows),
        "tiers": pd.concat(tier_tables, ignore_index=True),
        "matched_gpc": matched_gpc,
        "matched_gpc_boot": matched_gpc_boot,
        "weighting": weighting,
        "weighted_gpc": weighted_gpc,
        "weighted_gpc_boot": weighted_gpc_boot,
        "benchmarks": benchmark_table,
        "components": summarize_component_outcomes(matching.matched_df, matched_cfg),
        "metrics": summarize_wr_metrics_from_overall(matched_point),
        "support": _support_diagnostics(
            {"matching": matching},
            primary_dict,
        ),
    }


def plot_summary(results: Dict[str, Any], output_stem: Path) -> None:
    mpl.rcParams["svg.fonttype"] = "none"
    colors = {"primary": "#5E81AC", "sensitivity": "#88C0D0", "binary": "#A3BE8C", "null": "#4C566A"}
    sensitivities = results["sensitivities"]
    labels = ["Assigned-pair WR", "Matched all-pair GPC", "Overlap-weighted GPC"]
    estimates = [results["matched_point"]["wr"], results["matched_gpc"]["wr"], results["weighted_gpc"]["wr"]]
    intervals = [results["matched_boot"]["ci"], results["matched_gpc_boot"]["ci"], results["weighted_gpc_boot"]["ci"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [1.0, 1.2]})
    ax = axes[0]
    y = list(range(len(labels)))[::-1]
    xerr = [
        [max(0.0, estimate - interval[0]) for estimate, interval in zip(estimates, intervals)],
        [max(0.0, interval[1] - estimate) for estimate, interval in zip(estimates, intervals)],
    ]
    ax.errorbar(estimates, y, xerr=xerr, fmt="o", color=colors["primary"], ecolor="#81A1C1", capsize=3)
    ax.axvline(1.0, color=colors["null"], linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Win ratio (higher-volume / other)")
    ax.set_title("Estimands")

    ax = axes[1]
    display = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "plot_label": results["primary_plot_label"],
                        "wr": results["matched_point"]["wr"],
                        "ci_lower": results["matched_boot"]["ci"][0],
                        "ci_upper": results["matched_boot"]["ci"][1],
                    }
                ]
            ),
            sensitivities[["plot_label", "wr", "ci_lower", "ci_upper"]],
        ],
        ignore_index=True,
    )
    y = list(range(len(display)))[::-1]
    ax.errorbar(
        display["wr"],
        y,
        xerr=[
            (display["wr"] - display["ci_lower"]).clip(lower=0),
            (display["ci_upper"] - display["wr"]).clip(lower=0),
        ],
        fmt="o",
        color=colors["sensitivity"],
        ecolor="#8FBCBB",
        capsize=3,
    )
    ax.axvline(1.0, color=colors["null"], linewidth=1, linestyle="--")
    ax.set_yticks(y, display["plot_label"])
    ax.set_xlabel("Win ratio (higher-volume / other)")
    ax.set_title("Hierarchy and LOS sensitivity")
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "pywinratio"},
    )
    svg_path = output_stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public distal pancreatectomy analysis.")
    parser.add_argument("--dataset", type=Path, default=Path("data/public/distal_pancreatectomy_public.csv"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("studies/distal_pancreatectomy/config/primary_analysis.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/public"))
    parser.add_argument("--n-boot", type=int, default=400, help="Bootstrap replicates for the public demonstrator.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    df = pd.read_csv(args.dataset)
    results = run_public_analysis(df, config, n_boot=args.n_boot, seed=args.seed)

    def public_bootstrap_payload(bootstrap: Dict[str, Any]) -> Dict[str, Any]:
        excluded = {"original_matching", "original_weighting", "wr_samples", "net_benefit_samples", "pair_samples"}
        return {key: value for key, value in bootstrap.items() if key not in excluded}

    matched_boot_public = public_bootstrap_payload(results["matched_boot"])
    weighted_boot_public = public_bootstrap_payload(results["weighted_gpc_boot"])
    summary_payload = {
        "study_name": config["study_name"],
        "analysis_version": config["analysis_version"],
        "dataset_path": str(args.dataset),
        "volume_definition": config["volume_definition"],
        "unadjusted_all_pairs": {
            "point": results["unadjusted_point"],
            "bootstrap": public_bootstrap_payload(results["unadjusted_boot"]),
        },
        "primary_assigned_pairs": {
            "point": results["matched_point"],
            "bootstrap": matched_boot_public,
            "pairs": results["matching"].pairs,
            "missing_strategy": results["matching"].missing,
            "rows_dropped_missing": results["matching"].rows_dropped_missing,
        },
        "matched_sample_all_pair_gpc": {
            "point": results["matched_gpc"],
            "bootstrap": public_bootstrap_payload(results["matched_gpc_boot"]),
        },
        "overlap_weighted_gpc": {
            "point": results["weighted_gpc"],
            "bootstrap": weighted_boot_public,
            "ess_by_arm": {str(key): value for key, value in results["weighting"].ess_by_arm.items()},
        },
    }
    (args.output_dir / "win_ratio_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    results["components"].to_csv(args.output_dir / "component_outcomes.csv", index=False)
    results["metrics"].to_csv(args.output_dir / "matched_metrics.csv", index=False)
    results["primary_risk_differences"].to_csv(args.output_dir / "matched_risk_differences.csv", index=False)
    results["matching_balance"].to_csv(args.output_dir / "matching_balance.csv", index=False)
    results["support"].to_csv(args.output_dir / "matching_support.csv", index=False)
    results["sensitivities"].to_csv(args.output_dir / "hierarchy_sensitivities.csv", index=False)
    results["tiers"].to_csv(args.output_dir / "tier_resolution.csv", index=False)
    results["benchmarks"].to_csv(args.output_dir / "binary_benchmarks.csv", index=False)
    results["weighting"].balance_after.to_csv(args.output_dir / "overlap_weighting_balance.csv", index=False)
    results["weighting"].weight_summary.to_csv(args.output_dir / "overlap_weight_summary.csv", index=False)
    plot_summary(results, args.output_dir / "win_ratio_overview")

    public_outputs = [
        "win_ratio_summary.json",
        "component_outcomes.csv",
        "matched_metrics.csv",
        "matched_risk_differences.csv",
        "matching_balance.csv",
        "matching_support.csv",
        "hierarchy_sensitivities.csv",
        "tier_resolution.csv",
        "binary_benchmarks.csv",
        "overlap_weighting_balance.csv",
        "overlap_weight_summary.csv",
        "win_ratio_overview.png",
        "win_ratio_overview.svg",
    ]
    manifest = {
        "public_dataset": str(args.dataset),
        "config": str(args.config),
        "analysis_version": config["analysis_version"],
        "reproducible_with_public_data": [str(args.output_dir / name) for name in public_outputs],
        "dataset_type": "synthetic",
        "notes": [
            "The committed data set is synthetic; its estimates illustrate the software only.",
            "Increase --n-boot (e.g. 2000-5000) for publication-grade intervals.",
        ],
    }
    (args.output_dir / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
