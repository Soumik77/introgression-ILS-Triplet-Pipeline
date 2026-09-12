#!/usr/bin/env python3
"""Paired, backbone-clustered comparison of rooted benchmark methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=/path/to/per_simulation.csv")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Both method name and path are required")
    return name.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rooted methods against one reference.")
    parser.add_argument("--reference", type=parse_method, required=True)
    parser.add_argument("--baseline", type=parse_method, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def load_method(name: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "simulation_id",
        "scenario_group",
        "backbone_id",
        "exact_match",
        "normalized_rooted_rf",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame["simulation_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate simulation_id values")
    if frame["exact_match"].dtype != bool:
        frame["exact_match"] = (
            frame["exact_match"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        )
    if frame["exact_match"].isna().any():
        raise ValueError(f"{name} has invalid exact_match values")
    if frame["normalized_rooted_rf"].isna().any():
        raise ValueError(f"{name} has missing normalized_rooted_rf values")
    if "evaluation_status" in frame.columns and (frame["evaluation_status"] != "ok").any():
        failures = int((frame["evaluation_status"] != "ok").sum())
        raise ValueError(f"{name} contains {failures} failed evaluations")
    return frame


def method_summary(name: str, frame: pd.DataFrame) -> dict:
    runtime_column = next(
        (column for column in ("assembly_seconds", "runtime_seconds") if column in frame.columns),
        None,
    )
    runtimes = (
        pd.to_numeric(frame[runtime_column], errors="coerce")
        if runtime_column
        else pd.Series(dtype=float)
    )
    return {
        "method": name,
        "simulations": len(frame),
        "exact_tree_count": int(frame["exact_match"].sum()),
        "rooted_exact_accuracy": float(frame["exact_match"].mean()),
        "mean_normalized_rooted_rf": float(frame["normalized_rooted_rf"].mean()),
        "runtime_component": (
            "assembly" if runtime_column == "assembly_seconds" else "external_inference"
            if runtime_column == "runtime_seconds"
            else "unavailable"
        ),
        "mean_runtime_seconds": float(runtimes.mean()) if not runtimes.empty else np.nan,
        "median_runtime_seconds": float(runtimes.median()) if not runtimes.empty else np.nan,
    }


def cluster_bootstrap_interval(
    frame: pd.DataFrame,
    value_column: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    cluster_table = (
        frame.groupby("backbone_id", sort=True)[value_column]
        .agg(["sum", "count"])
        .reset_index(drop=True)
    )
    sums = cluster_table["sum"].to_numpy(dtype=float)
    counts = cluster_table["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    batch_size = 1000
    for start in range(0, repetitions, batch_size):
        end = min(repetitions, start + batch_size)
        sampled = rng.integers(0, len(sums), size=(end - start, len(sums)))
        estimates[start:end] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def one_sided_wilcoxon(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    if np.allclose(array, 0.0):
        return 1.0
    return float(wilcoxon(array, alternative="greater", zero_method="wilcox").pvalue)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def main() -> int:
    args = parse_args()
    reference_name, reference_path = args.reference
    reference = load_method(reference_name, reference_path)
    reference_ids = set(reference["simulation_id"])

    summary_rows = [method_summary(reference_name, reference)]
    comparisons = []
    scenario_outputs = []

    reference_columns = [
        "simulation_id",
        "scenario_group",
        "backbone_id",
        "exact_match",
        "normalized_rooted_rf",
    ]
    for comparison_index, (baseline_name, baseline_path) in enumerate(args.baseline):
        baseline = load_method(baseline_name, baseline_path)
        baseline_ids = set(baseline["simulation_id"])
        if baseline_ids != reference_ids:
            missing = len(reference_ids - baseline_ids)
            extra = len(baseline_ids - reference_ids)
            raise ValueError(
                f"{baseline_name} is not paired with {reference_name}: missing={missing}, extra={extra}"
            )
        summary_rows.append(method_summary(baseline_name, baseline))

        paired = reference[reference_columns].merge(
            baseline[["simulation_id", "exact_match", "normalized_rooted_rf"]],
            on="simulation_id",
            suffixes=("_reference", "_baseline"),
            validate="one_to_one",
        )
        paired["exact_gain"] = (
            paired["exact_match_reference"].astype(float)
            - paired["exact_match_baseline"].astype(float)
        )
        paired["nrf_reduction"] = (
            paired["normalized_rooted_rf_baseline"]
            - paired["normalized_rooted_rf_reference"]
        )
        scenario = (
            paired.groupby("scenario_group", as_index=False)
            .agg(
                backbone_id=("backbone_id", "first"),
                simulations=("simulation_id", "count"),
                exact_gain=("exact_gain", "mean"),
                nrf_reduction=("nrf_reduction", "mean"),
            )
        )
        scenario.insert(0, "baseline", baseline_name)
        scenario_outputs.append(scenario)

        for metric_index, metric in enumerate(("exact_gain", "nrf_reduction")):
            low, high = cluster_bootstrap_interval(
                scenario,
                metric,
                args.bootstrap_reps,
                args.seed + comparison_index * 100 + metric_index,
            )
            backbone_values = (
                scenario.groupby("backbone_id", sort=True)[metric]
                .mean()
                .reset_index(drop=True)
            )
            comparisons.append(
                {
                    "comparison": f"{reference_name} vs {baseline_name}",
                    "metric": metric,
                    "scenario_groups": len(scenario),
                    "backbones": scenario["backbone_id"].nunique(),
                    "wilcoxon_pairs": len(backbone_values),
                    "estimate": float(scenario[metric].mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "one_sided_wilcoxon_p": one_sided_wilcoxon(backbone_values),
                }
            )

    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["holm_adjusted_p"] = holm_adjust(
        comparison_frame["one_sided_wilcoxon_p"].to_numpy(dtype=float)
    )
    summary_frame = pd.DataFrame(summary_rows).drop_duplicates("method", keep="first")
    scenario_frame = pd.concat(scenario_outputs, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.out_dir / "rooted_method_summary.csv", index=False)
    comparison_frame.to_csv(args.out_dir / "rooted_pairwise_comparisons.csv", index=False)
    scenario_frame.to_csv(args.out_dir / "rooted_scenario_group_differences.csv", index=False)
    run_summary = {
        "reference": reference_name,
        "baselines": [name for name, _ in args.baseline],
        "simulations": len(reference),
        "scenario_groups": int(reference["scenario_group"].nunique()),
        "backbones": int(reference["backbone_id"].nunique()),
        "bootstrap_repetitions": args.bootstrap_reps,
        "wilcoxon_unit": "backbone-level mean paired difference",
        "holm_family_size": len(comparison_frame),
    }
    (args.out_dir / "comparison_config.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\n===== ROOTED METHOD SUMMARY =====")
    print(summary_frame.to_string(index=False))
    print("\n===== PAIRED ROOTED COMPARISONS =====")
    print(comparison_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
