#!/usr/bin/env python3
"""Create final taxon-stratified tables and figures from a completed run.

This script performs reporting only. It does not simulate gene trees, retrain
the classifier, assemble trees, or modify any completed evaluation file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = {
    "ml-soft": "Enhanced17 ML soft FM",
    "ml-hard": "Enhanced17 ML hard FM",
    "ml-confidence": "Enhanced17 ML confidence FM",
    "frequency": "Frequency FM",
    "majority": "Majority FM",
}

DISPLAY_ORDER = list(METHODS.values())
PRIMARY_DISPLAY = "Enhanced17 ML soft FM"
MAIN_FIGURE_METHODS = [PRIMARY_DISPLAY, "Frequency FM", "Majority FM"]
REGIME_LABELS = ["4-10", "11-20", "21-40", "41-80", "81-120", "121-180"]
REGIME_BINS = [3, 10, 20, 40, 80, 120, 180]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "outputs/supervisor_scope/"
            "supervisor_scope_corrected_deadline_normalized"
        ),
        help="Root of the corrected normalized run.",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        help=(
            "Oracle per-simulation CSV. By default, use "
            "RUN_ROOT/diagnostics/oracle_fm_4_180_normalized/"
            "oracle_per_simulation.csv."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory. Defaults to RUN_ROOT/comparison/final_report.",
    )
    parser.add_argument(
        "--expected-simulations",
        type=int,
        default=5310,
        help="Required number of rows for each completed method.",
    )
    return parser.parse_args()


def as_boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    result = values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    return result | numeric.eq(1).fillna(False)


def standardize(frame: pd.DataFrame, source: Path, oracle: bool = False) -> pd.DataFrame:
    required = {"simulation_id", "exact_match", "normalized_rooted_rf"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")

    tree_size_column = next(
        (name for name in ("tree_size", "taxon_count") if name in frame.columns),
        None,
    )
    if tree_size_column is None:
        raise ValueError(f"{source} has neither tree_size nor taxon_count")

    if oracle and "oracle_status" in frame.columns:
        bad = ~frame["oracle_status"].astype(str).eq("completed")
        if bad.any():
            raise ValueError(f"{source} contains {int(bad.sum())} incomplete oracle rows")
    if not oracle and "evaluation_status" in frame.columns:
        bad = ~frame["evaluation_status"].astype(str).eq("ok")
        if bad.any():
            raise ValueError(f"{source} contains {int(bad.sum())} failed evaluation rows")

    result = frame.copy()
    result["simulation_id"] = result["simulation_id"].astype(str)
    result["tree_size"] = pd.to_numeric(result[tree_size_column], errors="raise").astype(int)
    result["exact_match"] = as_boolean(result["exact_match"])
    result["normalized_rooted_rf"] = pd.to_numeric(
        result["normalized_rooted_rf"], errors="raise"
    )
    if result["simulation_id"].duplicated().any():
        raise ValueError(f"{source} contains duplicate simulation IDs")
    if not result["tree_size"].between(4, 180).all():
        raise ValueError(f"{source} contains a taxon count outside 4 through 180")
    if not result["normalized_rooted_rf"].between(0.0, 1.0).all():
        raise ValueError(f"{source} contains normalized rooted RF outside [0, 1]")
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def summarize_group(group: pd.DataFrame) -> pd.Series:
    total = int(len(group))
    exact = int(group["exact_match"].sum())
    low, high = wilson_interval(exact, total)
    return pd.Series(
        {
            "simulations": total,
            "exact_tree_count": exact,
            "exact_match_accuracy": exact / total if total else math.nan,
            "exact_ci_low": low,
            "exact_ci_high": high,
            "mean_normalized_rooted_rf": float(group["normalized_rooted_rf"].mean()),
            "median_normalized_rooted_rf": float(group["normalized_rooted_rf"].median()),
        }
    )


def summarize_table(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = frame.groupby(keys, sort=False, observed=True)
    for group_key, group in grouped:
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        record = dict(zip(keys, values))
        record.update(summarize_group(group).to_dict())
        rows.append(record)
    result = pd.DataFrame(rows)
    for column in ("simulations", "exact_tree_count"):
        if column in result.columns:
            result[column] = result[column].astype(int)
    return result


def add_taxon_regime(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["taxon_group"] = pd.cut(
        result["tree_size"],
        bins=REGIME_BINS,
        labels=REGIME_LABELS,
        include_lowest=True,
        ordered=True,
    )
    if result["taxon_group"].isna().any():
        raise ValueError("At least one row could not be assigned to a taxon group")
    return result


def method_summary(all_methods: pd.DataFrame) -> pd.DataFrame:
    return summarize_table(all_methods, ["method"])


def by_taxon(all_methods: pd.DataFrame) -> pd.DataFrame:
    return summarize_table(all_methods, ["method", "tree_size"]).sort_values(
        ["method", "tree_size"]
    )


def by_regime(all_methods: pd.DataFrame) -> pd.DataFrame:
    with_regime = add_taxon_regime(all_methods)
    return summarize_table(with_regime, ["method", "taxon_group"])


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.9f")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def plot_internal_taxon_curves(taxon_table: pd.DataFrame, out_dir: Path) -> None:
    configure_plotting()
    colors = {
        PRIMARY_DISPLAY: "#1f4e79",
        "Frequency FM": "#b45f06",
        "Majority FM": "#548235",
    }
    markers = {PRIMARY_DISPLAY: "o", "Frequency FM": "s", "Majority FM": "^"}
    fig, axes = plt.subplots(2, 1, figsize=(8.1, 7.2), sharex=True)
    for method in MAIN_FIGURE_METHODS:
        subset = taxon_table.loc[taxon_table["method"].eq(method)].sort_values("tree_size")
        axes[0].plot(
            subset["tree_size"],
            subset["exact_match_accuracy"],
            label=method,
            color=colors[method],
            linewidth=1.8,
            marker=markers[method],
            markersize=2.2,
            markevery=5,
        )
        axes[1].plot(
            subset["tree_size"],
            subset["mean_normalized_rooted_rf"],
            label=method,
            color=colors[method],
            linewidth=1.8,
            marker=markers[method],
            markersize=2.2,
            markevery=5,
        )

    axes[0].set_ylabel("Exact-match accuracy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("A. Exact recovery")
    axes[1].set_ylabel("Mean normalized rooted RF")
    axes[1].set_xlabel("Number of taxa")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("B. Topological error")
    for axis in axes:
        axis.axvline(20, color="#777777", linestyle="--", linewidth=0.9)
        axis.axvline(40, color="#777777", linestyle=":", linewidth=0.9)
        axis.set_xlim(4, 180)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    fig.suptitle("Internal-method performance by taxon count", y=0.995, fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"figure_internal_by_taxon_count.{suffix}", bbox_inches="tight")
    plt.close(fig)


def oracle_regime_table(primary: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    primary_labeled = primary.copy()
    primary_labeled["method"] = "Model"
    oracle_labeled = oracle.copy()
    oracle_labeled["method"] = "Oracle"
    primary_summary = by_regime(primary_labeled).drop(columns="method")
    oracle_summary = by_regime(oracle_labeled).drop(columns="method")
    merged = primary_summary.merge(
        oracle_summary,
        on="taxon_group",
        suffixes=("_model", "_oracle"),
        validate="one_to_one",
    )
    return merged


def plot_oracle_ceiling(regime: pd.DataFrame, out_dir: Path) -> None:
    configure_plotting()
    positions = np.arange(len(regime))
    width = 0.37
    fig, axes = plt.subplots(2, 1, figsize=(8.1, 7.1), sharex=True)
    axes[0].bar(
        positions - width / 2,
        regime["exact_match_accuracy_model"],
        width,
        label="OOF ML soft FM",
        color="#1f4e79",
    )
    axes[0].bar(
        positions + width / 2,
        regime["exact_match_accuracy_oracle"],
        width,
        label="Oracle triplet labels",
        color="#9dc3e6",
    )
    axes[1].bar(
        positions - width / 2,
        regime["mean_normalized_rooted_rf_model"],
        width,
        label="OOF ML soft FM",
        color="#1f4e79",
    )
    axes[1].bar(
        positions + width / 2,
        regime["mean_normalized_rooted_rf_oracle"],
        width,
        label="Oracle triplet labels",
        color="#9dc3e6",
    )
    axes[0].set_ylabel("Exact-match accuracy")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_title("A. Exact recovery")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_ylabel("Mean normalized rooted RF")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title("B. Topological error")
    axes[1].set_xlabel("Taxon-count stratum")
    axes[1].set_xticks(positions, regime["taxon_group"].astype(str))
    fig.suptitle("Observed performance and the fixed-triplet-budget ceiling", y=0.995, fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"figure_model_vs_oracle_by_stratum.{suffix}", bbox_inches="tight")
    plt.close(fig)


def results_note(
    full_summary: pd.DataFrame,
    primary_summary: pd.DataFrame,
    regime: pd.DataFrame,
    oracle_available: bool,
) -> str:
    full_soft = full_summary.loc[full_summary["method"].eq(PRIMARY_DISPLAY)].iloc[0]
    primary_soft = primary_summary.loc[primary_summary["method"].eq(PRIMARY_DISPLAY)].iloc[0]
    lines = [
        "FINAL CORRECTED INTERNAL RESULTS",
        "",
        (
            f"The corrected evaluation contains {int(full_soft.simulations):,} simulations. "
            f"Enhanced17 ML soft FM recovered {int(full_soft.exact_tree_count):,} exact trees "
            f"({100 * full_soft.exact_match_accuracy:.2f}%) and had mean normalized rooted RF "
            f"{full_soft.mean_normalized_rooted_rf:.4f} across 4 through 180 taxa."
        ),
        "",
        (
            f"Within the 4 through 20 taxon accuracy-focused stratum, the method recovered "
            f"{int(primary_soft.exact_tree_count):,} of {int(primary_soft.simulations):,} trees "
            f"({100 * primary_soft.exact_match_accuracy:.2f}%) and had mean normalized rooted "
            f"RF {primary_soft.mean_normalized_rooted_rf:.4f}."
        ),
        "",
    ]
    if oracle_available:
        low = regime.loc[regime["taxon_group"].astype(str).isin(["4-10", "11-20"])]
        model_exact = int(low["exact_tree_count_model"].sum())
        oracle_exact = int(low["exact_tree_count_oracle"].sum())
        simulations = int(low["simulations_model"].sum())
        lines.extend(
            [
                (
                    f"Across 4 through 20 taxa, model predictions recovered {model_exact} of "
                    f"{simulations} trees ({100 * model_exact / simulations:.2f}%), while the "
                    f"same assembler supplied with true triplet labels recovered {oracle_exact} "
                    f"of {simulations} trees ({100 * oracle_exact / simulations:.2f}%)."
                ),
                "",
                (
                    "The oracle decline at larger taxon counts identifies the fixed 1,000-triplet "
                    "cap as the dominant information constraint in that range."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "REPORTING NOTE",
            "",
            (
                "The 4 through 20, 21 through 40, and above-40 interpretations were developed "
                "during post-run diagnosis. Describe them as diagnostic strata, not as "
                "prespecified confirmatory strata."
            ),
            (
                "The 41 through 180 taxon results measure behavior under a fixed 1,000-triplet "
                "budget. They should be reported as a scalability stress test, not as evidence "
                "of accurate large-tree reconstruction."
            ),
            (
                "External STELAR, MP-EST, and Triplet MaxCut results remain incomplete unless "
                "their per-simulation evaluation files contain all 5,310 simulations."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    oracle_path = (
        args.oracle.resolve()
        if args.oracle is not None
        else run_root / "diagnostics" / "oracle_fm_4_180_normalized" / "oracle_per_simulation.csv"
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else run_root / "comparison" / "final_report"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    method_frames = []
    reference_ids: set[str] | None = None
    for mode, display in METHODS.items():
        source = run_root / "evaluation" / mode / "per_simulation.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Missing completed evaluation: {source}")
        frame = standardize(pd.read_csv(source), source)
        if len(frame) != args.expected_simulations:
            raise ValueError(
                f"{source} contains {len(frame):,} rows; expected "
                f"{args.expected_simulations:,}"
            )
        ids = set(frame["simulation_id"])
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"Simulation IDs in {source} do not match the reference method")
        frame["method"] = display
        method_frames.append(frame)

    all_methods = pd.concat(method_frames, ignore_index=True)
    all_methods["method"] = pd.Categorical(
        all_methods["method"], categories=DISPLAY_ORDER, ordered=True
    )
    per_taxon_counts = all_methods.groupby(
        ["method", "tree_size"], observed=True
    ).size()
    if not per_taxon_counts.eq(30).all():
        bad = per_taxon_counts.loc[~per_taxon_counts.eq(30)]
        raise ValueError(
            "Expected 30 simulations per method and taxon count; unexpected counts: "
            f"{bad.head(10).to_dict()}"
        )
    full = method_summary(all_methods)
    primary = method_summary(all_methods.loc[all_methods["tree_size"].between(4, 20)])
    taxon = by_taxon(all_methods)
    regime = by_regime(all_methods)

    write_csv(full, out_dir / "table_internal_all_taxa.csv")
    write_csv(primary, out_dir / "table_internal_4_20.csv")
    write_csv(taxon, out_dir / "table_internal_by_taxon_count.csv")
    write_csv(regime, out_dir / "table_internal_by_taxon_stratum.csv")
    plot_internal_taxon_curves(taxon, out_dir)

    primary_rows = all_methods.loc[all_methods["method"].eq(PRIMARY_DISPLAY)].copy()
    oracle_available = oracle_path.is_file()
    oracle_regime = pd.DataFrame()
    if oracle_available:
        oracle = standardize(pd.read_csv(oracle_path), oracle_path, oracle=True)
        if len(oracle) != args.expected_simulations:
            raise ValueError(
                f"{oracle_path} contains {len(oracle):,} rows; expected "
                f"{args.expected_simulations:,}"
            )
        if set(oracle["simulation_id"]) != set(primary_rows["simulation_id"]):
            raise ValueError("Oracle and ML-soft simulation IDs do not match")
        oracle_regime = oracle_regime_table(primary_rows, oracle)
        write_csv(oracle_regime, out_dir / "table_ml_soft_vs_oracle_by_taxon_stratum.csv")
        plot_oracle_ceiling(oracle_regime, out_dir)

    note = results_note(full, primary, oracle_regime, oracle_available)
    (out_dir / "results_note.txt").write_text(note, encoding="utf-8")

    audit = {
        "run_root": str(run_root),
        "output_directory": str(out_dir),
        "methods": len(METHODS),
        "simulations_per_method": args.expected_simulations,
        "unique_taxon_counts": int(all_methods["tree_size"].nunique()),
        "minimum_taxa": int(all_methods["tree_size"].min()),
        "maximum_taxa": int(all_methods["tree_size"].max()),
        "oracle_included": oracle_available,
        "oracle_path": str(oracle_path),
    }
    (out_dir / "report_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    print("FINAL REPORTING STAGE COMPLETE")
    print(json.dumps(audit, indent=2))
    print("\nPrimary 4 through 20 taxon table")
    print(
        primary[
            [
                "method",
                "simulations",
                "exact_tree_count",
                "exact_match_accuracy",
                "mean_normalized_rooted_rf",
            ]
        ].to_string(index=False)
    )
    print(f"\nFiles written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
