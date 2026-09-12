#!/usr/bin/env python3
"""Evaluate the existing FM assembler with true rooted-triplet labels.

This diagnostic reuses retained feature shards. It does not simulate gene trees,
extract new features, retrain the classifier, or overwrite the completed run.
For every selected feature row, the stored true label is converted to a one-hot
probability vector and passed to the same assembly implementation used by the
main pipeline.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Permit direct execution from an unpacked source checkout without requiring a
# separate editable installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"
if SOURCE_DIRECTORY.is_dir() and str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from introgression_triplets.assembly import assemble_tree, probability_topologies
from introgression_triplets.evaluate import is_rooted_binary, leaf_names, rooted_rf
from introgression_triplets.newick import parse_newick


DEFAULT_MANIFEST = Path(
    "data/manifests/supervisor_scope_corrected_deadline_analysis.csv"
)
DEFAULT_FEATURES = Path("data/features/supervisor_scope_corrected_shards")
DEFAULT_BASELINE = Path(
    "outputs/supervisor_scope/supervisor_scope_corrected_deadline/"
    "evaluation/ml-soft/per_simulation.csv"
)
DEFAULT_OUTPUT = Path(
    "outputs/supervisor_scope/supervisor_scope_corrected_deadline/"
    "diagnostics/oracle_fm_11_20"
)

FEATURE_COLUMNS = [
    "simulation_id",
    "triplet_id",
    "taxon_a",
    "taxon_b",
    "taxon_c",
    "label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current tree assembler with stored true triplet labels. "
            "This isolates classifier error from assembly and coverage error."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--baseline-evaluation", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-taxa", type=int, default=11)
    parser.add_argument("--max-taxa", type=int, default=20)
    parser.add_argument("--method", choices=["fm", "exact"], default="fm")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print progress after this many completed simulations.",
    )
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def coverage_statistics(features: pd.DataFrame, taxa: set[str]) -> dict[str, float | int]:
    taxon_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for row in features.itertuples(index=False):
        triple = (str(row.taxon_a), str(row.taxon_b), str(row.taxon_c))
        taxon_counts.update(triple)
        pair_counts.update(itertools.combinations(sorted(triple), 2))

    n = len(taxa)
    possible_triplets = math.comb(n, 3)
    possible_pairs = math.comb(n, 2)
    taxon_values = [taxon_counts[taxon] for taxon in sorted(taxa)]
    pair_values = [pair_counts[pair] for pair in itertools.combinations(sorted(taxa), 2)]
    return {
        "selected_triplets": int(len(features)),
        "possible_triplets": int(possible_triplets),
        "triplet_coverage": float(len(features) / possible_triplets),
        "covered_taxa": int(sum(value > 0 for value in taxon_values)),
        "minimum_triplets_per_taxon": int(min(taxon_values)),
        "maximum_triplets_per_taxon": int(max(taxon_values)),
        "covered_pairs": int(sum(value > 0 for value in pair_values)),
        "possible_pairs": int(possible_pairs),
        "pair_coverage": float(sum(value > 0 for value in pair_values) / possible_pairs),
        "minimum_triplets_per_pair": int(min(pair_values)),
        "maximum_triplets_per_pair": int(max(pair_values)),
    }


def oracle_task(payload: tuple[dict, str, str]) -> dict:
    row_dict, feature_directory, method = payload
    simulation_id = str(row_dict["simulation_id"])
    started = time.perf_counter()
    record = dict(row_dict)
    record.update(
        {
            "oracle_method": method,
            "oracle_status": "failed",
            "oracle_error": "",
            "inferred_newick": "",
            "rooted_rf": np.nan,
            "normalized_rooted_rf": np.nan,
            "exact_match": False,
            "fully_resolved_inferred_tree": False,
        }
    )

    try:
        feature_path = Path(feature_directory) / f"{simulation_id}.csv.gz"
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing feature shard: {feature_path}")
        features = pd.read_csv(feature_path, usecols=FEATURE_COLUMNS)
        if features.empty:
            raise ValueError("Feature shard is empty")
        observed_ids = set(features["simulation_id"].astype(str))
        if observed_ids != {simulation_id}:
            raise ValueError(f"Feature shard contains simulation IDs: {sorted(observed_ids)}")
        if features["triplet_id"].duplicated().any():
            raise ValueError("Feature shard contains duplicate triplet IDs")

        numeric_labels = pd.to_numeric(features["label"], errors="raise")
        labels = numeric_labels.to_numpy(dtype=int)
        if not np.array_equal(labels.astype(float), numeric_labels.to_numpy(dtype=float)):
            raise ValueError("Triplet labels are not integers")
        invalid_labels = sorted(set(labels) - {0, 1, 2})
        if invalid_labels:
            raise ValueError(f"Invalid triplet labels: {invalid_labels}")

        true_tree = parse_newick(str(row_dict["backbone_newick"]))
        true_taxa = set(leaf_names(true_tree))
        feature_taxa = set(
            features[["taxon_a", "taxon_b", "taxon_c"]]
            .astype(str)
            .to_numpy()
            .ravel()
        )
        if true_taxa != feature_taxa:
            raise ValueError(
                "Feature taxa do not match the backbone; "
                f"missing={sorted(true_taxa - feature_taxa)}, "
                f"extra={sorted(feature_taxa - true_taxa)}"
            )
        manifest_size = int(row_dict["tree_size"])
        if len(true_taxa) != manifest_size:
            raise ValueError(
                f"Manifest tree_size is {manifest_size}, but the backbone has "
                f"{len(true_taxa)} taxa"
            )

        record.update(coverage_statistics(features, true_taxa))
        probabilities = np.eye(3, dtype=float)[labels]
        topologies = []
        for index, row in enumerate(features.itertuples(index=False)):
            topologies.extend(
                probability_topologies(
                    (str(row.taxon_a), str(row.taxon_b), str(row.taxon_c)),
                    probabilities[index],
                )
            )

        inferred_newick = assemble_tree(true_taxa, topologies, method=method).to_newick()
        rf_distance, normalized_rf, exact_match = rooted_rf(
            str(row_dict["backbone_newick"]), inferred_newick
        )
        record.update(
            {
                "oracle_status": "completed",
                "inferred_newick": inferred_newick,
                "rooted_rf": int(rf_distance),
                "normalized_rooted_rf": float(normalized_rf),
                "exact_match": bool(exact_match),
                "fully_resolved_inferred_tree": bool(
                    is_rooted_binary(parse_newick(inferred_newick))
                ),
            }
        )
    except Exception as exc:
        record["oracle_error"] = f"{type(exc).__name__}: {exc}"
    record["oracle_seconds"] = float(time.perf_counter() - started)
    return record


def summarize_by_taxon(results: pd.DataFrame) -> pd.DataFrame:
    completed = results.loc[results["oracle_status"].eq("completed")].copy()
    if completed.empty:
        return pd.DataFrame()
    return (
        completed.groupby("tree_size", sort=True)
        .agg(
            simulations=("simulation_id", "count"),
            exact_tree_count=("exact_match", "sum"),
            exact_match_accuracy=("exact_match", "mean"),
            mean_normalized_rooted_rf=("normalized_rooted_rf", "mean"),
            median_normalized_rooted_rf=("normalized_rooted_rf", "median"),
            mean_triplet_coverage=("triplet_coverage", "mean"),
            minimum_pair_coverage=("pair_coverage", "min"),
            mean_oracle_seconds=("oracle_seconds", "mean"),
        )
        .reset_index()
    )


def compare_with_baseline(
    results: pd.DataFrame, baseline_path: Path
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not baseline_path.is_file():
        return None, None
    baseline = pd.read_csv(
        baseline_path,
        usecols=lambda column: column
        in {
            "simulation_id",
            "exact_match",
            "normalized_rooted_rf",
            "evaluation_status",
        },
    )
    required = {"simulation_id", "exact_match", "normalized_rooted_rf"}
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(f"Baseline evaluation is missing columns: {sorted(missing)}")
    if baseline["simulation_id"].astype(str).duplicated().any():
        raise ValueError("Baseline evaluation contains duplicate simulation IDs")
    baseline = baseline.copy()
    baseline["simulation_id"] = baseline["simulation_id"].astype(str)
    baseline["model_exact_match"] = as_boolean(baseline["exact_match"])
    baseline = baseline.rename(
        columns={"normalized_rooted_rf": "model_normalized_rooted_rf"}
    )
    baseline = baseline[
        ["simulation_id", "model_exact_match", "model_normalized_rooted_rf"]
    ]

    selected = results.loc[
        results["oracle_status"].eq("completed"),
        [
            "simulation_id",
            "tree_size",
            "exact_match",
            "normalized_rooted_rf",
        ],
    ].copy()
    selected["simulation_id"] = selected["simulation_id"].astype(str)
    selected = selected.rename(
        columns={
            "exact_match": "oracle_exact_match",
            "normalized_rooted_rf": "oracle_normalized_rooted_rf",
        }
    )
    comparison = selected.merge(
        baseline,
        on="simulation_id",
        how="left",
        validate="one_to_one",
    )
    if comparison["model_exact_match"].isna().any():
        missing_ids = comparison.loc[
            comparison["model_exact_match"].isna(), "simulation_id"
        ].head(5)
        raise ValueError(
            "Baseline does not cover all selected simulations. First missing IDs: "
            + ", ".join(missing_ids)
        )
    comparison["model_exact_match"] = comparison["model_exact_match"].astype(bool)
    comparison["oracle_exact_match"] = comparison["oracle_exact_match"].astype(bool)
    comparison["diagnostic_class"] = np.select(
        [
            comparison["oracle_exact_match"] & comparison["model_exact_match"],
            comparison["oracle_exact_match"] & ~comparison["model_exact_match"],
            ~comparison["oracle_exact_match"] & comparison["model_exact_match"],
        ],
        [
            "both_exact",
            "classifier_or_weighting_limited",
            "model_exact_oracle_fm_not_exact",
        ],
        default="assembly_or_coverage_limited",
    )
    grouped = (
        comparison.groupby("tree_size", sort=True)
        .agg(
            simulations=("simulation_id", "count"),
            model_exact_accuracy=("model_exact_match", "mean"),
            oracle_exact_accuracy=("oracle_exact_match", "mean"),
            model_mean_normalized_rooted_rf=("model_normalized_rooted_rf", "mean"),
            oracle_mean_normalized_rooted_rf=("oracle_normalized_rooted_rf", "mean"),
            classifier_or_weighting_limited=(
                "diagnostic_class",
                lambda values: int((values == "classifier_or_weighting_limited").sum()),
            ),
            assembly_or_coverage_limited=(
                "diagnostic_class",
                lambda values: int((values == "assembly_or_coverage_limited").sum()),
            ),
        )
        .reset_index()
    )
    grouped["exact_accuracy_gain_with_oracle_labels"] = (
        grouped["oracle_exact_accuracy"] - grouped["model_exact_accuracy"]
    )
    return comparison, grouped


def main() -> int:
    args = parse_args()
    if args.min_taxa < 3 or args.max_taxa < args.min_taxa:
        raise ValueError("Require 3 <= min-taxa <= max-taxa")
    if args.jobs < 1:
        raise ValueError("--jobs must be at least one")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least one")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.features_dir.is_dir():
        raise FileNotFoundError(args.features_dir)

    manifest = pd.read_csv(args.manifest)
    required = {"simulation_id", "tree_size", "backbone_newick"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest["simulation_id"].astype(str).duplicated().any():
        raise ValueError("Manifest contains duplicate simulation IDs")
    selected = manifest.loc[
        manifest["tree_size"].between(args.min_taxa, args.max_taxa)
    ].copy()
    if selected.empty:
        raise ValueError("No manifest rows fall within the requested taxon range")
    if args.method == "exact" and int(selected["tree_size"].max()) > 12:
        raise ValueError("The package restricts method=exact to at most 12 taxa")
    selected["_manifest_order"] = np.arange(len(selected), dtype=int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected = len(selected)
    print("Oracle triplet assembly diagnostic", flush=True)
    print(f"Simulations: {expected}", flush=True)
    print(f"Taxa: {args.min_taxa} to {args.max_taxa}", flush=True)
    print(f"Method: {args.method}", flush=True)
    print("Gene-tree simulation: not run", flush=True)
    print("Model training: not run", flush=True)

    payloads = [
        (row, str(args.features_dir), args.method)
        for row in selected.to_dict(orient="records")
    ]
    started = time.perf_counter()
    records: list[dict] = []

    if args.jobs == 1:
        iterator = (oracle_task(payload) for payload in payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.jobs)
        futures = [executor.submit(oracle_task, payload) for payload in payloads]
        iterator = (future.result() for future in as_completed(futures))

    try:
        for completed, record in enumerate(iterator, start=1):
            records.append(record)
            if completed % args.progress_every == 0 or completed == expected:
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = (expected - completed) / rate if rate > 0 else float("nan")
                percent = 100.0 * completed / expected
                print(
                    f"Progress: {completed}/{expected} ({percent:.1f}%), "
                    f"elapsed {elapsed / 60:.1f} min, ETA {remaining / 60:.1f} min",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    results = pd.DataFrame(records).sort_values("_manifest_order").reset_index(drop=True)
    results = results.drop(columns=["_manifest_order"])
    atomic_csv(results, args.out_dir / "oracle_per_simulation.csv")
    by_taxon = summarize_by_taxon(results)
    if not by_taxon.empty:
        atomic_csv(by_taxon, args.out_dir / "oracle_by_taxon_count.csv")

    comparison, comparison_by_taxon = compare_with_baseline(
        results, args.baseline_evaluation
    )
    if comparison is not None and comparison_by_taxon is not None:
        atomic_csv(comparison, args.out_dir / "oracle_vs_model_per_simulation.csv")
        atomic_csv(
            comparison_by_taxon,
            args.out_dir / "oracle_vs_model_by_taxon_count.csv",
        )

    completed_results = results.loc[results["oracle_status"].eq("completed")]
    failures = int(len(results) - len(completed_results))
    summary = {
        "method": args.method,
        "minimum_taxa": int(args.min_taxa),
        "maximum_taxa": int(args.max_taxa),
        "requested_simulations": int(len(results)),
        "completed_simulations": int(len(completed_results)),
        "failed_simulations": failures,
        "oracle_exact_tree_count": int(completed_results["exact_match"].sum()),
        "oracle_exact_match_accuracy": (
            float(completed_results["exact_match"].mean())
            if not completed_results.empty
            else None
        ),
        "oracle_mean_normalized_rooted_rf": (
            float(completed_results["normalized_rooted_rf"].mean())
            if not completed_results.empty
            else None
        ),
        "wall_seconds": float(time.perf_counter() - started),
        "baseline_comparison_created": comparison is not None,
    }
    if comparison is not None:
        summary.update(
            {
                "model_exact_match_accuracy": float(
                    comparison["model_exact_match"].mean()
                ),
                "model_mean_normalized_rooted_rf": float(
                    comparison["model_normalized_rooted_rf"].mean()
                ),
                "classifier_or_weighting_limited_count": int(
                    (comparison["diagnostic_class"] == "classifier_or_weighting_limited").sum()
                ),
                "assembly_or_coverage_limited_count": int(
                    (comparison["diagnostic_class"] == "assembly_or_coverage_limited").sum()
                ),
            }
        )
    atomic_json(summary, args.out_dir / "oracle_summary.json")

    print("\nORACLE SUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        print(
            f"\nWarning: {failures} simulations failed. Inspect oracle_per_simulation.csv.",
            flush=True,
        )
        return 1
    print(f"\nResults: {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
