#!/usr/bin/env python3
"""Evaluate one-tree-per-file rooted species-tree benchmark outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from introgression_triplets.evaluate import rooted_rf  # noqa: E402
from introgression_triplets.newick import NewickError, parse_newick, to_newick  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rooted external species-tree estimates.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trees-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return success even when output trees are missing or invalid.",
    )
    return parser.parse_args()


def first_newick(text: str) -> str:
    start = text.find("(")
    if start < 0:
        raise NewickError("No Newick opening parenthesis found")
    bracket_depth = 0
    quote: str | None = None
    for index in range(start, len(text)):
        character = text[index]
        if quote:
            if character == quote:
                quote = None
            continue
        if character in ("'", '"'):
            quote = character
        elif character == "[":
            bracket_depth += 1
        elif character == "]" and bracket_depth:
            bracket_depth -= 1
        elif character == ";" and bracket_depth == 0:
            return text[start : index + 1]
    raise NewickError("Newick statement has no terminating semicolon")


def normalized_tree(path: Path) -> str:
    root = parse_newick(first_newick(path.read_text(encoding="utf-8")))
    return to_newick(root, include_lengths=False).strip()


def taxa(root) -> set[str]:
    labels = [leaf.name for leaf in root.leaves()]
    if any(label is None for label in labels):
        raise NewickError("Unnamed leaf")
    if len(labels) != len(set(labels)):
        raise NewickError("Duplicate leaf labels")
    return set(labels)


def is_fully_resolved(root) -> bool:
    return all(node.is_leaf or len(node.children) == 2 for node in root.preorder())


def main() -> int:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    if manifest.empty:
        raise ValueError("Manifest is empty")
    if manifest["simulation_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate simulation_id values")

    evaluated: list[dict] = []
    failures: list[dict] = []
    for row in manifest.to_dict(orient="records"):
        simulation_id = row["simulation_id"]
        path = args.trees_dir / f"{simulation_id}.tre"
        try:
            if not path.is_file():
                raise FileNotFoundError(str(path))
            inferred_newick = normalized_tree(path)
            true_root = parse_newick(row["backbone_newick"])
            inferred_root = parse_newick(inferred_newick)
            true_taxa = taxa(true_root)
            inferred_taxa = taxa(inferred_root)
            if inferred_taxa != true_taxa:
                raise ValueError(
                    f"taxon mismatch: expected={sorted(true_taxa)}, observed={sorted(inferred_taxa)}"
                )
            distance, normalized, exact = rooted_rf(row["backbone_newick"], inferred_newick)
            evaluated.append(
                {
                    **row,
                    "method": args.method,
                    "inferred_newick": inferred_newick,
                    "rooted_rf": distance,
                    "normalized_rooted_rf": normalized,
                    "exact_match": bool(exact),
                    "fully_resolved": is_fully_resolved(inferred_root),
                }
            )
        except (OSError, ValueError, NewickError) as error:
            failures.append(
                {
                    "simulation_id": simulation_id,
                    "path": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    results = pd.DataFrame(evaluated)
    failure_frame = pd.DataFrame(failures, columns=("simulation_id", "path", "error"))
    run_records_path = args.trees_dir / "run_records.csv"
    if not results.empty and run_records_path.is_file():
        run_records = pd.read_csv(run_records_path, usecols=["simulation_id", "runtime_seconds"])
        results = results.merge(run_records, on="simulation_id", how="left", validate="one_to_one")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "per_simulation.csv", index=False)
    failure_frame.to_csv(args.out_dir / "failures.csv", index=False)

    strata = []
    for columns in (
        ["tree_size"],
        ["triplets_per_simulation"],
        ["gamma"],
        ["ils_length"],
        ["gene_tree_count"],
        ["pair_type"],
        ["backbone_id"],
        ["backbone_shape_class"],
        ["backbone_balance_bin"],
        ["gamma", "ils_length"],
    ):
        if results.empty or any(column not in results.columns for column in columns):
            continue
        grouped = (
            results.groupby(columns, dropna=False)
            .agg(
                simulations=("simulation_id", "count"),
                exact_tree_count=("exact_match", "sum"),
                exact_match_accuracy=("exact_match", "mean"),
                mean_normalized_rooted_rf=("normalized_rooted_rf", "mean"),
                median_normalized_rooted_rf=("normalized_rooted_rf", "median"),
                fully_resolved_trees=("fully_resolved", "sum"),
            )
            .reset_index()
        )
        grouped.insert(0, "stratum", "+".join(columns))
        strata.append(grouped)
    if strata:
        pd.concat(strata, ignore_index=True).to_csv(
            args.out_dir / "stratified_metrics.csv", index=False
        )

    summary = {
        "method": args.method,
        "requested_simulations": int(len(manifest)),
        "evaluated_simulations": int(len(results)),
        "failed_simulations": int(len(failures)),
        "exact_tree_count": int(results["exact_match"].sum()) if not results.empty else 0,
        "rooted_exact_match_accuracy": (
            float(results["exact_match"].mean()) if not results.empty else None
        ),
        "mean_normalized_rooted_rf": (
            float(results["normalized_rooted_rf"].mean()) if not results.empty else None
        ),
        "median_normalized_rooted_rf": (
            float(results["normalized_rooted_rf"].median()) if not results.empty else None
        ),
        "fully_resolved_inferred_trees": (
            int(results["fully_resolved"].sum()) if not results.empty else 0
        ),
        "taxon_mismatches": int(
            failure_frame["error"].str.contains("taxon mismatch", case=False).sum()
        )
        if not failure_frame.empty
        else 0,
    }
    if "runtime_seconds" in results.columns:
        runtimes = pd.to_numeric(results["runtime_seconds"], errors="coerce")
        summary.update(
            {
                "total_runtime_seconds": float(runtimes.sum()),
                "mean_runtime_seconds": float(runtimes.mean()),
                "median_runtime_seconds": float(runtimes.median()),
            }
        )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    incomplete = len(failures) > 0 or len(results) != len(manifest)
    return 2 if incomplete and not args.allow_incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
