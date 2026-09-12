from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .newick import Node, parse_newick


def leaf_names(root: Node) -> frozenset[str]:
    names = [leaf.name for leaf in root.leaves()]
    if any(name is None for name in names):
        raise ValueError("Tree contains an unnamed leaf")
    if len(names) != len(set(names)):
        raise ValueError("Tree contains duplicate taxon labels")
    return frozenset(str(name) for name in names)


def is_rooted_binary(root: Node) -> bool:
    return all(node.is_leaf or len(node.children) == 2 for node in root.preorder())


def rooted_clades(root: Node) -> set[frozenset[str]]:
    all_taxa = frozenset(x.name for x in root.leaves() if x.name is not None)
    clades: set[frozenset[str]] = set()

    def visit(node: Node) -> frozenset[str]:
        if node.is_leaf:
            return frozenset((node.name,))
        taxa = frozenset().union(*(visit(child) for child in node.children))
        if 1 < len(taxa) < len(all_taxa):
            clades.add(taxa)
        return taxa

    visit(root)
    return clades


def rooted_rf(true_newick: str, inferred_newick: str) -> tuple[int, float, bool]:
    true_tree = parse_newick(true_newick)
    inferred_tree = parse_newick(inferred_newick)
    true_taxa = leaf_names(true_tree)
    inferred_taxa = leaf_names(inferred_tree)
    if true_taxa != inferred_taxa:
        missing = sorted(true_taxa - inferred_taxa)
        extra = sorted(inferred_taxa - true_taxa)
        raise ValueError(f"Taxon mismatch; missing={missing}, extra={extra}")
    true_clades = rooted_clades(true_tree)
    inferred_clades = rooted_clades(inferred_tree)
    distance = len(true_clades.symmetric_difference(inferred_clades))
    denominator = len(true_clades) + len(inferred_clades)
    normalized = distance / denominator if denominator else 0.0
    return distance, normalized, true_clades == inferred_clades


def evaluate_assembled(assembled_path: str | Path, out_dir: str | Path) -> dict:
    frame = pd.read_csv(assembled_path)
    rows = []
    for row in frame.itertuples(index=False):
        record = row._asdict()
        try:
            true_tree = parse_newick(row.backbone_newick)
            inferred_tree = parse_newick(row.inferred_newick)
            true_taxa = leaf_names(true_tree)
            inferred_taxa = leaf_names(inferred_tree)
            taxon_match = true_taxa == inferred_taxa
            if not taxon_match:
                raise ValueError(
                    f"Taxon mismatch; missing={sorted(true_taxa - inferred_taxa)}, "
                    f"extra={sorted(inferred_taxa - true_taxa)}"
                )
            rf, nrf, exact = rooted_rf(row.backbone_newick, row.inferred_newick)
            record.update(
                {
                    "evaluation_status": "ok",
                    "evaluation_error": "",
                    "taxon_match": True,
                    "fully_resolved_inferred_tree": is_rooted_binary(inferred_tree),
                    "rooted_rf": rf,
                    "normalized_rooted_rf": nrf,
                    "exact_match": exact,
                }
            )
        except Exception as exc:
            message = str(exc)
            record.update(
                {
                    "evaluation_status": "taxon_mismatch" if "Taxon mismatch" in message else "failed",
                    "evaluation_error": message,
                    "taxon_match": False if "Taxon mismatch" in message else np.nan,
                    "fully_resolved_inferred_tree": False,
                    "rooted_rf": np.nan,
                    "normalized_rooted_rf": np.nan,
                    "exact_match": False,
                }
            )
        rows.append(record)
    results = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "per_simulation.csv", index=False)

    strata = []
    candidate_strata = [
        ["tree_size"],
        ["triplets_per_simulation"],
        ["gamma"],
        ["ils_length"],
        ["gene_tree_count"],
        ["pair_type"],
        ["backbone_id"],
        ["backbone_shape_class"],
        ["backbone_balance_bin"],
        ["perturbation", "perturbation_level"],
        ["gamma", "ils_length"],
    ]
    for columns in [values for values in candidate_strata if all(value in results.columns for value in values)]:
        grouped = (
            results.groupby(columns, dropna=False)
            .agg(
                simulations=("simulation_id", "count"),
                evaluated_simulations=("evaluation_status", lambda values: int((values == "ok").sum())),
                failed_simulations=("evaluation_status", lambda values: int((values != "ok").sum())),
                exact_match_accuracy=("exact_match", "mean"),
                mean_normalized_rooted_rf=("normalized_rooted_rf", "mean"),
                median_normalized_rooted_rf=("normalized_rooted_rf", "median"),
                fully_resolved_rate=("fully_resolved_inferred_tree", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "stratum", "+".join(columns))
        strata.append(grouped)
    if strata:
        pd.concat(strata, ignore_index=True).to_csv(out_dir / "stratified_metrics.csv", index=False)

    requested = int(len(results))
    evaluated = int((results["evaluation_status"] == "ok").sum())
    exact_count = int(results["exact_match"].sum())
    valid_nrf = results.loc[results["evaluation_status"] == "ok", "normalized_rooted_rf"]
    summary = {
        "simulation_count": requested,
        "requested_simulations": requested,
        "evaluated_simulations": evaluated,
        "failed_simulations": requested - evaluated,
        "failure_rate": float((requested - evaluated) / requested) if requested else 0.0,
        "exact_tree_count": exact_count,
        "exact_match_accuracy": float(exact_count / requested) if requested else np.nan,
        "rooted_exact_match_accuracy": float(exact_count / requested) if requested else np.nan,
        "mean_normalized_rooted_rf": float(valid_nrf.mean()) if not valid_nrf.empty else None,
        "median_normalized_rooted_rf": float(valid_nrf.median()) if not valid_nrf.empty else None,
        "fully_resolved_inferred_trees": int(results["fully_resolved_inferred_tree"].sum()),
        "taxon_mismatches": int((results["evaluation_status"] == "taxon_mismatch").sum()),
    }
    if "assembly_seconds" in results.columns:
        summary.update(
            {
                "total_assembly_seconds": float(results["assembly_seconds"].sum()),
                "mean_assembly_seconds": float(results["assembly_seconds"].mean()),
                "median_assembly_seconds": float(results["assembly_seconds"].median()),
            }
        )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
