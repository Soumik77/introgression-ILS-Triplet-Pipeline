from __future__ import annotations

import itertools
import math
import os
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .newick import parse_newick, read_newick_lines
from .triplets import TripletBatchPlan, TripletTreeIndex, topology_labels, triplet_combinations


BASE13_FEATURES = [
    "p0",
    "p1",
    "p2",
    "g_asym_major",
    "mean_bl0",
    "median_bl0",
    "std_bl0",
    "mean_bl1",
    "median_bl1",
    "std_bl1",
    "mean_bl2",
    "median_bl2",
    "std_bl2",
]

ENHANCED_FEATURES = BASE13_FEATURES + ["g_asym0", "g_asym1", "g_asym2", "log_gene_tree_count"]

FREQUENCY_FEATURES = ["p0", "p1", "p2"]
BRANCH_LENGTH_FEATURES = [
    "mean_bl0",
    "median_bl0",
    "std_bl0",
    "mean_bl1",
    "median_bl1",
    "std_bl1",
    "mean_bl2",
    "median_bl2",
    "std_bl2",
]
ASYMMETRY_FEATURES = ["g_asym_major", "g_asym0", "g_asym1", "g_asym2"]

FEATURE_SETS = {
    "base13": BASE13_FEATURES,
    "enhanced": ENHANCED_FEATURES,
    "frequency_only": FREQUENCY_FEATURES,
    "branch_only": BRANCH_LENGTH_FEATURES,
    "frequency_branch": FREQUENCY_FEATURES + BRANCH_LENGTH_FEATURES,
    "frequency_asymmetry": FREQUENCY_FEATURES + ASYMMETRY_FEATURES,
}


def feature_columns(name: str) -> list[str]:
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {name}")
    return FEATURE_SETS[name].copy()


def _stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (np.nan, np.nan, np.nan)
    array = np.asarray(values, dtype=float)
    return (float(np.mean(array)), float(np.median(array)), float(np.std(array, ddof=0)))


def _asymmetry_for_candidate(p: np.ndarray, candidate: int) -> float:
    j, k = [i for i in range(3) if i != candidate]
    denominator = p[j] + p[k]
    return float(abs(p[j] - p[k]) / denominator) if denominator > 0 else 0.0


def _selected_triplet_combinations(
    taxa: list[str], manifest_row: pd.Series
) -> list[tuple[str, str, str]]:
    if "triplets_per_simulation" not in manifest_row.index or pd.isna(
        manifest_row["triplets_per_simulation"]
    ):
        return triplet_combinations(taxa)
    requested = int(manifest_row["triplets_per_simulation"])
    total = math.comb(len(taxa), 3)
    if requested >= total:
        return triplet_combinations(taxa)
    if requested < (len(taxa) + 2) // 3:
        raise ValueError("triplets_per_simulation is too small to cover every taxon")

    rng = random.Random(int(manifest_row["seed"]) + 17_171)
    shuffled_taxa = sorted(taxa)
    rng.shuffle(shuffled_taxa)
    selected: set[tuple[str, str, str]] = set()
    for start in range(0, len(shuffled_taxa), 3):
        group = shuffled_taxa[start : start + 3]
        while len(group) < 3:
            candidate = rng.choice(taxa)
            if candidate not in group:
                group.append(candidate)
        selected.add(tuple(sorted(group)))

    if requested > total // 5:
        candidates = triplet_combinations(taxa)
        rng.shuffle(candidates)
        for combination in candidates:
            selected.add(combination)
            if len(selected) == requested:
                break
    else:
        while len(selected) < requested:
            selected.add(tuple(sorted(rng.sample(taxa, 3))))
    return sorted(selected)


def summarize_gene_trees(tree_path: str | Path, manifest_row: pd.Series) -> pd.DataFrame:
    backbone = parse_newick(str(manifest_row.backbone_newick))
    taxa = sorted(x.name for x in backbone.leaves() if x.name is not None)
    combos = _selected_triplet_combinations(taxa, manifest_row)
    plan = TripletBatchPlan.from_combinations(combos)
    backbone_index = TripletTreeIndex(backbone)
    true_indices, _ = backbone_index.extract_many(plan)

    expected_g = int(manifest_row.gene_tree_count)
    topology_matrix = np.empty((expected_g, len(combos)), dtype=np.int8)
    length_matrix = np.empty((expected_g, len(combos)), dtype=float)
    observed_g = 0
    for tree in read_newick_lines(str(tree_path)):
        if observed_g >= expected_g:
            raise ValueError(f"{tree_path}: expected {expected_g} gene trees, found more")
        observed_g += 1
        index = TripletTreeIndex(tree)
        indices, branch_lengths = index.extract_many(plan)
        topology_matrix[observed_g - 1] = indices
        length_matrix[observed_g - 1] = branch_lengths

    if observed_g != expected_g:
        raise ValueError(f"{tree_path}: expected {expected_g} gene trees, found {observed_g}")

    counts = np.stack(
        [(topology_matrix == index).sum(axis=0) for index in range(3)], axis=1
    )
    means = np.full((3, len(combos)), np.nan)
    medians = np.full((3, len(combos)), np.nan)
    standard_deviations = np.full((3, len(combos)), np.nan)
    for topology_index in range(3):
        mask = topology_matrix == topology_index
        values = np.where(mask, length_matrix, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means[topology_index] = np.nanmean(values, axis=0)
            medians[topology_index] = np.nanmedian(values, axis=0)
            standard_deviations[topology_index] = np.nanstd(values, axis=0, ddof=0)

    records: list[dict] = []
    metadata_columns = [
        "simulation_id",
        "experiment_id",
        "backbone_id",
        "backbone_shape",
        "backbone_shape_class",
        "backbone_newick",
        "donor",
        "recipient",
        "direction",
        "gamma",
        "ils_length",
        "gene_tree_count",
        "replicate",
        "scenario_group",
        "seed",
    ]
    optional_metadata = [
        "taxon_set_id",
        "tree_size",
        "triplets_per_simulation",
        "backbone_balance_score",
        "backbone_balance_bin",
        "pair_type",
        "perturbation",
        "perturbation_level",
        "triplets_per_simulation",
    ]
    metadata = {name: manifest_row[name] for name in metadata_columns if name in manifest_row.index}
    metadata.update(
        {name: manifest_row[name] for name in optional_metadata if name in manifest_row.index}
    )

    for combo_index, combo in enumerate(combos):
        n = counts[combo_index]
        p = n / observed_g
        majority = int(np.argmax(p))
        triplet_id = ",".join(combo)
        record = {
            **metadata,
            "triplet_id": triplet_id,
            "taxon_a": combo[0],
            "taxon_b": combo[1],
            "taxon_c": combo[2],
            "topology0": topology_labels(combo)[0],
            "topology1": topology_labels(combo)[1],
            "topology2": topology_labels(combo)[2],
            "n0": int(n[0]),
            "n1": int(n[1]),
            "n2": int(n[2]),
            "p0": float(p[0]),
            "p1": float(p[1]),
            "p2": float(p[2]),
            "g_asym_major": _asymmetry_for_candidate(p, majority),
            "g_asym0": _asymmetry_for_candidate(p, 0),
            "g_asym1": _asymmetry_for_candidate(p, 1),
            "g_asym2": _asymmetry_for_candidate(p, 2),
            "log_gene_tree_count": float(np.log(observed_g)),
            "label": int(true_indices[combo_index]),
        }
        for i in range(3):
            record[f"mean_bl{i}"] = float(means[i, combo_index])
            record[f"median_bl{i}"] = float(medians[i, combo_index])
            record[f"std_bl{i}"] = float(standard_deviations[i, combo_index])
        records.append(record)
    return pd.DataFrame(records)


def extract_feature_table(manifest_path: str | Path, trees_dir: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    trees_dir = Path(trees_dir)
    frames = []
    for row in manifest.itertuples(index=False):
        tree_path = trees_dir / f"{row.simulation_id}.tre"
        if not tree_path.exists():
            raise FileNotFoundError(tree_path)
        frames.append(summarize_gene_trees(tree_path, pd.Series(row._asdict())))
    if not frames:
        raise ValueError("Manifest is empty")
    return pd.concat(frames, ignore_index=True)


def extract_feature_table_to_csv(
    manifest_path: str | Path, trees_dir: str | Path, output_path: str | Path
) -> dict[str, int | float]:
    """Extract features incrementally and atomically write a potentially large CSV."""

    started = time.perf_counter()
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError("Manifest is empty")
    trees_dir = Path(trees_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.name}.partial")
    partial_path.unlink(missing_ok=True)
    total_rows = 0
    minimum_triplets: int | None = None
    maximum_triplets = 0
    try:
        for index, row in enumerate(manifest.itertuples(index=False)):
            tree_path = trees_dir / f"{row.simulation_id}.tre"
            if not tree_path.exists():
                raise FileNotFoundError(tree_path)
            features = summarize_gene_trees(tree_path, pd.Series(row._asdict()))
            features.to_csv(partial_path, mode="a", header=index == 0, index=False)
            count = len(features)
            total_rows += count
            minimum_triplets = count if minimum_triplets is None else min(minimum_triplets, count)
            maximum_triplets = max(maximum_triplets, count)
        os.replace(partial_path, output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return {
        "rows": total_rows,
        "simulations": int(len(manifest)),
        "minimum_triplets_per_simulation": int(minimum_triplets or 0),
        "maximum_triplets_per_simulation": int(maximum_triplets),
        "wall_seconds": time.perf_counter() - started,
    }


def _extract_feature_shard_task(
    payload: tuple[dict, str, str, bool, bool]
) -> tuple[str, int, int]:
    row_dict, trees_name, output_name, delete_tree_files, overwrite = payload
    manifest_row = pd.Series(row_dict)
    simulation_id = str(manifest_row["simulation_id"])
    tree_path = Path(trees_name) / f"{simulation_id}.tre"
    shard_path = Path(output_name) / f"{simulation_id}.csv.gz"
    if shard_path.is_file() and shard_path.stat().st_size > 0 and not overwrite:
        check = pd.read_csv(shard_path, usecols=["simulation_id", "triplet_id"])
        if check.empty or set(check["simulation_id"].astype(str)) != {simulation_id}:
            raise RuntimeError(f"Existing feature shard verification failed: {simulation_id}")
        deleted = 0
        if delete_tree_files and tree_path.is_file():
            tree_path.unlink()
            deleted = 1
        return "skipped", 0, deleted
    if not tree_path.is_file():
        raise FileNotFoundError(tree_path)

    features = summarize_gene_trees(tree_path, manifest_row)
    partial_path = Path(output_name) / f".{simulation_id}.csv.gz.partial"
    partial_path.unlink(missing_ok=True)
    try:
        features.to_csv(
            partial_path,
            index=False,
            compression="gzip",
            float_format="%.10g",
        )
        check = pd.read_csv(
            partial_path,
            compression="gzip",
            usecols=["simulation_id", "triplet_id"],
        )
        if len(check) != len(features) or set(check["simulation_id"].astype(str)) != {
            simulation_id
        }:
            raise RuntimeError(f"Feature shard verification failed: {simulation_id}")
        os.replace(partial_path, shard_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    deleted = 0
    if delete_tree_files:
        tree_path.unlink()
        deleted = 1
    return "completed", len(features), deleted


def extract_feature_shards(
    manifest_path: str | Path,
    trees_dir: str | Path,
    output_dir: str | Path,
    *,
    start_row: int | None = None,
    end_row: int | None = None,
    delete_tree_files: bool = False,
    overwrite: bool = False,
    jobs: int = 1,
) -> dict[str, int | float | bool]:
    """Write one atomic gzip feature shard per simulation.

    This mode bounds peak disk use. A raw gene-tree file is deleted only after
    its compressed feature shard has been written and read back successfully.
    Row bounds are one-based and inclusive, matching the Julia simulator.
    """

    started = time.perf_counter()
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError("Manifest is empty")
    first = 1 if start_row is None else int(start_row)
    last = len(manifest) if end_row is None else int(end_row)
    if not 1 <= first <= last <= len(manifest):
        raise ValueError("Invalid one-based manifest row range")
    if jobs < 1:
        raise ValueError("jobs must be at least one")

    trees_dir = Path(trees_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    completed = 0
    skipped = 0
    deleted = 0
    minimum_triplets: int | None = None
    maximum_triplets = 0

    payloads = [
        (
            manifest.iloc[row_number - 1].to_dict(),
            str(trees_dir),
            str(output_dir),
            delete_tree_files,
            overwrite,
        )
        for row_number in range(first, last + 1)
    ]
    if jobs == 1:
        results = map(_extract_feature_shard_task, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=jobs)
        results = executor.map(_extract_feature_shard_task, payloads)
    try:
        for status, count, deleted_count in results:
            deleted += deleted_count
            if status == "skipped":
                skipped += 1
                continue
            completed += 1
            total_rows += count
            minimum_triplets = count if minimum_triplets is None else min(minimum_triplets, count)
            maximum_triplets = max(maximum_triplets, count)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    return {
        "requested_simulations": last - first + 1,
        "completed_simulations": completed,
        "skipped_existing": skipped,
        "rows_written": total_rows,
        "minimum_triplets_per_simulation": int(minimum_triplets or 0),
        "maximum_triplets_per_simulation": int(maximum_triplets),
        "deleted_tree_files": deleted,
        "delete_tree_files": delete_tree_files,
        "jobs": jobs,
        "wall_seconds": time.perf_counter() - started,
    }
