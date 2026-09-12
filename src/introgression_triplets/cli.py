from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .assembly import assemble_tree, probability_topologies
from .evaluate import evaluate_assembled
from .features import FEATURE_SETS, extract_feature_shards, extract_feature_table_to_csv
from .model import load_model, predict_frame, raw_predict_frame, train_grouped_cv
from .scalable import assemble_feature_shards, train_sharded_grouped_cv
from .simulation import make_manifest, select_balanced_analysis_manifest


def _write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def command_make_manifest(args) -> None:
    frame = make_manifest(args.config)
    _write_csv(frame, args.out)
    print(json.dumps({"manifest": str(args.out), "simulation_runs": len(frame)}, indent=2))


def command_select_manifest(args) -> None:
    manifest = pd.read_csv(args.manifest)
    selected = select_balanced_analysis_manifest(
        manifest, args.runs_per_backbone, args.seed
    )
    _write_csv(selected, args.out)
    summary = {
        "source_manifest": str(args.manifest),
        "selected_manifest": str(args.out),
        "candidate_runs": len(manifest),
        "selected_runs": len(selected),
        "backbones": int(selected["backbone_id"].nunique()),
        "runs_per_backbone": args.runs_per_backbone,
        "tree_sizes": sorted(selected["tree_size"].astype(int).unique().tolist()),
        "gene_trees": int(selected["gene_tree_count"].sum()),
    }
    print(json.dumps(summary, indent=2))


def command_extract_features(args) -> None:
    summary = extract_feature_table_to_csv(args.manifest, args.trees_dir, args.out)
    summary_path = Path(args.out).with_suffix(".extraction.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        json.dumps(
            {
                "feature_table": str(args.out),
                "extraction_summary": str(summary_path),
                **summary,
            },
            indent=2,
        )
    )


def command_extract_feature_shards(args) -> None:
    summary = extract_feature_shards(
        args.manifest,
        args.trees_dir,
        args.out_dir,
        start_row=args.start_row,
        end_row=args.end_row,
        delete_tree_files=args.delete_tree_files,
        overwrite=args.overwrite,
        jobs=args.jobs,
    )
    summary_path = Path(args.out_dir) / "extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"feature_shards": str(args.out_dir), **summary}, indent=2))


def command_train(args) -> None:
    metrics = train_grouped_cv(
        args.features,
        args.out_dir,
        args.feature_set,
        args.folds,
        args.seed,
        args.cv_mode,
        args.calibration,
        args.inner_folds,
        args.strata_column,
    )
    print(json.dumps(metrics, indent=2))


def command_train_sharded(args) -> None:
    metrics = train_sharded_grouped_cv(
        args.manifest,
        args.features_dir,
        args.out_dir,
        feature_set=args.feature_set,
        folds=args.folds,
        inner_folds=args.inner_folds,
        random_state=args.seed,
        calibration=args.calibration,
        training_rows_per_simulation=args.training_rows_per_simulation,
        final_calibration_rows_per_simulation=args.final_calibration_rows_per_simulation,
    )
    print(json.dumps(metrics, indent=2))


def _probability_matrix(frame: pd.DataFrame, args) -> np.ndarray:
    if args.weight_mode == "frequency":
        return frame[["p0", "p1", "p2"]].to_numpy(dtype=float)
    if args.weight_mode == "majority":
        raw = frame[["p0", "p1", "p2"]].to_numpy(dtype=float)
        result = np.zeros_like(raw)
        result[np.arange(len(raw)), raw.argmax(axis=1)] = 1.0
        return result
    if args.probabilities:
        predictions = pd.read_csv(args.probabilities)
        keys = ["simulation_id", "triplet_id"]
        prefix = "raw_prob" if args.probability_source == "raw" else "prob"
        probability_columns = [f"{prefix}{index}" for index in range(3)]
        needed = keys + probability_columns
        missing = [c for c in needed if c not in predictions.columns]
        if missing:
            raise ValueError(f"Probability file is missing columns: {missing}")
        merged = frame[keys].merge(predictions[needed], on=keys, how="left", validate="one_to_one")
        if merged[probability_columns].isna().any().any():
            raise ValueError("Probability file does not cover every feature row")
        probabilities = merged[probability_columns].to_numpy(dtype=float)
    elif args.model:
        bundle = load_model(args.model)
        probabilities = (
            raw_predict_frame(frame, bundle)
            if args.probability_source == "raw"
            else predict_frame(frame, bundle)
        )
    else:
        raise ValueError("ML weights require --probabilities (evaluation) or --model (new data)")

    mode = "ml-soft" if args.weight_mode == "ml" else args.weight_mode
    if mode == "ml-soft":
        return probabilities
    result = np.zeros_like(probabilities)
    winners = probabilities.argmax(axis=1)
    if mode == "ml-hard":
        result[np.arange(len(result)), winners] = 1.0
        return result
    if mode == "ml-confidence":
        result[np.arange(len(result)), winners] = probabilities.max(axis=1)
        return result
    raise ValueError(f"Unknown ML weight mode: {args.weight_mode}")


def command_assemble(args) -> None:
    frame = pd.read_csv(args.features)
    probabilities = _probability_matrix(frame, args)
    frame = frame.copy()
    frame[["weight0", "weight1", "weight2"]] = probabilities
    rows = []
    metadata = [
        "simulation_id",
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
        "tree_size",
        "triplets_per_simulation",
        "backbone_balance_score",
        "backbone_balance_bin",
        "pair_type",
        "perturbation",
        "perturbation_level",
    ]
    metadata = [column for column in metadata if column in frame.columns]

    total_start = time.perf_counter()
    for simulation_id, group in frame.groupby("simulation_id", sort=True):
        simulation_start = time.perf_counter()
        first = group.iloc[0]
        record = {
            **{column: first[column] for column in metadata},
            "method": args.method,
            "weight_mode": args.weight_mode,
            "probability_source": args.probability_source,
        }
        try:
            topologies = []
            for row in group.itertuples(index=False):
                taxa = (row.taxon_a, row.taxon_b, row.taxon_c)
                topologies.extend(
                    probability_topologies(taxa, [row.weight0, row.weight1, row.weight2])
                )
            taxa = set(group[["taxon_a", "taxon_b", "taxon_c"]].to_numpy().ravel())
            tree = assemble_tree(taxa, topologies, method=args.method)
            record.update(
                {"assembly_status": "completed", "assembly_error": "", "inferred_newick": tree.to_newick()}
            )
        except Exception as exc:
            record.update(
                {
                    "assembly_status": "failed",
                    "assembly_error": f"{type(exc).__name__}: {exc}",
                    "inferred_newick": "",
                }
            )
        record["assembly_seconds"] = time.perf_counter() - simulation_start
        rows.append(record)
    result = pd.DataFrame(rows)
    _write_csv(result, args.out)
    print(
        json.dumps(
            {
                "assembled": str(args.out),
                "simulations": len(result),
                "failed_simulations": int((result["assembly_status"] == "failed").sum()),
                "total_wall_seconds": time.perf_counter() - total_start,
            },
            indent=2,
        )
    )


def command_assemble_sharded(args) -> None:
    summary = assemble_feature_shards(
        args.manifest,
        args.features_dir,
        args.out,
        prediction_dir=args.predictions_dir,
        weight_mode=args.weight_mode,
        probability_source=args.probability_source,
        method=args.method,
        overwrite=args.overwrite,
        jobs=args.jobs,
    )
    print(json.dumps(summary, indent=2))


def command_evaluate(args) -> None:
    summary = evaluate_assembled(args.assembled, args.out_dir)
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="introgression-triplets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("make-manifest")
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--out", required=True)
    manifest.set_defaults(func=command_make_manifest)

    selection = subparsers.add_parser("select-manifest")
    selection.add_argument("--manifest", required=True)
    selection.add_argument("--out", required=True)
    selection.add_argument("--runs-per-backbone", type=int, default=12)
    selection.add_argument("--seed", type=int, default=20260828)
    selection.set_defaults(func=command_select_manifest)

    features = subparsers.add_parser("extract-features")
    features.add_argument("--manifest", required=True)
    features.add_argument("--trees-dir", required=True)
    features.add_argument("--out", required=True)
    features.set_defaults(func=command_extract_features)

    shards = subparsers.add_parser("extract-feature-shards")
    shards.add_argument("--manifest", required=True)
    shards.add_argument("--trees-dir", required=True)
    shards.add_argument("--out-dir", required=True)
    shards.add_argument("--start-row", type=int)
    shards.add_argument("--end-row", type=int)
    shards.add_argument("--delete-tree-files", action="store_true")
    shards.add_argument("--overwrite", action="store_true")
    shards.add_argument("--jobs", type=int, default=1)
    shards.set_defaults(func=command_extract_feature_shards)

    train = subparsers.add_parser("train")
    train.add_argument("--features", required=True)
    train.add_argument("--out-dir", required=True)
    train.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="base13")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--inner-folds", type=int, default=3)
    train.add_argument("--seed", type=int, default=20260828)
    train.add_argument("--cv-mode", choices=["backbone", "scenario"], default="backbone")
    train.add_argument("--calibration", choices=["temperature", "none"], default="temperature")
    train.add_argument("--strata-column")
    train.set_defaults(func=command_train)

    sharded_train = subparsers.add_parser("train-sharded")
    sharded_train.add_argument("--manifest", required=True)
    sharded_train.add_argument("--features-dir", required=True)
    sharded_train.add_argument("--out-dir", required=True)
    sharded_train.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="enhanced")
    sharded_train.add_argument("--folds", type=int, default=5)
    sharded_train.add_argument("--inner-folds", type=int, default=3)
    sharded_train.add_argument("--seed", type=int, default=20260828)
    sharded_train.add_argument("--calibration", choices=["temperature", "none"], default="temperature")
    sharded_train.add_argument("--training-rows-per-simulation", type=int, default=20)
    sharded_train.add_argument("--final-calibration-rows-per-simulation", type=int, default=10)
    sharded_train.set_defaults(func=command_train_sharded)

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--features", required=True)
    assemble.add_argument("--out", required=True)
    assemble.add_argument("--method", choices=["fm", "exact"], default="fm")
    assemble.add_argument(
        "--weight-mode",
        choices=["ml", "ml-soft", "ml-hard", "ml-confidence", "frequency", "majority"],
        default="ml-soft",
    )
    assemble.add_argument("--probability-source", choices=["calibrated", "raw"], default="calibrated")
    source = assemble.add_mutually_exclusive_group()
    source.add_argument("--probabilities")
    source.add_argument("--model")
    assemble.set_defaults(func=command_assemble)

    sharded_assemble = subparsers.add_parser("assemble-sharded")
    sharded_assemble.add_argument("--manifest", required=True)
    sharded_assemble.add_argument("--features-dir", required=True)
    sharded_assemble.add_argument("--predictions-dir")
    sharded_assemble.add_argument("--out", required=True)
    sharded_assemble.add_argument("--method", choices=["fm", "exact"], default="fm")
    sharded_assemble.add_argument(
        "--weight-mode",
        choices=["ml", "ml-soft", "ml-hard", "ml-confidence", "frequency", "majority"],
        default="ml-soft",
    )
    sharded_assemble.add_argument(
        "--probability-source", choices=["calibrated", "raw"], default="calibrated"
    )
    sharded_assemble.add_argument("--overwrite", action="store_true")
    sharded_assemble.add_argument("--jobs", type=int, default=1)
    sharded_assemble.set_defaults(func=command_assemble_sharded)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--assembled", required=True)
    evaluate.add_argument("--out-dir", required=True)
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
