#!/usr/bin/env python3
"""Run the corrected 4 to 180 taxon study with bounded disk use.

The full candidate grid is retained as a compressed manifest. Deterministic
3-run deadline and 12-run compact analysis manifests cover every requested
taxon size and the planned factor margins. Raw gene trees are processed in
small batches and removed only after requested external methods finish and a
feature shard is verified.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "manifest",
    "select",
    "preprocess",
    "train",
    "assemble",
    "evaluate",
    "external-evaluate",
    "compare",
    "ablations",
)
ASSEMBLY_MODES = ("ml-soft", "ml-hard", "ml-confidence", "frequency", "majority")
ASSEMBLY_LABELS = {
    "ml-soft": "Enhanced17-ML-FM",
    "ml-hard": "Enhanced17-ML-hard-FM",
    "ml-confidence": "Enhanced17-ML-confidence-FM",
    "frequency": "Frequency-FM",
    "majority": "Majority-FM",
}
ABLATION_FEATURE_SETS = (
    "base13",
    "frequency_only",
    "branch_only",
    "frequency_branch",
    "frequency_asymmetry",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "supervisor_scope_corrected.yaml",
    )
    parser.add_argument("--name", help="Output name. Defaults to experiment_id in the config.")
    parser.add_argument(
        "--execution-mode",
        choices=("deadline", "compact", "full"),
        default="compact",
        help=(
            "deadline runs 3 balanced cells per backbone; compact runs the "
            "preregistered 12-cell-per-backbone subset; "
            "full runs all 191,160 candidate cells and requires server-scale resources"
        ),
    )
    parser.add_argument("--start-at", choices=STAGES, default="manifest")
    parser.add_argument("--stop-after", choices=STAGES, default="compare")
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--disk-budget-gb", type=float, default=11.0)
    parser.add_argument("--minimum-free-gb", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--training-rows-per-simulation", type=int, default=20)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--external-jobs", type=int, default=4)
    parser.add_argument("--feature-jobs", type=int, default=4)
    parser.add_argument("--external-timeout", type=float, default=1800.0)
    parser.add_argument("--mpest-starts", type=int, default=5)
    parser.add_argument("--stelar-executable", type=Path)
    parser.add_argument(
        "--stelar-exact",
        action="store_true",
        help="Use STELAR exact mode. Heuristic mode is the scalable default.",
    )
    parser.add_argument("--mpest-executable", type=Path)
    parser.add_argument("--tmc-executable", type=Path)
    parser.add_argument("--tmc-docker", action="store_true")
    parser.add_argument("--keep-raw-trees", action="store_true")
    parser.add_argument("--keep-oof-predictions", action="store_true")
    parser.add_argument("--keep-training-samples", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stage_enabled(stage: str, args: argparse.Namespace) -> bool:
    return STAGES.index(args.start_at) <= STAGES.index(stage) <= STAGES.index(args.stop_after)


def display_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in command)


def run(command: list[str], dry_run: bool) -> None:
    print(f"\n$ {display_command(command)}", flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    source_path = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def remove_file(path: Path) -> None:
    if path.is_file():
        path.unlink()


def remove_directory(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)


def path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            candidate = Path(root) / filename
            try:
                total += candidate.stat().st_size
            except FileNotFoundError:
                pass
    return total


def enforce_storage_budget(paths: list[Path], budget_gb: float, minimum_free_gb: float) -> None:
    used = sum(path_bytes(path) for path in paths)
    budget = int(budget_gb * 1024**3)
    free = shutil.disk_usage(REPOSITORY_ROOT).free
    if used > budget:
        raise RuntimeError(
            f"This experiment uses {used / 1024**3:.2f} GB, above the {budget_gb:.2f} GB budget"
        )
    if free < minimum_free_gb * 1024**3:
        raise RuntimeError(
            f"Only {free / 1024**3:.2f} GB is free, below the {minimum_free_gb:.2f} GB reserve"
        )
    print(
        f"storage check: experiment={used / 1024**3:.2f} GB, "
        f"free={free / 1024**3:.2f} GB",
        flush=True,
    )


def expected_paths(directory: Path, simulation_ids: list[str], suffix: str) -> list[Path]:
    return [directory / f"{simulation_id}{suffix}" for simulation_id in simulation_ids]


def external_definitions(args: argparse.Namespace, output_root: Path) -> list[dict]:
    definitions = []
    if args.stelar_executable is not None:
        definitions.append(
            {
                "method": "stelar",
                "label": "STELAR-exact" if args.stelar_exact else "STELAR-heuristic",
                "executable": args.stelar_executable,
                "tree_dir": output_root / "external" / "stelar" / "trees",
            }
        )
    if args.mpest_executable is not None:
        definitions.append(
            {
                "method": "mpest",
                "label": "MP-EST-v3",
                "executable": args.mpest_executable,
                "tree_dir": output_root / "external" / "mpest" / "trees",
            }
        )
    if args.tmc_executable is not None:
        definitions.append(
            {
                "method": "tmc",
                "label": "Triplet-MaxCut",
                "executable": args.tmc_executable,
                "tree_dir": output_root / "external" / "tmc" / "trees",
            }
        )
    return definitions


def external_runner_command(
    definition: dict,
    batch_manifest: Path,
    trees: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_external_rooted.py",
        "--method",
        definition["method"],
        "--manifest",
        str(batch_manifest),
        "--trees-dir",
        str(trees),
        "--out-dir",
        str(definition["tree_dir"]),
        "--executable",
        str(definition["executable"]),
        "--jobs",
        str(args.external_jobs),
        "--timeout",
        str(args.external_timeout),
    ]
    if definition["method"] == "stelar" and not args.stelar_exact:
        command.append("--stelar-heuristic")
    if definition["method"] == "mpest":
        command += ["--mpest-starts", str(args.mpest_starts)]
    if definition["method"] == "tmc" and args.tmc_docker:
        executable = Path(definition["executable"]).resolve()
        try:
            relative = executable.relative_to(REPOSITORY_ROOT)
        except ValueError as error:
            raise ValueError("Docker TMC executable must be inside the repository") from error
        command[command.index(str(definition["executable"]))] = f"/work/{relative}"
        mapped = [value.replace(str(REPOSITORY_ROOT), "/work") for value in command[1:]]
        return [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "-v",
            f"{REPOSITORY_ROOT}:/work",
            "-w",
            "/work",
            "python:3.12-slim",
            "python",
            *mapped,
        ]
    return command


def preprocess_batches(
    manifest_path: Path,
    trees: Path,
    features: Path,
    output_root: Path,
    definitions: list[dict],
    storage_paths: list[Path],
    args: argparse.Namespace,
) -> None:
    if args.dry_run and not manifest_path.is_file():
        print(
            "dry-run: preprocessing batch commands require the manifest created by the "
            "manifest/select stages",
            flush=True,
        )
        return
    manifest = pd.read_csv(manifest_path)
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True, exist_ok=True)
    trees.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(manifest), args.batch_size):
        end = min(len(manifest), start + args.batch_size)
        batch = manifest.iloc[start:end].copy()
        pending_rows = []
        for row in batch.itertuples(index=False):
            simulation_id = str(row.simulation_id)
            feature_ready = complete(features / f"{simulation_id}.csv.gz")
            externals_ready = all(
                complete(definition["tree_dir"] / f"{simulation_id}.tre")
                for definition in definitions
            )
            if not (feature_ready and externals_ready):
                pending_rows.append(row._asdict())
        if not pending_rows:
            print(f"skip completed batch {start + 1}:{end}", flush=True)
            continue

        batch_index = start // args.batch_size
        if batch_index == 0 or batch_index % 20 == 0:
            enforce_storage_budget(storage_paths, args.disk_budget_gb, args.minimum_free_gb)
        active_manifest = checkpoints / "active_batch.csv"
        pd.DataFrame(pending_rows).to_csv(active_manifest, index=False)
        run(
            [
                args.julia,
                "--project=julia",
                "julia/simulate_grid.jl",
                "--manifest",
                str(active_manifest),
                "--out-dir",
                str(trees),
            ],
            args.dry_run,
        )
        for definition in definitions:
            run(external_runner_command(definition, active_manifest, trees, args), args.dry_run)
        feature_command = [
            sys.executable,
            "-m",
            "introgression_triplets.cli",
            "extract-feature-shards",
            "--manifest",
            str(active_manifest),
            "--trees-dir",
            str(trees),
            "--out-dir",
            str(features),
            "--jobs",
            str(args.feature_jobs),
        ]
        if not args.keep_raw_trees:
            feature_command.append("--delete-tree-files")
        run(feature_command, args.dry_run)
        if not args.dry_run:
            marker = {
                "first_analysis_row": start + 1,
                "last_analysis_row": end,
                "simulations": [str(row["simulation_id"]) for row in pending_rows],
                "external_methods": [definition["label"] for definition in definitions],
            }
            (checkpoints / f"batch_{start + 1:06d}_{end:06d}.json").write_text(
                json.dumps(marker, indent=2) + "\n", encoding="utf-8"
            )
            active_manifest.unlink(missing_ok=True)


def run_sharded_training(
    manifest: Path,
    features: Path,
    model_dir: Path,
    feature_set: str,
    args: argparse.Namespace,
) -> None:
    prediction_dir = model_dir / "oof_predictions"
    expected = (
        len(pd.read_csv(manifest, usecols=["simulation_id"]))
        if manifest.is_file()
        else None
    )
    predictions = len(list(prediction_dir.glob("*.csv.gz"))) if prediction_dir.is_dir() else 0
    if (
        expected is not None
        and complete(model_dir / "metrics.json")
        and complete(model_dir / "model.joblib")
        and predictions == expected
    ):
        print(f"skip existing model and OOF predictions: {model_dir}")
        return
    run(
        [
            sys.executable,
            "-m",
            "introgression_triplets.cli",
            "train-sharded",
            "--manifest",
            str(manifest),
            "--features-dir",
            str(features),
            "--out-dir",
            str(model_dir),
            "--feature-set",
            feature_set,
            "--folds",
            str(args.folds),
            "--inner-folds",
            str(args.inner_folds),
            "--calibration",
            "temperature",
            "--training-rows-per-simulation",
            str(args.training_rows_per_simulation),
        ],
        args.dry_run,
    )


def run_assembly(
    manifest: Path,
    features: Path,
    prediction_dir: Path,
    output: Path,
    mode: str,
    args: argparse.Namespace,
) -> None:
    if complete(output):
        print(f"skip existing: {output}")
        return
    command = [
        sys.executable,
        "-m",
        "introgression_triplets.cli",
        "assemble-sharded",
        "--manifest",
        str(manifest),
        "--features-dir",
        str(features),
        "--out",
        str(output),
        "--weight-mode",
        mode,
        "--method",
        "fm",
        "--jobs",
        str(args.feature_jobs),
    ]
    if mode.startswith("ml"):
        command += ["--predictions-dir", str(prediction_dir)]
    run(command, args.dry_run)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least one")
    if args.feature_jobs < 1 or args.external_jobs < 1:
        raise ValueError("--feature-jobs and --external-jobs must be at least one")
    if args.folds < 2 or args.inner_folds < 2:
        raise ValueError("--folds and --inner-folds must be at least two")
    if STAGES.index(args.start_at) > STAGES.index(args.stop_after):
        raise ValueError("--start-at occurs after --stop-after")
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    name = args.name or str(config["experiment_id"])

    candidate_manifest = REPOSITORY_ROOT / "data" / "manifests" / f"{name}_candidates.csv.gz"
    compact_manifest = REPOSITORY_ROOT / "data" / "manifests" / f"{name}_analysis.csv"
    deadline_manifest = (
        REPOSITORY_ROOT / "data" / "manifests" / f"{name}_deadline_analysis.csv"
    )
    analysis_manifest = (
        deadline_manifest if args.execution_mode == "deadline" else compact_manifest
    )
    run_name = name if args.execution_mode == "compact" else f"{name}_{args.execution_mode}"
    working_manifest = candidate_manifest if args.execution_mode == "full" else analysis_manifest
    shared_bounded_name = name if args.execution_mode in {"deadline", "compact"} else run_name
    trees = REPOSITORY_ROOT / "data" / "gene_trees" / f"{shared_bounded_name}_batch"
    features = REPOSITORY_ROOT / "data" / "features" / f"{shared_bounded_name}_shards"
    output_root = REPOSITORY_ROOT / "outputs" / "supervisor_scope" / run_name
    model_dir = output_root / "models" / "enhanced"
    definitions = external_definitions(args, output_root)
    storage_paths = [candidate_manifest, analysis_manifest, trees, features, output_root]

    if stage_enabled("manifest", args):
        if complete(candidate_manifest):
            print(f"skip existing: {candidate_manifest}")
        else:
            run(
                [
                    sys.executable,
                    "-m",
                    "introgression_triplets.cli",
                    "make-manifest",
                    "--config",
                    str(config_path),
                    "--out",
                    str(candidate_manifest),
                ],
                args.dry_run,
            )

    if stage_enabled("select", args):
        if args.execution_mode == "full":
            print("full execution mode uses the complete candidate manifest")
        elif complete(analysis_manifest):
            print(f"skip existing: {analysis_manifest}")
        else:
            run(
                [
                    sys.executable,
                    "-m",
                    "introgression_triplets.cli",
                    "select-manifest",
                    "--manifest",
                    str(candidate_manifest),
                    "--out",
                    str(analysis_manifest),
                    "--runs-per-backbone",
                    str(
                        config.get("deadline_runs_per_backbone", 3)
                        if args.execution_mode == "deadline"
                        else config.get("analysis_runs_per_backbone", 12)
                    ),
                    "--seed",
                    str(config["base_seed"]),
                ],
                args.dry_run,
            )

    if not args.dry_run and complete(candidate_manifest):
        candidate_columns = [
            "simulation_id",
            "backbone_id",
            "tree_size",
            "gamma",
            "pair_type",
            "direction",
        ]
        candidates = pd.read_csv(candidate_manifest, usecols=candidate_columns)
        expected_candidates = int(config.get("expected_candidate_runs", 191_160))
        candidate_audit = {
            "candidate_runs": len(candidates),
            "unique_simulation_ids": int(candidates["simulation_id"].nunique()),
            "backbones": int(candidates["backbone_id"].nunique()),
            "taxon_sizes": int(candidates["tree_size"].nunique()),
            "minimum_taxa": int(candidates["tree_size"].min()),
            "maximum_taxa": int(candidates["tree_size"].max()),
        }
        print(json.dumps({"candidate_manifest_audit": candidate_audit}, indent=2), flush=True)
        if len(candidates) != expected_candidates:
            raise ValueError(
                f"Expected {expected_candidates} candidate runs, found {len(candidates)}"
            )
        if not candidates["simulation_id"].is_unique:
            raise ValueError("Candidate simulation_id values are not unique")
        null_rows = candidates["gamma"].astype(float).eq(0.0)
        if not (
            candidates.loc[null_rows, "pair_type"].astype(str).eq("none").all()
            and candidates.loc[null_rows, "direction"].astype(str).eq("none").all()
        ):
            raise ValueError("Gamma-zero candidates contain false introgression labels")

    if not args.dry_run and complete(working_manifest):
        analysis = pd.read_csv(working_manifest)
        audit = {
            "analysis_runs": len(analysis),
            "backbones": int(analysis["backbone_id"].nunique()),
            "taxon_sizes": int(analysis["tree_size"].nunique()),
            "minimum_taxa": int(analysis["tree_size"].min()),
            "maximum_taxa": int(analysis["tree_size"].max()),
            "gene_trees": int(analysis["gene_tree_count"].sum()),
            "maximum_triplet_rows": int(analysis["triplets_per_simulation"].sum()),
        }
        print(json.dumps(audit, indent=2), flush=True)
        expected_key = {
            "deadline": "expected_deadline_runs",
            "compact": "expected_analysis_runs",
            "full": "expected_candidate_runs",
        }[args.execution_mode]
        expected_runs = int(config[expected_key])
        if len(analysis) != expected_runs:
            raise ValueError(f"Expected {expected_runs} analysis runs, found {len(analysis)}")

    if stage_enabled("preprocess", args):
        if not definitions:
            print(
                "No external executables were supplied. Internal features will be retained; "
                "external methods can be added later by regenerating deterministic batches.",
                flush=True,
            )
        preprocess_batches(
            working_manifest,
            trees,
            features,
            output_root,
            definitions,
            storage_paths,
            args,
        )

    if stage_enabled("train", args):
        run_sharded_training(working_manifest, features, model_dir, "enhanced", args)

    if stage_enabled("assemble", args):
        for mode in ASSEMBLY_MODES:
            run_assembly(
                working_manifest,
                features,
                model_dir / "oof_predictions",
                output_root / "assembled" / f"{mode}.csv",
                mode,
                args,
            )
        if not args.keep_oof_predictions and not args.dry_run:
            remove_directory(model_dir / "oof_predictions")
        if not args.keep_training_samples and not args.dry_run:
            remove_file(model_dir / "training_sample.csv.gz")

    if stage_enabled("evaluate", args):
        for mode in ASSEMBLY_MODES:
            evaluation = output_root / "evaluation" / mode
            if complete(evaluation / "summary.json"):
                print(f"skip existing: {evaluation / 'summary.json'}")
                continue
            run(
                [
                    sys.executable,
                    "-m",
                    "introgression_triplets.cli",
                    "evaluate",
                    "--assembled",
                    str(output_root / "assembled" / f"{mode}.csv"),
                    "--out-dir",
                    str(evaluation),
                ],
                args.dry_run,
            )

    if stage_enabled("external-evaluate", args):
        for definition in definitions:
            evaluation = output_root / "external" / definition["method"] / "evaluation"
            if complete(evaluation / "summary.json"):
                print(f"skip existing: {evaluation / 'summary.json'}")
                continue
            run(
                [
                    sys.executable,
                    "scripts/evaluate_external_rooted.py",
                    "--manifest",
                    str(working_manifest),
                    "--trees-dir",
                    str(definition["tree_dir"]),
                    "--method",
                    definition["label"],
                    "--out-dir",
                    str(evaluation),
                ],
                args.dry_run,
            )

    if stage_enabled("compare", args):
        comparison = output_root / "comparison"
        command = [
            sys.executable,
            "scripts/compare_rooted_benchmarks.py",
            "--reference",
            f"{ASSEMBLY_LABELS['ml-soft']}={output_root / 'evaluation' / 'ml-soft' / 'per_simulation.csv'}",
        ]
        for mode in ASSEMBLY_MODES[1:]:
            command += [
                "--baseline",
                f"{ASSEMBLY_LABELS[mode]}={output_root / 'evaluation' / mode / 'per_simulation.csv'}",
            ]
        for definition in definitions:
            command += [
                "--baseline",
                f"{definition['label']}={output_root / 'external' / definition['method'] / 'evaluation' / 'per_simulation.csv'}",
            ]
        command += [
            "--out-dir",
            str(comparison),
            "--bootstrap-reps",
            str(args.bootstrap_reps),
        ]
        run(command, args.dry_run)

    if stage_enabled("ablations", args):
        for feature_set in ABLATION_FEATURE_SETS:
            ablation_root = output_root / "ablations" / feature_set
            ablation_model = ablation_root / "model"
            run_sharded_training(
                working_manifest, features, ablation_model, feature_set, args
            )
            assembled = ablation_root / "assembled.csv"
            run_assembly(
                working_manifest,
                features,
                ablation_model / "oof_predictions",
                assembled,
                "ml-soft",
                args,
            )
            evaluation = ablation_root / "evaluation"
            if not complete(evaluation / "summary.json"):
                run(
                    [
                        sys.executable,
                        "-m",
                        "introgression_triplets.cli",
                        "evaluate",
                        "--assembled",
                        str(assembled),
                        "--out-dir",
                        str(evaluation),
                    ],
                    args.dry_run,
                )
            if not args.keep_oof_predictions and not args.dry_run:
                remove_directory(ablation_model / "oof_predictions")
            if not args.keep_training_samples and not args.dry_run:
                remove_file(ablation_model / "training_sample.csv.gz")

        ablation_compare = [
            sys.executable,
            "scripts/compare_rooted_benchmarks.py",
            "--reference",
            f"enhanced={output_root / 'evaluation' / 'ml-soft' / 'per_simulation.csv'}",
        ]
        for feature_set in ABLATION_FEATURE_SETS:
            ablation_compare += [
                "--baseline",
                f"{feature_set}={output_root / 'ablations' / feature_set / 'evaluation' / 'per_simulation.csv'}",
            ]
        ablation_compare += [
            "--out-dir",
            str(output_root / "ablation_comparison"),
            "--bootstrap-reps",
            str(args.bootstrap_reps),
        ]
        run(ablation_compare, args.dry_run)

    if not args.dry_run:
        enforce_storage_budget(storage_paths, args.disk_budget_gb, args.minimum_free_gb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
