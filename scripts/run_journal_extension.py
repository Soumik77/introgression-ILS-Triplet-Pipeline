#!/usr/bin/env python3
"""Execute one journal-extension experiment in a fixed, resumable order."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "manifest",
    "simulate",
    "features",
    "train",
    "assemble",
    "evaluate",
    "external",
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--name", help="Output name. Defaults to experiment_id in the YAML file.")
    parser.add_argument("--start-at", choices=STAGES, default="manifest")
    parser.add_argument("--stop-after", choices=STAGES, default="compare")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--feature-set", default="enhanced")
    parser.add_argument("--assembly-method", choices=("fm", "exact"), default="fm")
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--stelar-executable", type=Path)
    parser.add_argument("--stelar-heuristic", action="store_true")
    parser.add_argument("--mpest-executable", type=Path)
    parser.add_argument("--tmc-executable", type=Path)
    parser.add_argument("--external-jobs", type=int, default=4)
    parser.add_argument("--external-timeout", type=float, default=300.0)
    parser.add_argument("--mpest-starts", type=int, default=5)
    parser.add_argument("--tmc-docker", action="store_true")
    parser.add_argument("--start-row", type=int)
    parser.add_argument("--end-row", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def display_command(command: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in command)


def run(command: list[str], dry_run: bool) -> None:
    print(f"\n$ {display_command(command)}", flush=True)
    if not dry_run:
        environment = os.environ.copy()
        source_path = str(REPOSITORY_ROOT / "src")
        environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
        subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def stage_enabled(stage: str, start_at: str, stop_after: str) -> bool:
    return STAGES.index(start_at) <= STAGES.index(stage) <= STAGES.index(stop_after)


def complete(path: Path, force: bool, dry_run: bool) -> bool:
    return path.exists() and path.stat().st_size > 0 and not force and not dry_run


def audit_manifest(path: Path) -> None:
    frame = pd.read_csv(path)
    if frame["simulation_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate simulation_id values")
    summary = {
        "simulations": int(len(frame)),
        "scenario_groups": int(frame["scenario_group"].nunique()),
        "backbones": int(frame["backbone_id"].nunique()),
        "tree_sizes": sorted(frame["tree_size"].dropna().astype(int).unique().tolist()),
        "gene_trees": int(frame["gene_tree_count"].sum()),
    }
    if "backbone_balance_bin" in frame.columns:
        summary["backbones_by_balance_bin"] = {
            str(key): int(value)
            for key, value in frame.groupby("backbone_balance_bin")["backbone_id"].nunique().items()
        }
    print(json.dumps(summary, indent=2), flush=True)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    name = args.name or str(config["experiment_id"])
    if args.start_row or args.end_row:
        if args.stop_after != "simulate":
            raise ValueError("Row-batched simulation runs must use --stop-after simulate")
        if not (args.start_row and args.end_row):
            raise ValueError("Specify both --start-row and --end-row")
    if STAGES.index(args.start_at) > STAGES.index(args.stop_after):
        raise ValueError("--start-at occurs after --stop-after")

    manifest = REPOSITORY_ROOT / "data" / "manifests" / f"{name}.csv"
    trees = REPOSITORY_ROOT / "data" / "gene_trees" / name
    features = REPOSITORY_ROOT / "data" / "features" / f"{name}.csv"
    output_root = REPOSITORY_ROOT / "outputs" / "journal_extension" / name
    model_dir = output_root / "models" / args.feature_set
    primary_label = "Enhanced17-ML-FM" if args.feature_set == "enhanced" else f"{args.feature_set}-ML-FM"

    python = sys.executable
    module = [python, "-m", "introgression_triplets.cli"]

    if stage_enabled("manifest", args.start_at, args.stop_after):
        if complete(manifest, args.force, args.dry_run):
            print(f"skip existing: {manifest}")
        else:
            run(module + ["make-manifest", "--config", str(config_path), "--out", str(manifest)], args.dry_run)
        if not args.dry_run:
            audit_manifest(manifest)

    if stage_enabled("simulate", args.start_at, args.stop_after):
        command = [
            args.julia,
            "--project=julia",
            "julia/simulate_grid.jl",
            "--manifest",
            str(manifest),
            "--out-dir",
            str(trees),
        ]
        if args.start_row and args.end_row:
            command += ["--start-row", str(args.start_row), "--end-row", str(args.end_row)]
        if args.force:
            command.append("--overwrite")
        run(command, args.dry_run)

    if stage_enabled("features", args.start_at, args.stop_after):
        if complete(features, args.force, args.dry_run):
            print(f"skip existing: {features}")
        else:
            run(
                module
                + [
                    "extract-features",
                    "--manifest",
                    str(manifest),
                    "--trees-dir",
                    str(trees),
                    "--out",
                    str(features),
                ],
                args.dry_run,
            )

    if stage_enabled("train", args.start_at, args.stop_after):
        model_marker = model_dir / "oof_predictions.csv"
        if complete(model_marker, args.force, args.dry_run):
            print(f"skip existing: {model_marker}")
        else:
            run(
                module
                + [
                    "train",
                    "--features",
                    str(features),
                    "--out-dir",
                    str(model_dir),
                    "--feature-set",
                    args.feature_set,
                    "--folds",
                    str(args.folds),
                    "--inner-folds",
                    str(args.inner_folds),
                    "--cv-mode",
                    "backbone",
                    "--calibration",
                    "temperature",
                ],
                args.dry_run,
            )

    if stage_enabled("assemble", args.start_at, args.stop_after):
        for mode in ASSEMBLY_MODES:
            assembled = output_root / "assembled" / f"{mode}.csv"
            if complete(assembled, args.force, args.dry_run):
                print(f"skip existing: {assembled}")
                continue
            command = module + [
                "assemble",
                "--features",
                str(features),
                "--out",
                str(assembled),
                "--weight-mode",
                mode,
                "--method",
                args.assembly_method,
            ]
            if mode.startswith("ml"):
                command += ["--probabilities", str(model_dir / "oof_predictions.csv")]
            run(command, args.dry_run)

    if stage_enabled("evaluate", args.start_at, args.stop_after):
        for mode in ASSEMBLY_MODES:
            evaluation = output_root / "evaluation" / mode
            marker = evaluation / "summary.json"
            if complete(marker, args.force, args.dry_run):
                print(f"skip existing: {marker}")
                continue
            run(
                module
                + [
                    "evaluate",
                    "--assembled",
                    str(output_root / "assembled" / f"{mode}.csv"),
                    "--out-dir",
                    str(evaluation),
                ],
                args.dry_run,
            )

    external_definitions = [
        (
            "stelar",
            "STELAR-heuristic" if args.stelar_heuristic else "STELAR-exact",
            args.stelar_executable,
        ),
        ("mpest", "MP-EST-v3", args.mpest_executable),
        ("tmc", "Triplet-MaxCut", args.tmc_executable),
    ]
    if stage_enabled("external", args.start_at, args.stop_after):
        selected = [definition for definition in external_definitions if definition[2] is not None]
        if not selected:
            print("No external executable paths were supplied. External baselines were not run.")
        for method, label, executable in selected:
            assert executable is not None
            tree_dir = output_root / "external" / method / "trees"
            evaluation_dir = output_root / "external" / method / "evaluation"
            marker = evaluation_dir / "summary.json"
            if complete(marker, args.force, args.dry_run):
                print(f"skip existing: {marker}")
                continue
            runner_arguments = [
                "scripts/run_external_rooted.py",
                "--method",
                method,
                "--manifest",
                str(manifest),
                "--trees-dir",
                str(trees),
                "--out-dir",
                str(tree_dir),
                "--executable",
                str(executable),
                "--jobs",
                str(args.external_jobs),
                "--timeout",
                str(args.external_timeout),
            ]
            if method == "mpest":
                runner_arguments += ["--mpest-starts", str(args.mpest_starts)]
            if method == "stelar" and args.stelar_heuristic:
                runner_arguments.append("--stelar-heuristic")
            if args.force:
                runner_arguments.append("--overwrite")
            if method == "tmc" and args.tmc_docker:
                resolved_executable = executable.resolve()
                try:
                    relative_executable = resolved_executable.relative_to(REPOSITORY_ROOT)
                except ValueError as exc:
                    raise ValueError("Docker TMC executable must be inside the repository") from exc
                executable_index = runner_arguments.index(str(executable))
                runner_arguments[executable_index] = f"/work/{relative_executable}"
                docker_arguments = [
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
                ]
                mapped = [
                    value.replace(str(REPOSITORY_ROOT), "/work") for value in runner_arguments
                ]
                run(docker_arguments + mapped, args.dry_run)
            else:
                run([python] + runner_arguments, args.dry_run)
            run(
                [
                    python,
                    "scripts/evaluate_external_rooted.py",
                    "--manifest",
                    str(manifest),
                    "--trees-dir",
                    str(tree_dir),
                    "--method",
                    label,
                    "--out-dir",
                    str(evaluation_dir),
                ],
                args.dry_run,
            )

    if stage_enabled("compare", args.start_at, args.stop_after):
        comparison_dir = output_root / "comparison"
        command = [
            python,
            "scripts/compare_rooted_benchmarks.py",
            "--reference",
            f"{primary_label}={output_root / 'evaluation' / 'ml-soft' / 'per_simulation.csv'}",
        ]
        for mode in ASSEMBLY_MODES[1:]:
            command += [
                "--baseline",
                f"{ASSEMBLY_LABELS[mode]}={output_root / 'evaluation' / mode / 'per_simulation.csv'}",
            ]
        for method, label, executable in external_definitions:
            external_results = output_root / "external" / method / "evaluation" / "per_simulation.csv"
            if executable is not None or external_results.is_file():
                command += ["--baseline", f"{label}={external_results}"]
        command += [
            "--out-dir",
            str(comparison_dir),
            "--bootstrap-reps",
            str(args.bootstrap_reps),
        ]
        run(command, args.dry_run)

    if stage_enabled("ablations", args.start_at, args.stop_after):
        ablation_sets = [value for value in ABLATION_FEATURE_SETS if value != args.feature_set]
        for feature_set in ablation_sets:
            ablation_model = output_root / "ablations" / feature_set / "model"
            marker = ablation_model / "oof_predictions.csv"
            if not complete(marker, args.force, args.dry_run):
                run(
                    module
                    + [
                        "train",
                        "--features",
                        str(features),
                        "--out-dir",
                        str(ablation_model),
                        "--feature-set",
                        feature_set,
                        "--folds",
                        str(args.folds),
                        "--inner-folds",
                        str(args.inner_folds),
                        "--cv-mode",
                        "backbone",
                        "--calibration",
                        "temperature",
                    ],
                    args.dry_run,
                )
            assembled = output_root / "ablations" / feature_set / "assembled.csv"
            if not complete(assembled, args.force, args.dry_run):
                run(
                    module
                    + [
                        "assemble",
                        "--features",
                        str(features),
                        "--probabilities",
                        str(marker),
                        "--out",
                        str(assembled),
                        "--weight-mode",
                        "ml-soft",
                        "--method",
                        args.assembly_method,
                    ],
                    args.dry_run,
                )
            evaluation = output_root / "ablations" / feature_set / "evaluation"
            if not complete(evaluation / "summary.json", args.force, args.dry_run):
                run(
                    module
                    + ["evaluate", "--assembled", str(assembled), "--out-dir", str(evaluation)],
                    args.dry_run,
                )
        comparison_command = [
            python,
            "scripts/compare_rooted_benchmarks.py",
            "--reference",
            f"{args.feature_set}={output_root / 'evaluation' / 'ml-soft' / 'per_simulation.csv'}",
        ]
        for feature_set in ablation_sets:
            comparison_command += [
                "--baseline",
                f"{feature_set}={output_root / 'ablations' / feature_set / 'evaluation' / 'per_simulation.csv'}",
            ]
        comparison_command += [
            "--out-dir",
            str(output_root / "ablation_comparison"),
            "--bootstrap-reps",
            str(args.bootstrap_reps),
        ]
        run(comparison_command, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
