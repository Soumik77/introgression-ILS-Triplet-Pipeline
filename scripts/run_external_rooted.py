#!/usr/bin/env python3
"""Run rooted species-tree benchmarks over a simulation manifest.

Supported methods:

* STELAR (exact mode by default)
* MP-EST 3.0 (multiple deterministic random starts)
* Triplet MaxCut (TMC)

Each successful result is normalized to one plain rooted Newick tree named
``<simulation_id>.tre`` in the output directory. The runner is resumable and
never uses the generating backbone to configure an inference program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from introgression_triplets.newick import NewickError, parse_newick, to_newick  # noqa: E402


NEWICK_COMMENT = re.compile(r"\[[^\[\]]*\]")
BRANCH_LENGTH = re.compile(r":[^,()\[\];\s]+")
TREE_STATEMENT = re.compile(
    r"\btree\s+\S+\s+\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\]"
    r"\s*=\s*(.+?;)",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


@dataclass
class TaskResult:
    simulation_id: str
    status: str
    runtime_seconds: float = 0.0
    score: float | None = None
    successful_starts: int | None = None
    unique_topologies: int | None = None
    candidate_scores: list[float] | None = None
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STELAR, MP-EST, or Triplet MaxCut on manifest gene trees."
    )
    parser.add_argument("--method", choices=("stelar", "mpest", "tmc"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trees-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--executable",
        type=Path,
        required=True,
        help="STELAR.jar, the mpest executable, or treeFromTriplets.",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-success-logs", action="store_true")
    parser.add_argument(
        "--stelar-heuristic",
        action="store_true",
        help="Use STELAR heuristic mode instead of exact -xt mode.",
    )
    parser.add_argument(
        "--mpest-starts",
        type=int,
        default=5,
        help="Independent, data-blind rooted starting trees; best likelihood is retained.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    required = {"simulation_id"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    ids = [row["simulation_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate simulation_id values")
    return rows


def first_newick(text: str) -> str:
    """Return the first complete Newick statement in arbitrary program output."""

    start = text.find("(")
    if start < 0:
        raise NewickError("No Newick opening parenthesis found")
    in_quote: str | None = None
    bracket_depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if in_quote:
            if character == in_quote:
                in_quote = None
            continue
        if character in ("'", '"'):
            in_quote = character
        elif character == "[":
            bracket_depth += 1
        elif character == "]" and bracket_depth:
            bracket_depth -= 1
        elif character == ";" and bracket_depth == 0:
            return text[start : index + 1]
    raise NewickError("Newick statement has no terminating semicolon")


def normalize_newick(text: str) -> str:
    statement = first_newick(text)
    statement = NEWICK_COMMENT.sub("", statement)
    root = parse_newick(statement)
    return to_newick(root, include_lengths=False).strip()


def taxa_in_tree(newick: str) -> set[str]:
    root = parse_newick(newick)
    names = [leaf.name for leaf in root.leaves()]
    if any(name is None for name in names):
        raise NewickError("Tree contains an unnamed leaf")
    if len(names) != len(set(names)):
        raise NewickError("Tree contains duplicate leaf labels")
    return set(names)  # type: ignore[arg-type]


def read_gene_trees(path: Path) -> tuple[list[str], list[str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No gene trees found in {path}")
    normalized_first = normalize_newick(lines[0])
    taxa = sorted(taxa_in_tree(normalized_first))
    return lines, taxa


def valid_existing_tree(path: Path, expected_taxa: set[str] | None = None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        tree = normalize_newick(path.read_text(encoding="utf-8"))
        observed = taxa_in_tree(tree)
    except (OSError, ValueError, NewickError):
        return False
    return expected_taxa is None or observed == expected_taxa


def run_command(command: list[str], cwd: Path, timeout: float) -> CommandResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_seconds=time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        return CommandResult(
            command=command,
            returncode=124,
            stdout=stdout,
            stderr=stderr + f"\nTimed out after {timeout:g} seconds.",
            elapsed_seconds=time.perf_counter() - started,
        )


def stable_seed(simulation_id: str, start_index: int = 0) -> int:
    digest = hashlib.sha256(f"{simulation_id}|{start_index}".encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % 2_000_000_000


def random_rooted_binary_tree(taxa: list[str], seed: int) -> str:
    rng = random.Random(seed)
    forest = list(taxa)
    rng.shuffle(forest)
    while len(forest) > 1:
        first_index, second_index = sorted(rng.sample(range(len(forest)), 2), reverse=True)
        first = forest.pop(first_index)
        second = forest.pop(second_index)
        if rng.random() < 0.5:
            first, second = second, first
        forest.append(f"({first},{second})")
    return forest[0] + ";"


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def format_log(result: CommandResult) -> str:
    return (
        "COMMAND\n"
        + " ".join(result.command)
        + f"\n\nRETURN CODE\n{result.returncode}"
        + f"\n\nELAPSED SECONDS\n{result.elapsed_seconds:.6f}"
        + "\n\nSTDOUT\n"
        + result.stdout
        + "\n\nSTDERR\n"
        + result.stderr
    )


def run_stelar(
    gene_tree_path: Path,
    executable: Path,
    simulation_id: str,
    timeout: float,
    exact: bool,
) -> tuple[str, TaskResult, str]:
    with tempfile.TemporaryDirectory(prefix="stelar-") as temporary_name:
        temporary = Path(temporary_name)
        raw_output = temporary / "species_tree.tre"
        command = ["java", "-jar", str(executable), "-i", str(gene_tree_path), "-o", str(raw_output)]
        if exact:
            command.append("-xt")
        result = run_command(command, temporary, timeout)
        log = format_log(result)
        if result.returncode != 0:
            raise RuntimeError(f"STELAR returned {result.returncode}\n{log}")
        if not raw_output.is_file():
            raise RuntimeError(f"STELAR did not create {raw_output}\n{log}")
        tree = normalize_newick(raw_output.read_text(encoding="utf-8"))
        return tree, TaskResult(simulation_id, "completed", result.elapsed_seconds), log


def mpest_nexus(gene_trees: list[str], taxa: list[str]) -> str:
    outgroup = taxa[0]
    lines = [
        "#NEXUS",
        "Begin data;",
        f"dimension ngene={len(gene_trees)} ntaxa={len(taxa)};",
        f"Format tree=rooted outgroup={outgroup};",
        "matrix",
    ]
    lines.extend(f"{taxon} 1 {taxon}" for taxon in taxa)
    lines.extend((";", "End;", "Begin trees;"))
    lines.extend(gene_trees)
    lines.extend(("End;", ""))
    return "\n".join(lines)


def parse_mpest_best_trees(text: str) -> list[tuple[float, str]]:
    translate_match = re.search(r"\btranslate\b(.*?);", text, flags=re.IGNORECASE | re.DOTALL)
    if not translate_match:
        raise ValueError("MP-EST best-tree file has no translate block")
    translation: dict[str, str] = {}
    for number, label in re.findall(r"(?:^|,)\s*(\d+)\s+([^,;\s]+)", translate_match.group(1)):
        translation[number] = label.strip("'\"")
    if not translation:
        raise ValueError("MP-EST translate block is empty")

    candidates: list[tuple[float, str]] = []
    for match in TREE_STATEMENT.finditer(text):
        score = float(match.group(1))
        tree = NEWICK_COMMENT.sub("", match.group(2))

        def replace_numeric_leaf(token: re.Match[str]) -> str:
            prefix, number = token.group(1), token.group(2)
            return prefix + translation.get(number, number)

        tree = re.sub(r"([\(,])\s*(\d+)\s*(?=[:,\)])", replace_numeric_leaf, tree)
        tree = BRANCH_LENGTH.sub("", tree)
        candidates.append((score, normalize_newick(tree)))
    if not candidates:
        raise ValueError("MP-EST best-tree file has no scored tree statements")
    return candidates


def run_mpest(
    gene_tree_path: Path,
    executable: Path,
    simulation_id: str,
    timeout: float,
    starts: int,
) -> tuple[str, TaskResult, str]:
    gene_trees, taxa = read_gene_trees(gene_tree_path)
    expected_taxa = set(taxa)
    all_candidates: list[tuple[float, str]] = []
    log_parts: list[str] = []
    total_runtime = 0.0

    with tempfile.TemporaryDirectory(prefix="mpest-") as temporary_name:
        temporary = Path(temporary_name)
        nexus_path = temporary / "genes.nex"
        nexus_path.write_text(mpest_nexus(gene_trees, taxa), encoding="utf-8")
        besttree_path = Path(str(nexus_path) + "_besttree.tre")
        output_path = Path(str(nexus_path) + "_output.tre")

        for start_index in range(starts):
            seed = stable_seed(simulation_id, start_index)
            start_path = temporary / f"start_{start_index + 1}.tre"
            start_path.write_text(random_rooted_binary_tree(taxa, seed) + "\n", encoding="utf-8")
            besttree_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            command = [
                str(executable),
                "-i",
                str(nexus_path),
                "-n",
                "1",
                "-s",
                str(seed),
                "-u",
                str(start_path),
            ]
            result = run_command(command, temporary, timeout)
            total_runtime += result.elapsed_seconds
            log_parts.append(f"\n===== START {start_index + 1} =====\n{format_log(result)}")
            if result.returncode != 0 or not besttree_path.is_file():
                continue
            try:
                candidates = parse_mpest_best_trees(besttree_path.read_text(encoding="utf-8"))
                candidates = [item for item in candidates if taxa_in_tree(item[1]) == expected_taxa]
                all_candidates.extend(candidates)
            except (ValueError, NewickError) as error:
                log_parts.append(f"\nPARSE ERROR\n{error}\n")

    if not all_candidates:
        raise RuntimeError("Every MP-EST start failed.\n" + "".join(log_parts))
    best_score, best_tree = max(all_candidates, key=lambda item: item[0])
    unique_topologies = len({tree for _, tree in all_candidates})
    task = TaskResult(
        simulation_id=simulation_id,
        status="completed",
        runtime_seconds=total_runtime,
        score=best_score,
        successful_starts=len(all_candidates),
        unique_topologies=unique_topologies,
        candidate_scores=[score for score, _ in all_candidates],
    )
    return best_tree, task, "".join(log_parts)


def run_tmc(
    gene_tree_path: Path,
    executable: Path,
    simulation_id: str,
    timeout: float,
) -> tuple[str, TaskResult, str]:
    with tempfile.TemporaryDirectory(prefix="tmc-") as temporary_name:
        temporary = Path(temporary_name)
        raw_output = temporary / "species_tree.tre"
        numeric_output = temporary / "species_tree_numeric.tre"
        triplets_output = temporary / "triplets.dat"
        program_log = temporary / "tmc.log"
        command = [
            str(executable),
            "-fit",
            str(gene_tree_path),
            "-frt",
            str(raw_output),
            "-frtN",
            str(numeric_output),
            "-fsd",
            str(triplets_output),
            "-flg",
            str(program_log),
        ]
        result = run_command(command, temporary, timeout)
        log = format_log(result)
        if program_log.is_file():
            log += "\n\nTMC LOG\n" + program_log.read_text(encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"Triplet MaxCut returned {result.returncode}\n{log}")
        if not raw_output.is_file():
            raise RuntimeError(f"Triplet MaxCut did not create {raw_output}\n{log}")
        tree = normalize_newick(raw_output.read_text(encoding="utf-8"))
        return tree, TaskResult(simulation_id, "completed", result.elapsed_seconds), log


def run_one(row: dict[str, str], args: argparse.Namespace) -> TaskResult:
    simulation_id = row["simulation_id"]
    gene_tree_path = (args.trees_dir / f"{simulation_id}.tre").resolve()
    output_path = args.out_dir / f"{simulation_id}.tre"
    log_path = args.out_dir / "logs" / f"{simulation_id}.log"
    if not gene_tree_path.is_file():
        return TaskResult(simulation_id, "failed", message=f"Missing input: {gene_tree_path}")

    try:
        _, expected_taxa_list = read_gene_trees(gene_tree_path)
        expected_taxa = set(expected_taxa_list)
        if not args.overwrite and valid_existing_tree(output_path, expected_taxa):
            return TaskResult(simulation_id, "skipped_existing")

        if args.method == "stelar":
            tree, task, log = run_stelar(
                gene_tree_path,
                args.executable,
                simulation_id,
                args.timeout,
                exact=not args.stelar_heuristic,
            )
        elif args.method == "mpest":
            tree, task, log = run_mpest(
                gene_tree_path,
                args.executable,
                simulation_id,
                args.timeout,
                args.mpest_starts,
            )
        else:
            tree, task, log = run_tmc(
                gene_tree_path,
                args.executable,
                simulation_id,
                args.timeout,
            )

        observed_taxa = taxa_in_tree(tree)
        if observed_taxa != expected_taxa:
            raise ValueError(
                f"Taxon mismatch: expected {sorted(expected_taxa)}, observed {sorted(observed_taxa)}"
            )
        write_atomic(output_path, tree)
        if args.keep_success_logs:
            write_atomic(log_path, log)
        else:
            log_path.unlink(missing_ok=True)
        return task
    except Exception as error:  # Keep a 30,100-item batch resumable after individual failures.
        message = f"{type(error).__name__}: {error}"
        try:
            write_atomic(log_path, message)
        except OSError:
            pass
        return TaskResult(simulation_id, "failed", message=message)


def check_environment(args: argparse.Namespace) -> None:
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.mpest_starts < 1:
        raise ValueError("--mpest-starts must be at least 1")
    args.manifest = args.manifest.resolve()
    args.trees_dir = args.trees_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.executable = args.executable.resolve()
    if not args.executable.is_file():
        raise FileNotFoundError(f"Executable not found: {args.executable}")
    if args.method == "stelar" and shutil.which("java") is None:
        raise RuntimeError("java is not on PATH")
    if args.method in {"mpest", "tmc"} and not os.access(args.executable, os.X_OK):
        raise PermissionError(f"Executable bit is not set: {args.executable}")


def merge_records(path: Path, current: list[TaskResult]) -> None:
    fields = [
        "simulation_id",
        "status",
        "runtime_seconds",
        "score",
        "successful_starts",
        "unique_topologies",
        "candidate_scores",
        "message",
    ]
    records: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            records.update({row["simulation_id"]: row for row in csv.DictReader(handle)})
    for task in current:
        record = {
            "simulation_id": task.simulation_id,
            "status": task.status,
            "runtime_seconds": f"{task.runtime_seconds:.6f}" if task.runtime_seconds else "",
            "score": "" if task.score is None else f"{task.score:.12g}",
            "successful_starts": "" if task.successful_starts is None else task.successful_starts,
            "unique_topologies": "" if task.unique_topologies is None else task.unique_topologies,
            "candidate_scores": "" if task.candidate_scores is None else json.dumps(task.candidate_scores),
            "message": task.message,
        }
        if task.status != "skipped_existing" or task.simulation_id not in records:
            records[task.simulation_id] = record
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records[key] for key in sorted(records))


def main() -> int:
    args = parse_args()
    check_environment(args)
    rows = read_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)

    configuration = {
        "method": args.method,
        "manifest": str(args.manifest),
        "trees_dir": str(args.trees_dir),
        "executable": str(args.executable),
        "jobs": args.jobs,
        "timeout_seconds": args.timeout,
        "stelar_exact": args.method == "stelar" and not args.stelar_heuristic,
        "mpest_independent_starts": args.mpest_starts if args.method == "mpest" else None,
        "ground_truth_used_for_inference": False,
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )

    started = time.perf_counter()
    results: list[TaskResult] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_map = {executor.submit(run_one, row, args): row["simulation_id"] for row in rows}
        total = len(future_map)
        for completed_count, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            results.append(result)
            if completed_count == 1 or completed_count % 100 == 0 or completed_count == total:
                failed = sum(item.status == "failed" for item in results)
                print(f"[{completed_count}/{total}] failed={failed}", flush=True)

    merge_records(args.out_dir / "run_records.csv", results)
    failures = [result for result in results if result.status == "failed"]
    with (args.out_dir / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("simulation_id", "message"))
        writer.writeheader()
        writer.writerows(
            {"simulation_id": result.simulation_id, "message": result.message} for result in failures
        )

    valid_outputs = sum(
        valid_existing_tree(args.out_dir / f"{row['simulation_id']}.tre") for row in rows
    )
    summary = {
        "method": args.method,
        "requested": len(rows),
        "completed": sum(result.status == "completed" for result in results),
        "skipped_existing": sum(result.status == "skipped_existing" for result in results),
        "valid_outputs": valid_outputs,
        "failed": len(failures),
        "jobs": args.jobs,
        "total_wall_seconds": time.perf_counter() - started,
    }
    (args.out_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 2 if failures or valid_outputs != len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
