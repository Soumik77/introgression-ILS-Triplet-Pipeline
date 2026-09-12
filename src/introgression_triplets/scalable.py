from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .assembly import assemble_tree, probability_topologies
from .features import feature_columns
from .model import (
    _inner_oof_temperature,
    _three_class_probabilities,
    apply_temperature,
    augment_topology_coordinates,
    balanced_group_folds,
    fit_temperature,
    make_classifier,
)
from .simulation import deterministic_seed


def _feature_shard_path(directory: str | Path, simulation_id: str) -> Path:
    return Path(directory) / f"{simulation_id}.csv.gz"


def _atomic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    try:
        frame.to_csv(partial, index=False, compression="gzip", float_format="%.10g")
        os.replace(partial, path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(partial, path)


@dataclass
class ProbabilityAccumulator:
    bins: int = 10
    rows: int = 0
    correct: int = 0
    log_loss_sum: float = 0.0
    brier_sum: float = 0.0
    bin_rows: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.int64))
    bin_confidence: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=float))
    bin_correct: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=float))

    def __post_init__(self) -> None:
        if self.bins < 1:
            raise ValueError("bins must be at least one")
        self.bin_rows = np.zeros(self.bins, dtype=np.int64)
        self.bin_confidence = np.zeros(self.bins, dtype=float)
        self.bin_correct = np.zeros(self.bins, dtype=float)

    def update(self, labels: np.ndarray, probabilities: np.ndarray) -> None:
        labels = np.asarray(labels, dtype=int)
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(labels), 3):
            raise ValueError("Expected a three-class probability matrix")
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        correct = predictions == labels
        clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1.0)
        one_hot = np.eye(3)[labels]
        indices = np.minimum((confidence * self.bins).astype(int), self.bins - 1)

        self.rows += len(labels)
        self.correct += int(correct.sum())
        self.log_loss_sum += float((-np.log(clipped)).sum())
        self.brier_sum += float(np.sum((probabilities - one_hot) ** 2))
        for index in range(self.bins):
            mask = indices == index
            if mask.any():
                self.bin_rows[index] += int(mask.sum())
                self.bin_confidence[index] += float(confidence[mask].sum())
                self.bin_correct[index] += float(correct[mask].sum())

    def metrics(self) -> dict[str, float]:
        if self.rows == 0:
            raise ValueError("No probability rows were accumulated")
        ece = 0.0
        for count, confidence_sum, correct_sum in zip(
            self.bin_rows, self.bin_confidence, self.bin_correct
        ):
            if count:
                ece += (count / self.rows) * abs(correct_sum / count - confidence_sum / count)
        return {
            "accuracy": self.correct / self.rows,
            "log_loss": self.log_loss_sum / self.rows,
            "multiclass_brier": self.brier_sum / self.rows,
            "top_label_ece_10": float(ece),
        }

    def calibration_table(self, source: str) -> pd.DataFrame:
        rows = []
        for index in range(self.bins):
            count = int(self.bin_rows[index])
            mean_confidence = self.bin_confidence[index] / count if count else np.nan
            empirical_accuracy = self.bin_correct[index] / count if count else np.nan
            rows.append(
                {
                    "source": source,
                    "bin": index + 1,
                    "lower_bound": index / self.bins,
                    "upper_bound": (index + 1) / self.bins,
                    "rows": count,
                    "mean_confidence": mean_confidence,
                    "empirical_accuracy": empirical_accuracy,
                    "absolute_gap": (
                        abs(empirical_accuracy - mean_confidence) if count else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows)


def _manifest_with_backbone_folds(
    manifest: pd.DataFrame, folds: int, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["simulation_id", "backbone_id", "tree_size"]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    fold_ids, assignments = balanced_group_folds(
        manifest,
        "backbone_id",
        folds,
        random_state,
        "tree_size",
    )
    result = manifest.copy()
    result["outer_fold"] = fold_ids
    return result, assignments


def _training_sample(
    manifest: pd.DataFrame,
    feature_dir: str | Path,
    columns: list[str],
    rows_per_simulation: int,
    random_state: int,
) -> pd.DataFrame:
    if rows_per_simulation < 1:
        raise ValueError("training_rows_per_simulation must be at least one")
    usecols = list(dict.fromkeys(columns + ["label", "simulation_id", "backbone_id", "tree_size"]))
    frames = []
    missing_paths = []
    for row in manifest.itertuples(index=False):
        path = _feature_shard_path(feature_dir, str(row.simulation_id))
        if not path.is_file():
            missing_paths.append(path)
            if len(missing_paths) == 5:
                break
            continue
        frame = pd.read_csv(path, usecols=usecols)
        if set(frame["simulation_id"].astype(str)) != {str(row.simulation_id)}:
            raise ValueError(f"Feature shard has the wrong simulation_id: {path}")
        if len(frame) > rows_per_simulation:
            frame = frame.sample(
                n=rows_per_simulation,
                random_state=deterministic_seed(
                    random_state, f"training-sample|{row.simulation_id}"
                ),
            )
        frame = frame.copy()
        frame["outer_fold"] = int(row.outer_fold)
        frames.append(frame)
    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing feature shards, first paths: {preview}")
    if not frames:
        raise ValueError("No feature shards were loaded")
    return pd.concat(frames, ignore_index=True)


def train_sharded_grouped_cv(
    manifest_path: str | Path,
    feature_dir: str | Path,
    out_dir: str | Path,
    *,
    feature_set: str = "enhanced",
    folds: int = 5,
    inner_folds: int = 3,
    random_state: int = 20260828,
    calibration: str = "temperature",
    training_rows_per_simulation: int = 20,
    final_calibration_rows_per_simulation: int = 10,
) -> dict:
    """Train on bounded samples and predict every held-out feature shard.

    Complete backbones stay in one outer fold. Each outer-fold temperature is
    learned from grouped inner folds. Only the fit sample is held in memory;
    held-out predictions are generated and written one simulation at a time.
    """

    started = time.perf_counter()
    if calibration not in {"none", "temperature"}:
        raise ValueError("calibration must be 'none' or 'temperature'")
    if final_calibration_rows_per_simulation < 1:
        raise ValueError("final_calibration_rows_per_simulation must be at least one")
    manifest = pd.read_csv(manifest_path)
    if manifest.empty or manifest["simulation_id"].duplicated().any():
        raise ValueError("Manifest must contain unique simulation rows")
    manifest, assignments = _manifest_with_backbone_folds(manifest, folds, random_state)
    columns = feature_columns(feature_set)
    out_dir = Path(out_dir)
    prediction_dir = out_dir / "oof_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    sample = _training_sample(
        manifest,
        feature_dir,
        columns,
        training_rows_per_simulation,
        random_state,
    )
    _atomic_gzip_csv(sample, out_dir / "training_sample.csv.gz")
    assignments.to_csv(out_dir / "fold_assignments.csv", index=False)

    raw_accumulator = ProbabilityAccumulator()
    calibrated_accumulator = ProbabilityAccumulator()
    calibration_labels: list[np.ndarray] = []
    calibration_probabilities: list[np.ndarray] = []
    fold_records = []

    for fold in range(1, folds + 1):
        fold_started = time.perf_counter()
        train_sample = sample.loc[sample["outer_fold"] != fold].reset_index(drop=True)
        if train_sample.empty:
            raise ValueError(f"Outer fold {fold} has no training rows")
        temperature = 1.0
        if calibration == "temperature":
            temperature = _inner_oof_temperature(
                train_sample,
                columns,
                "backbone_id",
                "tree_size",
                inner_folds,
                random_state + fold * 100,
            )
        augmented = augment_topology_coordinates(train_sample)
        model = make_classifier(random_state + fold)
        model.fit(augmented[columns], augmented["label"].to_numpy(dtype=int))

        fold_raw = ProbabilityAccumulator()
        fold_calibrated = ProbabilityAccumulator()
        test_rows = 0
        fold_manifest = manifest.loc[manifest["outer_fold"] == fold]
        for manifest_row in fold_manifest.itertuples(index=False):
            simulation_id = str(manifest_row.simulation_id)
            feature_path = _feature_shard_path(feature_dir, simulation_id)
            usecols = list(dict.fromkeys(columns + ["simulation_id", "triplet_id", "label"]))
            frame = pd.read_csv(feature_path, usecols=usecols)
            labels = frame["label"].to_numpy(dtype=int)
            raw = _three_class_probabilities(model, frame[columns])
            calibrated = apply_temperature(raw, temperature)
            predictions = frame[["simulation_id", "triplet_id", "label"]].copy()
            predictions[["raw_prob0", "raw_prob1", "raw_prob2"]] = raw
            predictions[["prob0", "prob1", "prob2"]] = calibrated
            predictions["fold"] = fold
            predictions["raw_prediction"] = raw.argmax(axis=1)
            predictions["prediction"] = calibrated.argmax(axis=1)
            _atomic_gzip_csv(
                predictions, _feature_shard_path(prediction_dir, simulation_id)
            )

            raw_accumulator.update(labels, raw)
            calibrated_accumulator.update(labels, calibrated)
            fold_raw.update(labels, raw)
            fold_calibrated.update(labels, calibrated)
            test_rows += len(frame)

            calibration_count = min(final_calibration_rows_per_simulation, len(frame))
            if calibration_count:
                rng = np.random.default_rng(
                    deterministic_seed(
                        random_state, f"final-calibration|{simulation_id}"
                    )
                )
                indices = rng.choice(len(frame), size=calibration_count, replace=False)
                calibration_labels.append(labels[indices])
                calibration_probabilities.append(raw[indices])

        fold_records.append(
            {
                "fold": fold,
                "train_sample_rows": len(train_sample),
                "augmented_train_rows": len(augmented),
                "test_rows": test_rows,
                "train_backbones": int(train_sample["backbone_id"].nunique()),
                "test_backbones": int(fold_manifest["backbone_id"].nunique()),
                "temperature": temperature,
                "wall_seconds": time.perf_counter() - fold_started,
                **{f"raw_{key}": value for key, value in fold_raw.metrics().items()},
                **fold_calibrated.metrics(),
            }
        )

    final_temperature = 1.0
    if calibration == "temperature":
        final_temperature = fit_temperature(
            np.concatenate(calibration_labels),
            np.concatenate(calibration_probabilities),
        )
    augmented_all = augment_topology_coordinates(sample)
    final_model = make_classifier(random_state)
    final_model.fit(augmented_all[columns], augmented_all["label"].to_numpy(dtype=int))
    bundle = {
        "pipeline": final_model,
        "feature_set": feature_set,
        "feature_columns": columns,
        "cv_mode": "backbone",
        "calibration": calibration,
        "temperature": final_temperature,
        "strata_column": "tree_size",
        "training_mode": "sharded_sample_fit_full_oof_prediction",
        "training_rows_per_simulation": training_rows_per_simulation,
    }
    joblib.dump(bundle, out_dir / "model.joblib")
    pd.DataFrame(fold_records).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.concat(
        [
            raw_accumulator.calibration_table("raw"),
            calibrated_accumulator.calibration_table("calibrated"),
        ],
        ignore_index=True,
    ).to_csv(out_dir / "calibration_curve.csv", index=False)

    raw_metrics = raw_accumulator.metrics()
    calibrated_metrics = calibrated_accumulator.metrics()
    metrics = {
        "feature_set": feature_set,
        "cv_mode": "backbone",
        "group_column": "backbone_id",
        "strata_column": "tree_size",
        "folds": folds,
        "inner_folds": inner_folds,
        "calibration": calibration,
        "final_temperature": final_temperature,
        "rows": calibrated_accumulator.rows,
        "training_sample_rows": len(sample),
        "augmented_training_rows": len(augmented_all),
        "training_rows_per_simulation": training_rows_per_simulation,
        "simulation_count": int(manifest["simulation_id"].nunique()),
        "cv_group_count": int(manifest["backbone_id"].nunique()),
        "total_training_wall_seconds": time.perf_counter() - started,
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
        **calibrated_metrics,
    }
    _atomic_json(metrics, out_dir / "metrics.json")
    return metrics


def _probabilities_for_shard(
    features: pd.DataFrame,
    prediction_path: Path | None,
    weight_mode: str,
    probability_source: str,
) -> np.ndarray:
    if weight_mode == "frequency":
        return features[["p0", "p1", "p2"]].to_numpy(dtype=float)
    if weight_mode == "majority":
        raw = features[["p0", "p1", "p2"]].to_numpy(dtype=float)
        result = np.zeros_like(raw)
        result[np.arange(len(result)), raw.argmax(axis=1)] = 1.0
        return result
    if prediction_path is None or not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    predictions = pd.read_csv(prediction_path)
    prefix = "raw_prob" if probability_source == "raw" else "prob"
    columns = [f"{prefix}{index}" for index in range(3)]
    merged = features[["simulation_id", "triplet_id"]].merge(
        predictions[["simulation_id", "triplet_id", *columns]],
        on=["simulation_id", "triplet_id"],
        how="left",
        validate="one_to_one",
    )
    if merged[columns].isna().any().any():
        raise ValueError(f"Prediction shard does not cover all feature rows: {prediction_path}")
    probabilities = merged[columns].to_numpy(dtype=float)
    if weight_mode in {"ml", "ml-soft"}:
        return probabilities
    winners = probabilities.argmax(axis=1)
    result = np.zeros_like(probabilities)
    if weight_mode == "ml-hard":
        result[np.arange(len(result)), winners] = 1.0
    elif weight_mode == "ml-confidence":
        result[np.arange(len(result)), winners] = probabilities.max(axis=1)
    else:
        raise ValueError(f"Unknown weight mode: {weight_mode}")
    return result


_ASSEMBLY_METADATA_COLUMNS = [
    "simulation_id",
    "experiment_id",
    "taxon_set_id",
    "tree_size",
    "triplets_per_simulation",
    "backbone_id",
    "backbone_shape",
    "backbone_shape_class",
    "backbone_newick",
    "backbone_balance_score",
    "backbone_balance_bin",
    "donor",
    "recipient",
    "direction",
    "pair_type",
    "gamma",
    "ils_length",
    "gene_tree_count",
    "replicate",
    "scenario_group",
    "seed",
]


def _assemble_feature_shard_task(
    payload: tuple[dict, str, str | None, str, str, str]
) -> dict:
    row_dict, feature_name, prediction_name, weight_mode, probability_source, method = payload
    simulation_id = str(row_dict["simulation_id"])
    features = pd.read_csv(
        feature_name,
        usecols=[
            "simulation_id",
            "triplet_id",
            "taxon_a",
            "taxon_b",
            "taxon_c",
            "p0",
            "p1",
            "p2",
        ],
    )
    if features.empty or set(features["simulation_id"].astype(str)) != {simulation_id}:
        raise ValueError(f"Feature shard has the wrong simulation_id: {feature_name}")
    probabilities = _probabilities_for_shard(
        features,
        Path(prediction_name) if prediction_name is not None else None,
        weight_mode,
        probability_source,
    )
    record = {
        **{column: row_dict[column] for column in _ASSEMBLY_METADATA_COLUMNS if column in row_dict},
        "method": method,
        "weight_mode": weight_mode,
        "probability_source": probability_source,
    }
    simulation_started = time.perf_counter()
    try:
        topologies = []
        for row_index, row in enumerate(features.itertuples(index=False)):
            taxa = (row.taxon_a, row.taxon_b, row.taxon_c)
            topologies.extend(probability_topologies(taxa, probabilities[row_index]))
        taxa = set(features[["taxon_a", "taxon_b", "taxon_c"]].to_numpy().ravel())
        inferred = assemble_tree(taxa, topologies, method=method).to_newick()
        record.update(
            {
                "assembly_status": "completed",
                "assembly_error": "",
                "inferred_newick": inferred,
            }
        )
    except Exception as exc:
        record.update(
            {
                "assembly_status": "failed",
                "assembly_error": f"{type(exc).__name__}: {exc}",
                "inferred_newick": "",
            }
        )
    record["assembly_seconds"] = time.perf_counter() - simulation_started
    return record


def assemble_feature_shards(
    manifest_path: str | Path,
    feature_dir: str | Path,
    output_path: str | Path,
    *,
    prediction_dir: str | Path | None = None,
    weight_mode: str = "ml-soft",
    probability_source: str = "calibrated",
    method: str = "fm",
    overwrite: bool = False,
    jobs: int = 1,
) -> dict:
    """Assemble trees one simulation at a time from feature and OOF shards."""

    started = time.perf_counter()
    if jobs < 1:
        raise ValueError("jobs must be at least one")
    manifest = pd.read_csv(manifest_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and output_path.stat().st_size > 0 and not overwrite:
        existing = pd.read_csv(output_path)
        if len(existing) == len(manifest) and existing["simulation_id"].is_unique:
            return {
                "assembled": str(output_path),
                "simulations": len(existing),
                "failed_simulations": int((existing["assembly_status"] == "failed").sum()),
                "skipped_existing": True,
                "total_wall_seconds": time.perf_counter() - started,
            }
        raise ValueError(f"Existing assembly output is incomplete: {output_path}")

    partial = output_path.with_name(f".{output_path.name}.partial")
    if overwrite:
        partial.unlink(missing_ok=True)
    completed: set[str] = set()
    if partial.is_file() and partial.stat().st_size > 0:
        checkpoint = pd.read_csv(partial, usecols=["simulation_id"])
        if checkpoint["simulation_id"].duplicated().any():
            raise ValueError(f"Assembly checkpoint has duplicate simulations: {partial}")
        completed = set(checkpoint["simulation_id"].astype(str))

    new_rows = 0
    payloads = []
    for row_dict in manifest.to_dict(orient="records"):
        simulation_id = str(row_dict["simulation_id"])
        if simulation_id in completed:
            continue
        payloads.append(
            (
                row_dict,
                str(_feature_shard_path(feature_dir, simulation_id)),
                str(_feature_shard_path(prediction_dir, simulation_id))
                if prediction_dir is not None
                else None,
                weight_mode,
                probability_source,
                method,
            )
        )

    if jobs == 1:
        records = map(_assemble_feature_shard_task, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=jobs)
        records = executor.map(_assemble_feature_shard_task, payloads, chunksize=1)
    try:
        for record in records:
            pd.DataFrame([record]).to_csv(
                partial,
                mode="a",
                header=not partial.exists() or partial.stat().st_size == 0,
                index=False,
            )
            new_rows += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    result = pd.read_csv(partial)
    expected = set(manifest["simulation_id"].astype(str))
    observed = set(result["simulation_id"].astype(str))
    if len(result) != len(manifest) or not result["simulation_id"].is_unique or observed != expected:
        raise RuntimeError("Assembly checkpoint does not exactly cover the manifest")
    os.replace(partial, output_path)
    return {
        "assembled": str(output_path),
        "simulations": len(result),
        "new_simulations": new_rows,
        "failed_simulations": int((result["assembly_status"] == "failed").sum()),
        "skipped_existing": False,
        "jobs": jobs,
        "total_wall_seconds": time.perf_counter() - started,
    }
