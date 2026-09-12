from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline

from .features import feature_columns


CYCLIC_PERMUTATIONS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


def augment_topology_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """Cyclically permute topology-indexed columns and the class label.

    A fixed labeled backbone may not place examples in every canonical class.
    This augmentation encodes the biological exchangeability of the three
    candidate rooted topologies without moving rows across CV folds.
    """

    augmented = []
    for permutation in CYCLIC_PERMUTATIONS:
        copy = frame.copy()
        for prefix in ["p", "n", "mean_bl", "median_bl", "std_bl", "g_asym"]:
            source_columns = [f"{prefix}{i}" for i in range(3)]
            if all(column in frame.columns for column in source_columns):
                for new_index, old_index in enumerate(permutation):
                    copy[f"{prefix}{new_index}"] = frame[f"{prefix}{old_index}"].to_numpy()
        inverse = {old_index: new_index for new_index, old_index in enumerate(permutation)}
        copy["label"] = frame["label"].map(inverse).astype(int)
        copy["topology_permutation"] = "".join(map(str, permutation))
        augmented.append(copy)
    return pd.concat(augmented, ignore_index=True)


def make_classifier(random_state: int = 20260828) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=250,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    early_stopping=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _three_class_probabilities(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = model.named_steps["classifier"].classes_
    result = np.zeros((len(x), 3), dtype=float)
    for source_index, label in enumerate(classes):
        result[:, int(label)] = raw[:, source_index]
    return result


def multiclass_brier(y: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(3)[y.astype(int)]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def top_label_ece(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def calibration_table(
    y: np.ndarray, probabilities: np.ndarray, source: str, bins: int = 10
) -> pd.DataFrame:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else np.nan
        empirical_accuracy = float(correct[mask].mean()) if count else np.nan
        rows.append(
            {
                "source": source,
                "bin": index,
                "lower_bound": low,
                "upper_bound": high,
                "rows": count,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "absolute_gap": abs(empirical_accuracy - mean_confidence) if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def probability_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, probabilities.argmax(axis=1))),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "multiclass_brier": multiclass_brier(y, probabilities),
        "top_label_ece_10": top_label_ece(y, probabilities, bins=10),
    }


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    log_probabilities = np.log(np.clip(probabilities, 1e-15, 1.0)) / float(temperature)
    log_probabilities -= log_probabilities.max(axis=1, keepdims=True)
    scaled = np.exp(log_probabilities)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(y: np.ndarray, probabilities: np.ndarray) -> float:
    """Fit one multiclass temperature by deterministic log-loss minimization."""

    candidates = np.exp(np.linspace(np.log(0.25), np.log(4.0), 161))
    losses = [log_loss(y, apply_temperature(probabilities, value), labels=[0, 1, 2]) for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def _default_strata_column(frame: pd.DataFrame, cv_mode: str, folds: int) -> str | None:
    if cv_mode != "backbone":
        return None
    if "tree_size" in frame.columns and set(frame["tree_size"].dropna().astype(int)) == {6}:
        if "backbone_shape_class" in frame.columns:
            counts = frame.drop_duplicates("backbone_id").groupby("backbone_shape_class").size()
            if not counts.empty and int(counts.min()) >= folds:
                return "backbone_shape_class"
    if "backbone_balance_bin" in frame.columns and frame["backbone_balance_bin"].nunique() > 1:
        counts = frame.drop_duplicates("backbone_id").groupby("backbone_balance_bin").size()
        if not counts.empty and int(counts.min()) >= folds:
            return "backbone_balance_bin"
    if "backbone_shape_class" in frame.columns:
        counts = frame.drop_duplicates("backbone_id").groupby("backbone_shape_class").size()
        if not counts.empty and int(counts.min()) >= folds:
            return "backbone_shape_class"
    return None


def balanced_group_folds(
    frame: pd.DataFrame,
    group_column: str,
    folds: int,
    random_state: int,
    strata_column: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assign complete groups to folds, balanced within an optional group stratum."""

    if folds < 2:
        raise ValueError("folds must be at least 2")
    group_columns = [group_column] + ([strata_column] if strata_column else [])
    group_frame = frame[group_columns].copy()
    if strata_column:
        counts = group_frame.groupby(group_column)[strata_column].nunique(dropna=False)
        if (counts != 1).any():
            raise ValueError(f"Each {group_column} must have one {strata_column}")
    group_sizes = frame.groupby(group_column).size().rename("rows").reset_index()
    group_frame = group_frame.drop_duplicates(group_column).merge(
        group_sizes, on=group_column, how="left", validate="one_to_one"
    )
    if len(group_frame) < folds:
        raise ValueError(f"folds={folds} exceeds unique {group_column} groups={len(group_frame)}")
    if strata_column:
        minimum = int(group_frame.groupby(strata_column, dropna=False).size().min())
        if minimum < folds:
            raise ValueError(
                f"Every {strata_column} stratum needs at least {folds} groups; smallest has {minimum}"
            )
        strata = list(group_frame.groupby(strata_column, dropna=False, sort=True))
    else:
        strata = [("all", group_frame)]

    rng = np.random.default_rng(random_state)
    assignments = []
    total_rows = np.zeros(folds, dtype=int)
    for stratum_value, values in strata:
        records = values.to_dict("records")
        rng.shuffle(records)
        records.sort(key=lambda record: int(record["rows"]), reverse=True)
        stratum_rows = np.zeros(folds, dtype=int)
        stratum_groups = np.zeros(folds, dtype=int)
        for record in records:
            fold_index = min(
                range(folds),
                key=lambda index: (stratum_groups[index], stratum_rows[index], total_rows[index], index),
            )
            assignments.append(
                {
                    group_column: record[group_column],
                    "stratum": stratum_value,
                    "fold": fold_index + 1,
                    "rows": int(record["rows"]),
                }
            )
            stratum_groups[fold_index] += 1
            stratum_rows[fold_index] += int(record["rows"])
            total_rows[fold_index] += int(record["rows"])

    assignment_frame = pd.DataFrame(assignments)
    mapping = assignment_frame.set_index(group_column)["fold"]
    fold_ids = frame[group_column].map(mapping).to_numpy(dtype=int)
    if np.any(fold_ids < 1):
        raise RuntimeError("Fold assignment is incomplete")
    return fold_ids, assignment_frame.sort_values(["fold", "stratum", group_column]).reset_index(drop=True)


def _inner_oof_temperature(
    frame: pd.DataFrame,
    columns: list[str],
    group_column: str,
    strata_column: str | None,
    folds: int,
    random_state: int,
) -> float:
    unique_groups = frame[group_column].nunique()
    inner_folds = min(folds, int(unique_groups))
    if inner_folds < 2:
        return 1.0
    if strata_column:
        smallest_stratum = int(frame.drop_duplicates(group_column).groupby(strata_column).size().min())
        inner_folds = min(inner_folds, smallest_stratum)
    if inner_folds < 2:
        return 1.0
    fold_ids, _ = balanced_group_folds(
        frame, group_column, inner_folds, random_state, strata_column
    )
    y = frame["label"].to_numpy(dtype=int)
    raw_oof = np.full((len(frame), 3), np.nan)
    for fold in range(1, inner_folds + 1):
        train_index = np.flatnonzero(fold_ids != fold)
        validation_index = np.flatnonzero(fold_ids == fold)
        augmented = augment_topology_coordinates(frame.iloc[train_index])
        model = make_classifier(random_state + fold)
        model.fit(augmented[columns], augmented["label"].to_numpy(dtype=int))
        raw_oof[validation_index] = _three_class_probabilities(
            model, frame.iloc[validation_index][columns]
        )
    if np.isnan(raw_oof).any():
        raise RuntimeError("Inner out-of-fold probability matrix is incomplete")
    return fit_temperature(y, raw_oof)


def train_grouped_cv(
    features_path: str | Path,
    out_dir: str | Path,
    feature_set: str = "base13",
    folds: int = 5,
    random_state: int = 20260828,
    cv_mode: str = "backbone",
    calibration: str = "temperature",
    inner_folds: int = 3,
    strata_column: str | None = None,
) -> dict:
    training_started = time.perf_counter()
    frame = pd.read_csv(features_path)
    columns = feature_columns(feature_set)
    if cv_mode not in {"backbone", "scenario"}:
        raise ValueError("cv_mode must be 'backbone' or 'scenario'")
    if calibration not in {"none", "temperature"}:
        raise ValueError("calibration must be 'none' or 'temperature'")
    group_column = "backbone_id" if cv_mode == "backbone" else "scenario_group"
    missing = [c for c in columns + ["label", group_column] if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    unique_groups = int(frame[group_column].nunique())
    strata_column = strata_column or _default_strata_column(frame, cv_mode, folds)
    if strata_column and strata_column not in frame.columns:
        raise ValueError(f"Missing strata column: {strata_column}")

    x = frame[columns]
    y = frame["label"].to_numpy(dtype=int)
    groups = frame[group_column].astype(str).to_numpy()
    fold_ids, fold_assignments = balanced_group_folds(
        frame, group_column, folds, random_state, strata_column
    )
    raw_oof = np.full((len(frame), 3), np.nan)
    calibrated_oof = np.full((len(frame), 3), np.nan)
    fold_rows = []

    for fold in range(1, folds + 1):
        fold_started = time.perf_counter()
        train_index = np.flatnonzero(fold_ids != fold)
        test_index = np.flatnonzero(fold_ids == fold)
        train_frame = frame.iloc[train_index].reset_index(drop=True)
        temperature = 1.0
        if calibration == "temperature":
            temperature = _inner_oof_temperature(
                train_frame,
                columns,
                group_column,
                strata_column,
                inner_folds,
                random_state + fold * 100,
            )
        augmented_train = augment_topology_coordinates(frame.iloc[train_index])
        model = make_classifier(random_state + fold)
        model.fit(augmented_train[columns], augmented_train["label"].to_numpy(dtype=int))
        raw_probabilities = _three_class_probabilities(model, x.iloc[test_index])
        probabilities = apply_temperature(raw_probabilities, temperature)
        raw_oof[test_index] = raw_probabilities
        calibrated_oof[test_index] = probabilities
        raw_metrics = probability_metrics(y[test_index], raw_probabilities)
        calibrated_metrics = probability_metrics(y[test_index], probabilities)
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "augmented_train_rows": len(augmented_train),
                "test_rows": len(test_index),
                "train_groups": len(set(groups[train_index])),
                "test_groups": len(set(groups[test_index])),
                "temperature": temperature,
                "wall_seconds": time.perf_counter() - fold_started,
                **{f"raw_{name}": value for name, value in raw_metrics.items()},
                **calibrated_metrics,
            }
        )

    if np.isnan(raw_oof).any() or np.isnan(calibrated_oof).any():
        raise RuntimeError("Out-of-fold probability matrix is incomplete")

    final_temperature = fit_temperature(y, raw_oof) if calibration == "temperature" else 1.0
    augmented_all = augment_topology_coordinates(frame)
    final_model = make_classifier(random_state)
    final_model.fit(augmented_all[columns], augmented_all["label"].to_numpy(dtype=int))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": final_model,
        "feature_set": feature_set,
        "feature_columns": columns,
        "cv_mode": cv_mode,
        "calibration": calibration,
        "temperature": final_temperature,
        "strata_column": strata_column,
    }
    joblib.dump(bundle, out_dir / "model.joblib")

    metadata_columns = [
        "simulation_id",
        "scenario_group",
        "tree_size",
        "triplets_per_simulation",
        "backbone_id",
        "backbone_shape_class",
        "backbone_balance_bin",
        "triplet_id",
        "label",
        "gamma",
        "ils_length",
        "gene_tree_count",
        "direction",
        "pair_type",
    ]
    prediction_frame = frame[[name for name in metadata_columns if name in frame.columns]].copy()
    prediction_frame[["raw_prob0", "raw_prob1", "raw_prob2"]] = raw_oof
    prediction_frame[["prob0", "prob1", "prob2"]] = calibrated_oof
    prediction_frame["fold"] = fold_ids
    prediction_frame["raw_prediction"] = raw_oof.argmax(axis=1)
    prediction_frame["prediction"] = calibrated_oof.argmax(axis=1)
    prediction_frame.to_csv(out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    fold_assignments.to_csv(out_dir / "fold_assignments.csv", index=False)
    pd.concat(
        [
            calibration_table(y, raw_oof, "raw"),
            calibration_table(y, calibrated_oof, "calibrated"),
        ],
        ignore_index=True,
    ).to_csv(out_dir / "calibration_curve.csv", index=False)

    raw_metrics = probability_metrics(y, raw_oof)
    calibrated_metrics = probability_metrics(y, calibrated_oof)
    metrics = {
        "feature_set": feature_set,
        "cv_mode": cv_mode,
        "group_column": group_column,
        "strata_column": strata_column,
        "folds": folds,
        "inner_folds": inner_folds,
        "calibration": calibration,
        "final_temperature": final_temperature,
        "rows": len(frame),
        "augmented_training_rows": len(augmented_all),
        "simulation_count": int(frame["simulation_id"].nunique()),
        "cv_group_count": int(unique_groups),
        "total_training_wall_seconds": time.perf_counter() - training_started,
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
        **calibrated_metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def load_model(path: str | Path) -> dict:
    bundle = joblib.load(path)
    required = {"pipeline", "feature_set", "feature_columns"}
    if not required.issubset(bundle):
        raise ValueError("Invalid model bundle")
    return bundle


def predict_frame(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    raw = raw_predict_frame(frame, bundle)
    return apply_temperature(raw, float(bundle.get("temperature", 1.0)))


def raw_predict_frame(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    return _three_class_probabilities(bundle["pipeline"], frame[bundle["feature_columns"]])
