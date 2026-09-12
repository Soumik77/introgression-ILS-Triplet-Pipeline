import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from introgression_triplets.assembly import assemble_tree, probability_topologies
from introgression_triplets.cli import _probability_matrix
from introgression_triplets.evaluate import evaluate_assembled, rooted_rf
from introgression_triplets.features import extract_feature_shards, summarize_gene_trees
from introgression_triplets.model import apply_temperature, balanced_group_folds, fit_temperature
from introgression_triplets.newick import parse_newick
from introgression_triplets.scalable import assemble_feature_shards, train_sharded_grouped_cv
from introgression_triplets.simulation import (
    backbone_from_topology,
    enumerate_rooted_binary_topologies,
    make_manifest,
    render_backbone,
    render_network,
    select_balanced_analysis_manifest,
    six_taxon_backbone,
    topology_shape,
)
from introgression_triplets.triplets import (
    TripletTreeIndex,
    extract_triplet,
    topology_labels,
    triplet_combinations,
    true_triplet_labels,
)


ROOT = Path(__file__).resolve().parents[1]


class SimulationTests(unittest.TestCase):
    def test_manifest_counts(self):
        pilot = make_manifest(ROOT / "configs/pilot.yaml")
        full = make_manifest(ROOT / "configs/full_corrected.yaml")
        self.assertEqual(len(pilot), 36)
        self.assertEqual(pilot.backbone_id.nunique(), 6)
        self.assertEqual(pilot.backbone_shape_class.nunique(), 6)
        self.assertEqual(len(full), 30100)
        self.assertEqual(full.backbone_id.nunique(), 60)
        self.assertEqual(len(make_manifest(ROOT / "configs/full_legacy_33000.yaml")), 33000)

    def test_all_six_taxon_topologies_are_enumerated(self):
        topologies = enumerate_rooted_binary_topologies(tuple("ABCDEF"))
        self.assertEqual(len(topologies), 945)
        self.assertEqual(len({topology_shape(topology) for topology in topologies}), 6)

    def test_journal_manifests_have_planned_counts(self):
        all_945 = make_manifest(ROOT / "configs/journal_all_945.yaml")
        n10 = make_manifest(ROOT / "configs/journal_n10.yaml")
        self.assertEqual(len(all_945), 34020)
        self.assertEqual(all_945.backbone_id.nunique(), 945)
        self.assertEqual(len(n10), 10080)
        self.assertEqual(n10.tree_size.unique().tolist(), [10])
        self.assertEqual(n10.groupby("backbone_balance_bin").backbone_id.nunique().tolist(), [10] * 4)

    def test_network_has_reticulation_and_gamma(self):
        network = render_network(six_taxon_backbone(0.1), "A", "C", 0.2, 0.25)
        self.assertEqual(network.count("#H1"), 2)
        self.assertIn("::0.2", network)

    def test_multisize_supervisor_grid_and_balanced_selection(self):
        configuration = """
experiment_id: test_scope
taxon_count_range: {start: 4, end: 5}
taxon_prefix: A
taxon_width: 1
backbone_strategy: balance_stratified_random
backbones_per_taxon_count: 4
backbone_candidate_multiplier: 4
backbone_assignment: cross_product
pendant_base: 1.0
gene_flow_height: 0.25
direction_strategy: distance_stratified
ils_levels: [0.02, 0.10, 0.50]
gammas: [0.00, 0.25, 0.50]
gene_tree_counts: [500, 1000, 2000, 2500]
null_replicates: 3
positive_replicates: 1
deduplicate_gamma_zero: true
max_triplets_per_simulation: 1000
base_seed: 20260828
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scope.yaml"
            path.write_text(configuration, encoding="utf-8")
            manifest = make_manifest(path)
        self.assertEqual(len(manifest), 8 * 108)
        self.assertEqual(manifest.backbone_id.nunique(), 8)
        self.assertEqual(sorted(manifest.tree_size.unique()), [4, 5])
        self.assertEqual(set(manifest.loc[manifest.gamma == 0, "pair_type"]), {"none"})
        self.assertEqual(set(manifest.loc[manifest.gamma == 0, "direction"]), {"none"})
        self.assertTrue(manifest.simulation_id.is_unique)

        selected = select_balanced_analysis_manifest(manifest, 12, 123)
        self.assertEqual(len(selected), 8 * 12)
        self.assertTrue((selected.groupby("backbone_id").size() == 12).all())
        gamma_counts = selected.groupby(["backbone_id", "gamma"]).size().unstack(fill_value=0)
        gene_tree_counts = (
            selected.groupby(["backbone_id", "gene_tree_count"]).size().unstack(fill_value=0)
        )
        self.assertTrue((gamma_counts.to_numpy() == 4).all())
        self.assertTrue((gene_tree_counts.to_numpy() == 3).all())
        positive_pair_counts = (
            selected.loc[selected.gamma > 0]
            .groupby(["backbone_id", "pair_type"])
            .size()
            .unstack(fill_value=0)
        )
        self.assertTrue(
            ((positive_pair_counts.max(axis=1) - positive_pair_counts.min(axis=1)) <= 1).all()
        )

        deadline = select_balanced_analysis_manifest(manifest, 3, 123)
        self.assertEqual(len(deadline), 8 * 3)
        self.assertTrue((deadline.groupby("backbone_id").size() == 3).all())
        deadline_gamma_counts = (
            deadline.groupby(["backbone_id", "gamma"]).size().unstack(fill_value=0)
        )
        deadline_ils_counts = (
            deadline.groupby(["backbone_id", "ils_length"]).size().unstack(fill_value=0)
        )
        self.assertTrue((deadline_gamma_counts.to_numpy() == 1).all())
        self.assertTrue((deadline_ils_counts.to_numpy() == 1).all())
        self.assertTrue(
            (deadline.groupby("backbone_id")["gene_tree_count"].nunique() == 3).all()
        )
        self.assertTrue(
            (
                deadline.loc[deadline.gamma > 0]
                .groupby("backbone_id")["pair_type"]
                .nunique()
                == 2
            ).all()
        )
        deadline_gene_margins = (
            deadline.groupby(["tree_size", "gene_tree_count"])
            .size()
            .unstack(fill_value=0)
        )
        deadline_pair_margins = (
            deadline.loc[deadline.gamma > 0]
            .groupby(["tree_size", "pair_type"])
            .size()
            .unstack(fill_value=0)
        )
        self.assertTrue(
            ((deadline_gene_margins.max(axis=1) - deadline_gene_margins.min(axis=1)) <= 1).all()
        )
        self.assertTrue(
            ((deadline_pair_margins.max(axis=1) - deadline_pair_margins.min(axis=1)) <= 1).all()
        )
        self.assertTrue(set(deadline.simulation_id).issubset(set(selected.simulation_id)))


class TripletTests(unittest.TestCase):
    def test_triplet_topology_and_length(self):
        tree = parse_newick("((A:1,B:1):0.5,C:1.5);")
        observation = extract_triplet(tree, ("A", "B", "C"))
        self.assertEqual(observation.topology_index, 0)
        self.assertEqual(observation.topology_label, "AB|C")
        self.assertAlmostEqual(observation.internal_branch_length, 0.5)

    def test_indexed_extraction_matches_reference_path(self):
        tree = parse_newick("(((A:1,B:1):0.2,(C:1,D:1):0.2):0.3,(E:1.2,F:1.2):0.3);")
        index = TripletTreeIndex(tree)
        for combo in triplet_combinations(list("ABCDEF")):
            reference = extract_triplet(tree, combo)
            indexed = index.extract(combo)
            self.assertEqual(indexed.topology_index, reference.topology_index)
            self.assertAlmostEqual(indexed.internal_branch_length, reference.internal_branch_length)

    def test_feature_summary(self):
        manifest = make_manifest(ROOT / "configs/pilot.yaml").iloc[0].copy()
        manifest["gene_tree_count"] = 3
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "three.tre"
            path.write_text((manifest.backbone_newick + "\n") * 3, encoding="utf-8")
            features = summarize_gene_trees(path, manifest)
        self.assertEqual(len(features), 20)
        abc = features.loc[features.triplet_id == "A,B,C"].iloc[0]
        self.assertEqual(abc[f"p{int(abc.label)}"], 1.0)
        self.assertEqual(abc[f"n{int(abc.label)}"], 3)

    def test_deterministic_triplet_subsample_covers_taxa(self):
        manifest = make_manifest(ROOT / "configs/pilot.yaml").iloc[0].copy()
        manifest["gene_tree_count"] = 1
        manifest["triplets_per_simulation"] = 5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.tre"
            path.write_text(manifest.backbone_newick + "\n", encoding="utf-8")
            features = summarize_gene_trees(path, manifest)
        self.assertEqual(len(features), 5)
        observed = set(features[["taxon_a", "taxon_b", "taxon_c"]].to_numpy().ravel())
        self.assertEqual(observed, set("ABCDEF"))

    def test_feature_shard_is_verified_before_raw_tree_deletion(self):
        manifest = make_manifest(ROOT / "configs/pilot.yaml").iloc[[0]].copy()
        manifest["gene_tree_count"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            trees_dir = root / "trees"
            shards_dir = root / "shards"
            trees_dir.mkdir()
            manifest.to_csv(manifest_path, index=False)
            tree_path = trees_dir / f"{manifest.iloc[0].simulation_id}.tre"
            tree_path.write_text(manifest.iloc[0].backbone_newick + "\n", encoding="utf-8")
            summary = extract_feature_shards(
                manifest_path, trees_dir, shards_dir, delete_tree_files=True
            )
            shard = shards_dir / f"{manifest.iloc[0].simulation_id}.csv.gz"
            extracted = pd.read_csv(shard)
            raw_tree_was_deleted = not tree_path.exists()
        self.assertEqual(summary["completed_simulations"], 1)
        self.assertEqual(summary["deleted_tree_files"], 1)
        self.assertTrue(raw_tree_was_deleted)
        self.assertEqual(len(extracted), 20)


class ModelTests(unittest.TestCase):
    def test_shape_balanced_backbone_folds(self):
        frame = pd.DataFrame(
            [
                {"backbone_id": f"b{shape}_{index}", "backbone_shape_class": f"s{shape}"}
                for shape in range(6)
                for index in range(10)
            ]
        )
        fold_ids, assignments = balanced_group_folds(
            frame, "backbone_id", 5, 123, "backbone_shape_class"
        )
        self.assertEqual(set(fold_ids), {1, 2, 3, 4, 5})
        table = assignments.groupby(["fold", "stratum"]).size().unstack(fill_value=0)
        self.assertTrue((table.to_numpy() == 2).all())

    def test_temperature_scaling_is_normalized(self):
        probabilities = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.2, 0.2]])
        labels = np.array([0, 1, 2])
        temperature = fit_temperature(labels, probabilities)
        calibrated = apply_temperature(probabilities, temperature)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
        self.assertGreater(temperature, 0.0)

    def test_sharded_backbone_cv_and_parallel_assembly(self):
        manifest = make_manifest(ROOT / "configs/pilot.yaml").drop_duplicates(
            "backbone_id"
        )
        manifest = manifest.copy()
        manifest["gene_tree_count"] = 3
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.csv"
            trees_dir = root / "trees"
            feature_dir = root / "features"
            model_dir = root / "model"
            assembled_path = root / "assembled.csv"
            trees_dir.mkdir()
            manifest.to_csv(manifest_path, index=False)
            for row in manifest.itertuples(index=False):
                (trees_dir / f"{row.simulation_id}.tre").write_text(
                    (row.backbone_newick + "\n") * 3,
                    encoding="utf-8",
                )

            extraction = extract_feature_shards(
                manifest_path, trees_dir, feature_dir, jobs=2
            )
            metrics = train_sharded_grouped_cv(
                manifest_path,
                feature_dir,
                model_dir,
                folds=3,
                inner_folds=2,
                training_rows_per_simulation=5,
                final_calibration_rows_per_simulation=2,
            )
            assembly = assemble_feature_shards(
                manifest_path,
                feature_dir,
                assembled_path,
                prediction_dir=model_dir / "oof_predictions",
                jobs=2,
            )
            assignments = pd.read_csv(model_dir / "fold_assignments.csv")
            assembled = pd.read_csv(assembled_path)

        self.assertEqual(extraction["completed_simulations"], len(manifest))
        self.assertEqual(metrics["simulation_count"], len(manifest))
        self.assertEqual(metrics["cv_group_count"], manifest.backbone_id.nunique())
        self.assertEqual(assignments.groupby("backbone_id").fold.nunique().max(), 1)
        self.assertEqual(assembly["failed_simulations"], 0)
        self.assertEqual(assembly["jobs"], 2)
        self.assertEqual(set(assembled.simulation_id), set(manifest.simulation_id))


class EvaluationTests(unittest.TestCase):
    def test_taxon_mismatch_is_reported(self):
        frame = pd.DataFrame(
            [
                {
                    "simulation_id": "s1",
                    "backbone_newick": "((A,B),C);",
                    "inferred_newick": "((A,B),D);",
                    "gamma": 0.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            assembled = Path(tmp) / "assembled.csv"
            output = Path(tmp) / "evaluation"
            frame.to_csv(assembled, index=False)
            summary = evaluate_assembled(assembled, output)
            results = pd.read_csv(output / "per_simulation.csv")
        self.assertEqual(summary["taxon_mismatches"], 1)
        self.assertEqual(summary["failed_simulations"], 1)
        self.assertEqual(results.loc[0, "evaluation_status"], "taxon_mismatch")


class AssemblyTests(unittest.TestCase):
    def _true_weighted_topologies(self):
        true_newick = "(((A:1,B:1):0.1,(C:1,D:1):0.1):0.1,(E:1.1,F:1.1):0.1);"
        tree = parse_newick(true_newick)
        labels = true_triplet_labels(tree)
        topologies = []
        for combo in triplet_combinations(list("ABCDEF")):
            probabilities = [0.0, 0.0, 0.0]
            probabilities[labels[",".join(combo)]] = 1.0
            topologies.extend(probability_topologies(combo, probabilities))
        return true_newick, topologies

    def test_exact_assembly_recovers_true_tree(self):
        true_newick, topologies = self._true_weighted_topologies()
        inferred = assemble_tree(set("ABCDEF"), topologies, method="exact").to_newick()
        self.assertTrue(rooted_rf(true_newick, inferred)[2])

    def test_fm_assembly_recovers_true_tree(self):
        true_newick, topologies = self._true_weighted_topologies()
        inferred = assemble_tree(set("ABCDEF"), topologies, method="fm").to_newick()
        self.assertTrue(rooted_rf(true_newick, inferred)[2])

    def test_both_assemblers_recover_all_945_backbones_from_true_triplets(self):
        for topology in enumerate_rooted_binary_topologies(tuple("ABCDEF")):
            true_newick = render_backbone(backbone_from_topology(topology, 0.1))
            labels = true_triplet_labels(parse_newick(true_newick))
            weighted = []
            for combo in triplet_combinations(list("ABCDEF")):
                probabilities = [0.0, 0.0, 0.0]
                probabilities[labels[",".join(combo)]] = 1.0
                weighted.extend(probability_topologies(combo, probabilities))
            for method in ("exact", "fm"):
                inferred = assemble_tree(set("ABCDEF"), weighted, method=method).to_newick()
                self.assertTrue(rooted_rf(true_newick, inferred)[2], (topology, method))

    def test_ml_weight_modes(self):
        features = pd.DataFrame([{"simulation_id": "s", "triplet_id": "A,B,C"}])
        predictions = pd.DataFrame(
            [{"simulation_id": "s", "triplet_id": "A,B,C", "prob0": 0.2, "prob1": 0.7, "prob2": 0.1}]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probabilities.csv"
            predictions.to_csv(path, index=False)
            common = {
                "probabilities": path,
                "model": None,
                "probability_source": "calibrated",
            }
            soft = _probability_matrix(features, SimpleNamespace(weight_mode="ml-soft", **common))
            hard = _probability_matrix(features, SimpleNamespace(weight_mode="ml-hard", **common))
            confidence = _probability_matrix(
                features, SimpleNamespace(weight_mode="ml-confidence", **common)
            )
        np.testing.assert_allclose(soft, [[0.2, 0.7, 0.1]])
        np.testing.assert_allclose(hard, [[0.0, 1.0, 0.0]])
        np.testing.assert_allclose(confidence, [[0.0, 0.7, 0.0]])


if __name__ == "__main__":
    unittest.main()
