# Journal extension execution protocol

Run every command from the repository root. Install the Python package and Julia environment first:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
julia julia/setup.jl
```

## Implemented experimental blocks

| Block | Taxa | Backbones | Runs | Gene trees | Purpose |
|---|---:|---:|---:|---:|---|
| Existing corrected experiment | 6 | 60 | 30,100 | 13,921,250 | Dense six-taxon benchmark |
| All six-taxon topologies | 6 | 945 | 34,020 | 18,144,000 | Complete topology coverage |
| Scaling experiment | 10 | 40 | 10,080 | 4,662,000 | Larger-tree evaluation |
| Scaling experiment | 20 | 40 | 10,080 | 4,662,000 | Larger-tree evaluation |
| Scaling experiment | 50 | 40 | 10,080 | 4,662,000 | Larger-tree evaluation with 1,000 sampled triplets per run |
| Total | | | 94,360 | 46,051,250 | Existing and new principal experiments |

The 10, 20, and 50 taxon profiles use four Colless-imbalance strata, with 10 backbones per stratum. Each positive-introgression setting selects one close, one intermediate, and one distant tip pair. Pair orientation is deterministic from the backbone identifier and seed.

## 1. Run the smoke test

The smoke profile contains 24 simulations and 480 gene trees. Run it before submitting large jobs:

```bash
python scripts/run_journal_extension.py \
  --config configs/journal_smoke.yaml \
  --stop-after compare
```

The runner performs these stages in order: manifest generation, coalescent simulation, feature extraction, nested backbone-grouped training, five assembly variants, rooted-tree evaluation, and paired comparison.

## 2. Regenerate the existing model with leakage-safe validation

Use the existing 30,100-run feature table. This command does not resimulate gene trees:

```bash
python scripts/run_journal_extension.py \
  --config configs/full_corrected.yaml \
  --name full_corrected \
  --start-at train \
  --stop-after ablations
```

The primary model uses the observable `enhanced` feature set, which corresponds to Enhanced17-ML-FM. None of its 17 predictors contains gamma, ILS, donor, recipient, backbone identity, or the true topology. The outer folds hold out complete backbone identifiers. Temperature scaling is fitted from inner grouped out-of-fold predictions. The ablation stage trains `base13`, `frequency_only`, `branch_only`, `frequency_branch`, and `frequency_asymmetry` models with the same deterministic outer-fold assignments.

## 3. Generate and inspect each new manifest

```bash
python scripts/run_journal_extension.py --config configs/journal_all_945.yaml --stop-after manifest
python scripts/run_journal_extension.py --config configs/journal_n10.yaml --stop-after manifest
python scripts/run_journal_extension.py --config configs/journal_n20.yaml --stop-after manifest
python scripts/run_journal_extension.py --config configs/journal_n50.yaml --stop-after manifest
```

The runner prints the run count, scenario-group count, backbone count, taxon count, total gene-tree count, and backbone balance distribution. Do not proceed if a printed count differs from the table above.

## 4. Simulate the new gene trees in batches

The full extension should run on a server or cluster. The following example processes rows 1 through 500. Submit nonoverlapping ranges until the final manifest row is complete:

```bash
python scripts/run_journal_extension.py \
  --config configs/journal_all_945.yaml \
  --start-at simulate \
  --stop-after simulate \
  --start-row 1 \
  --end-row 500
```

Repeat the same pattern for `journal_n10.yaml`, `journal_n20.yaml`, and `journal_n50.yaml`. The Julia simulator skips a complete tree file, so interrupted batches are resumable. Do not use `--force` unless the relevant outputs must be regenerated.

## 5. Extract, train, assemble, evaluate, and compare

After all gene-tree files for one manifest exist, continue from feature extraction:

```bash
python scripts/run_journal_extension.py \
  --config configs/journal_all_945.yaml \
  --start-at features \
  --stop-after compare
```

Run the same command for the three scaling configurations. Feature extraction writes incrementally to a partial file and renames it only after every simulation succeeds.

The assembly comparison contains:

| Mode | Triplet weights |
|---|---|
| `ml-soft` | All three calibrated class probabilities |
| `ml-hard` | Weight 1 for the most probable topology, 0 otherwise |
| `ml-confidence` | Maximum calibrated probability for the selected topology, 0 otherwise |
| `frequency` | Raw topology frequencies |
| `majority` | Weight 1 for the most frequent topology, 0 otherwise |

The exact split oracle is restricted to at most 12 taxa. Use FM assembly for the 20 and 50 taxon experiments.

## 6. Run external species-tree methods

Supply executable paths only after the internal analysis completes. For the six-taxon experiment:

```bash
python scripts/run_journal_extension.py \
  --config configs/journal_all_945.yaml \
  --start-at external \
  --stop-after compare \
  --stelar-executable tools/external/STELAR/STELAR.jar \
  --mpest-executable tools/external/mp-est/src/mpest \
  --tmc-executable tools/external/tmc/treeFromTriplets \
  --tmc-docker \
  --external-jobs 4 \
  --mpest-starts 5
```

For large taxon counts, add `--stelar-heuristic`. Record this algorithmic change in the methods section and do not label those results as STELAR exact.

## 7. Output locations

For an experiment named `journal_n10`, the main outputs are:

```text
data/manifests/journal_n10.csv
data/gene_trees/journal_n10/
data/features/journal_n10.csv
data/features/journal_n10.extraction.json
outputs/journal_extension/journal_n10/models/enhanced/
outputs/journal_extension/journal_n10/assembled/
outputs/journal_extension/journal_n10/evaluation/
outputs/journal_extension/journal_n10/comparison/
outputs/journal_extension/journal_n10/ablations/
```

Each trained model directory contains `model.joblib`, `oof_predictions.csv`, `fold_assignments.csv`, `fold_metrics.csv`, `calibration_curve.csv`, and `metrics.json`. Each evaluation directory contains `per_simulation.csv`, `stratified_metrics.csv`, and `summary.json`.

## External-data blocks not generated by this runner

Sequence-derived gene-tree error requires a sequence simulator, an alignment model, and a gene-tree estimator. Empirical validation requires selected datasets and taxon mappings. Network-level evaluation requires SNaQ or PhyloNet output conventions. These blocks cannot be executed from the current inputs because the required sequence parameters, empirical datasets, and network truth files are not present. Add them after fixing those inputs in a separate preregistered configuration.
