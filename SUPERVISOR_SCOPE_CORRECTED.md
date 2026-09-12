# Supervisor simulation scope with the corrected evaluation protocol

This protocol combines the supervisor's broad simulation domain with the
leakage-safe, rooted evaluation used by the corrected journal extension. It is
implemented by `configs/supervisor_scope_corrected.yaml` and
`scripts/run_supervisor_scope.py`.

## 1. Scope represented by the design

| Design factor | Values |
|---|---|
| Taxon count | Every integer from 4 through 180 |
| Backbones | 10 seeded, unique rooted binary backbones per taxon count |
| Backbone balance | Deterministic sampling across four Colless-imbalance strata |
| ILS internal length | 0.02, 0.10, 0.50 coalescent units |
| Positive introgression strength | 0.25, 0.50 |
| Positive pair class | close, intermediate, distant |
| Gene-tree count | 500, 1000, 2000, 2500 |
| Replication | 1 per positive cell, 3 independent null replicates per null cell |
| Triplets | Exhaustive for small trees, deterministic taxon-covering sample capped at 1000 for large trees |

There are 177 taxon counts and 1,770 backbones. Each backbone has 108 full-grid
runs:

- 36 null runs: 3 null replicates x 3 ILS levels x 4 gene-tree counts;
- 72 positive runs: 2 gamma levels x 3 pair classes x 3 ILS levels x 4
  gene-tree counts.

The complete candidate grid therefore contains 191,160 simulations and
286,740,000 gene trees. With the 1,000-triplet cap, it contains at most
179,111,520 simulation-triplet feature rows.

### Correction to the gamma-zero cells

At gamma = 0 there is no biological donor, recipient, direction, or pair
distance. The supervisor count effectively contains three slots at gamma = 0.
The corrected design preserves the count by treating those slots as three
independently seeded null replicates. Their `donor`, `recipient`, `direction`,
and `pair_type` fields are all `none`. It does not attach false introgression
labels to a null network.

## 2. Three execution modes

| Mode | Simulations | Gene trees | Maximum feature rows | Intended machine |
|---|---:|---:|---:|---|
| `deadline` | 5,310 | 7,976,000 | 4,975,320 | Emergency 36-hour MacBook profile |
| `compact` | 21,240 | 31,860,000 | 19,901,280 | The 11 GB bounded workflow |
| `full` | 191,160 | 286,740,000 | 179,111,520 | Server or cluster |

Deadline mode selects three cells per backbone before any outcomes are
simulated. Every backbone contributes one observation at each gamma level and
one at each ILS level. Within every taxon count, the four gene-tree counts and
three positive pair classes are balanced to differ by at most one. All 177
taxon counts and all 1,770 backbones remain represented. The selected cells
are a strict subset of the compact manifest, so verified feature shards from
an interrupted compact run can be reused.

Compact mode selects 12 cells before any outcomes are simulated. For every
backbone it includes four observations at each gamma level, four at each ILS
level, three at each gene-tree count, and a 3/3/2 rotation across the three
positive pair classes. Every taxon count and every requested factor level is
represented.

Deadline and compact modes are deterministic, factor-balanced subsets, not the
full Cartesian experiment. A paper must report the mode and actual simulation
count, and must not claim that all 191,160 cells were executed. Use
`--execution-mode full` only when the complete Cartesian execution is required
and server-scale resources are available.

## 3. Corrected evaluation protocol

The following rules are enforced for all execution modes:

1. The target is the true rooted topology for each sampled taxon triplet.
2. Model predictors are observable summaries of estimated gene trees only.
   Gamma, ILS, pair class, donor, recipient, direction, backbone ID, and the
   true tree are never classifier inputs.
3. The same `HistGradientBoostingClassifier` configuration is used for the
   main model and every feature ablation. Only the selected feature columns
   change.
4. Complete backbone IDs are held out. A backbone and all of its scenarios
   appear in exactly one outer fold.
5. Outer folds are stratified by taxon count. With 10 backbones per size and
   five folds, each fold receives two unseen backbones at every taxon count.
6. Topology-coordinate augmentation is applied only to outer-fold training
   data.
7. Temperature calibration is fitted from grouped inner out-of-fold
   predictions. Test-fold labels are never used for fold calibration.
8. Every reported proposed-method tree is assembled from outer-fold
   probabilities, not probabilities from a model fitted on that backbone.
9. Complete trees are evaluated with rooted exact match and normalized rooted
   Robinson-Foulds distance. Taxon mismatches, unresolved trees, and failures
   are recorded explicitly.
10. Baselines use the identical simulation IDs. Comparisons are paired, the
    bootstrap is clustered by backbone, the Wilcoxon unit is a backbone-level
    paired mean, and Holm correction covers the reported comparison family.

The primary method is `Enhanced17-ML-FM`. The five assembly controls are
calibrated soft ML weights, hard ML labels, ML confidence weights, raw triplet
frequencies, and raw majority triplets. The ablations are `base13`,
`frequency_only`, `branch_only`, `frequency_branch`, and
`frequency_asymmetry`.

## 4. Code changes that implement the combined design

| File | Modification |
|---|---|
| `src/introgression_triplets/simulation.py` | Multi-size backbones, corrected gamma-zero replication, exact full-grid manifest, and deterministic deadline and compact selection |
| `src/introgression_triplets/features.py` | Taxon-covering triplet sampling, vectorized extraction, compressed per-simulation shards, verification before raw-tree deletion, and parallel workers |
| `src/introgression_triplets/triplets.py` | Batched rooted-triplet indexing for large trees |
| `src/introgression_triplets/scalable.py` | Bounded-memory backbone-grouped CV, nested calibration, full held-out predictions, resumable sharded FM assembly, and parallel assembly workers |
| `src/introgression_triplets/evaluate.py` | Rooted metrics, explicit failures, and taxon-size and simulation-factor strata |
| `scripts/run_supervisor_scope.py` | Resumable stages, deadline, compact, and full modes, external methods in each raw-tree batch, disk-budget checks, and output pruning |
| `scripts/compare_rooted_benchmarks.py` | Paired estimates, backbone-clustered bootstrap intervals, Wilcoxon tests, and Holm adjustment |
| `tests/test_core.py` | Scope-count, null-label, fold-isolation, shard-integrity, and end-to-end sharded workflow tests |

The existing six-taxon configurations and outputs use different names and are
not overwritten.

## 5. Install and validate

Run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
julia julia/setup.jl
PYTHONPATH=src python -m unittest discover -s tests -v
```

All tests must report `OK` before the large run.

## 6. Emergency 36-hour workflow

Use deadline mode if compact preprocessing has not already completed. It is
the largest profile recommended for a 36-hour MacBook deadline. Runtime cannot
be guaranteed because Julia simulation and large-tree parsing depend strongly
on CPU, thermal throttling, and disk speed. A practical planning range is:

| Stage | Planning range |
|---|---:|
| Manifest and deterministic selection | 2 to 5 minutes |
| Simulate and extract 5,310 feature shards | 8 to 24 hours |
| Grouped training and full held-out prediction | 1 to 3 hours |
| Five assemblies, evaluation, and comparison | 2 to 6 hours |
| Total | about 11 to 33 hours |

First create and audit the deadline manifest:

```bash
python3 scripts/run_supervisor_scope.py \
  --config configs/supervisor_scope_corrected.yaml \
  --execution-mode deadline \
  --stop-after select
```

The printed analysis audit must show:

```text
analysis_runs: 5310
backbones: 1770
taxon_sizes: 177
minimum_taxa: 4
maximum_taxa: 180
gene_trees: 7976000
maximum_triplet_rows: 4975320
```

Then run internal preprocessing without external programs:

```bash
caffeinate -i python3 scripts/run_supervisor_scope.py \
  --config configs/supervisor_scope_corrected.yaml \
  --execution-mode deadline \
  --start-at preprocess \
  --stop-after preprocess \
  --batch-size 5 \
  --feature-jobs 2 \
  --external-jobs 2 \
  --disk-budget-gb 11 \
  --minimum-free-gb 2
```

Deadline and compact modes share their deterministic feature-shard directory.
Therefore, this command skips any compatible verified shards that an earlier
compact preprocessing run already completed. Raw tree files are removed only
after each compressed shard passes a read-back check. Do not add
`--keep-raw-trees` under the 11 GB constraint.

When preprocessing finishes, run the corrected evaluation:

```bash
caffeinate -i python3 scripts/run_supervisor_scope.py \
  --config configs/supervisor_scope_corrected.yaml \
  --execution-mode deadline \
  --start-at train \
  --stop-after compare \
  --feature-jobs 2 \
  --disk-budget-gb 11 \
  --minimum-free-gb 2
```

For the deadline submission, do not run multi-taxon external programs or the
five ablations unless the main comparison finishes with substantial time left.
Previously completed six-taxon external and ablation results may be reported
as a separate experiment, but must not be merged into the 4-to-180-taxon
paired comparison.

If compact preprocessing is already complete and only its train-to-compare
command is running, allow that command to finish instead of restarting in
deadline mode. The expensive simulation and feature extraction have already
been paid in that case.

## 7. Compact 11 GB workflow

Create and audit the compact design with:

```bash
python3 scripts/run_supervisor_scope.py \
  --config configs/supervisor_scope_corrected.yaml \
  --execution-mode compact \
  --stop-after select
```

Its analysis audit must report 21,240 simulations, 31,860,000 gene trees, and
at most 19,901,280 feature rows. Run bounded preprocessing and the remaining
stages using the same commands in Section 6 with `compact` substituted for
`deadline`. External executable options may be supplied during preprocessing
when there is enough time. Repeat those options at the train-to-compare stage
so the paired comparison includes the generated external outputs.

Run ablations only after the main comparison:

```bash
caffeinate -i python3 scripts/run_supervisor_scope.py \
  --config configs/supervisor_scope_corrected.yaml \
  --execution-mode compact \
  --start-at ablations \
  --stop-after ablations \
  --feature-jobs 2 \
  --disk-budget-gb 11 \
  --minimum-free-gb 2
```

## 8. Monitor storage and completion

In another terminal, use:

```bash
du -sh \
  data/manifests/supervisor_scope_corrected* \
  data/features/supervisor_scope_corrected_shards \
  data/gene_trees/supervisor_scope_corrected_batch \
  outputs/supervisor_scope/supervisor_scope_corrected*

df -h .
```

The runner is resumable. Repeating a preprocessing command skips verified
feature shards and valid external trees. After successful preprocessing, the
raw batch directory should be empty or contain only files from an interrupted
batch.

## 9. Full Cartesian execution

Use this only on a server or cluster:

```bash
python3 scripts/run_supervisor_scope.py \
  --config configs/supervisor_scope_corrected.yaml \
  --execution-mode full \
  --start-at manifest \
  --stop-after compare \
  --batch-size 25 \
  --feature-jobs 8 \
  --external-jobs 8 \
  --disk-budget-gb 200 \
  --minimum-free-gb 20
```

Supply the external executable arguments when those methods are required. The
full mode is supported by the code, but its 191,160 simulations and nearly 287
million gene trees are not a realistic 11 GB MacBook workload.

## 10. Claims allowed in the paper

For deadline mode, state that the study evaluates a deterministic reduced
design of 5,310 simulations spanning every taxon count from 4 through 180 and
all 1,770 seeded backbones. State that each backbone contributes one cell at
each gamma and ILS level, while gene-tree-count and positive-pair margins are
balanced within taxon size. Do not describe deadline mode as the full
Cartesian design.

For compact mode, state that the study evaluates a deterministic,
factor-balanced sample of 21,240 simulations spanning all taxon counts from 4
through 180 and all requested factor levels. State the 1,000-triplet analysis
cap and the 20-row-per-simulation classifier fit sample. Report full held-out
prediction and complete-tree evaluation counts separately.

Only full mode permits the statement that all 191,160 Cartesian candidate
simulations were executed. In every mode, do not claim donor-recipient
inference: the current supervised target is rooted backbone-triplet topology.
