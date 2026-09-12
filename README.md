# Introgression-aware species-tree inference from rooted gene-tree triplets

This repository implements the six-step framework from Soumik Datta and Anika
Tasnim's BUET bioinformatics presentation:

1. design network-multispecies-coalescent simulation scenarios;
2. simulate rooted gene trees with `PhyloCoalSimulations.jl`;
3. extract all rooted triplets and their internal branch lengths;
4. build topology-frequency, discordance-asymmetry, and branch-length features;
5. learn explicitly evaluated probabilities for the
   three candidate backbone topologies;
6. assemble the probability-weighted triplets into one rooted species tree
   using a weighted FM-style recursive bipartition heuristic.

The code includes a small pilot profile, the corrected 30,100-run experiment,
and the presentation's legacy 33,000-run grid. Start with the pilot. Do not run
the full grid until the pilot outputs and scientific assumptions have been
checked.

The combined supervisor-scope implementation covers taxon counts 4 through
180, preserves the corrected gamma-zero semantics, and uses the leakage-safe
rooted evaluation protocol. See
[SUPERVISOR_SCOPE_CORRECTED.md](SUPERVISOR_SCOPE_CORRECTED.md) for its exact
counts, 36-hour deadline profile, compact 11 GB workflow, full-grid server
mode, and reporting rules.

## Important corrections and assumptions

- In the corrected grid, gamma = 0 is simulated once per ILS/G/replicate
  setting. Repeating it for all 30 donor-recipient directions would create
  2,900 duplicate baseline runs. The corrected total is therefore 30,100.
- The legacy profile preserves the presentation's 33,000-run count for exact
  comparison.
- There are 945 rooted binary labeled topologies for six taxa. The corrected
  experiment deterministically selects 60 backbones, stratified across all six
  unlabeled tree-shape classes, and rotates them across biological scenarios.
  This changes backbone diversity without increasing the 30,100-run budget.
- The pilot uses six backbones: one from each tree-shape class.
- Introgression is placed at a fixed height of 0.25 coalescent units above the
  tips, on terminal donor and recipient branches. This makes the generated
  network time-consistent for the supplied ultrametric backbone.
- The presentation's original 13 features are the default (`base13`). An
  optional `enhanced` feature set adds three candidate-relative asymmetry
  values and log gene-tree count.
- Some labeled backbones may not naturally occupy all three canonical class
  indices. Training therefore applies three cyclic permutations of the
  topology-indexed feature coordinates *inside each training fold*. This
  balances the multiclass task without leaking held-out scenarios.
- The weighted FM assembler is custom rooted-triplet code. For six taxa, an
  exact split oracle is also provided to check whether the heuristic missed a
  better split.
- Cross-validation holds out complete backbone IDs by default. Thus no triplet
  from a test backbone is present while fitting that fold's model.

## Software

- Julia 1.10+ with the packages in `julia/Project.toml`
- Python 3.10+ with the packages in `requirements.txt`

Create the Python environment:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install Julia, then create the simulation environment (the package manager
records the official package UUIDs and versions in the local project):

```bash
julia julia/setup.jl
```

## Pilot: exact command sequence

Run all commands from the repository root.

### 1. Generate the scenario manifest

```bash
python -m introgression_triplets.cli make-manifest \
  --config configs/pilot.yaml \
  --out data/manifests/pilot.csv
```

The pilot contains 36 simulation runs over six structurally different
backbones: four selected directions, two positive gamma values, two ILS levels,
one gene-tree count, two replicates, plus four deduplicated no-introgression
baselines.

### 2. Simulate gene trees

```bash
julia --project=julia julia/simulate_grid.jl \
  --manifest data/manifests/pilot.csv \
  --out-dir data/gene_trees/pilot
```

Each manifest row produces one `.tre` file containing one Newick gene tree per
line. Re-running the command skips complete files unless `--overwrite` is
given.

### 3. Extract the 20 rooted-triplet feature rows per simulation

```bash
python -m introgression_triplets.cli extract-features \
  --manifest data/manifests/pilot.csv \
  --trees-dir data/gene_trees/pilot \
  --out data/features/pilot.csv
```

### 4. Train and validate the triplet classifier

```bash
python -m introgression_triplets.cli train \
  --features data/features/pilot.csv \
  --out-dir outputs/pilot/model \
  --feature-set base13 \
  --folds 3 \
  --cv-mode backbone
```

Cross-validation holds out entire backbone topologies, providing a direct test
of generalization to unseen species-tree structures. The command writes the
final model, out-of-fold probabilities, fold metrics, and aggregate metrics.
Use `--cv-mode scenario` only for the secondary scenario-held-out analysis.

### 5. Assemble full six-taxon trees

```bash
python -m introgression_triplets.cli assemble \
  --features data/features/pilot.csv \
  --probabilities outputs/pilot/model/oof_predictions.csv \
  --out outputs/pilot/assembled_trees.csv \
  --weight-mode ml \
  --method fm
```

Using out-of-fold probabilities keeps this evaluation independent of the rows
used to fit each fold's model. For validation, repeat with `--method exact` and
compare the outputs. Use `--model outputs/pilot/model/model.joblib` only when
predicting genuinely new simulations or empirical data.

### 6. Evaluate the assembled trees

```bash
python -m introgression_triplets.cli evaluate \
  --assembled outputs/pilot/assembled_trees.csv \
  --out-dir outputs/pilot/evaluation
```

The main outputs are rooted-tree exact-match accuracy, normalized rooted
Robinson-Foulds distance, and results stratified by gamma, ILS, G, backbone ID,
and backbone-shape class.

## Full experiment

After the pilot passes, replace `configs/pilot.yaml` with
`configs/full_corrected.yaml` and use new output directories. The full grid
contains 30,100 simulations and 13,921,250 gene trees across the four G values,
so run it as batches on a server or cluster:

```bash
python -m introgression_triplets.cli make-manifest \
  --config configs/full_corrected.yaml \
  --out data/manifests/full_corrected.csv
```

```bash
julia --project=julia julia/simulate_grid.jl \
  --manifest data/manifests/full_corrected.csv \
  --out-dir data/gene_trees/full_corrected \
  --start-row 1 --end-row 500
```

Use non-overlapping row ranges for parallel jobs. Seeds are deterministic from
the manifest, so batches are reproducible.

## Changing the backbone pool

The recommended configuration is:

```yaml
backbone_strategy: stratified
backbone_count: 60
```

`stratified` samples equally from the six possible unlabeled rooted-binary
shape classes. Other supported strategies are `random`, `all`, `fixed`, and
`explicit`. To supply your own trees, use:

```yaml
backbone_strategy: explicit
backbone_topologies:
  - "(((A,B),(C,D)),(E,F))"
  - "((((A,B),C),D),(E,F))"
  - "(((A,C),(B,E)),(D,F))"
```

Backbones are assigned at the scenario level, so all replicates of a biological
scenario use the same backbone. Increasing backbone diversity does not
multiply the configured run count.

## Recommended comparisons for the paper

Report at least these four methods on identical held-out scenarios:

1. raw majority triplet + exact/weighted assembly;
2. raw frequency weights + weighted FM assembly;
3. ML probability weights + weighted FM assembly (proposed);
4. ML probability weights + exact split oracle (six-taxon diagnostic).

Report triplet accuracy, multiclass log loss, multiclass Brier score,
calibration error, complete-tree exact match, and normalized rooted RF distance.
Stratify every result by gamma, ILS branch length, gene-tree count, backbone
shape, and whether introgression is between sister or non-sister taxa. Report
the backbone-held-out analysis as the primary generalization result.

## Scope of the current implementation

The pipeline estimates the backbone topology. It does **not** yet infer the
donor-recipient pair, although that output appears on slide 6. That should be a
separate simulation-level classification task and evaluated independently; it
must not be claimed as an output of the current topology model.

## Journal extension

The journal extension adds complete coverage of all 945 six-taxon rooted binary topologies, separate 10, 20, and 50 taxon experiments, deterministic topology-balance strata, distance-stratified introgression pairs, balanced backbone-grouped folds, nested temperature calibration, component ablations, hard and soft triplet weighting, runtime measurement, and explicit failure accounting.

Read [JOURNAL_EXTENSION.md](JOURNAL_EXTENSION.md), then run the smoke profile:

```bash
python scripts/run_journal_extension.py \
  --config configs/journal_smoke.yaml \
  --stop-after compare
```
# introgression-ILS-Triplet-Pipeline
