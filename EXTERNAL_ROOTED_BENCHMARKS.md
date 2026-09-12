# Rooted external benchmark protocol

This protocol evaluates STELAR, MP-EST 3.0, and Triplet MaxCut against the
Enhanced17 ML-FM estimator on the same 30,100 simulated rooted gene-tree data
sets. The generating backbone is used only during evaluation, never during
inference.

## Scientific settings

- STELAR is run in exact mode (`-xt`). With six taxa this searches only 63
  possible clusters and avoids an unnecessary heuristic approximation.
- MP-EST is run from five deterministic random rooted starting trees. The best
  pseudo-likelihood candidate is retained. This avoids giving MP-EST the true
  outgroup or the true backbone. Its required Nexus `outgroup` field is only a
  syntactic placeholder when a user starting tree is supplied.
- Triplet MaxCut receives the original rooted gene trees, decomposes them into
  rooted triplets, and uses their observed multiplicities as support. It does
  not receive Enhanced17 probabilities.
- Primary metrics are rooted exact-match accuracy and normalized rooted RF.
- All methods must have 30,100 valid outputs and zero taxon mismatches before
  final statistical comparison.

Official software:

- STELAR: <https://github.com/islamazhar/STELAR>
- MP-EST: <https://github.com/lliu1871/mp-est>
- Triplet MaxCut archive: <https://doi.org/10.5061/dryad.qt26f>

## Install STELAR and MP-EST

Run from the repository root:

```bash
mkdir -p tools/external

git clone --depth 1 \
  https://github.com/islamazhar/STELAR.git \
  tools/external/STELAR

git clone --depth 1 \
  https://github.com/lliu1871/mp-est.git \
  tools/external/mp-est

make -C tools/external/mp-est/src
```

Confirm both tools:

```bash
java -jar tools/external/STELAR/STELAR.jar 2>&1 | head -20
tools/external/mp-est/src/mpest -h | head -30
```

## Install Triplet MaxCut

Download `TMC.tar.gz` from the official Dryad page above, save it under
`tools/external`, then run:

```bash
mkdir -p tools/external/tmc
tar -xzf tools/external/TMC.tar.gz -C tools/external/tmc
chmod +x tools/external/tmc/treeFromTriplets
file tools/external/tmc/treeFromTriplets
```

The published executable is an x86-64 Linux ELF binary. On an arm64 Mac, run it
through Docker as shown below. Do not try to execute it directly in macOS.

## Pilot benchmark

Use the same 80-row pilot manifest already used for the ASTRAL smoke benchmark:

```bash
PILOT=outputs/full_corrected/external_benchmarks/astral/pilot/pilot_manifest.csv
```

Run STELAR exact:

```bash
python scripts/run_external_rooted.py \
  --method stelar \
  --manifest "$PILOT" \
  --trees-dir data/gene_trees/full_corrected \
  --out-dir outputs/full_corrected/external_benchmarks/stelar/pilot \
  --executable tools/external/STELAR/STELAR.jar \
  --jobs 4 \
  --timeout 300
```

Run MP-EST with five independent starts:

```bash
python scripts/run_external_rooted.py \
  --method mpest \
  --manifest "$PILOT" \
  --trees-dir data/gene_trees/full_corrected \
  --out-dir outputs/full_corrected/external_benchmarks/mpest/pilot \
  --executable tools/external/mp-est/src/mpest \
  --mpest-starts 5 \
  --jobs 4 \
  --timeout 300
```

Run Triplet MaxCut in an amd64 Linux container:

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD:/work" \
  -w /work \
  python:3.12-slim \
  python scripts/run_external_rooted.py \
    --method tmc \
    --manifest "$PILOT" \
    --trees-dir data/gene_trees/full_corrected \
    --out-dir outputs/full_corrected/external_benchmarks/tmc/pilot \
    --executable tools/external/tmc/treeFromTriplets \
    --jobs 4 \
    --timeout 300
```

Evaluate the pilot outputs:

```bash
python scripts/evaluate_external_rooted.py \
  --manifest "$PILOT" \
  --trees-dir outputs/full_corrected/external_benchmarks/stelar/pilot \
  --method STELAR-exact \
  --out-dir outputs/full_corrected/external_benchmarks/stelar/pilot_evaluation

python scripts/evaluate_external_rooted.py \
  --manifest "$PILOT" \
  --trees-dir outputs/full_corrected/external_benchmarks/mpest/pilot \
  --method MP-EST-v3 \
  --out-dir outputs/full_corrected/external_benchmarks/mpest/pilot_evaluation

python scripts/evaluate_external_rooted.py \
  --manifest "$PILOT" \
  --trees-dir outputs/full_corrected/external_benchmarks/tmc/pilot \
  --method Triplet-MaxCut \
  --out-dir outputs/full_corrected/external_benchmarks/tmc/pilot_evaluation
```

Each pilot summary must report 80 evaluated simulations, zero failures, and
zero taxon mismatches. Inspect MP-EST `run_records.csv`; if the five starts
often disagree, retain all five starts for the full run.

## Full benchmark

Run methods sequentially so they do not compete for CPU:

```bash
python scripts/run_external_rooted.py \
  --method stelar \
  --manifest data/manifests/full_corrected.csv \
  --trees-dir data/gene_trees/full_corrected \
  --out-dir outputs/full_corrected/external_benchmarks/stelar/full \
  --executable tools/external/STELAR/STELAR.jar \
  --jobs 4 \
  --timeout 300

python scripts/run_external_rooted.py \
  --method mpest \
  --manifest data/manifests/full_corrected.csv \
  --trees-dir data/gene_trees/full_corrected \
  --out-dir outputs/full_corrected/external_benchmarks/mpest/full \
  --executable tools/external/mp-est/src/mpest \
  --mpest-starts 5 \
  --jobs 4 \
  --timeout 300

docker run --rm --platform linux/amd64 \
  -v "$PWD:/work" \
  -w /work \
  python:3.12-slim \
  python scripts/run_external_rooted.py \
    --method tmc \
    --manifest data/manifests/full_corrected.csv \
    --trees-dir data/gene_trees/full_corrected \
    --out-dir outputs/full_corrected/external_benchmarks/tmc/full \
    --executable tools/external/tmc/treeFromTriplets \
    --jobs 4 \
    --timeout 300
```

All three commands are resumable. Re-running skips valid existing trees.

Evaluate by repeating the three pilot evaluation commands with `pilot` changed
to `full` and the manifest changed to `data/manifests/full_corrected.csv`.

## Final paired comparison

After all rooted evaluators report 30,100 valid simulations, run one comparison
family so Holm correction covers every rooted baseline together:

```bash
python scripts/compare_rooted_benchmarks.py \
  --reference "Enhanced17 ML-FM=outputs/full_corrected/evaluation_ml_fm_enhanced17/per_simulation.csv" \
  --baseline "Majority-FM=outputs/full_corrected/evaluation_majority_fm/per_simulation.csv" \
  --baseline "Frequency-FM=outputs/full_corrected/evaluation_frequency_fm/per_simulation.csv" \
  --baseline "STELAR-exact=outputs/full_corrected/external_benchmarks/stelar/full_evaluation/per_simulation.csv" \
  --baseline "MP-EST-v3=outputs/full_corrected/external_benchmarks/mpest/full_evaluation/per_simulation.csv" \
  --baseline "Triplet-MaxCut=outputs/full_corrected/external_benchmarks/tmc/full_evaluation/per_simulation.csv" \
  --out-dir outputs/full_corrected/rooted_external_comparisons \
  --bootstrap-reps 10000
```

Adjust the three internal evaluation paths if their directory names differ.
ASTRAL is not included here because it outputs an unrooted tree; keep ASTRAL in
the separate common-unrooted comparison.
