# Face Verification System — Milestone 2

This repository evaluates a simple face-verification pipeline on the Labeled Faces in the Wild (LFW) dataset. The Milestone 2 system uses a non-learnable image embedding:

- convert image to grayscale
- resize to `32x32`
- flatten
- L2 normalize
- compare left/right images with cosine similarity

The milestone focuses on:

- deterministic pair generation
- tracked evaluation runs
- validation checks for configs and pair files
- threshold sweep on validation
- one data-centric iteration
- error analysis and reporting

## Milestone 2 Summary

### Baseline

The baseline uses the original pair policy reproduced in `outputs/pairs` and evaluates a fixed-threshold and validation-sweep version of the same raw-pixel cosine verifier.


### Data-Centric Improvement

The data-centric change creates a second pair-set version in `outputs/pairs_v2`.

Implemented change:

- keep training pair construction unchanged
- remove positive self-pairs from validation and test

This makes the evaluation set less artificially easy and gives a clearer picture of how brittle the raw-pixel baseline is.


## Repository Structure

```text
face-verification-system/
├── configs/
├── reports/
│   ├── evaluation_report.md
├── scripts/
│   ├── benchmark_similarity.py
│   ├── evaluator.py
│   ├── generate_pairs.py
│   ├── validate_pipeline.py
│   └── falseneg_falsepos.py  # script to pull out some examples for false neg and false positives for error analysis
├── src/
│   ├── similarity_metrics.py
│   └── validation.py
├── tests/
├── outputs/                # generated, gitignored
│   ├── pairs/              # reproducible baseline pair set
│   ├── pairs_v2/           # reproducible data-centric pair set
│   ├── bench/
│   └── runs/
└── README.md
```

## Environment Setup

Run all commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## How To Run 

### 1. Generate pair sets

Generate the original baseline pair version:

```bash
python scripts/generate_pairs.py --pair-version baseline
```
Generate the data-centric pair version:

```bash
python scripts/generate_pairs.py --pair-version v2

```
Generated files:

Baseline pair version:
- `outputs/pairs/manifest.json`
- `outputs/pairs/train.jsonl`
- `outputs/pairs/val.jsonl`
- `outputs/pairs/test.jsonl`

and Data-centric pair version:
- `outputs/pairs_v2/manifest.json`
- `outputs/pairs_v2/train.jsonl`
- `outputs/pairs_v2/val.jsonl`
- `outputs/pairs_v2/test.jsonl`

### 2. Validate the pipeline inputs

```bash
python scripts/validate_pipeline.py --config configs/after_change_sweep.json
```

This checks:

- config validity
- pair-file schema
- binary labels
- valid split names
- referenced image-path existence
- val/test disjointness

### 3. Run the main reproducible evaluation

Validation sweep plus test evaluation on the current pair set:

```bash
python scripts/evaluator.py --config configs/after_change_sweep.json
```

Then run the fixed-threshold follow-up using the selected validation threshold:

```bash
python scripts/evaluator.py --config configs/after_change_best.json
```

### 4. Run the historical baseline configs

These configs point to `outputs/pairs` and are kept for baseline comparison:

```bash
python scripts/evaluator.py --config configs/baseline.json
python scripts/evaluator.py --config configs/baseline_sweep.json
python scripts/evaluator.py --config configs/baseline_best.json
```

Note:

- the current `generate_pairs.py` reproduces both `pairs` and `pairs_v2` depending on the selected `--pair-version`
- `outputs/pairs` is the baseline pair version
- `outputs/pairs_v2` is the data-centric pair version with self-pairs removed from validation and test

### 5. Run the benchmark

```bash
python scripts/benchmark_similarity.py
```

### 6. Run tests

```bash
pytest -q
```

If you want to run the three Milestone 2 reliability pieces separately:

Unit tests:

```bash
pytest -q tests/test_similarity_metrics.py tests/test_validation.py tests/test_evaluator.py
```

Small integration test only:

```bash
pytest -q tests/test_evaluator.py -k integration
```

Pipeline validation checks only:

```bash
python scripts/validate_pipeline.py --config configs/after_change_sweep.json
python scripts/validate_pipeline.py --config configs/baseline.json
```

## Main Artifacts

### Report

- `reports/evaluation_report.md`


### Important Run Outputs

Tracked runs:

- `baseline`
  - files:
    - `outputs/runs/baseline/baseline_summary.json`
    - `outputs/runs/baseline/baseline_val_scores.jsonl`
    - `outputs/runs/baseline/baseline_test_scores.jsonl`
  - purpose:
    - fixed-threshold baseline on the original pair set using threshold `0.9`

- `baseline_sweep`
  - files:
    - `outputs/runs/baseline_sweep/baseline_sweep_summary.json`
    - `outputs/runs/baseline_sweep/baseline_sweep_val_scores.jsonl`
    - `outputs/runs/baseline_sweep/baseline_sweep_val_threshold_sweep.jsonl`
    - `outputs/runs/baseline_sweep/baseline_sweep_val_roc.png`
    - `outputs/runs/baseline_sweep/baseline_sweep_test_scores.jsonl`
  - purpose:
    - sweeps thresholds on validation, selects the best threshold by validation F1, and applies that threshold to test on the original pair set

- `baseline_best`
  - files:
    - `outputs/runs/baseline_best/baseline_best_summary.json`
    - `outputs/runs/baseline_best/baseline_best_val_scores.jsonl`
    - `outputs/runs/baseline_best/baseline_best_test_scores.jsonl`
  - purpose:
    - fixed-threshold rerun on the original pair set using the selected threshold from `baseline_sweep`

- `after_change_sweep`
  - files:
    - `outputs/runs/after_change_sweep/after_change_sweep_summary.json`
    - `outputs/runs/after_change_sweep/after_change_sweep_val_scores.jsonl`
    - `outputs/runs/after_change_sweep/after_change_sweep_val_threshold_sweep.jsonl`
    - `outputs/runs/after_change_sweep/after_change_sweep_val_roc.png`
    - `outputs/runs/after_change_sweep/after_change_sweep_test_scores.jsonl`
  - purpose:
    - threshold sweep on the data-centric pair set `outputs/pairs_v2`, where positive self-pairs were removed from validation and test

- `after_change_best`
  - files:
    - `outputs/runs/after_change_best/after_change_best_summary.json`
    - `outputs/runs/after_change_best/after_change_best_val_scores.jsonl`
    - `outputs/runs/after_change_best/after_change_best_test_scores.jsonl`
  - purpose:
    - fixed-threshold rerun on `outputs/pairs_v2` using the selected threshold from `after_change_sweep`

### Selected Thresholds

Original pair set:

- selected on validation: `0.9266837672022352`

Current data-centric pair set:

- selected on validation: `0.5853148971300512`

## Reproducing The Main Reported Result

The main reported result in the report is the stricter data-centric evaluation on `outputs/pairs_v2`.

Generate the data-centric pair set and run the evaluation:

```bash
python scripts/generate_pairs.py --pair-version v2
python scripts/validate_pipeline.py --config configs/after_change_sweep.json
python scripts/evaluator.py --config configs/after_change_sweep.json
python scripts/evaluator.py --config configs/after_change_best.json
```

For baseline comparison, the original pair version is also reproducible with:
```bash
python scripts/generate_pairs.py --pair-version baseline
python scripts/evaluator.py --config configs/baseline_sweep.json
python scripts/evaluator.py --config configs/baseline_best.json
```

The resulting main artifacts will be:

- `outputs/pairs_v2/manifest.json`
- `outputs/runs/after_change_sweep/after_change_sweep_summary.json`
- `outputs/runs/after_change_sweep/after_change_sweep_val_roc.png`
- `outputs/runs/after_change_best/after_change_best_summary.json`

## Notes On Threshold Reproducibility

- fixed-threshold runs read the threshold directly from config
- sweep runs choose the threshold on validation by:
  - maximizing validation F1
  - breaking ties with higher accuracy
- the selected threshold is written into the summary JSON under `threshold_information`

## Clean-Clone Reproducibility Note

Before tagging the milestone, the intended clean-clone check is:

1. start from a fresh clone
2. follow the setup commands above exactly
3. generate both pair versions
    ```bash
    python scripts/generate_pairs.py --pair-version baseline
    python scripts/generate_pairs.py --pair-version v2
   ```
4. Validate inputs by running:
    ```bash   
    python scripts/validate_pipeline.py --config configs/after_change_sweep.json
    python scripts/validate_pipeline.py --config configs/baseline.json
   ```
5. run evaluation configs 
    ```bash
    python scripts/evaluator.py --config configs/after_change_sweep.json
    python scripts/evaluator.py --config configs/after_change_best.json
    python scripts/evaluator.py --config configs/baseline_sweep.json
    python scripts/evaluator.py --config configs/baseline_best.json
    python scripts/evaluator.py --config configs/baseline.json
    ```
6. run tests using `pytest -q`
7. confirm expected artifacts exist under:
    ```bash
    outputs/pairs/
    outputs/pairs_v2/
    outputs/runs/
    ```
Both `outputs/pairs` and `outputs/pairs_v2` are reproducible from the current generator by choosing the appropriate `--pair-version`.