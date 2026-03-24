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

The baseline uses the original pair set in `outputs/pairs` generated in Milestone 1 and evaluates a fixed-threshold and validation-sweep version of the same raw-pixel cosine verifier.


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
├── notes/
├── scripts/
│   ├── benchmark_similarity.py
│   ├── evaluator.py
│   ├── generate_pairs.py
│   └── validate_pipeline.py
├── src/
│   ├── similarity_metrics.py
│   └── validation.py
├── tests/
├── outputs/                # generated, gitignored
│   ├── pairs/              # original pair set (historical baseline)
│   ├── pairs_v2/           # current reproducible data-centric pair set
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

### 1. Generate the current pair set

The current generator writes the data-centric version to `outputs/pairs_v2/`.

```bash
python scripts/generate_pairs.py
```

Generated files:

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

- the current `generate_pairs.py` reproduces `pairs_v2`
- `outputs/pairs` is the original pair set generated in Milestone 1(v0.1) used for the tracked baseline comparison

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

- `notes/evaluation_report.md`


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

- selected on validation: `0.9266850288062412`

Current data-centric pair set:

- selected on validation: `0.5853091582168102`

## Reproducing The Main Reported Result

For the current clean-clone workflow, the most reproducible result is the data-centric sweep on `pairs_v2`, because the current generator recreates that dataset version directly.

Use:

```bash
python scripts/generate_pairs.py
python scripts/validate_pipeline.py --config configs/after_change_sweep.json
python scripts/evaluator.py --config configs/after_change_sweep.json
python scripts/evaluator.py --config configs/after_change_best.json
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
3. run `generate_pairs.py`
4. run `validate_pipeline.py`
5. run the `after_change_sweep` and `after_change_best` configs
6. run `pytest -q`
7. confirm the expected files in `outputs/pairs_v2/` and `outputs/runs/after_change_*`

The historical `outputs/pairs` baseline is kept for comparison, but the current generator is centered on reproducing `pairs_v2`.
