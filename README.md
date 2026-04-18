# Face Verification System — Milestone 3

This repository implements a face verification system on the Labeled Faces in the Wild (LFW) dataset.

In Milestone 3, we extend the earlier evaluation pipeline into a more complete inference system using embedding based representations instead of just raw pixel features.

The system now supports:

- ArcFace based face embeddings
- deterministic evaluation using pre generated pairs
- threshold based verification decisions
- CLI based inference for individual pairs or batch inputs
- confidence scoring and latency measurement for each prediction

Earlier milestones focused on building a reproducible pipeline and evaluating it carefully. In this milestone, the focus is more toward making the system usable and easier to run for inference.

## Milestone 3 Summary

Milestone 3 builds on the Milestone 2 pipeline and replaces the raw pixel representation with an embedding based approach.

Main changes in this milestone:

- switched to ArcFace embeddings for face representation
- reselected the decision threshold based on the new score distribution
- added a CLI inference mode for running predictions on individual pairs
- added confidence scoring based on distance from the decision boundary
- added latency measurement for each inference call

The deterministic pair generation and evaluation setup from previous milestones are reused so results remain reproducible.

## Representation Overview

The system now supports two types of representations:

Baseline (Milestone 2):
- grayscale → resize → flatten → normalize → cosine similarity  

Milestone 3:
- ArcFace embeddings → cosine similarity → threshold decision  


## Repository Structure

```text
face-verification-system/
├── configs/
├── reports/
│   ├── evaluation_report.md
├── scripts/
│   ├── benchmark_similarity.py
│   ├── evaluator.py
|   ├── preprocessing.py
|   ├── similarity_scoring.py
|   ├── embedding_generation.py  
|   ├── threshold_decision.py
|   ├── latency_measurement.py
|   ├── confidence_computation.py
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


## Milestone 2 Summary

### Baseline

The baseline uses the original pair policy reproduced in `outputs/pairs` and evaluates a fixed-threshold and validation-sweep version of the same raw-pixel cosine verifier.


### Data-Centric Improvement

The data-centric change creates a second pair-set version in `outputs/pairs_v2`.

Implemented change:

- keep training pair construction unchanged
- remove positive self-pairs from validation and test

This makes the evaluation set less artificially easy and gives a clearer picture of how brittle the raw-pixel baseline is.



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

Use evaluation mode when you want metrics on validation/test splits and tracked run artifacts.

Validation sweep plus test evaluation on the current pair set:

```bash
python -m scripts.evaluator --config configs/after_change_sweep.json
```

Then run the fixed-threshold follow-up using the selected validation threshold:

```bash
python -m scripts.evaluator --config configs/after_change_best.json
```

ArcFace evaluation examples:

```bash
python -m scripts.evaluator --config configs/arcface_sweep.json
python -m scripts.evaluator --config configs/arcface_best.json
```

Main evaluation outputs are written to `outputs/runs/<run_name>/`:

- `<run_name>_summary.json`
- `<run_name>_val_scores.jsonl`
- `<run_name>_test_scores.jsonl`
- `<run_name>_val_roc.png` (for sweep runs)

### 3.1 Run inference mode (CLI)

Use inference mode when you want to run pair-level predictions directly from the terminal.

Single pair inference:

```bash
python -m scripts.evaluator \
  --config configs/arcface_best.json \
  --embedding-backend arcface \
  --left-image data/lfw/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
  --right-image data/lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg \
  --pair-id demo_pair_1 \
  --threshold 0.2658909489140911
```

Batch inference from a pairs file (`.jsonl` or `.csv`):

```bash
python -m scripts.evaluator \
  --config configs/arcface_best.json \
  --embedding-backend arcface \
  --pairs-file outputs/pairs/inference_pairs.jsonl
```

Each pair prints:

- pair identifier / input paths
- similarity score
- threshold used
- binary decision
- calibrated confidence
- latency for that inference

Note:
- If no face is detected in an image, a zero embedding is used as a fallback.

### Docker

Build the Docker image from the repository root:

```bash
docker build -t face-verifier-m3 .
```

Run single pair ArcFace inference inside Docker:

```bash
docker run --rm \
  -v "$(pwd)":/app \
  face-verifier-m3 \
  python -m scripts.evaluator \
    --config configs/arcface_best.json \
    --embedding-backend arcface \
    --left-image data/lfw/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
    --right-image data/lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg \
    --pair-id demo_pair_1 \
    --threshold 0.2658909489140911
```

Notes:

- The repository is mounted into /app using `-v "$(pwd)":/app`
- This allows the container to access local files including the dataset `(data/lfw)`
- The first ArcFace run inside Docker may download the InsightFace model and take longer
- The CLI output includes score, threshold, decision, confidence, and latency
- Docker uses a mounted volume for dataset access instead of copying data into the image

---

### Load Test

Run a small local load test on a deterministic set of face pairs:

```bash
python -m scripts.load_test \
  --config configs/arcface_best.json \
  --embedding-backend arcface \
  --threshold 0.2658909489140911 \
  --workers 4 \
  --repeats 5
```

This writes a summary file to `outputs/load_test/load_test_summary.json`

The load test reports:
- total requests
- successful and failed requests
- total wall clock time
- throughput in requests per second
- latency statistics including mean, median, and p95

Notes:

- The request set is deterministic and is reused across runs
- Some requests may fail if face detection doesn't succeed on a given image

### System Behavior and Analysis

#### Inference Behavior

At inference time, the system processes one pair of images at a time. For each pair:

- ArcFace embeddings are generated for both images  
- cosine similarity is computed between the embeddings  
- the score is compared to a threshold to produce a binary decision  
- confidence and latency are reported for each prediction  

This replaces the raw pixel pipeline from Milestone 2 with a learned approach that's based on embedding

---

#### Confidence Interpretation

Confidence is based on how far the similarity score is from the threshold:

- margin = |score - threshold|  
- confidence = margin / 2  

Since cosine similarity lies in [-1, 1], this keeps values in [0, 1].

Interpretation:

- higher confidence → prediction is farther from the decision boundary  
- lower confidence → prediction is closer to the boundary  

This is just a simple deterministic heuristic and not a calibrated probability.

---

#### Load Testing Observations

A small local load test was run using a deterministic set of face pairs with multiple worker threads.

Results:

- throughput around 1–1.3 requests per second  
- mean latency around 2 seconds  
- median latency around 1.5 seconds  
- p95 latency around 4 seconds  

Some requests failed during testing mostly due to face detection issues on certain images. These failures were tracked and included in the results.

---

#### Limitations

A few limitations were observed:

- face detection is not guaranteed to succeed for all images  
- inference latency is relatively high due to CPU only execution  
- confidence scores are heuristic and not probabilistically calibrated  
- performance depends on image quality and face detectability  

Despite this, the system provides a complete and reproducible inference pipeline.

### 4. Run the historical baseline configs

These configs point to `outputs/pairs` and are kept for baseline comparison:

```bash
python -m scripts.evaluator --config configs/baseline.json
python -m scripts.evaluator --config configs/baseline_sweep.json
python -m scripts.evaluator --config configs/baseline_best.json
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

Run all tests:

```bash
pytest -q
```
The test suite covers:
- Similarity metric correctness
- validation checks
- thresholding behavior
- evaluator output structure
- CLI single pair and batch inference output
- 1 small integration run

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

- `arcface_sweep`
  - files:
    - `outputs/runs/arcface_sweep/arcface_sweep_summary.json`
    - `outputs/runs/arcface_sweep/arcface_sweep_val_scores.jsonl`
    - `outputs/runs/arcface_sweep/arcface_sweep_val_threshold_sweep.jsonl`
    - `outputs/runs/arcface_sweep/arcface_sweep_val_roc.png`
    - `outputs/runs/arcface_sweep/arcface_sweep_test_scores.jsonl`
    - purpose:
    - threshold sweep using ArcFace embeddings on the original pair set

- `arcface_best`
  - files:
    - `outputs/runs/arcface_best/arcface_best_summary.json`
    - `outputs/runs/arcface_best/arcface_best_val_scores.jsonl`
    - `outputs/runs/arcface_best/arcface_best_test_scores.jsonl`
  - purpose:
    - fixed-threshold rerun using ArcFace embeddings on the original pair set with the selected


### Selected Thresholds

Original pair set:

- selected on validation: `0.9266837672022352`

Current data-centric pair set:

- selected on validation: `0.5853148971300512`

ArcFace embedding  with original pair set:

- selected on validation: `0.2658909489140911`

## Reproducing The Main Reported Result

The main reported result in the report is the stricter data-centric evaluation on `outputs/pairs_v2`.

Generate the data-centric pair set and run the evaluation:

```bash
python scripts/generate_pairs.py --pair-version v2
python scripts/validate_pipeline.py --config configs/after_change_sweep.json
python -m scripts.evaluator --config configs/after_change_sweep.json
python -m scripts.evaluator --config configs/after_change_best.json
```

For baseline comparison, the original pair version is also reproducible with:
```bash
python scripts/generate_pairs.py --pair-version baseline
python -m scripts.evaluator --config configs/baseline_sweep.json
python -m scripts.evaluator --config configs/baseline_best.json
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
    python -m scripts.evaluator --config configs/after_change_sweep.json
    python -m scripts.evaluator --config configs/after_change_best.json
    python -m scripts.evaluator --config configs/baseline_sweep.json
    python -m scripts.evaluator --config configs/baseline_best.json
    python -m scripts.evaluator --config configs/baseline.json
    ```
6. run tests using `pytest -q`
7. confirm expected artifacts exist under:
    ```bash
    outputs/pairs/
    outputs/pairs_v2/
    outputs/runs/
    ```
Both `outputs/pairs` and `outputs/pairs_v2` are reproducible from the current generator by choosing the appropriate `--pair-version`.