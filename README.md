# Face Verification System — Milestone 1

This project builds a fully reproducible face verification data pipeline using the Labeled Faces in the Wild (LFW) dataset.

Face verification is the task of determining whether two images belong to the same person or to different people.

In this milestone, we:

- Ingest and split the dataset deterministically  
- Generate verification pairs (positive + negative)  
- Implement loop-based and vectorized similarity metrics  
- Benchmark performance and verify numerical correctness  

All artifacts are saved locally and can be reproduced from scratch using the commands below.

## Repository Structure
```
face-verification-system/
│
├── src/
│ └── similarity_metrics.py
│
├── scripts/
│ ├── generate_pairs.py
│ └── benchmark_similarity.py
│
├── outputs/ # generated (gitignored)
│ ├── manifest.json
│ ├── pairs/
│ └── bench/
│
├── requirements.txt
└── pyproject.toml
```
- `src/` contains similarity implementations
- `scripts/` contains runnable pipeline scripts
- `outputs/` is generated when scripts are run

## Setup

Run all commands from the repository root.

### Step 1: Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```
### Step 2: Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```
'pip install -e .' ensures that the src package can be imported by the scripts.

# Ingest LFW and generate pairs

### Step 3: Ingest dataset + generate verification pairs

Run:

```bash
python scripts/generate_pairs.py
```
This script will:
- Download LFW
- Perform a deterministic identity level split (80%/10%/10%) for (train/val/test)
- Create a dataset manifest
- Generate verfification pairs
- Save all outputs under the ouputs/ directory

### Files generated
After running 'generate_pairs.py', you will see:
- `outputs/manifest.json`
- `outputs/pairs/train.jsonl`
- `outputs/pairs/val.jsonl`
- `outputs/pairs/test.jsonl`

Each JSONL file contains one pair per line:
```json
{
   "left_path": "data/lfw/<identity>/<image>.jpg",
   "right_path": "data/lfw/<identity>/<image>.jpg",
   "label": 0, 
   "split": "train"
}
```
- if label = 1, then that means same identity (positive pair)
- if label = 0, different identities (negative pair)
- "split" will be "train", "test", or "val" depending on which of the 3 jsonls  you are looking at

### Step 4: Run similarity benchmark

Run:
```bash
python scripts/benchmark_similarity.py
```

This script:
- Generates random embedding matrices
- Benchmarks loop vs vectorized implementations for:
   - Euclidean distance
   - Cosine similarity
- Verifies if they're numerically equal using np.allclose
- Prints timing and speedup information
- Saves benchmark artifacts under `outputs/bench/`

### Files generated

Example files generated:
- `outputs/bench/similarity_benchmark_np10000_d128.json`
- `outputs/bench/similarity_benchmark_np10000_d512.json`

Each file includes:

- execution time (for both loop and vectorized)
- speedup
- correctness check result
- maximum absolute difference

## How this pipeline is deterministic

We made this pipeline deterministic in the following ways:

- Identity split is based on sorted identity names.
- Split ratios are fixed (80%/10%/10%)
- Pair generation uses 'numpy.random.default_rng(seed=42)'
- Benchmark generation also uses a fixed seed
- TFDS loading uses 'shuffle_files=False'

Running the scripts multiple times is guaranteed to produce identical outputs.

## Additional Notes

- The `outputs/` directory is generated and gitignored.
- Dataset images are not committed to the repository.
- Re-running scripts overwrites existing artifacts.