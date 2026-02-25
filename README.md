# face-verification-system

# Requirements
Install project dependencies from the repository root:

```bash
pip install -r requirements.txt
```
Activate virtual environment:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# LFW Ingest + Pair Generation (`generate_pairs.py`)

This script ingests the LFW dataset, creates an identity-level train/val/test split, writes a manifest, and generates verification pairs for each split.

## What it does

1. Checks for `data/lfw`.
2. If missing, downloads LFW via TensorFlow Datasets into `data_cache`, finds the extracted LFW folder, and copies it to `data/lfw`.
3. Scans `data/lfw` and builds `(image_path, identity)` records.
4. Performs a deterministic identity split:
   - Sort identities alphabetically
   - Split by identity into 80% train / 10% val / 10% test
5. Writes `outputs/manifest.json` with seed, split policy, and image/identity counts.
6. Generates verification pairs per split (roughly 50/50 positive/negative):
   - Train: 10,000 pairs
   - Val: 2,000 pairs
   - Test: 2,000 pairs
7. Writes JSONL files:
   - `outputs/train.jsonl`
   - `outputs/val.jsonl`
   - `outputs/test.jsonl`

## Pair format

Each line in the JSONL files is:

```json
{
  "left_path": "data/lfw/<identity>/<image>.jpg",
  "right_path": "data/lfw/<identity>/<image>.jpg",
  "label": 0,
  "split": "train"
}
```

- `label = 1`: same identity (positive pair)
- `label = 0`: different identities (negative pair)

## Determinism

The script is deterministic for split and pair generation by design:

- Identity split is based on sorted identity names.
- Pair generation uses `numpy.random.default_rng(seed=42)`.
- TFDS loading uses `shuffle_files=False`.


## Run

From repository root:

```bash
python scripts/generate_pairs.py
```


## Output paths

- Manifest: `outputs/manifest.json`
- Pairs: `outputs/{train,val,test}.jsonl`

These paths are relative to the current working directory when you run the script.
