# Reproducibility Checklist (Milestone 4)

The following steps reproduce the final ArcFace-based face verification system from a clean clone.

## 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Generate deterministic pair set

```bash
python scripts/generate_pairs.py --pair-version baseline
```

### Expected outputs:

- `outputs/pairs/manifest.json`
- `outputs/pairs/train.jsonl`
- `outputs/pairs/val.jsonl`
- `outputs/pairs/test.jsonl`

## 3. Validate pipeline inputs

```bash
python scripts/validate_pipeline.py --config configs/arcface_sweep.json
```

## 4. Run evaluation (threshold selection + final system)

```bash
python -m scripts.evaluator --config configs/arcface_sweep.json
python -m scripts.evaluator --config configs/arcface_best.json
```

### Expected outputs:

- `outputs/runs/arcface_sweep/arcface_sweep_summary.json`
- `outputs/runs/arcface_best/acrface_best_summary.json`

## 5. Run CLI inference (for sanity check)

```bash
python -m scripts.evaluator \
  --config configs/arcface_best.json \
  --embedding-backend arcface \
  --left-image data/lfw/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
  --right-image data/lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg \
  --pair-id demo_pair \
  --threshold 0.2658909489140911
```

## 6. Run profiling (CPU baseline)

```bash
python -m scripts.profile_system --limit 25
```

### Expected output:

- `outputs/profiling/profile_cpu.json`

## 7. Run tests

```bash
pytest -q
```

## 8. Final artifacts

1) System card: `reports/system_card.md`
2) Profiling report: `reports/profiling_report.md`
3) Reproducibility checklist: `reports/reproducibility_checklist.md`
4) Profiling output: `outputs/profiling/profile_cpu.json`



