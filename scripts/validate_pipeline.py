import argparse
from pathlib import Path

from src.validation import load_config, read_pairs, validate_config, validate_split_disjointness


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate face-verification pipeline inputs and artifacts.")
    parser.add_argument(
        "--config",
        default="configs/baseline.json",
        help="Path to config JSON.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg, args.config)

    pairs_dir = Path(cfg.get("pairs_dir", "outputs/pairs"))
    val_split = cfg.get("split_for_threshold_selection", "val")
    test_split = cfg.get("split_for_final_reporting", "test")

    val_rows = read_pairs(pairs_dir / f"{val_split}.jsonl", expected_split=val_split)
    test_rows = read_pairs(pairs_dir / f"{test_split}.jsonl", expected_split=test_split)
    validate_split_disjointness(val_rows, test_rows)

    print(f"Config valid: {args.config}")
    print(f"Validation pairs valid: {pairs_dir / f'{val_split}.jsonl'} ({len(val_rows)} rows)")
    print(f"Test pairs valid: {pairs_dir / f'{test_split}.jsonl'} ({len(test_rows)} rows)")
    print("Validation checks passed.")


if __name__ == "__main__":
    main()
