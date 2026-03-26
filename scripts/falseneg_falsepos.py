import json
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

baseline_threshold = 0.9266837672022352
after_change_threshold = 0.5853148971300512

baseline_rows = load_jsonl("outputs/runs/baseline_best/baseline_best_test_scores.jsonl")
after_change_rows = load_jsonl("outputs/runs/after_change_best/after_change_best_test_scores.jsonl")

baseline_false_negatives = [
    row for row in baseline_rows
    if row["label"] == 1 and row["score"] < baseline_threshold
]

after_change_false_positives = [
    row for row in after_change_rows
    if row["label"] == 0 and row["score"] >= after_change_threshold
]

print("Baseline false negatives:", len(baseline_false_negatives))
for row in baseline_false_negatives[:5]:
    print(row)

print("\nAfter-change false positives:", len(after_change_false_positives))
for row in after_change_false_positives[:5]:
    print(row)