import numpy as np

from src.validation import validate_metrics, validate_threshold


def compute_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
	tp = int(np.sum((predictions == 1) & (labels == 1)))
	tn = int(np.sum((predictions == 0) & (labels == 0)))
	fp = int(np.sum((predictions == 1) & (labels == 0)))
	fn = int(np.sum((predictions == 0) & (labels == 1)))

	accuracy = (tp + tn) / max(labels.shape[0], 1)
	true_positive_rate = tp / max(tp + fn, 1)
	true_negative_rate = tn / max(tn + fp, 1)
	balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate)
	precision = tp / max(tp + fp, 1)
	recall = true_positive_rate
	if precision + recall == 0.0:
		f1 = 0.0
	else:
		f1 = 2.0 * precision * recall / (precision + recall)

	metrics = {
		"tp": tp,
		"tn": tn,
		"fp": fp,
		"fn": fn,
		"accuracy": float(accuracy),
		"balanced_accuracy": float(balanced_accuracy),
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
	}
	validate_metrics(metrics)
	return metrics


def evaluate_scored_split(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
	validate_threshold(threshold)
	predictions = (scores >= threshold).astype(np.int64)
	return compute_metrics(labels=labels, predictions=predictions)


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
	unique_scores = np.unique(scores)
	if unique_scores.size == 1:
		score = float(unique_scores[0])
		candidates = np.asarray([score - 1e-6, score, score + 1e-6], dtype=np.float64)
		return np.clip(np.unique(candidates), -1.0, 1.0)

	midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
	candidates = np.concatenate(
		[
			np.asarray([unique_scores[0] - 1e-6], dtype=np.float64),
			unique_scores,
			midpoints,
			np.asarray([unique_scores[-1] + 1e-6], dtype=np.float64),
		]
	)
	return np.clip(np.unique(candidates), -1.0, 1.0)


def threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict, list[dict]]:
	best_threshold = None
	best_metrics = None
	sweep_rows = []

	for threshold in threshold_candidates(scores):
		metrics = evaluate_scored_split(labels=labels, scores=scores, threshold=float(threshold))
		row = {"threshold": float(threshold), **metrics}
		sweep_rows.append(row)

		if best_metrics is None:
			best_threshold = float(threshold)
			best_metrics = metrics
			continue

		if metrics["f1"] > best_metrics["f1"]:
			best_threshold = float(threshold)
			best_metrics = metrics
		elif metrics["f1"] == best_metrics["f1"] and metrics["accuracy"] > best_metrics["accuracy"]:
			best_threshold = float(threshold)
			best_metrics = metrics

	validate_threshold(float(best_threshold))
	return float(best_threshold), best_metrics, sweep_rows
