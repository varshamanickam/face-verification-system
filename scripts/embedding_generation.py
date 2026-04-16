from pathlib import Path
from typing import Any
import warnings

import numpy as np
from tqdm import tqdm

from scripts.preprocessing import preprocess_image

_ARC_FACE_APP: Any = None


def _get_arcface_app(cfg: dict) -> Any:
	global _ARC_FACE_APP
	if _ARC_FACE_APP is None:
		from insightface.app import FaceAnalysis

		model_name = str(cfg.get("arcface_model_name", "buffalo_l"))
		ctx_id = int(cfg.get("arcface_ctx_id", -1))
		det_size_cfg = cfg.get("arcface_det_size", [640, 640])
		if not isinstance(det_size_cfg, list) or len(det_size_cfg) != 2:
			raise ValueError("arcface_det_size must be a list of length 2")
		det_h, det_w = int(det_size_cfg[0]), int(det_size_cfg[1])

		_ARC_FACE_APP = FaceAnalysis(name=model_name)
		_ARC_FACE_APP.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
	return _ARC_FACE_APP


def _face_bbox_area(face: Any) -> float:
	bbox = np.asarray(getattr(face, "bbox", [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
	if bbox.shape[0] < 4:
		return 0.0
	w = max(float(bbox[2] - bbox[0]), 0.0)
	h = max(float(bbox[3] - bbox[1]), 0.0)
	return w * h


def _face_center_distance(face: Any, image_h: int, image_w: int) -> float:
	bbox = np.asarray(getattr(face, "bbox", [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
	if bbox.shape[0] < 4:
		return float("inf")
	fx = float((bbox[0] + bbox[2]) / 2.0)
	fy = float((bbox[1] + bbox[3]) / 2.0)
	cx = image_w / 2.0
	cy = image_h / 2.0
	return float(np.hypot(fx - cx, fy - cy))


def _select_primary_face(faces: list[Any], image_h: int, image_w: int) -> Any:
	# Choose the most complete/prominent face: largest box, then highest score, then center-prioritized.
	return max(
		faces,
		key=lambda face: (
			_face_bbox_area(face),
			float(getattr(face, "det_score", 0.0)),
			-_face_center_distance(face, image_h=image_h, image_w=image_w),
		),
	)


def _no_face_fallback_embedding(image_path: Path, cfg: dict) -> np.ndarray:
	dim = int(cfg.get("arcface_embedding_dim", 512))
	if dim <= 0:
		raise ValueError(f"arcface_embedding_dim must be positive, got {dim}")
	warnings.warn(
		f"No face detected for image: {image_path}. Using zero embedding fallback with dim={dim}.",
		RuntimeWarning,
	)
	return np.zeros((dim,), dtype=np.float64)


def preprocess_image_arcface(image_path: str | Path, cfg: dict) -> np.ndarray:
	import cv2

	image_path = Path(image_path)
	image = cv2.imread(str(image_path))
	if image is None:
		raise ValueError(f"Failed to read image: {image_path}")

	app = _get_arcface_app(cfg)
	faces = app.get(image)
	if len(faces) == 0:
		return _no_face_fallback_embedding(image_path=image_path, cfg=cfg)

	require_single_face = bool(cfg.get("arcface_require_single_face", False))
	if require_single_face and len(faces) != 1:
		raise ValueError(f"Expected 1 face, got {len(faces)} for image: {image_path}")

	face = faces[0] if len(faces) == 1 else _select_primary_face(faces, image_h=image.shape[0], image_w=image.shape[1])

	embedding = np.asarray(face.normed_embedding, dtype=np.float64)
	norm = np.linalg.norm(embedding)
	if norm > 0:
		embedding = embedding / norm
	return embedding


def generate_embedding(
	image_path: str | Path,
	embedding_backend: str,
	cfg: dict,
	image_mode: str,
	resize: tuple[int, int],
) -> np.ndarray:
	if embedding_backend == "arcface":
		return preprocess_image_arcface(image_path, cfg=cfg)
	return preprocess_image(image_path, image_mode=image_mode, resize=resize)


def build_image_cache(
	pairs: list[dict],
	image_mode: str,
	resize: tuple[int, int],
	embedding_backend: str,
	cfg: dict,
) -> dict[str, np.ndarray]:
	image_cache: dict[str, np.ndarray] = {}
	unique_paths = sorted(
		{item["left_path"] for item in pairs}.union({item["right_path"] for item in pairs})
	)
	for image_path in tqdm(unique_paths, desc="Embedding cache", unit="image"):
		image_cache[image_path] = generate_embedding(
			image_path=image_path,
			embedding_backend=embedding_backend,
			cfg=cfg,
			image_mode=image_mode,
			resize=resize,
		)
	return image_cache


def pairs_to_arrays(
	pairs: list[dict],
	image_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	left = np.stack([image_cache[item["left_path"]] for item in pairs], axis=0)
	right = np.stack([image_cache[item["right_path"]] for item in pairs], axis=0)
	labels = np.asarray([item["label"] for item in pairs], dtype=np.int64)
	return left, right, labels
