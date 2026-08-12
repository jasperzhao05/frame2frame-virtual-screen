"""Optional deep backend: Hopenet (Ruiz et al., 2018).

The original prototype imported a `hopenet` module that was never committed, so
the repo could not actually run this path. The package carries the compatible
network definition and fetches the pretrained weights on first use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .._downloads import ensure_download
from ._hopenet_model import build_hopenet as _build_model
from .base import FaceObservation, HeadPose, PoseEstimator

_CACHE_DIR = Path(
    os.environ.get("FRAME2FRAME_CACHE", Path.home() / ".cache" / "frame2frame")
).expanduser()
# Pretrained "robust" snapshot published with the original paper.
_DEFAULT_GDRIVE_ID = "1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR"
_DEFAULT_WEIGHTS = "hopenet_robust_alpha1.pkl"
_DEFAULT_WEIGHTS_SHA256 = "1e0c6ddfda0e19a679607480c10875020de29b3984f187ec311c5e0802b6b6d5"
_DEFAULT_WEIGHTS_SIZE = 95_924_799


def _resolve_weights(
    weights: str | os.PathLike[str] | None,
    gdrive_id: str | None,
) -> Path:
    if weights:
        path = Path(weights).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"weights not found or not a file: {path}")
        return path
    path = _CACHE_DIR / _DEFAULT_WEIGHTS
    file_id = gdrive_id or _DEFAULT_GDRIVE_ID
    if file_id != _DEFAULT_GDRIVE_ID:
        raise ValueError("custom gdrive_id is unsupported; download it and pass weights=<path>")
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
    return ensure_download(
        url,
        path,
        sha256=_DEFAULT_WEIGHTS_SHA256,
        expected_size=_DEFAULT_WEIGHTS_SIZE,
    )


class HopenetEstimator(PoseEstimator):
    def __init__(
        self,
        weights: str | os.PathLike[str] | None = None,
        gdrive_id: str | None = None,
        device: Any = None,
        margin: float = 20,
        fps: float = 30.0,
        face_model_path: str | os.PathLike[str] | None = None,
    ) -> None:
        import torch
        from torchvision import transforms

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device: Any = torch.device(device)
        self.margin: float = margin

        self._model: Any = _build_model()
        # The published snapshot is a plain state dict; refuse anything that
        # needs arbitrary unpickling (torch >= 2.6 also defaults to this).
        state = torch.load(
            _resolve_weights(weights, gdrive_id), map_location="cpu", weights_only=True
        )
        self._model.load_state_dict(state)
        self._model.eval().to(self.device)

        self._tf: Any = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._bins: Any = torch.arange(66, dtype=torch.float32, device=self.device)

        from ._facemesh import FaceMeshDetector

        self._detector = FaceMeshDetector(model_path=face_model_path, fps=fps)

    def _decode(self, logits: Any) -> float:
        import torch.nn.functional as F

        prob = F.softmax(logits, dim=1)
        return float((prob[0] * self._bins).sum() * 3 - 99)

    def estimate(self, frame_bgr: np.ndarray) -> FaceObservation | None:
        return self._estimate(frame_bgr, None)

    def estimate_at(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: float | None,
    ) -> FaceObservation | None:
        return self._estimate(frame_bgr, timestamp_ms)

    def _estimate(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: float | None,
    ) -> FaceObservation | None:
        import torch

        from ._facemesh import _detected_face_crop

        detected = _detected_face_crop(
            self._detector,
            frame_bgr,
            self.margin,
            timestamp_ms,
        )
        if detected is None:
            return None

        rgb = cv2.cvtColor(detected.image, cv2.COLOR_BGR2RGB)
        tensor = self._tf(rgb).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            yaw, pitch, roll = self._model(tensor)
        pose = HeadPose(self._decode(yaw), self._decode(pitch), self._decode(roll))
        return FaceObservation.from_bbox(pose, detected.bbox, detected.landmarks)

    def close(self) -> None:
        self._detector.close()
