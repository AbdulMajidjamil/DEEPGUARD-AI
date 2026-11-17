# SPDX-License-Identifier: Apache-2.0
# Originally part of SENTINEL, Copyright 2026 Dev Parth.
# Modified 2026 by Abdul Majid Jamil for DEEPGUARD AI: strict checkpoint validation, CPU/CUDA
# selection, explicit demo mode, per-person results, frame-level indicators (see DERIVATION.md).
"""Model loading and inference.

Two honest modes:

``trained_model``
    A checkpoint in ``weights/`` passed strict validation. The verdict is the
    ResNeXt-50 + LSTM logit.

``demo_heuristic``
    No usable checkpoint. The verdict is a rule-based temporal-consistency
    heuristic over ImageNet embeddings. It is *not* a trained classifier and
    every result says so (``is_demo`` + warnings).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import CheckpointError, load_checkpoint
from config import (
    ALLOW_PICKLE_WEIGHTS,
    BACKBONE,
    DEVICE,
    DISCLAIMER,
    MIN_FACES,
    PRETRAINED_BACKBONE,
    SPATIAL_BATCH,
    TORCH_THREADS,
    WEIGHTS_PATH,
)
from model import DeepfakeDetector
from pipeline import ExtractionResult, FaceTrack, faces_to_tensor, thumb_to_data_uri
from security import assert_safe_weights_path

logger = logging.getLogger("deepguard.inference")

DEMO_WARNING = (
    "No trained checkpoint is loaded. This verdict comes from a rule-based temporal heuristic, "
    "not a trained deepfake classifier - treat it as a demonstration only."
)

_ENGINE: Optional["InferenceEngine"] = None
_ENGINE_LOCK = threading.Lock()


def configure_torch() -> None:
    torch.set_num_threads(max(1, TORCH_THREADS))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set once in this process


def resolve_device(requested: str = DEVICE) -> tuple[torch.device, Optional[str]]:
    """Return (device, note). ``cuda``/``auto`` fall back to CPU when no GPU is present."""
    want = (requested or "cpu").lower()
    if want in {"cuda", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda"), None
        note = "CUDA requested but no GPU is available; running on CPU." if want == "cuda" else None
        return torch.device("cpu"), note
    if want != "cpu":
        return torch.device("cpu"), f"Unknown DEVICE '{requested}'; running on CPU."
    return torch.device("cpu"), None


class InferenceEngine:
    def __init__(
        self,
        backbone: str = BACKBONE,
        weights: Path | str = WEIGHTS_PATH,
        device: str = DEVICE,
        pretrained: bool = PRETRAINED_BACKBONE,
    ) -> None:
        configure_torch()
        self.device, self.device_note = resolve_device(device)
        self.backbone = backbone
        self.weights_path = Path(weights)
        self.weights_loaded = False
        self.checkpoint: Optional[dict[str, Any]] = None
        self.checkpoint_error: Optional[str] = None
        self.pretrained_ok = True
        self._lock = threading.Lock()

        try:
            self.weights_path = assert_safe_weights_path(self.weights_path)
        except PermissionError:
            self.checkpoint_error = "Weights path is outside the weights/ directory and was ignored."
            self.weights_path = Path("__missing__")

        has_ckpt = self.weights_path.is_file()
        # With a checkpoint the ImageNet download is pointless (overwritten) - skip it.
        self.model = self._build(pretrained=pretrained and not has_ckpt)
        if has_ckpt:
            try:
                self.checkpoint = load_checkpoint(
                    self.model, self.weights_path, backbone=backbone, allow_pickle=ALLOW_PICKLE_WEIGHTS
                )
                self.weights_loaded = True
            except CheckpointError as exc:
                self.checkpoint_error = str(exc)
                logger.error("Checkpoint rejected: %s", exc)
                self.model = self._build(pretrained=pretrained)  # demo mode needs real features
        else:
            logger.warning("No checkpoint at weights/%s - running in DEMO heuristic mode.", WEIGHTS_PATH.name)

        self.model.eval().to(self.device)
        self.mode = "trained_model" if self.weights_loaded else "demo_heuristic"
        self.n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Engine ready backbone=%s params=%.1fM mode=%s device=%s",
            self.backbone, self.n_params / 1e6, self.mode, self.device,
        )

    # -- construction ------------------------------------------------------
    def _build(self, pretrained: bool) -> DeepfakeDetector:
        try:
            return DeepfakeDetector(backbone=self.backbone, lstm_hidden=512, lstm_layers=2, dropout=0.5, pretrained=pretrained)
        except Exception as exc:
            if not pretrained:
                raise
            logger.warning("ImageNet backbone weights unavailable (%s); using random init", type(exc).__name__)
            self.pretrained_ok = False
            return DeepfakeDetector(backbone=self.backbone, lstm_hidden=512, lstm_layers=2, dropout=0.5, pretrained=False)

    # -- public ------------------------------------------------------------
    @property
    def model_id(self) -> str:
        if self.weights_loaded and self.checkpoint:
            return f"deepguard-{self.backbone}-lstm@{self.checkpoint['sha256']}"
        return f"heuristic-demo({self.backbone}-features)"

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "inference_mode": self.mode,
            "is_demo": not self.weights_loaded,
            "backbone": self.backbone,
            "device": str(self.device),
            "device_note": self.device_note,
            "parameters_millions": round(self.n_params / 1e6, 1),
            "weights_present": self.weights_path.is_file(),
            "weights_loaded": self.weights_loaded,
            "checkpoint": self.checkpoint,
            "checkpoint_error": self.checkpoint_error,
            "imagenet_backbone": self.pretrained_ok and not self.weights_loaded,
        }

    @torch.inference_mode()
    def run(self, extraction: ExtractionResult) -> dict[str, Any]:
        tracks = [t for t in extraction.tracks if t.n_faces >= MIN_FACES]
        if not tracks:
            have = max((t.n_faces for t in extraction.tracks), default=0)
            raise RuntimeError(
                f"Need at least {MIN_FACES} frames with a detectable face to classify a clip (found {have})."
            )
        with self._lock:
            scored = [self._score_track(t) for t in tracks]

        focus = max(scored, key=lambda s: s["probability_fake"])
        probability = focus["probability_fake"]
        label = "FAKE" if probability >= 0.5 else "REAL"
        confidence = probability if label == "FAKE" else 1.0 - probability

        warnings = list(extraction.warnings)
        if not self.weights_loaded:
            warnings.insert(0, DEMO_WARNING)
        if self.checkpoint_error:
            warnings.insert(0, f"Checkpoint rejected: {self.checkpoint_error}")
        if self.device_note:
            warnings.append(self.device_note)
        if not self.pretrained_ok and not self.weights_loaded:
            warnings.append("ImageNet backbone weights could not be downloaded; features are randomly initialised.")
        if self.weights_loaded and not (self.checkpoint or {}).get("val"):
            warnings.append("This checkpoint carries no recorded validation metrics.")
        if extraction.n_sampled and extraction.n_faces < 0.5 * extraction.n_sampled:
            warnings.append(
                f"A face was found in only {extraction.n_faces} of {extraction.n_sampled} sampled frames; "
                "treat the verdict with extra caution."
            )
        if len(scored) > 1 and label == "FAKE":
            warnings.append("The verdict reflects the most suspicious of several people in the clip.")

        track = next(t for t in tracks if t.track_id == focus["track_id"])
        suspicious = self._suspicious_frames(track, focus["anomaly"])
        persons = [
            {
                "track_id": s["track_id"],
                "label": "FAKE" if s["probability_fake"] >= 0.5 else "REAL",
                "probability_fake": s["probability_fake"],
                "faces_analysed": s["faces"],
            }
            for s in scored
        ]
        return {
            "label": label,
            "probability_fake": round(probability, 4),
            "manipulation_probability": round(probability, 4),
            "confidence": round(float(confidence), 4),
            "logit": focus["logit"],
            "hybrid_logit": focus["hybrid_logit"],
            "score_source": "cnn_rnn" if self.weights_loaded else "temporal_forensics",
            "mode": self.mode,
            "inference_mode": self.mode,
            "is_demo": not self.weights_loaded,
            "model_id": self.model_id,
            "backbone": self.backbone,
            "device": str(self.device),
            "weights_loaded": self.weights_loaded,
            "forensics": focus["forensics"],
            "cosine_series": focus["forensics"].pop("cosine_series"),
            "frame_scores": [
                {"index": track.frame_meta[i].index, "timestamp": track.frame_meta[i].timestamp, "anomaly": round(float(a), 3)}
                for i, a in enumerate(focus["anomaly"])
            ],
            "suspicious_frames": suspicious,
            "persons": persons,
            "warnings": warnings,
            "disclaimer": DISCLAIMER,
        }

    # -- internals ---------------------------------------------------------
    def _score_track(self, track: FaceTrack) -> dict[str, Any]:
        tensor = faces_to_tensor(track.faces_bgr).to(self.device)  # [1, T, 3, H, W]
        t = tensor.shape[1]
        flat = tensor.reshape(t, *tensor.shape[2:])
        raw = self.model.extract_spatial(flat, chunk_size=SPATIAL_BATCH)  # [T, D]
        norm = F.normalize(raw, dim=-1).cpu()
        forensic = self._forensic_scores(norm, track)

        hybrid: Optional[float] = None
        if self.weights_loaded:
            lengths = torch.tensor([t], dtype=torch.long)
            hybrid = float(self.model.temporal_logits(raw.unsqueeze(0), lengths).view(-1)[0].cpu())
            logit = hybrid
        else:
            logit = forensic["logit"]
        prob = float(1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20))))
        anomaly = forensic.pop("anomaly")
        logit_r = round(float(logit), 4)
        forensic.pop("logit", None)
        return {
            "track_id": track.track_id,
            "faces": track.n_faces,
            "probability_fake": prob,
            "logit": logit_r,
            "hybrid_logit": None if hybrid is None else round(hybrid, 4),
            "forensics": forensic,
            "anomaly": anomaly,
        }

    @staticmethod
    def _forensic_scores(feats: torch.Tensor, track: FaceTrack) -> dict[str, Any]:
        """Temporal-consistency indicators over L2-normalised embeddings + sharpness."""
        t = feats.shape[0]
        laps = np.array([m.laplacian_var for m in track.frame_meta], dtype=np.float64)
        mean_lap = float(laps.mean()) if len(laps) else 0.0
        std_lap = float(laps.std()) if len(laps) else 0.0
        if t < 2:
            return {
                "logit": 0.0, "mean_cosine": 1.0, "std_cosine": 0.0,
                "mean_laplacian": round(mean_lap, 2), "std_laplacian": round(std_lap, 2),
                "flicker": 0.0, "oversmooth": 0.0, "cosine_series": [1.0], "anomaly": np.zeros(t),
            }
        sims = (feats[:-1] * feats[1:]).sum(dim=-1).numpy().astype(np.float64)
        mean_cos, std_cos = float(sims.mean()), float(sims.std())

        flicker = float(np.clip((std_cos - 0.02) / 0.10, 0.0, 1.0))
        oversmooth = float(np.clip((80.0 - mean_lap) / 80.0, 0.0, 1.0))
        frozen = float(np.clip((mean_cos - 0.992) / 0.008, 0.0, 1.0)) if mean_cos > 0.99 else 0.0
        evidence = float(np.clip(0.50 * flicker + 0.30 * oversmooth + 0.20 * frozen, 0.02, 0.98))
        logit = float(np.log(evidence / (1.0 - evidence)))

        # Per-frame anomaly index: how far this frame departs from its neighbours
        # (embedding jump) and from the clip's typical sharpness. A heuristic cue for
        # *where to look*, not a per-frame classifier output.
        jump = np.zeros(t)
        jump[1:] += 1.0 - sims
        jump[:-1] += 1.0 - sims
        jump[1:-1] /= 2.0
        med = float(np.median(jump))
        mad = float(np.median(np.abs(jump - med))) * 1.4826 + 1e-3
        jump_z = np.clip((jump - med) / mad / 6.0, 0.0, 1.0)
        if len(laps) == t and std_lap > 1e-6:
            lap_z = np.clip(np.abs(laps - float(np.median(laps))) / (std_lap * 4.0 + 1e-6), 0.0, 1.0)
        else:
            lap_z = np.zeros(t)
        anomaly = np.clip(0.7 * jump_z + 0.3 * lap_z, 0.0, 1.0)

        return {
            "logit": logit,
            "mean_cosine": round(mean_cos, 4),
            "std_cosine": round(std_cos, 4),
            "mean_laplacian": round(mean_lap, 2),
            "std_laplacian": round(std_lap, 2),
            "flicker": round(flicker, 4),
            "oversmooth": round(oversmooth, 4),
            "cosine_series": [round(float(s), 4) for s in sims.tolist()],
            "anomaly": anomaly,
        }

    @staticmethod
    def _suspicious_frames(track: FaceTrack, anomaly: np.ndarray, k: int = 6) -> list[dict[str, Any]]:
        order = np.argsort(-np.asarray(anomaly))[:k]
        out = []
        for i in sorted(int(j) for j in order):
            meta = track.frame_meta[i]
            if float(anomaly[i]) < 0.05 or not meta.thumb_jpeg:
                continue
            out.append(
                {
                    "index": meta.index,
                    "timestamp": meta.timestamp,
                    "anomaly": round(float(anomaly[i]), 3),
                    "thumb": thumb_to_data_uri(meta.thumb_jpeg),
                }
            )
        return out


def get_engine() -> InferenceEngine:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = InferenceEngine()
    return _ENGINE


def engine_loaded() -> Optional[InferenceEngine]:
    return _ENGINE


_LIGHT_CACHE: dict[str, Any] = {}


def light_status() -> dict[str, Any]:
    """Cheap description of the configured model without keeping an engine alive.

    Validates the checkpoint (cached per file mtime/size) so the API can report
    ``trained_model`` vs ``demo_heuristic`` truthfully before the first job runs.
    """
    if _ENGINE is not None:
        return _ENGINE.describe()
    path = Path(WEIGHTS_PATH)
    present = path.is_file()
    key = f"{path}:{path.stat().st_mtime_ns if present else 0}:{path.stat().st_size if present else 0}:{BACKBONE}"
    cached = _LIGHT_CACHE.get("key") == key
    if cached:
        return dict(_LIGHT_CACHE["value"])
    device, note = resolve_device()
    value: dict[str, Any] = {
        "model_id": f"heuristic-demo({BACKBONE}-features)",
        "inference_mode": "demo_heuristic",
        "is_demo": True,
        "backbone": BACKBONE,
        "device": str(device),
        "device_note": note,
        "parameters_millions": None,
        "weights_present": present,
        "weights_loaded": False,
        "checkpoint": None,
        "checkpoint_error": None,
        "imagenet_backbone": None,
    }
    if present:
        try:
            probe = DeepfakeDetector(backbone=BACKBONE, lstm_hidden=512, lstm_layers=2, pretrained=False)
            info = load_checkpoint(probe, assert_safe_weights_path(path), backbone=BACKBONE, allow_pickle=ALLOW_PICKLE_WEIGHTS)
            value.update(
                model_id=f"deepguard-{BACKBONE}-lstm@{info['sha256']}",
                inference_mode="trained_model",
                is_demo=False,
                weights_loaded=True,
                checkpoint=info,
                parameters_millions=round(sum(p.numel() for p in probe.parameters()) / 1e6, 1),
            )
        except (CheckpointError, PermissionError) as exc:
            value["checkpoint_error"] = str(exc)
    _LIGHT_CACHE["key"] = key
    _LIGHT_CACHE["value"] = value
    return dict(value)
