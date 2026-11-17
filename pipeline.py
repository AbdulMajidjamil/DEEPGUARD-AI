# SPDX-License-Identifier: Apache-2.0
# Originally part of SENTINEL, Copyright 2026 Dev Parth.
# Modified 2026 by Abdul Majid Jamil for DEEPGUARD AI: multi-face tracking, safer decoding,
# frame-rate edge cases (see DERIVATION.md).
"""OpenCV frame sampling + YuNet / MTCNN / Haar face detection and alignment.

:func:`extract_face_sequence`

1. uniformly samples ``SEQ_LEN`` frames across the video;
2. detects **every** face in each sampled frame;
3. links detections into per-person tracks (nearest-centre matching);
4. aligns each track's faces (eye landmarks when available) to 224x224 crops.

The longest track is the *primary* face (``ExtractionResult.faces_bgr``); further
tracks are kept in ``ExtractionResult.tracks`` so a multi-person clip is analysed
per person instead of silently looking at one face only.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from config import (
    FACE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_DECODE_FRAMES,
    SEQ_LEN,
    YUNET_PATH,
)
from security import validate_video_meta

logger = logging.getLogger("deepguard.pipeline")

ProgressFn = Callable[[str, float, str], None]

MAX_TRACKS = 3  # analysed persons per clip (CPU budget)


@dataclass
class FrameFace:
    index: int
    timestamp: float
    has_face: bool
    box: Optional[list[float]] = None
    score: float = 0.0
    laplacian_var: float = 0.0
    thumb_jpeg: Optional[bytes] = None
    n_faces: int = 0  # faces visible in this sampled frame (all people)


@dataclass
class FaceTrack:
    """One person followed across the sampled frames."""

    track_id: int
    faces_bgr: list[np.ndarray] = field(default_factory=list)  # aligned 224x224 BGR
    frame_meta: list[FrameFace] = field(default_factory=list)  # frames where the person is present

    @property
    def n_faces(self) -> int:
        return len(self.faces_bgr)


@dataclass
class ExtractionResult:
    faces_bgr: list[np.ndarray]          # primary track, aligned 224x224 BGR uint8
    frame_meta: list[FrameFace]          # every sampled frame (primary track view)
    video: dict = field(default_factory=dict)
    detector_name: str = "none"
    n_sampled: int = 0
    n_faces: int = 0                     # sampled frames in which the primary face was found
    tracks: list[FaceTrack] = field(default_factory=list)
    max_faces_in_frame: int = 0
    warnings: list[str] = field(default_factory=list)


def _noop_progress(stage: str, pct: float, message: str) -> None:
    return None


# ---------------------------------------------------------------------------
# Face detectors
# ---------------------------------------------------------------------------

Detection = tuple[list[float], Optional[np.ndarray], float]  # xywh, landmarks, confidence


class FaceDetector:
    """YuNet (preferred) -> optional facenet-pytorch MTCNN -> Haar cascade."""

    def __init__(self) -> None:
        self.name = "haar"
        self._yunet = None
        self._haar = None
        self._mtcnn = None
        self._load()

    def _load(self) -> None:
        if YUNET_PATH.exists():
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(YUNET_PATH), "", (320, 320), 0.65, 0.3, 5000
                )
                self.name = "yunet"
                logger.info("Face detector: YuNet")
                return
            except Exception as exc:
                logger.warning("YuNet failed (%s); falling back", exc)

        try:
            from facenet_pytorch import MTCNN  # type: ignore

            self._mtcnn = MTCNN(keep_all=True, device="cpu", post_process=False)
            self.name = "mtcnn"
            logger.info("Face detector: MTCNN")
            return
        except Exception:
            pass

        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(str(cascade))
        self.name = "haar"
        logger.info("Face detector: Haar cascade")

    def detect_all(self, bgr: np.ndarray) -> list[Detection]:
        """Every face in the frame, largest first."""
        h, w = bgr.shape[:2]
        found: list[Detection] = []
        if self._yunet is not None:
            try:
                self._yunet.setInputSize((w, h))
                _retval, faces = self._yunet.detect(bgr)
            except cv2.error:
                faces = None
            if faces is not None:
                for f in faces:
                    box = [float(f[0]), float(f[1]), float(f[2]), float(f[3])]
                    lms = np.array(f[4:14], dtype=np.float32).reshape(5, 2)
                    found.append((box, lms, float(f[14]) if len(f) > 14 else 0.9))
        elif self._mtcnn is not None:
            import torch

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                boxes, probs, landmarks = self._mtcnn.detect(rgb, landmarks=True)
            if boxes is not None:
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    lms = landmarks[i].astype(np.float32) if landmarks is not None else None
                    found.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], lms, float(probs[i])))
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            hits = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            for x, y, bw, bh in (hits if hits is not None else []):
                found.append(([float(x), float(y), float(bw), float(bh)], None, 0.75))
        found.sort(key=lambda d: d[0][2] * d[0][3], reverse=True)
        return found

    def detect(self, bgr: np.ndarray) -> tuple[Optional[list[float]], Optional[np.ndarray], float]:
        """Primary (largest) face only. Kept for callers that want one face."""
        found = self.detect_all(bgr)
        if not found:
            return None, None, 0.0
        return found[0]


_DETECTOR: Optional[FaceDetector] = None


def get_detector() -> FaceDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = FaceDetector()
    return _DETECTOR


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _clip_box(box: list[float], w: int, h: int, expand: float = 1.35) -> tuple[int, int, int, int]:
    x, y, bw, bh = box
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(bw, bh) * expand
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0, 0, w, h
    return x1, y1, x2, y2


def align_face(bgr: np.ndarray, box: list[float], landmarks: Optional[np.ndarray], out: int = FACE_SIZE) -> np.ndarray:
    """Rotate so the eyes sit on a horizontal line, then crop/resize to `out`."""
    h, w = bgr.shape[:2]
    if landmarks is not None and landmarks.shape[0] >= 2:
        # YuNet: right eye, left eye, nose, right mouth, left mouth.
        right_eye, left_eye = landmarks[0], landmarks[1]
        dy = left_eye[1] - right_eye[1]
        dx = left_eye[0] - right_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        center = (float(w / 2.0), float(h / 2.0))
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(bgr, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
        homo = np.hstack([landmarks.astype(np.float32), ones])
        rotated_lms = (rot @ homo.T).T
        rx, ry, rw, rh = box
        new_box = [float(rx), float(ry), float(rw), float(rh)]
        mid = (rotated_lms[0] + rotated_lms[1]) / 2.0
        new_box[0] = float(mid[0] - rw / 2.0)
        new_box[1] = float(mid[1] - rh * 0.38)
        x1, y1, x2, y2 = _clip_box(new_box, w, h, expand=1.45)
        crop = rotated[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = _clip_box(box, w, h, expand=1.40)
        crop = bgr[y1:y2, x1:x2]

    if crop.size == 0:
        crop = bgr
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA)


def _jpeg_thumb(bgr: np.ndarray, size: int = 96, quality: int = 60) -> bytes:
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def laplacian_var(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def probe_video(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("The file could not be opened as a video (it may be corrupted).")
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    meta = {
        "fps": raw_fps,
        "frame_count": frame_count if 0 < frame_count < 10**9 else 0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC) or 0),
        "fps_estimated": False,
    }
    cap.release()
    if not math.isfinite(raw_fps) or raw_fps < 0.1 or raw_fps > 1000:
        meta["fps"] = 30.0  # container reports no usable rate: assume 30 fps
        meta["fps_estimated"] = True
    meta["duration"] = (meta["frame_count"] / meta["fps"]) if meta["frame_count"] else 0.0
    return meta


def _count_frames(path: str, limit: int) -> int:
    """Count decodable frames cheaply (grab() skips colour conversion)."""
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.grab():
        n += 1
        if n > limit:
            break
    cap.release()
    return n


def sample_frames(path: str | Path, n: int = SEQ_LEN) -> tuple[list[np.ndarray], list[int], list[float], dict]:
    """Uniformly sample up to ``n`` frames. Returns frames, indices, timestamps, meta."""
    path = str(path)
    meta = probe_video(path)
    validate_video_meta(meta)
    fps = meta["fps"]
    total = meta["frame_count"]

    if total <= 0:
        # Container does not declare a frame count: count first, never buffer every frame.
        total = _count_frames(path, MAX_DECODE_FRAMES)
        if total > MAX_DECODE_FRAMES:
            raise RuntimeError("Video has too many frames to process")
        if total == 0:
            raise RuntimeError("Video contained no decodable frames")
        meta["frame_count"] = total
        meta["duration"] = total / fps
        validate_video_meta(meta)

    picks = [int(i) for i in np.unique(np.linspace(0, max(total - 1, 0), num=min(n, total), dtype=int))]
    frames: list[np.ndarray] = []
    indices: list[int] = []
    times: list[float] = []

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("The file could not be opened as a video (it may be corrupted).")
    seek_ok = True
    for i in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok or frame is None:
            seek_ok = False
            break
        frames.append(frame)
        indices.append(i)
        times.append(i / fps)
    cap.release()
    if seek_ok and frames:
        return frames, indices, times, meta

    # Seeking is unreliable for some codecs: decode sequentially, keep only wanted frames.
    cap = cv2.VideoCapture(path)
    wanted = set(picks)
    frames, indices, times = [], [], []
    pos = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if pos in wanted:
            frames.append(frame)
            indices.append(pos)
            times.append(pos / fps)
            if len(frames) == len(wanted):
                break
        pos += 1
    cap.release()
    if not frames:
        raise RuntimeError("Video contained no decodable frames")
    return frames, indices, times, meta


# ---------------------------------------------------------------------------
# Face sequence extraction
# ---------------------------------------------------------------------------

@dataclass
class _Det:
    frame_k: int
    box: list[float]
    score: float
    aligned: np.ndarray


@dataclass
class _Track:
    dets: list[_Det] = field(default_factory=list)
    center: tuple[float, float] = (0.0, 0.0)
    size: float = 1.0


def _center(box: list[float]) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _link_tracks(per_frame: list[list[tuple[list[float], float, np.ndarray]]]) -> list[_Track]:
    """Greedy nearest-centre linking of per-frame detections into person tracks."""
    tracks: list[_Track] = []
    for k, dets in enumerate(per_frame):
        taken: set[int] = set()
        for box, score, aligned in dets:  # largest face first
            cx, cy = _center(box)
            size = max(box[2], box[3])
            best, best_d = None, None
            for ti, tr in enumerate(tracks):
                if ti in taken:
                    continue
                d = math.hypot(cx - tr.center[0], cy - tr.center[1])
                if d <= 1.6 * max(size, tr.size) and (best_d is None or d < best_d):
                    best, best_d = ti, d
            if best is None:
                tracks.append(_Track())
                best = len(tracks) - 1
            taken.add(best)
            tr = tracks[best]
            tr.dets.append(_Det(k, box, score, aligned))
            tr.center = (cx, cy)
            tr.size = size
    return tracks


def extract_face_sequence(
    path: str | Path,
    n_frames: int = SEQ_LEN,
    progress: ProgressFn = _noop_progress,
) -> ExtractionResult:
    """Full CV pipeline: sample -> detect all faces -> track -> align 224x224 crops."""
    progress("sampling_frames", 0.08, "Opening container and sampling frames")
    frames, indices, times, meta = sample_frames(path, n=n_frames)
    detector = get_detector()
    progress("detecting_faces", 0.22, f"Detecting faces with {detector.name.upper()}")

    warnings: list[str] = []
    if meta.get("fps_estimated"):
        warnings.append("The video reports no usable frame rate; 30 fps was assumed for timestamps.")

    n = len(frames)
    per_frame: list[list[tuple[list[float], float, np.ndarray]]] = []
    face_counts: list[int] = []
    for k, frame in enumerate(frames):
        detections = detector.detect_all(frame)
        face_counts.append(len(detections))
        per_frame.append(
            [
                (box, score, align_face(frame, box, lms, out=FACE_SIZE))
                for box, lms, score in detections[:6]
            ]
        )
        if k % 4 == 0:
            progress("detecting_faces", 0.22 + 0.38 * ((k + 1) / max(n, 1)), f"Frame {k + 1}/{n}")

    raw_tracks = _link_tracks(per_frame)
    raw_tracks.sort(key=lambda t: (len(t.dets), sum(d.box[2] * d.box[3] for d in t.dets)), reverse=True)
    raw_tracks = [t for t in raw_tracks if t.dets][:MAX_TRACKS]

    def _meta_for(d: _Det) -> FrameFace:
        return FrameFace(
            index=indices[d.frame_k],
            timestamp=round(times[d.frame_k], 3),
            has_face=True,
            box=[round(v, 1) for v in d.box],
            score=round(float(d.score), 3),
            laplacian_var=round(laplacian_var(d.aligned), 2),
            thumb_jpeg=_jpeg_thumb(d.aligned),
            n_faces=face_counts[d.frame_k],
        )

    tracks: list[FaceTrack] = []
    for tid, rt in enumerate(raw_tracks):
        tracks.append(
            FaceTrack(
                track_id=tid,
                faces_bgr=[d.aligned for d in rt.dets],
                frame_meta=[_meta_for(d) for d in rt.dets],
            )
        )

    # Primary-track view across *all* sampled frames (frames without the face are kept).
    primary_by_frame = {indices[d.frame_k]: m for d, m in zip(raw_tracks[0].dets, tracks[0].frame_meta)} if tracks else {}
    frame_meta = [
        primary_by_frame.get(
            idx,
            FrameFace(index=idx, timestamp=round(ts, 3), has_face=False, n_faces=face_counts[k]),
        )
        for k, (idx, ts) in enumerate(zip(indices, times))
    ]

    max_faces = max(face_counts) if face_counts else 0
    if max_faces > 1:
        warnings.append(
            f"Up to {max_faces} faces appear in one frame; the {len(tracks)} most visible "
            "person(s) were analysed separately."
        )
    result = ExtractionResult(
        faces_bgr=tracks[0].faces_bgr if tracks else [],
        frame_meta=frame_meta,
        video=meta,
        detector_name=detector.name,
        n_sampled=n,
        n_faces=tracks[0].n_faces if tracks else 0,
        tracks=tracks,
        max_faces_in_frame=max_faces,
        warnings=warnings,
    )
    progress("detecting_faces", 0.60, f"Aligned {result.n_faces}/{n} faces")
    return result


def faces_to_tensor(faces_bgr: list[np.ndarray]):
    """Convert BGR uint8 crops to a torch tensor ``[1, T, 3, 224, 224]``."""
    import torch

    if not faces_bgr:
        raise RuntimeError("No faces to tensorise")
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    seq = []
    for bgr in faces_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - mean) / std
        seq.append(np.transpose(norm, (2, 0, 1)))
    arr = np.stack(seq, axis=0)  # [T, 3, H, W]
    return torch.from_numpy(arr).unsqueeze(0)  # [1, T, 3, H, W]


def thumb_to_data_uri(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def thumbs_to_data_uris(frame_meta: list[FrameFace], limit: int = 24) -> list[str]:
    uris: list[str] = []
    for item in frame_meta:
        if item.thumb_jpeg and len(uris) < limit:
            uris.append(thumb_to_data_uri(item.thumb_jpeg))
    return uris
