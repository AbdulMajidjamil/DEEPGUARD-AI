# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Abdul Majid Jamil
"""Dataset scanning, leakage detection and face-sequence caching for training / evaluation.

Expected layout::

    <root>/<split>/real/*.mp4      split in {train, val, test}
    <root>/<split>/fake/*.mp4
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from config import ALLOWED_VIDEO_EXT, FACE_SIZE, MIN_FACES, SEQ_LEN

logger = logging.getLogger("deepguard.dataprep")

LABELS = {"real": 0, "fake": 1}
_FFPP = re.compile(r"^(\d+)(?:_(\d+))?$")  # FaceForensics++: real "123", fake "123_456"


@dataclass
class Item:
    path: Path
    label: int
    split: str


def scan_split(root: Path, split: str) -> list[Item]:
    """List every supported video under ``<root>/<split>/{real,fake}``."""
    items: list[Item] = []
    for name, label in LABELS.items():
        folder = Path(root) / split / name
        if folder.is_dir():
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in ALLOWED_VIDEO_EXT:
                    items.append(Item(path, label, split))
    return items


def file_fingerprint(path: Path) -> str:
    """Cheap content fingerprint (size + first and last MiB) for duplicate detection."""
    size = path.stat().st_size
    digest = hashlib.sha1(str(size).encode())
    with path.open("rb") as fh:
        digest.update(fh.read(1 << 20))
        if size > (2 << 20):
            fh.seek(-(1 << 20), 2)
            digest.update(fh.read(1 << 20))
    return digest.hexdigest()


def identity_tokens(path: Path) -> set[str]:
    """Identities encoded in a FaceForensics++-style file name (``12`` or ``12_34``), else empty."""
    match = _FFPP.match(path.stem)
    return {g for g in match.groups() if g} if match else set()


def check_leakage(train: list[Item], other: list[Item], other_name: str = "val") -> dict[str, list[str]]:
    """Return ``{"errors": [...], "warnings": [...]}`` describing train/<other> contamination.

    errors   - byte-identical (fingerprint) duplicates across splits, and shared
               FaceForensics++ identities (the same person in train and evaluation data).
    warnings - the same file name in both splits (may be unrelated clips).
    """
    errors: list[str] = []
    warnings: list[str] = []
    train_fp: dict[str, Path] = {}
    for it in train:
        train_fp.setdefault(file_fingerprint(it.path), it.path)
    for it in other:
        match = train_fp.get(file_fingerprint(it.path))
        if match is not None:
            errors.append(f"duplicate content: {other_name}/{it.path.name} == train/{match.name}")
    train_ids: dict[str, str] = {}
    for it in train:
        for tok in identity_tokens(it.path):
            train_ids.setdefault(tok, it.path.name)
    for it in other:
        for tok in sorted(identity_tokens(it.path)):
            if tok in train_ids:
                errors.append(f"shared identity '{tok}': {other_name}/{it.path.name} vs train/{train_ids[tok]}")
                break
    names = {it.path.name for it in train}
    for it in other:
        if it.path.name in names:
            warnings.append(f"same file name in both splits: {it.path.name}")
    return {"errors": errors, "warnings": warnings}


def class_counts(items: list[Item]) -> dict[str, int]:
    return {"real": sum(1 for i in items if i.label == 0), "fake": sum(1 for i in items if i.label == 1)}


def cache_key(path: Path, seq_len: int) -> str:
    """Unique per file (full path, size, mtime) and extraction settings - never just the stem."""
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{seq_len}|{FACE_SIZE}"
    return hashlib.sha1(raw.encode()).hexdigest()[:24]


class VideoFaceDataset(Dataset):
    """Face-sequence tensors, cached on disk when ``cache_dir`` is given.

    ``prepare()`` extracts every video once. Clips that cannot be used (unreadable, or fewer
    than ``MIN_FACES`` aligned face frames) are dropped and recorded in ``skipped`` - they are
    never replaced by a zero tensor, which would silently teach the model that black frames
    carry a label. Sequences are stored un-padded; ``collate`` pads and returns true lengths.
    """

    def __init__(self, items: list[Item], cache_dir: Optional[Path] = None, seq_len: int = SEQ_LEN):
        self.items: list[Item] = list(items)
        self.seq_len = seq_len
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.skipped: list[dict[str, str]] = []
        self._mem: dict[str, torch.Tensor] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, item: Item) -> Optional[Path]:
        return self.cache_dir / f"{cache_key(item.path, self.seq_len)}.pt" if self.cache_dir else None

    def _extract(self, item: Item) -> torch.Tensor:
        from pipeline import extract_face_sequence, faces_to_tensor

        extraction = extract_face_sequence(item.path, n_frames=self.seq_len)
        if extraction.n_faces < MIN_FACES:
            raise ValueError(f"only {extraction.n_faces} face frame(s) found (need {MIN_FACES})")
        return faces_to_tensor(extraction.faces_bgr).squeeze(0)[: self.seq_len].contiguous()

    def prepare(self, progress: Optional[Callable] = None) -> "VideoFaceDataset":
        usable: list[Item] = []
        iterator = progress(self.items) if progress else self.items
        for item in iterator:
            cache = self._cache_path(item)
            if cache is not None and cache.exists():
                usable.append(item)
                continue
            try:
                tensor = self._extract(item)
            except Exception as exc:  # unreadable video / no faces: record and move on
                self.skipped.append({"file": item.path.name, "reason": str(exc)[:160]})
                logger.warning("skipping %s: %s", item.path.name, exc)
                continue
            if cache is not None:
                torch.save(tensor, cache)
            else:
                self._mem[str(item.path)] = tensor
            usable.append(item)
        self.items = usable
        return self

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        cache = self._cache_path(item)
        if cache is not None:
            tensor = torch.load(cache, map_location="cpu", weights_only=True)
        else:
            tensor = self._mem[str(item.path)]
        return tensor, torch.tensor(float(item.label))


def collate(batch):
    """Pad variable-length sequences by repeating the last frame; return true lengths."""
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    t = int(lengths.max())
    padded = [x if x.shape[0] == t else torch.cat([x, x[-1:].repeat(t - x.shape[0], 1, 1, 1)], 0) for x in xs]
    return torch.stack(padded, 0), torch.stack(ys, 0), lengths


def write_json(path: Path, data: dict) -> None:
    """Write JSON; NaN values (undefined metrics) become null."""
    def clean(o):
        if isinstance(o, float) and o != o:
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2, default=str), encoding="utf-8")
