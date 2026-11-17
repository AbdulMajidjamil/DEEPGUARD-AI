# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Abdul Majid Jamil
"""Safe, validated loading of DEEPGUARD AI model checkpoints.

A checkpoint is accepted only when

* it deserialises with ``torch.load(weights_only=True)`` (no pickle code
  execution) - unless the operator explicitly opts in with
  ``DEEPGUARD_ALLOW_PICKLE=1``;
* it is a dict holding a ``state_dict`` (or is itself a flat state dict);
* every value is a tensor and contains no NaN / Inf;
* the optional ``backbone`` metadata matches the backbone being built;
* ``load_state_dict(strict=True)`` succeeds, i.e. the keys *and* shapes match.

Anything else raises :class:`CheckpointError` with a short, path-free message,
so the engine can report "no usable weights" instead of silently pretending a
partially-loaded model is trained.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("deepguard.checkpoint")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, unsafe or incompatible."""


def _sha256_prefix(path: Path, n: int = 12) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:n]


def _read_blob(path: Path, allow_pickle: bool) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_pickle:
            raise CheckpointError(
                "Checkpoint is not a safe tensor-only file. Re-export it with "
                "torch.save(state_dict) or set DEEPGUARD_ALLOW_PICKLE=1 for a checkpoint you trust."
            ) from exc
        logger.warning("weights_only load failed; unpickling because DEEPGUARD_ALLOW_PICKLE is set")
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc2:  # pragma: no cover - depends on file
            raise CheckpointError("Checkpoint could not be read.") from exc2


def _summarise_keys(keys: list[str], limit: int = 3) -> str:
    shown = ", ".join(keys[:limit])
    more = f" (+{len(keys) - limit} more)" if len(keys) > limit else ""
    return shown + more


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    *,
    backbone: str,
    allow_pickle: bool = False,
) -> dict[str, Any]:
    """Load ``path`` into ``model`` with strict validation.

    Returns metadata (epoch, validation metrics, sha256 prefix...). Raises
    :class:`CheckpointError` on any problem; the model is left untouched when
    validation fails before ``load_state_dict`` is reached.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError("Checkpoint file not found.")

    blob = _read_blob(path, allow_pickle)
    meta_src: dict[str, Any] = blob if isinstance(blob, dict) else {}
    state = blob.get("state_dict") if isinstance(blob, dict) and "state_dict" in blob else blob
    if not isinstance(state, dict) or not state:
        raise CheckpointError("Checkpoint does not contain a state_dict.")

    bad = [k for k, v in state.items() if not isinstance(v, torch.Tensor)]
    if bad:
        raise CheckpointError(f"state_dict has non-tensor entries: {_summarise_keys(bad)}")
    for key, tensor in state.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise CheckpointError(f"state_dict contains NaN/Inf values (e.g. {key}).")

    declared = meta_src.get("backbone")
    if isinstance(declared, str) and declared.lower().replace("-", "_") != backbone.lower().replace("-", "_"):
        raise CheckpointError(
            f"Checkpoint was trained with backbone '{declared}' but the server is configured "
            f"for '{backbone}'. Set BACKBONE={declared} or supply a matching checkpoint."
        )

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    if missing or unexpected:
        raise CheckpointError(
            "Checkpoint does not match the model architecture "
            f"(missing: {_summarise_keys(missing) or 'none'}; unexpected: {_summarise_keys(unexpected) or 'none'})."
        )
    own = model.state_dict()
    wrong = [k for k in state if tuple(state[k].shape) != tuple(own[k].shape)]
    if wrong:
        raise CheckpointError(f"Checkpoint tensor shapes differ from the model: {_summarise_keys(wrong)}")

    model.load_state_dict(state, strict=True)

    val = meta_src.get("val")
    info = {
        "sha256": _sha256_prefix(path),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
        "epoch": meta_src.get("epoch") if isinstance(meta_src.get("epoch"), int) else None,
        "backbone": declared if isinstance(declared, str) else backbone,
        "val": val if isinstance(val, dict) else None,
        "seq_len": meta_src.get("seq_len") if isinstance(meta_src.get("seq_len"), int) else None,
        "trained_by_deepguard": bool(meta_src.get("deepguard_version")),
    }
    return info
