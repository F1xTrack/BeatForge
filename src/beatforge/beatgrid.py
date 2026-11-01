"""Normalized beat/downbeat contract for optional Beat This integration."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class BeatGrid:
    beats: tuple[float, ...]
    downbeats: tuple[float, ...]
    bpm: float | None
    source: str
    confidence: float | None = None


def _times(values: Iterable[Any], label: str) -> tuple[float, ...]:
    result = []
    for index, value in enumerate(values):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{label}[{index}] must be finite and non-negative")
        result.append(float(value))
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError(f"{label} must be strictly increasing")
    return tuple(result)


def normalize_beat_grid(payload: dict[str, Any], *, source: str = "beat-tracker") -> BeatGrid:
    """Normalize common tracker output without inventing missing BPM/confidence."""
    if not isinstance(payload, dict):
        raise ValueError("beat-grid payload must be an object")
    beats = _times(payload.get("beats", ()), "beats")
    downbeats = _times(payload.get("downbeats", ()), "downbeats")
    if any(value not in set(beats) for value in downbeats):
        raise ValueError("downbeats must be members of beats")
    bpm = payload.get("bpm")
    if bpm is not None and (not isinstance(bpm, (int, float)) or isinstance(bpm, bool) or not math.isfinite(bpm) or bpm <= 0):
        raise ValueError("bpm must be finite and positive when present")
    confidence = payload.get("confidence")
    if confidence is not None and (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                                   or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("confidence must be in [0, 1] when present")
    return BeatGrid(beats, downbeats, float(bpm) if bpm is not None else None,
                    source, float(confidence) if confidence is not None else None)
