"""Human-reviewable musical plan kept separate from generated map JSON."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any

from .analysis import SongAnalysis
from .beatgrid import BeatGrid


NPS_TARGET_RELATIVE_TOLERANCE = 0.15


@dataclass(frozen=True, slots=True)
class Section:
    start_beat: float
    end_beat: float
    label: str
    energy: float = 0.5
    density: float = 2.5
    locked: bool = False


@dataclass(slots=True)
class TrackPlan:
    bpm: float
    duration_seconds: float
    target_nps: float
    primary_layer: str = "mixed"
    style: str = "expressive-readable"
    scene_palette: dict[str, list[int]] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrackPlan":
        if not isinstance(payload, dict):
            raise ValueError("track plan must be a JSON object")
        try:
            default_density = float(payload["target_nps"])
            sections = [Section(
                float(item["start_beat"]), float(item["end_beat"]), str(item["label"]),
                float(item.get("energy", 0.5)), float(item.get("density", default_density)),
                bool(item.get("locked", False)),
            ) for item in payload.get("sections", [])]
            plan = cls(
                float(payload["bpm"]), float(payload["duration_seconds"]),
                default_density, str(payload.get("primary_layer", "mixed")),
                str(payload.get("style", "expressive-readable")),
                dict(payload.get("scene_palette") or {}), sections,
                str(payload.get("notes", "")),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid track plan: {exc}") from exc
        errors = validate_plan(plan)
        if errors:
            raise ValueError("; ".join(errors))
        return plan

    @classmethod
    def from_json(cls, text: str) -> "TrackPlan":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid track plan JSON: {exc}") from exc
        return cls.from_dict(payload)


def load_plan(path: str | Any) -> TrackPlan:
    """Load and validate a human-edited plan without model inference."""
    from pathlib import Path
    return TrackPlan.from_json(Path(path).read_text(encoding="utf-8"))


def validate_plan(plan: TrackPlan) -> list[str]:
    errors: list[str] = []
    for name, value in (("bpm", plan.bpm), ("duration_seconds", plan.duration_seconds),
                        ("target_nps", plan.target_nps)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            errors.append(f"{name} must be finite and positive")
    previous = 0.0
    for index, section in enumerate(plan.sections):
        if not math.isfinite(section.start_beat) or not math.isfinite(section.end_beat) or section.end_beat <= section.start_beat:
            errors.append(f"section {index} has invalid beat interval")
        if index and section.start_beat < previous:
            errors.append(f"section {index} overlaps or is out of order")
        if not 0 <= section.energy <= 1:
            errors.append(f"section {index} energy must be in [0, 1]")
        if not math.isfinite(section.density) or section.density < 0:
            errors.append(f"section {index} density must be non-negative")
        previous = section.end_beat
    return errors


def _section_signals(analysis: SongAnalysis, spans: list[tuple[float, float]],
                     bpm: float, target_nps: float,
                     label_prefix: str) -> list[Section]:
    """Measure overlap-weighted RMS per section and preserve the global NPS mean."""
    raw_rms: list[float] = []
    for start_beat, end_beat in spans:
        start_seconds = start_beat * 60.0 / bpm
        end_seconds = min(analysis.duration_seconds, end_beat * 60.0 / bpm)
        weighted_power = 0.0
        covered_seconds = 0.0
        for window in analysis.windows:
            overlap = max(0.0, min(end_seconds, window.end_seconds)
                          - max(start_seconds, window.start_seconds))
            if overlap:
                weighted_power += overlap * window.rms * window.rms
                covered_seconds += overlap
        raw_rms.append(math.sqrt(weighted_power / covered_seconds)
                       if covered_seconds else 0.0)

    peak = max(raw_rms, default=0.0)
    normalized = [value / peak if peak else 0.0 for value in raw_rms]
    # Keep quiet sections playable while shifting more of the requested density
    # toward louder sections. Duration-weighted normalization keeps the track's
    # average section target equal to target_nps.
    activity = [0.35 + math.sqrt(value) for value in normalized]
    durations = [(end - start) * 60.0 / bpm for start, end in spans]
    total_duration = math.fsum(durations)
    mean_activity = (math.fsum(value * duration
                               for value, duration in zip(activity, durations))
                     / total_duration if total_duration else 1.0)
    return [Section(start, end, f"{label_prefix}-{index:02d}",
                    round(normalized[index - 1], 4),
                    round(target_nps * activity[index - 1] / mean_activity, 4))
            for index, (start, end) in enumerate(spans, 1)]


def initial_plan_from_analysis(analysis: SongAnalysis, *, bpm: float, target_nps: float,
                               section_beats: float = 16.0) -> TrackPlan:
    """Create a reviewable plan from local energy windows without model inference."""
    if bpm <= 0 or target_nps <= 0 or section_beats <= 0:
        raise ValueError("bpm, target_nps and section_beats must be positive")
    final_beat = analysis.duration_seconds * bpm / 60.0
    if final_beat <= 0:
        raise ValueError("analysis contains no playable duration")
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start < final_beat:
        end = min(final_beat, start + section_beats)
        spans.append((start, end))
        start = end
    sections = _section_signals(analysis, spans, bpm, target_nps, "section")
    return TrackPlan(bpm, analysis.duration_seconds, target_nps, sections=sections)


def initial_plan_from_beat_grid(analysis: SongAnalysis, grid: BeatGrid, *, target_nps: float,
                                bars_per_section: int = 4) -> TrackPlan:
    """Create sections from tracker downbeats while preserving uncertainty.

    BeatGrid timestamps are seconds. A missing BPM is an explicit error rather
    than an inferred value; missing downbeats fall back to the RMS-window plan.
    """
    if grid.bpm is None:
        raise ValueError("BeatGrid BPM is required to build a beat-aligned plan")
    if target_nps <= 0 or bars_per_section <= 0:
        raise ValueError("target_nps and bars_per_section must be positive")
    downbeats = [float(value) for value in grid.downbeats
                 if 0 <= float(value) < analysis.duration_seconds]
    if len(downbeats) < 2:
        return initial_plan_from_analysis(analysis, bpm=grid.bpm, target_nps=target_nps)
    beat_per_second = grid.bpm / 60.0
    final_beat = analysis.duration_seconds * grid.bpm / 60.0
    boundaries = [0.0]
    for index in range(bars_per_section, len(downbeats), bars_per_section):
        beat = downbeats[index] * beat_per_second
        if beat > boundaries[-1] and beat < final_beat:
            boundaries.append(beat)
    if final_beat > boundaries[-1]:
        boundaries.append(final_beat)
    spans = list(zip(boundaries, boundaries[1:]))
    sections = _section_signals(analysis, spans, grid.bpm, target_nps,
                                "grid-section")
    confidence = "unknown" if grid.confidence is None else f"{grid.confidence:.3f}"
    return TrackPlan(grid.bpm, analysis.duration_seconds, target_nps,
                     sections=sections,
                     notes=f"BeatGrid source={grid.source}; confidence={confidence}")
