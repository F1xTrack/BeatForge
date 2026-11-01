"""Model-blind preflight diagnostics, deliberately not a gameplay rating."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import math
from collections import Counter
import re
from typing import Any

from .contracts import MapProgram
from .mapping_extensions import decode_position, direction_bucket
from .planning import NPS_TARGET_RELATIVE_TOLERANCE, TrackPlan
from .validation import validate_program


@dataclass(frozen=True, slots=True)
class QualityReport:
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    note_count: int
    chain_slices: int
    actual_nps: float
    occupied_sections: int
    total_sections: int
    direction_profiles: dict[str, dict[str, Any]]
    chain_opportunities: dict[str, Any]
    chain_usage: dict[str, Any]
    pattern_alignment: dict[str, Any]
    palette_contrast: dict[str, Any]
    rhythm_alignment: dict[str, Any]
    lighting_coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_AUDIO_GRID_ROW = re.compile(
    r"^beats\s+([0-9.]+)-([0-9.]+)\s+low=([0-9]+)\s+mid=([0-9]+)\s+"
    r"high=([0-9]+)\s+attack=([0-9]+)\s*$"
)
PATTERN_SUPPORT_WARNING_THRESHOLD = 0.20
CHAIN_OPPORTUNITY_MIN_HEADS = 3
CHAIN_OPPORTUNITY_MAX_GAP_BEATS = 0.25
CHAIN_SLICE_SHARE_REVIEW_THRESHOLD = 0.35


def _chain_opportunities(program: MapProgram) -> dict[str, Any]:
    """Find rapid same-hand note streams that could contain short chain bursts."""
    by_color: dict[int, list[float]] = {0: [], 1: []}
    for note in program.notes:
        if note.color in by_color:
            by_color[note.color].append(float(note.beat))

    runs: list[dict[str, Any]] = []
    for color, beats in by_color.items():
        ordered = sorted(beats)
        current: list[float] = []
        for beat in ordered:
            if not current:
                current = [beat]
                continue
            gap = beat - current[-1]
            if 1e-6 < gap <= CHAIN_OPPORTUNITY_MAX_GAP_BEATS + 1e-6:
                current.append(beat)
            else:
                if len(current) >= CHAIN_OPPORTUNITY_MIN_HEADS:
                    gaps = [right - left for left, right in zip(current, current[1:])]
                    runs.append({
                        "color": color,
                        "start_beat": round(current[0], 4),
                        "end_beat": round(current[-1], 4),
                        "head_count": len(current),
                        "max_gap_beats": round(max(gaps), 4),
                    })
                current = [beat]
        if len(current) >= CHAIN_OPPORTUNITY_MIN_HEADS:
            gaps = [right - left for left, right in zip(current, current[1:])]
            runs.append({
                "color": color,
                "start_beat": round(current[0], 4),
                "end_beat": round(current[-1], 4),
                "head_count": len(current),
                "max_gap_beats": round(max(gaps), 4),
            })

    runs.sort(key=lambda item: (item["start_beat"], item["color"]))
    # Timing and hand color are too weak to require conversion, but an explicit
    # local review can use the synchronized SG to decide whether a chain fits.
    review_recommended = bool(runs)
    return {
        "available": True,
        "minimum_heads": CHAIN_OPPORTUNITY_MIN_HEADS,
        "maximum_gap_beats": CHAIN_OPPORTUNITY_MAX_GAP_BEATS,
        "chains_present": len(program.chains),
        "eligible_run_count": len(runs),
        "unrepresented_run_count": len(runs),
        "eligible_runs": runs[:12],
        "truncated": len(runs) > 12,
        "warning": review_recommended,
        "review_recommended": review_recommended,
        "actionable_repair": review_recommended,
        "limitations": [
            "Timing and hand color identify candidates only; SG, swing feasibility, and phrase context still require review before conversion.",
            "Long rapid streams must be split into short phrase-local chain bursts, never emitted as one long chain.",
        ],
    }


def _chain_usage(program: MapProgram, plan: TrackPlan) -> dict[str, Any]:
    """Flag unusually dominant chain-slice share without imposing a quota."""
    chain_slices = sum(int(chain.slices) for chain in program.chains)
    total_hits = len(program.notes) + 2 * len(program.arcs) + chain_slices
    share = chain_slices / total_hits if total_hits else 0.0
    spans = sorted(max(0.0, float(chain.tail_beat) - float(chain.beat))
                   for chain in program.chains)
    median_span = (spans[(len(spans) - 1) // 2] if spans else None)
    window_beats = 4.0 * plan.bpm / 60.0
    windows: dict[int, dict[str, int]] = {}

    def bucket(beat: float) -> dict[str, int]:
        index = max(0, int(float(beat) // window_beats))
        return windows.setdefault(index, {"normal_notes": 0, "arc_heads": 0,
                                          "arc_tails": 0,
                                          "chain_heads": 0, "chain_slices": 0})

    for note in program.notes:
        bucket(note.beat)["normal_notes"] += 1
    for arc in program.arcs:
        bucket(arc.beat)["arc_heads"] += 1
        bucket(arc.tail_beat)["arc_tails"] += 1
    for chain in program.chains:
        bucket(chain.beat)["chain_heads"] += 1
        start, end = float(chain.beat), float(chain.tail_beat)
        for slice_index in range(int(chain.slices)):
            position = (start + (end - start) * slice_index / max(1, int(chain.slices) - 1))
            bucket(position)["chain_slices"] += 1
    window_profiles = []
    for index, counts in sorted(windows.items()):
        hits = (counts["normal_notes"] + counts["arc_heads"]
                + counts["arc_tails"] + counts["chain_slices"])
        window_profiles.append({
            "window": index,
            "beats": [round(index * window_beats, 4),
                      round((index + 1) * window_beats, 4)],
            **counts,
            "hit_equivalents": hits,
            "chain_slice_share": round(counts["chain_slices"] / hits if hits else 0.0, 6),
        })
    repair_targets = [item for item in window_profiles
                      if item["chain_slice_share"] > CHAIN_SLICE_SHARE_REVIEW_THRESHOLD]
    repair_targets.sort(key=lambda item: (-item["chain_slice_share"],
                                          -item["chain_slices"], item["window"]))
    return {
        "available": True,
        "chains_present": len(program.chains),
        "chain_slices": chain_slices,
        "total_hit_equivalents": total_hits,
        "chain_slice_share": round(share, 6),
        "mean_span_beats": round(sum(spans) / len(spans), 6) if spans else None,
        "median_span_beats": round(median_span, 6) if median_span is not None else None,
        "warning_threshold": CHAIN_SLICE_SHARE_REVIEW_THRESHOLD,
        "warning": total_hits >= 100 and share > CHAIN_SLICE_SHARE_REVIEW_THRESHOLD,
        "window_profiles": window_profiles,
        "repair_targets": repair_targets[:6],
        "truncated_repair_targets": len(repair_targets) > 6,
        "limitations": [
            "High slice share is a review signal, not proof that chains are musically wrong.",
            "Slice beats are linearly interpolated over each chain's head-tail span for coarse windows.",
            "Fast-burst eligibility, SG alignment, swing feasibility and phrase context need review.",
        ],
    }


SCENE_PALETTE_MIN_RGB_DISTANCE = 0.30
RHYTHM_AUDIO_ACTIVITY_CV_MIN = 0.75
RHYTHM_MAP_DENSITY_CV_MAX = 0.20
RHYTHM_DENSITY_CORRELATION_MAX = 0.10


def _palette_contrast(program: MapProgram) -> dict[str, Any]:
    """Flag environment pairs that are too close to read as distinct colors."""
    palette = program.scene_palette
    if not isinstance(palette, dict) or not {
        "environment_left", "environment_right",
    }.issubset(palette):
        return {"available": False, "reason": "environment palette unavailable"}
    left, right = palette["environment_left"], palette["environment_right"]
    if any(not isinstance(color, (list, tuple)) or len(color) != 3
           or any(not isinstance(channel, int) or isinstance(channel, bool)
                  or not 0 <= channel <= 255 for channel in color)
           for color in (left, right)):
        return {"available": False, "reason": "environment palette invalid"}
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right))) / (255 * math.sqrt(3))
    return {
        "available": True,
        "normalized_rgb_distance": round(distance, 6),
        "minimum_distance": SCENE_PALETTE_MIN_RGB_DISTANCE,
        "warning": distance < SCENE_PALETTE_MIN_RGB_DISTANCE,
        "limitations": [
            "Normalized sRGB distance is a coarse contrast proxy; it does not model environment lighting or visual adaptation.",
        ],
    }


LIGHTING_SECTION_COVERAGE_MIN = 0.50


def _lighting_coverage(program: MapProgram, plan: TrackPlan) -> dict[str, Any]:
    """Measure light-cue coverage only across sections with playable objects."""
    playable_beats = ([float(note.beat) for note in program.notes]
                      + [beat for arc in program.arcs
                         for beat in (float(arc.beat), float(arc.tail_beat))]
                      + [float(chain.beat) for chain in program.chains])
    active_sections = []
    covered_sections = []
    events_by_section: dict[str, int] = {}
    for index, section in enumerate(plan.sections):
        in_section = any(
            float(section.start_beat) <= beat < float(section.end_beat)
            or (index == len(plan.sections) - 1
                and math.isclose(beat, float(section.end_beat), abs_tol=1e-6))
            for beat in playable_beats
        )
        if not in_section:
            continue
        active_sections.append(section.label)
        count = sum(
            float(section.start_beat) <= float(light.beat) < float(section.end_beat)
            or (index == len(plan.sections) - 1
                and math.isclose(float(light.beat), float(section.end_beat), abs_tol=1e-6))
            for light in program.lights
        )
        events_by_section[section.label] = count
        if count:
            covered_sections.append(section.label)
    total = len(active_sections)
    ratio = len(covered_sections) / total if total else None
    warning = bool(
        program.lights and total > 0 and ratio is not None
        and ratio < LIGHTING_SECTION_COVERAGE_MIN
    )
    return {
        "available": bool(plan.sections),
        "light_event_count": len(program.lights),
        "active_sections": active_sections,
        "covered_sections": covered_sections,
        "missing_sections": [label for label in active_sections
                             if label not in covered_sections],
        "events_by_section": events_by_section,
        "coverage_ratio": round(ratio, 6) if ratio is not None else None,
        "minimum_coverage_ratio": LIGHTING_SECTION_COVERAGE_MIN,
        "warning": warning,
        "limitations": [
            "Section coverage is a sparse-lighting review signal; cue timing and brightness still need musical/visual review.",
            "Lighting coverage does not prove compatibility with the selected environment or installed mods.",
        ],
    }


def _rhythm_alignment(program: MapProgram, plan: TrackPlan,
                      audio_features: str | None) -> dict[str, Any]:
    """Find flat map density across changing band energy and transient activity.

    Quantized band energy distinguishes sustained sound from silence when the
    attack proxy is low. Neither signal is an onset detector or a note chart.
    """
    if not isinstance(audio_features, str) or not audio_features.strip():
        return {"available": False, "reason": "audio feature grid unavailable"}
    windows = int(plan.duration_seconds // 4.0)
    if windows < 6:
        return {"available": False, "reason": "fewer than six complete four-second windows"}

    audio_attack_cells = [0] * windows
    strong_spectral_change_beats: list[list[float]] = [[] for _ in range(windows)]
    audio_cell_counts = [0] * windows
    audio_energy_squares = [0.0] * windows
    for line in audio_features.splitlines():
        match = _AUDIO_GRID_ROW.fullmatch(line.strip())
        if not match:
            continue
        start_beat = float(match.group(1))
        bands = [match.group(index) for index in (3, 4, 5)]
        attacks = match.group(6)
        if len({len(values) for values in (*bands, attacks)}) != 1:
            continue
        for offset, (low, mid, high, attack) in enumerate(zip(*bands, attacks)):
            beat = start_beat + 0.5 * offset
            window = int((beat * 60.0 / plan.bpm) // 4.0)
            if 0 <= window < windows:
                audio_cell_counts[window] += 1
                audio_energy_squares[window] += (
                    int(low) ** 2 + int(mid) ** 2 + int(high) ** 2
                ) / 3.0
                if int(attack) >= 8:
                    audio_attack_cells[window] += 1
                    strong_spectral_change_beats[window].append(round(beat, 3))

    map_hit_equivalents = [0] * windows

    def add_hit(beat: float, count: int = 1) -> None:
        window = int((float(beat) * 60.0 / plan.bpm) // 4.0)
        if 0 <= window < windows:
            map_hit_equivalents[window] += count

    for note in program.notes:
        add_hit(note.beat)
    for arc in program.arcs:
        add_hit(arc.beat)
        add_hit(arc.tail_beat)
    for chain in program.chains:
        slices = max(1, int(chain.slices))
        span = float(chain.tail_beat) - float(chain.beat)
        for index in range(slices):
            beat = float(chain.beat) + (span * index / (slices - 1) if slices > 1 else 0.0)
            add_hit(beat)

    import numpy as np

    audio_band_energy = [
        math.sqrt(audio_energy_squares[index] / audio_cell_counts[index])
        if audio_cell_counts[index] else 0.0
        for index in range(windows)
    ]
    audio_attack_ratio = [
        audio_attack_cells[index] / audio_cell_counts[index]
        if audio_cell_counts[index] else 0.0
        for index in range(windows)
    ]
    # Both components are scaled to 0..1. A sustained high-energy window with
    # no strong attacks therefore remains active instead of being labelled quiet.
    audio_activity = [
        (audio_band_energy[index] / 9.0 + audio_attack_ratio[index]) / 2.0
        for index in range(windows)
    ]
    audio = np.asarray(audio_activity, dtype=float)
    energy = np.asarray(audio_band_energy, dtype=float)
    attacks = np.asarray(audio_attack_ratio, dtype=float)
    mapped = np.asarray(map_hit_equivalents, dtype=float)
    map_mean = float(np.mean(mapped))
    energy_mean = float(np.mean(energy))
    attack_mean = float(np.mean(attacks))
    energy_cv = float(np.std(energy) / energy_mean) if energy_mean > 0 else 0.0
    attack_cv = float(np.std(attacks) / attack_mean) if attack_mean > 0 else 0.0
    audio_cv = max(energy_cv, attack_cv)
    map_cv = float(np.std(mapped) / map_mean) if map_mean > 0 else 0.0
    correlation = (float(np.corrcoef(audio, mapped)[0, 1])
                   if audio.std() > 1e-8 and mapped.std() > 1e-8 else None)
    energy_correlation = (float(np.corrcoef(energy, mapped)[0, 1])
                          if energy.std() > 1e-8 and mapped.std() > 1e-8 else None)
    attack_correlation = (float(np.corrcoef(attacks, mapped)[0, 1])
                          if attacks.std() > 1e-8 and mapped.std() > 1e-8 else None)
    warning = (
        audio_cv >= RHYTHM_AUDIO_ACTIVITY_CV_MIN
        and map_cv <= RHYTHM_MAP_DENSITY_CV_MAX
        and (correlation is None or correlation <= RHYTHM_DENSITY_CORRELATION_MAX)
    )
    # Repair against a handful of contrasting high/low audio windows rather
    # than emitting every window as a prompt target.
    order = sorted(range(windows), key=lambda index: (audio_activity[index], index))
    target_indices = sorted(set(order[:3] + order[-3:]))
    profiles = [{
        "window": index,
        "beats": [round(index * 4.0 * plan.bpm / 60.0, 2),
                  round((index + 1) * 4.0 * plan.bpm / 60.0, 2)],
        "map_hit_equivalents": map_hit_equivalents[index],
        "audio_band_energy": round(audio_band_energy[index], 4),
        "audio_strong_attack_cells": audio_attack_cells[index],
        "strong_spectral_change_beats": strong_spectral_change_beats[index],
        "audio_attack_ratio": round(audio_attack_ratio[index], 4),
        "audio_activity": round(audio_activity[index], 4),
        "audio_character": (
            "quiet" if audio_band_energy[index] < 2.0 and audio_attack_ratio[index] < 0.125
            else "sustained" if audio_band_energy[index] >= 4.0 and audio_attack_ratio[index] < 0.125
            else "transient" if audio_attack_ratio[index] >= 0.25
            else "mixed"
        ),
        "repair_target": index in target_indices,
    } for index in range(windows)]
    return {
        "available": True,
        "windows": windows,
        "attack_quantization_threshold": 8,
        "audio_strong_attack_cells": audio_attack_cells,
        "map_hit_equivalents": map_hit_equivalents,
        "audio_band_energy": [round(value, 4) for value in audio_band_energy],
        "audio_attack_ratio": [round(value, 4) for value in audio_attack_ratio],
        "audio_activity": [round(value, 4) for value in audio_activity],
        "audio_activity_cv": round(audio_cv, 6),
        "audio_band_energy_cv": round(energy_cv, 6),
        "audio_attack_cv": round(attack_cv, 6),
        "map_density_cv": round(map_cv, 6),
        "density_activity_correlation": round(correlation, 6) if correlation is not None else None,
        "density_energy_correlation": round(energy_correlation, 6) if energy_correlation is not None else None,
        "density_attack_correlation": round(attack_correlation, 6) if attack_correlation is not None else None,
        "window_profiles": profiles,
        "repair_targets": [profiles[index] for index in target_indices],
        "warning_thresholds": {
            "audio_activity_cv_min": RHYTHM_AUDIO_ACTIVITY_CV_MIN,
            "audio_band_energy_cv_min": RHYTHM_AUDIO_ACTIVITY_CV_MIN,
            "audio_attack_cv_min": RHYTHM_AUDIO_ACTIVITY_CV_MIN,
            "map_density_cv_max": RHYTHM_MAP_DENSITY_CV_MAX,
            "density_activity_correlation_max": RHYTHM_DENSITY_CORRELATION_MAX,
        },
        "warning": warning,
        "limitations": [
            "Band energy and attack cells are quantized SG proxies, not calibrated loudness or detected note onsets.",
            "Four-second density/activity association is a coarse review signal and not a calibrated gameplay score.",
        ],
    }


def _normalized_pairs(matrix):
    import numpy as np

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 2:
        return np.asarray([], dtype=float)
    scale = np.std(matrix, axis=0)
    useful = scale > 1e-8
    if not useful.any():
        return np.zeros(len(matrix) * (len(matrix) - 1) // 2)
    normalized = (matrix[:, useful] - np.mean(matrix[:, useful], axis=0)) / scale[useful]
    output = []
    for left in range(len(normalized) - 1):
        distances = np.linalg.norm(normalized[left + 1:] - normalized[left], axis=1)
        output.extend(np.clip(distances / math.sqrt(int(useful.sum())) / 3.0, 0.0, 1.0))
    return np.asarray(output, dtype=float)


def _adjacent_change(matrix):
    """Return normalized structural change between neighboring rows."""
    import numpy as np

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 2:
        return np.asarray([], dtype=float)
    scale = np.std(matrix, axis=0)
    useful = scale > 1e-8
    if not useful.any():
        return np.zeros(len(matrix) - 1, dtype=float)
    normalized = (matrix[:, useful] - np.mean(matrix[:, useful], axis=0)) / scale[useful]
    changes = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
    return np.clip(changes / math.sqrt(int(useful.sum())) / 3.0, 0.0, 1.0)


def _audio_pattern_alignment(program: MapProgram, plan: TrackPlan,
                             audio_features: str | None, *,
                             mapping_extensions: bool = False) -> dict[str, Any]:
    """Compare four-second map-shape changes with the measured signal grid.

    The result is a directional diagnostic, not a calibrated quality score.
    """
    import numpy as np

    if not isinstance(audio_features, str) or not audio_features.strip():
        return {"available": False, "reason": "audio feature grid unavailable"}
    windows = int(plan.duration_seconds // 4.0)
    if windows < 4:
        return {"available": False, "reason": "fewer than four complete four-second windows"}
    audio_values: list[list[list[int]]] = [[] for _ in range(windows)]
    for line in audio_features.splitlines():
        match = _AUDIO_GRID_ROW.fullmatch(line.strip())
        if not match:
            continue
        start_beat, _end_beat = float(match.group(1)), float(match.group(2))
        bands = [match.group(index) for index in range(3, 7)]
        if len({len(values) for values in bands}) != 1:
            continue
        for offset, chars in enumerate(zip(*bands)):
            beat = start_beat + 0.5 * offset
            window = int((beat * 60.0 / plan.bpm) // 4.0)
            if 0 <= window < windows:
                audio_values[window].append([int(value) for value in chars])
    if sum(bool(values) for values in audio_values) < 4:
        return {"available": False, "reason": "audio grid covers fewer than four windows"}

    audio = np.zeros((windows, 4), dtype=float)
    for index, rows in enumerate(audio_values):
        if not rows:
            continue
        values = np.asarray(rows, dtype=float)
        bands = values[:, :3]
        # Quantized band energy stands in for the four-second RMS level; band
        # rises provide conservative attack-count proxies from the same grid.
        level = float(np.sqrt(np.mean(np.square(bands))))
        attacks = [float(np.count_nonzero(np.diff(bands[:, band]) >= 2))
                   for band in range(3)]
        audio[index] = (level, *attacks)

    # Phrase structure counts authored objects once. Chain subdivisions are
    # derived from their head/tail and slice count, so they remain in NPS but
    # must not overwhelm the authored pattern descriptor.
    hits: list[tuple[float, int, int | None, float, float, int, float]] = []
    for item in program.notes:
        hits.append((float(item.beat), item.color,
                     direction_bucket(item.direction, mapping_extensions=mapping_extensions),
                     decode_position(item.x), decode_position(item.y), 0, 0.0))
    for item in program.arcs:
        hits.append((float(item.beat), item.color,
                     direction_bucket(item.direction, mapping_extensions=mapping_extensions),
                     decode_position(item.x), decode_position(item.y), 1,
                     float(item.tail_beat) - float(item.beat)))
        hits.append((float(item.tail_beat), item.tail_color,
                     direction_bucket(item.tail_direction,
                                      mapping_extensions=mapping_extensions),
                     decode_position(item.tail_x), decode_position(item.tail_y), 1, 0.0))
    for item in program.bombs:
        hits.append((float(item.beat), -1, None, decode_position(item.x),
                     decode_position(item.y), 3, 0.0))
    hits.sort(key=lambda item: item[0])

    lane_count = 6 if mapping_extensions else 4
    lane_buckets = lane_count + (2 if mapping_extensions else 0)
    object_type_start = 15 + lane_buckets * 2
    duration_start = object_type_start + 4
    interval_start = duration_start + 3
    # [count, 8 directions, 6 color/row combinations, per-hand lane counts,
    # 4 object types, 3 duration buckets, 6 interval buckets].
    descriptors = np.zeros((windows, interval_start + 6), dtype=float)
    hand_counts = np.zeros((windows, 2), dtype=float)
    previous_seconds: float | None = None
    existing: set[tuple[float, int, float, float]] = set()
    projected: list[tuple[float, int, int | None, float, float, int, float]] = []
    for beat, color, direction, x, y, kind, duration in hits:
        key = (beat, color, x, y)
        if key not in existing:
            existing.add(key)
            projected.append((beat, color, direction, x, y, kind, duration))
    for chain in program.chains:
        beat, color = float(chain.beat), int(chain.color)
        x, y = decode_position(chain.x), decode_position(chain.y)
        key = (beat, color, x, y)
        if key not in existing:
            existing.add(key)
            projected.append((beat, color, direction_bucket(
                chain.direction, mapping_extensions=mapping_extensions), x, y, 2,
                              float(chain.tail_beat) - float(chain.beat)))
    projected.sort(key=lambda item: (item[0], item[1]))
    for beat, color, direction, _x, y, kind, duration in projected:
        window = int((beat * 60.0 / plan.bpm) // 4.0)
        if not 0 <= window < windows:
            continue
        descriptors[window, 0] += 1
        if color in (0, 1):
            hand_counts[window, color] += 1
        if direction is not None and 0 <= direction < 8:
            descriptors[window, 1 + direction] += 1
        current_seconds = beat * 60.0 / plan.bpm
        row = int(math.floor(y + 0.5))
        if color in (0, 1) and 0 <= row < 3:
            descriptors[window, 9 + color * 3 + row] += 1
        if color in (0, 1):
            if mapping_extensions:
                column = (0 if _x < -0.5 else lane_count + 1 if _x >= lane_count - 0.5
                          else 1 + int(math.floor(_x + 0.5)))
            else:
                column = int(_x) if 0 <= _x < lane_count else -1
            if 0 <= column < lane_buckets:
                descriptors[window, 15 + color * lane_buckets + column] += 1
        descriptors[window, object_type_start + kind] += 1
        duration_bucket = 0 if duration < 1.0 else 1 if duration < 4.0 else 2
        descriptors[window, duration_start + duration_bucket] += 1
        if previous_seconds is not None:
            gap = current_seconds - previous_seconds
            bucket = int(np.searchsorted([0.08, 0.2, 0.4, 0.7, 1.2], gap))
            descriptors[window, interval_start + bucket] += 1
        previous_seconds = current_seconds
    descriptors[:, 1:] /= np.maximum(descriptors[:, :1], 1.0)

    audio_pairs = _normalized_pairs(audio)
    map_pairs = _normalized_pairs(descriptors)
    if len(audio_pairs) != len(map_pairs) or len(audio_pairs) == 0:
        return {"available": False, "reason": "insufficient map/audio windows"}
    audio_variation = float(np.std(audio_pairs))
    map_variation = float(np.std(map_pairs))
    if audio_variation < 1e-8:
        return {"available": False, "reason": "audio feature profile is effectively uniform",
                "windows": windows}
    if map_variation < 1e-8:
        correlation = 0.0
        chance_mean = 0.0
        support = 0.0
    else:
        correlation = float(np.corrcoef(audio_pairs, map_pairs)[0, 1])
        if not math.isfinite(correlation):
            return {"available": False, "reason": "pairwise correlation is undefined",
                    "windows": windows}
        rng = np.random.default_rng(1138)
        chance = []
        for _ in range(32):
            permuted = _normalized_pairs(descriptors[rng.permutation(windows)])
            value = float(np.corrcoef(audio_pairs, permuted)[0, 1])
            if math.isfinite(value):
                chance.append(value)
        if not chance:
            return {"available": False, "reason": "permuted baseline is undefined",
                    "windows": windows}
        chance_mean = float(np.mean(chance))
        support = float(np.clip((correlation - chance_mean) / max(1e-9, 1 - chance_mean), 0, 1))
    audio_changes = _adjacent_change(audio)
    map_changes = _adjacent_change(descriptors)
    profiles = []
    for index in range(windows):
        values = np.asarray(audio_values[index], dtype=float)
        # Descriptor columns contain authored heads and proportions. Keep the
        # prompt-facing summary small and interpretable; chain slices are not
        # counted here because they are derived subdivisions, not new authored
        # musical events.
        direction_mix = descriptors[index, 1:9]
        left_columns = [round(float(value), 2)
                        for value in descriptors[index, 15:15 + lane_count]]
        right_columns = [round(float(value), 2)
                         for value in descriptors[index, 15 + lane_count:object_type_start]]
        profiles.append({
            "window": index,
            "beats": [round(index * 4.0 * plan.bpm / 60.0, 2),
                      round((index + 1) * 4.0 * plan.bpm / 60.0, 2)],
            "audio_mean_0_9": [round(float(value), 2) for value in
                               (np.mean(values[:, band]) for band in range(3))]
                               if len(values) else [0.0, 0.0, 0.0],
            "audio_attack_cells_ge_5": int(np.count_nonzero(values[:, 3] >= 5))
                                      if len(values) else 0,
            "map_heads": int(descriptors[index, 0]),
            "map_heads_by_type": {
                "notes": int(round(descriptors[index, object_type_start] * descriptors[index, 0])),
                "arc_heads": int(round(descriptors[index, object_type_start + 1] * descriptors[index, 0])),
                "chain_heads": int(round(descriptors[index, object_type_start + 2] * descriptors[index, 0])),
                "bombs": int(round(descriptors[index, object_type_start + 3] * descriptors[index, 0])),
            },
            "hands_left_right": [
                round(float(hand_counts[index, 0] / max(1, int(descriptors[index, 0]))), 2),
                round(float(hand_counts[index, 1] / max(1, int(descriptors[index, 0]))), 2),
            ],
            "grid_columns_left_right": [left_columns, right_columns],
            "dominant_direction_fraction": round(float(direction_mix.max()), 2),
            "audio_change_from_previous": (round(float(audio_changes[index - 1]), 3)
                                           if index else None),
            "map_change_from_previous": (round(float(map_changes[index - 1]), 3)
                                         if index else None),
        })
    if len(audio_changes):
        active_audio = audio_changes[audio_changes > 1e-8]
        audio_threshold = float(np.median(active_audio)) if len(active_audio) else 1.0
        targets = [
            {"between_windows": [index, index + 1],
             "audio_change": round(float(audio_changes[index]), 3),
             "map_change": round(float(map_changes[index]), 3),
             "unmatched_change": round(float(audio_changes[index] - map_changes[index]), 3)}
            for index in range(len(audio_changes))
            if audio_changes[index] >= audio_threshold
            and audio_changes[index] > map_changes[index]
        ]
        targets.sort(key=lambda item: item["unmatched_change"], reverse=True)
        targets = targets[:4]
        window_beats = 4.0 * plan.bpm / 60.0
        object_names = {0: "note", 1: "arc", 2: "chain", 3: "bomb"}

        def window_signature(window_index: int) -> dict[str, Any]:
            start = window_index * window_beats
            end = start + window_beats
            events = []
            for beat, color, direction, x, y, kind, duration in projected:
                if not start <= beat < end:
                    continue
                event = {
                    "beat_offset": round(float(beat - start), 2),
                    "hand": "left" if color == 0 else "right" if color == 1 else "none",
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "direction_bucket": direction,
                    "type": object_names.get(kind, "unknown"),
                }
                if kind in (1, 2):
                    event["duration_beats"] = round(float(duration), 2)
                events.append(event)
            return {
                "window": window_index,
                "beats": [round(start, 2), round(end, 2)],
                "total_events": len(events),
                "events": events[:16],
                "truncated": len(events) > 16,
            }

        for target in targets:
            left, right = target["between_windows"]
            target["window_signatures"] = [
                window_signature(int(left)), window_signature(int(right)),
            ]
    else:
        targets = []
    return {
        "available": True,
        "windows": windows,
        "pairwise_structure_correlation": round(correlation, 6),
        "permuted_baseline_mean": round(chance_mean, 6),
        "chance_corrected_structure_support": round(support, 6),
        "audio_pairwise_variation": round(audio_variation, 6),
        "map_pairwise_variation": round(map_variation, 6),
        "window_profiles": profiles,
        "repair_targets": targets,
        "warning": support < PATTERN_SUPPORT_WARNING_THRESHOLD,
        "warning_threshold": PATTERN_SUPPORT_WARNING_THRESHOLD,
        "limitations": [
            "Uses quantized half-beat audio features and four-second windows; it cannot recognize instruments or thematic motives.",
            "Counts authored arc/chain objects once and buckets their duration; chain-slice density remains a separate NPS measure.",
            "The metric is a development diagnostic, not a calibrated gameplay rating; repeated choruses may be intentional.",
        ],
    }


def preflight(program: MapProgram, plan: TrackPlan, *,
              audio_features: str | None = None,
              mapping_extensions: bool = False) -> QualityReport:
    errors = [error.code + ": " + error.message for error in validate_program(
        program, mapping_extensions=mapping_extensions)]
    warnings: list[str] = []
    pattern_alignment = _audio_pattern_alignment(
        program, plan, audio_features, mapping_extensions=mapping_extensions)
    if pattern_alignment.get("warning"):
        warnings.append(
            "map pattern changes show little relationship to measured audio changes "
            f"(chance-corrected support "
            f"{pattern_alignment['chance_corrected_structure_support']:.3f}); "
            "review for phrase-aware development"
        )
    chain_opportunities = _chain_opportunities(program)
    chain_usage = _chain_usage(program, plan)
    if chain_opportunities["warning"]:
        candidates = "; ".join(
            f"color {item['color']} beats {item['start_beat']:.2f}–{item['end_beat']:.2f} "
            f"({item['head_count']} notes)"
            for item in chain_opportunities["eligible_runs"][:6]
        )
        warnings.append(
            "native_chain_candidate_review: inspect rapid same-hand run(s) "
            + candidates
            + "; convert only a coherent SG-supported burst, with no chain quota"
        )
    if chain_usage["warning"]:
        warnings.append(
            "chain_slice_overuse: chain slices make up "
            f"{chain_usage['chain_slice_share']:.1%} of playable hit-equivalents "
            f"(review threshold {CHAIN_SLICE_SHARE_REVIEW_THRESHOLD:.0%}); "
            "review against SG and keep ordinary notes for isolated or slower hits"
        )
    palette_contrast = _palette_contrast(program)
    if palette_contrast.get("warning"):
        warnings.append(
            "scene_palette_low_contrast: environment colors have weak visual separation "
            f"(normalized RGB distance {palette_contrast['normalized_rgb_distance']:.3f}; "
            f"minimum {palette_contrast['minimum_distance']:.3f}); choose a clearly distinguishable pair"
        )
    rhythm_alignment = _rhythm_alignment(program, plan, audio_features)
    if rhythm_alignment.get("warning"):
        energy_correlation = rhythm_alignment.get("density_energy_correlation")
        attack_correlation = rhythm_alignment.get("density_attack_correlation")
        energy_text = "unavailable" if energy_correlation is None else f"{energy_correlation:.3f}"
        attack_text = "unavailable" if attack_correlation is None else f"{attack_correlation:.3f}"
        warnings.append(
            "rhythm_density_unresponsive: map hit density stays nearly flat while "
            "measured band-energy/transient activity varies "
            f"(map density CV {rhythm_alignment['map_density_cv']:.3f}, "
            f"band-energy CV {rhythm_alignment['audio_band_energy_cv']:.3f} "
            f"(density correlation {energy_text}), attack CV "
            f"{rhythm_alignment['audio_attack_cv']:.3f} "
            f"(density correlation {attack_text}); review the highlighted SG windows"
        )
    lighting_coverage = _lighting_coverage(program, plan)
    if lighting_coverage["warning"]:
        missing = ", ".join(lighting_coverage["missing_sections"][:6])
        omitted = (f" and {len(lighting_coverage['missing_sections']) - 6} more"
                   if len(lighting_coverage["missing_sections"]) > 6 else "")
        warnings.append(
            "lighting_section_coverage_low: authored lights cover "
            f"{len(lighting_coverage['covered_sections'])}/"
            f"{len(lighting_coverage['active_sections'])} active sections; "
            f"missing cues in {missing}{omitted}; review phrase and accent alignment"
        )
    # NPS is a full-track target. Using the last authored object as the
    # denominator lets a sparse/early-ending map appear artificially dense.
    seconds = plan.duration_seconds
    chain_slices = sum(chain.slices for chain in program.chains)
    arc_hits = 2 * len(program.arcs)
    note_count = len(program.notes) + arc_hits + chain_slices
    if not program.notes and not program.arcs and not program.chains:
        errors.append("no authored playable object")
    actual_nps = note_count / seconds if seconds > 0 else 0.0
    if (plan.target_nps > 0 and abs(actual_nps - plan.target_nps)
            > plan.target_nps * NPS_TARGET_RELATIVE_TOLERANCE):
        warnings.append(
            f"actual NPS {actual_nps:.3f} is outside the "
            f"±{NPS_TARGET_RELATIVE_TOLERANCE:.0%} target band")
    ordered_notes = sorted(program.notes, key=lambda note: note.beat)
    run_length = 0
    run_signature = None
    previous_beat = None
    longest_run = 0
    for note in ordered_notes:
        signature = (note.x, note.y)
        if (signature == run_signature and previous_beat is not None
                and math.isclose(note.beat - previous_beat, 1.0, abs_tol=1e-6)):
            run_length += 1
        else:
            run_signature = signature
            run_length = 1
        longest_run = max(longest_run, run_length)
        previous_beat = note.beat
    if longest_run >= 8:
        warnings.append(
            f"notes stay in the same grid cell every beat for {longest_run} notes; "
            "review for musical alignment and hand-pattern variety"
        )
    playable_heads = [(item.color, item.direction) for item in program.notes]
    playable_heads.extend((arc.color, arc.direction) for arc in program.arcs)
    playable_heads.extend((arc.tail_color, arc.tail_direction) for arc in program.arcs)
    playable_heads.extend((chain.color, chain.direction) for chain in program.chains)
    direction_profiles: dict[str, dict[str, Any]] = {}
    for color in (0, 1):
        directions = [direction for item_color, direction in playable_heads
                      if item_color == color]
        counts = Counter(directions)
        total = len(directions)
        dominant_direction, dominant_count = max(counts.items(), key=lambda item: item[1], default=(None, 0))
        fraction = dominant_count / total if total else 0.0
        profile = {
            "playable_heads": total,
            "direction_counts": {str(key): value for key, value in sorted(counts.items())},
            "dominant_direction": dominant_direction,
            "dominant_fraction": round(fraction, 6),
        }
        direction_profiles[str(color)] = profile
        if total >= 24 and fraction >= 0.9:
            warnings.append(
                f"severe cut-direction monotony in color {color} hand: direction "
                f"{dominant_direction} on {dominant_count}/{total} playable heads "
                f"({fraction:.0%}); review for repeated swing geometry"
            )
    total_sections = len(plan.sections)
    occupied_sections = 0
    active_sections = [section for section in plan.sections
                       if not section.locked and section.density > 0]
    occupied_active_sections = 0
    playable_beats = ([note.beat for note in program.notes]
                      + [beat for arc in program.arcs
                         for beat in (arc.beat, arc.tail_beat)]
                      + [chain.beat for chain in program.chains])
    for section in plan.sections:
        if any(section.start_beat <= beat <= section.end_beat for beat in playable_beats):
            occupied_sections += 1
            if section in active_sections:
                occupied_active_sections += 1
        elif not section.locked and section.density > 0:
            errors.append(
                f"active planned section {section.label!r} has no authored playable object "
                f"(target density {section.density:.3f} NPS)"
            )
    if active_sections and occupied_active_sections < len(active_sections):
        errors.append(
            "not all active planned sections contain playable objects "
            f"({occupied_active_sections}/{len(active_sections)} occupied)"
        )
    if not program.lights:
        warnings.append("no authored light events; inspect presentation before export")
    return QualityReport(not errors, tuple(errors), tuple(warnings), note_count, chain_slices,
                         actual_nps, occupied_sections, total_sections, direction_profiles,
                         chain_opportunities, chain_usage, pattern_alignment, palette_contrast,
                         rhythm_alignment, lighting_coverage)
