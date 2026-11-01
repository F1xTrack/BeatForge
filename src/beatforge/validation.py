"""Deterministic authoring checks; errors are explicit and machine-readable."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .contracts import MapProgram
from .mapping_extensions import (decode_position, is_coordinate,
                                 valid_direction, valid_wall_dimension)


@dataclass(frozen=True, slots=True)
class ValidationError:
    code: str
    message: str
    index: int | None = None


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_program(program: MapProgram, *, mapping_extensions: bool = False) -> list[ValidationError]:
    errors: list[ValidationError] = []
    lane_count = 6 if mapping_extensions else 4
    grid_description = ("Mapping Extensions integer encoding" if mapping_extensions
                        else f"x=0..{lane_count - 1} and y=0..2")
    if not _finite(program.bpm) or program.bpm <= 0:
        errors.append(ValidationError("invalid_bpm", "BPM must be finite and positive"))

    occupied: dict[tuple[float, float, float], tuple[str, int]] = {}
    for index, note in enumerate(program.notes):
        if not _finite(note.beat) or note.beat < 0:
            errors.append(ValidationError("invalid_note_beat", "note beat must be finite and non-negative", index))
        if (not is_coordinate(note.x, mapping_extensions=mapping_extensions)
                or not is_coordinate(note.y, mapping_extensions=mapping_extensions, axis="y")):
            errors.append(ValidationError("note_out_of_grid", f"note cell must be {grid_description}", index))
        if note.color not in (0, 1) or not valid_direction(
                note.direction, mapping_extensions=mapping_extensions):
            errors.append(ValidationError("invalid_note_shape", "note color/direction is outside V3 bounds", index))
        key = (float(note.beat), decode_position(note.x), decode_position(note.y))
        if key in occupied:
            errors.append(ValidationError("duplicate_cell", "two notes occupy one cell at the same beat", index))
        else:
            occupied[key] = ("note", index)

    for index, bomb in enumerate(program.bombs):
        if not _finite(bomb.beat) or bomb.beat < 0:
            errors.append(ValidationError("invalid_bomb_beat", "bomb beat must be finite and non-negative", index))
        if (not is_coordinate(bomb.x, mapping_extensions=mapping_extensions)
                or not is_coordinate(bomb.y, mapping_extensions=mapping_extensions, axis="y")):
            errors.append(ValidationError("bomb_out_of_grid", f"bomb cell must be {grid_description}", index))
        key = (float(bomb.beat), decode_position(bomb.x), decode_position(bomb.y))
        if key in occupied:
            previous_kind, _ = occupied[key]
            errors.append(ValidationError(
                "duplicate_cell", f"bomb overlaps a {previous_kind} at the same cell and beat", index))
        else:
            occupied[key] = ("bomb", index)

    for kind, elements in (("arc", program.arcs), ("chain", program.chains)):
        for index, element in enumerate(elements):
            if not _finite(element.beat) or not _finite(element.tail_beat) or element.tail_beat <= element.beat:
                errors.append(ValidationError(f"invalid_{kind}_duration", "tail beat must be after head beat", index))
            if (not is_coordinate(element.x, mapping_extensions=mapping_extensions)
                    or not is_coordinate(element.y, mapping_extensions=mapping_extensions, axis="y")
                    or not is_coordinate(element.tail_x, mapping_extensions=mapping_extensions)
                    or not is_coordinate(element.tail_y, mapping_extensions=mapping_extensions, axis="y")):
                errors.append(ValidationError(
                    f"{kind}_out_of_grid", f"head and tail cells must be inside the {lane_count}x3 grid", index))
            if element.color not in (0, 1) or getattr(element, "tail_color", element.color) not in (0, 1):
                errors.append(ValidationError(f"invalid_{kind}_color", "element colors must be 0 or 1", index))
            if not valid_direction(element.direction, mapping_extensions=mapping_extensions):
                errors.append(ValidationError(f"invalid_{kind}_direction", "element direction is outside V3 bounds", index))
            if (kind == "arc"
                    and not valid_direction(element.tail_direction,
                                            mapping_extensions=mapping_extensions)):
                errors.append(ValidationError(
                    "invalid_arc_tail_direction", "arc tail direction is outside V3 bounds", index))
            if kind == "arc" and (not isinstance(element.curve, int)
                                   or isinstance(element.curve, bool)
                                   or element.curve not in (0, 1)):
                errors.append(ValidationError("invalid_arc_curve", "arc curve must be 0 or 1", index))
            if kind == "chain" and (not isinstance(element.slices, int) or isinstance(element.slices, bool)
                                      or element.slices < 1 or not _finite(element.squish) or element.squish <= 0):
                errors.append(ValidationError("invalid_chain_slices", "chain slices and squish must be positive", index))
            endpoints = [(float(element.beat), element.x, element.y, "head")]
            if kind == "arc":
                endpoints.append((float(element.tail_beat), element.tail_x,
                                  element.tail_y, "tail"))
            for beat, x, y, endpoint_name in endpoints:
                key = (beat, decode_position(x), decode_position(y))
                if key in occupied:
                    previous_kind, _ = occupied[key]
                    errors.append(ValidationError(
                        "duplicate_cell",
                        f"{kind} {endpoint_name} overlaps a {previous_kind} at the same cell and beat", index,
                    ))
                else:
                    occupied[key] = (f"{kind} {endpoint_name}", index)

    for index, wall in enumerate(program.walls):
        if not _finite(wall.beat) or not _finite(wall.duration) or wall.beat < 0 or wall.duration <= 0:
            errors.append(ValidationError("invalid_wall_timing", "wall beat must be non-negative and duration positive", index))
        if (not is_coordinate(wall.x, mapping_extensions=mapping_extensions)
                or not is_coordinate(wall.y, mapping_extensions=mapping_extensions,
                                     axis="y", wall=True)
                or not valid_wall_dimension(wall.width, mapping_extensions=mapping_extensions)
                or not valid_wall_dimension(wall.height, mapping_extensions=mapping_extensions,
                                            height=True)):
            errors.append(ValidationError("invalid_wall_geometry", "wall geometry is outside the playable grid", index))
    for index, event in enumerate(program.custom_events):
        if not _finite(event.beat) or event.beat < 0:
            errors.append(ValidationError("invalid_custom_event_beat",
                                          "custom event beat must be finite and non-negative", index))
        if not isinstance(event.event_type, str) or not event.event_type.strip():
            errors.append(ValidationError("invalid_custom_event_type",
                                          "custom event type must be a non-empty string", index))
        if not isinstance(event.data, dict):
            errors.append(ValidationError("invalid_custom_event_data",
                                          "custom event data must be an object", index))
    palette = program.scene_palette
    if palette:
        allowed_palette_keys = {"environment_left", "environment_right",
                                "saber_left", "saber_right"}
        if not isinstance(palette, dict) or set(palette) - allowed_palette_keys:
            errors.append(ValidationError("invalid_scene_palette",
                                          "palette keys must be environment_left/right and optional saber_left/right"))
        else:
            if not {"environment_left", "environment_right"}.issubset(palette):
                errors.append(ValidationError("invalid_scene_palette",
                                              "both environment colors are required"))
            if ("saber_left" in palette) != ("saber_right" in palette):
                errors.append(ValidationError("invalid_scene_palette",
                                              "saber colors must be supplied as a left/right pair"))
            for key, color in palette.items():
                if (not isinstance(color, (list, tuple)) or len(color) != 3
                        or any(not isinstance(channel, int) or isinstance(channel, bool)
                               or not 0 <= channel <= 255 for channel in color)):
                    errors.append(ValidationError("invalid_scene_palette",
                                                  f"{key} must be an RGB triple of integers from 0 to 255"))
    return errors


def raise_if_invalid(program: MapProgram, *, mapping_extensions: bool = False) -> None:
    errors = validate_program(program, mapping_extensions=mapping_extensions)
    if errors:
        groups: dict[tuple[str, str], list[int | None]] = {}
        for error in errors:
            groups.setdefault((error.code, error.message), []).append(error.index)
        details = []
        for (code, message), indices in groups.items():
            if len(indices) == 1:
                details.append(f"{code}: {message}")
            else:
                examples = ", ".join(str(index) for index in indices[:5])
                suffix = f" (indices {examples}{', ...' if len(indices) > 5 else ''})"
                details.append(f"{code}: {message} ({len(indices)} occurrences){suffix}")
        raise ValueError("; ".join(details))
