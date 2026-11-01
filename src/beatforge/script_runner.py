"""Tiny safe authoring DSL for model-produced map scripts.

The first version intentionally permits only direct SDK calls. Loops, imports,
assignments, comprehensions and arbitrary Python expressions are rejected until
the future worker sandbox has resource limits and a separate process boundary.
"""
from __future__ import annotations

import ast
import json
import math
from typing import Any, Callable

from .compiler import compile_program
from .contracts import Arc, Bomb, Chain, CustomEvent, LightEvent, MapProgram, Note, Wall


class ScriptRejected(ValueError):
    pass


_CALLS: dict[str, tuple[type, tuple[str, ...]]] = {
    "note": (Note, ("beat", "x", "y", "color", "direction", "angle", "custom_data")),
    "bomb": (Bomb, ("beat", "x", "y", "custom_data")),
    "arc": (Arc, ("beat", "color", "x", "y", "direction", "tail_beat", "tail_x", "tail_y", "tail_color", "curve", "custom_data", "tail_direction")),
    "chain": (Chain, ("beat", "color", "x", "y", "direction", "tail_beat", "tail_x", "tail_y", "slices", "squish", "custom_data")),
    "wall": (Wall, ("beat", "x", "y", "duration", "width", "height", "custom_data")),
    "light": (LightEvent, ("beat", "event_type", "value", "float_value", "custom_data")),
    "custom_event": (CustomEvent, ("beat", "event_type", "data")),
}

_TARGETS = {"note": "notes", "bomb": "bombs", "arc": "arcs", "chain": "chains", "wall": "walls",
            "light": "lights", "custom_event": "custom_events"}
_MAX_REPEAT_EVENTS = 256
_MAX_REPEAT_COUNT = 512
_MAX_EXPANDED_OBJECTS = 10000


def remove_exact_duplicate_palette_calls(source: str) -> tuple[str, list[dict[str, Any]]]:
    """Remove repeated standalone palette calls only when their literal data matches."""
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, TypeError):
        return source, []
    lines = source.splitlines(keepends=True)
    if not lines:
        return source, []
    seen: dict[str, int] = {}
    remove_lines: set[int] = set()
    corrections: list[dict[str, Any]] = []
    allowed = {"environment_left", "environment_right", "saber_left", "saber_right"}
    for statement in tree.body:
        if (not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call)
                or not isinstance(statement.value.func, ast.Attribute)
                or not isinstance(statement.value.func.value, ast.Name)
                or statement.value.func.value.id != "map"
                or statement.value.func.attr != "palette"):
            continue
        call = statement.value
        if (call.args or not call.keywords or statement.lineno != statement.end_lineno
                or any(keyword.arg not in allowed for keyword in call.keywords)):
            continue
        try:
            palette: dict[str, Any] = {}
            for keyword in call.keywords:
                if keyword.arg in palette:
                    raise ValueError("repeated palette keyword")
                palette[str(keyword.arg)] = ast.literal_eval(keyword.value)
            signature = json.dumps(palette, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"))
        except (ValueError, TypeError, SyntaxError):
            continue
        line = lines[statement.lineno - 1]
        segment = ast.get_source_segment(source, statement)
        if segment is None or line.strip() != segment.strip():
            continue
        previous = seen.get(signature)
        if previous is None:
            seen[signature] = statement.lineno
            continue
        remove_lines.add(statement.lineno)
        corrections.append({
            "kind": "removed_exact_duplicate_palette",
            "line": statement.lineno,
            "duplicate_of_line": previous,
            "palette": palette,
        })
    if not remove_lines:
        return source, []
    corrected = "".join(line for number, line in enumerate(lines, start=1)
                        if number not in remove_lines)
    return corrected, corrections


def remove_exact_duplicate_note_calls(source: str) -> tuple[str, list[dict[str, Any]]]:
    """Remove a standalone note that exactly duplicates a note in a repeat or earlier call.

    This narrow normalization never chooses between notes with different colors,
    directions, angles, or custom data, and never edits a repeat template.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, TypeError):
        return source, []
    lines = source.splitlines(keepends=True)
    if not lines:
        return source, []
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}

    def signature(values: dict[str, Any], beat_offset: float = 0.0) -> tuple[Any, ...] | None:
        try:
            note = Note(**{**values, "beat": values["beat"] + beat_offset})
            return (float(note.beat), note.x, note.y, note.color, note.direction,
                    note.angle, json.dumps(note.custom_data, sort_keys=True, ensure_ascii=False))
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    direct_notes: list[tuple[ast.Expr, tuple[Any, ...]]] = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if (not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name)
                or call.func.value.id != "map"):
            continue
        if call.func.attr == "repeat" and len(call.args) == 4 and not call.keywords:
            try:
                start, repetitions, step, template = [_literal(arg) for arg in call.args]
            except ScriptRejected:
                continue
            if (not isinstance(start, (int, float)) or isinstance(start, bool)
                    or not math.isfinite(start)
                    or not isinstance(repetitions, int) or isinstance(repetitions, bool)
                    or not 1 <= repetitions <= _MAX_REPEAT_COUNT
                    or not isinstance(step, (int, float)) or isinstance(step, bool)
                    or not math.isfinite(step) or step <= 0
                    or not isinstance(template, list)
                    or not 1 <= len(template) <= _MAX_REPEAT_EVENTS):
                continue
            for repeat_index in range(repetitions):
                for event in template:
                    if not isinstance(event, dict) or event.get("kind") != "note":
                        continue
                    values = {key: value for key, value in event.items() if key != "kind"}
                    beat_offset = start + repeat_index * step
                    key = signature(values, beat_offset)
                    if key is not None:
                        seen.setdefault(key, {"kind": "map.repeat", "line": statement.lineno})
        elif call.func.attr == "note":
            try:
                cls, fields = _CALLS["note"]
                positional = [_literal(arg) for arg in call.args]
                values = dict(zip(fields, positional))
                values.update({keyword.arg: _literal(keyword.value) for keyword in call.keywords})
                note = cls(**values)
                key = (float(note.beat), note.x, note.y, note.color, note.direction,
                       note.angle, json.dumps(note.custom_data, sort_keys=True, ensure_ascii=False))
            except (KeyError, TypeError, ValueError, ScriptRejected, OverflowError):
                continue
            direct_notes.append((statement, key))

    remove_lines: set[int] = set()
    corrections: list[dict[str, Any]] = []
    for statement, key in direct_notes:
        if statement.lineno != statement.end_lineno:
            continue
        line = lines[statement.lineno - 1]
        segment = ast.get_source_segment(source, statement)
        if segment is None or line.strip() != segment.strip():
            continue
        previous = seen.get(key)
        if previous is None:
            seen[key] = {"kind": "map.note", "line": statement.lineno}
            continue
        remove_lines.add(statement.lineno)
        corrections.append({
            "kind": "removed_exact_duplicate_note", "line": statement.lineno,
            "duplicate_of": previous,
            "beat": key[0], "x": key[1], "y": key[2], "color": key[3],
            "direction": key[4],
        })
    if not remove_lines:
        return source, []
    return "".join(line for number, line in enumerate(lines, start=1)
                   if number not in remove_lines), corrections


def _append_literal_event(program: MapProgram, kind: str, values: dict[str, Any]) -> None:
    if kind not in _CALLS:
        raise ScriptRejected(f"unsupported pattern event type: {kind}")
    cls, fields = _CALLS[kind]
    if set(values) - set(fields):
        raise ScriptRejected(f"invalid fields for pattern event {kind}")
    try:
        event = cls(**values)
    except TypeError as exc:
        raise ScriptRejected(f"invalid pattern event {kind}: {exc}") from exc
    getattr(program, _TARGETS[kind]).append(event)


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ScriptRejected("only literal arguments are allowed") from exc


def normalize_vivify_setting_categories(
    source: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Move uniquely recognized SetRenderingSettings fields to their schema category.

    This is deliberately narrow: unknown keys, malformed values, ambiguous fields,
    and destination conflicts are left untouched for the validator to reject.
    """
    from .vivify import VIVIFY_SETTING_KEYS

    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, TypeError):
        return source, []
    lines = source.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))
    edits: list[tuple[int, int, str]] = []
    corrections: list[dict[str, Any]] = []

    def argument(call: ast.Call, name: str, position: int) -> ast.AST | None:
        named = [keyword.value for keyword in call.keywords if keyword.arg == name]
        if named:
            return named[0] if len(named) == 1 and len(call.args) <= position else None
        return call.args[position] if len(call.args) > position else None

    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if (not isinstance(call.func, ast.Attribute)
                or not isinstance(call.func.value, ast.Name)
                or call.func.value.id != "map"
                or call.func.attr != "custom_event"):
            continue
        event_node = argument(call, "event_type", 1)
        data_node = argument(call, "data", 2)
        if event_node is None or data_node is None:
            continue
        try:
            if _literal(event_node) != "SetRenderingSettings":
                continue
            data = _literal(data_node)
        except ScriptRejected:
            continue
        if not isinstance(data, dict):
            continue

        normalized = {key: (dict(value) if isinstance(value, dict) else value)
                      for key, value in data.items()}
        event_fixes: list[dict[str, Any]] = []
        for category, settings in list(normalized.items()):
            if category not in VIVIFY_SETTING_KEYS or not isinstance(settings, dict):
                continue
            for name, value in list(settings.items()):
                if name in VIVIFY_SETTING_KEYS[category]:
                    continue
                destinations = [target for target, keys in VIVIFY_SETTING_KEYS.items()
                                if name in keys]
                if len(destinations) != 1 or destinations[0] == category:
                    continue
                destination = destinations[0]
                target_settings = normalized.get(destination)
                if target_settings is None:
                    target_settings = {}
                    normalized[destination] = target_settings
                if not isinstance(target_settings, dict) or name in target_settings:
                    continue
                del settings[name]
                target_settings[name] = value
                event_fixes.append({
                    "kind": "moved_vivify_setting_to_schema_category",
                    "line": getattr(data_node, "lineno", statement.lineno),
                    "setting": name,
                    "from": category,
                    "to": destination,
                })
            if not settings:
                normalized.pop(category, None)
        if not event_fixes:
            continue
        segment = ast.get_source_segment(source, data_node)
        if segment is None:
            continue
        line_index = getattr(data_node, "lineno", statement.lineno) - 1
        if not 0 <= line_index < len(lines):
            continue
        line = lines[line_index]
        byte_column = getattr(data_node, "col_offset", 0)
        try:
            char_column = len(line.encode("utf-8")[:byte_column].decode("utf-8"))
        except UnicodeDecodeError:
            continue
        start = line_offsets[line_index] + char_column
        if source[start:start + len(segment)] != segment:
            continue
        edits.append((start, start + len(segment), repr(normalized)))
        corrections.extend(event_fixes)

    if not edits:
        return source, []
    corrected = source
    for start, end, replacement in sorted(edits, reverse=True):
        corrected = corrected[:start] + replacement + corrected[end:]
    return corrected, corrections


def run_script(source: str, *, bpm: float, mapping_extensions: bool = False) -> dict:
    """Run direct typed SDK calls and return a compiled V3 difficulty object."""
    if not isinstance(source, str) or len(source.encode("utf-8")) > 512 * 1024:
        raise ScriptRejected("script exceeds the 512 KiB authoring limit")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ScriptRejected(str(exc)) from exc
    if len(tree.body) > 10000:
        raise ScriptRejected("script has too many statements")
    program = MapProgram(bpm=bpm)
    expanded_objects = 0
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ScriptRejected("only direct SDK calls are allowed")
        call = statement.value
        if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name) or call.func.value.id != "map":
            raise ScriptRejected("calls must target map.<typed_method>")
        if call.func.attr == "repeat":
            if call.keywords or len(call.args) != 4:
                raise ScriptRejected("map.repeat requires four positional literal arguments")
            start, repetitions, step, template = [_literal(arg) for arg in call.args]
            if (not isinstance(start, (int, float)) or isinstance(start, bool) or not math.isfinite(start)
                    or not isinstance(repetitions, int) or isinstance(repetitions, bool)
                    or not 1 <= repetitions <= _MAX_REPEAT_COUNT
                    or not isinstance(step, (int, float)) or isinstance(step, bool)
                    or not math.isfinite(step) or step <= 0
                    or not isinstance(template, list) or not 1 <= len(template) <= _MAX_REPEAT_EVENTS):
                raise ScriptRejected("invalid map.repeat bounds or event template")
            expanded_objects += repetitions * len(template)
            if expanded_objects > _MAX_EXPANDED_OBJECTS:
                raise ScriptRejected("expanded map exceeds the 10000 object limit")
            for repeat_index in range(repetitions):
                offset = start + repeat_index * step
                for raw_event in template:
                    if not isinstance(raw_event, dict):
                        raise ScriptRejected("map.repeat events must be literal dictionaries")
                    event = dict(raw_event)
                    kind = event.pop("kind", None)
                    if not isinstance(kind, str):
                        raise ScriptRejected("map.repeat event requires a string kind")
                    for beat_field in ("beat", "tail_beat"):
                        if beat_field in event:
                            value = event[beat_field]
                            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                                    or not math.isfinite(value)):
                                raise ScriptRejected(f"pattern {beat_field} must be finite")
                            event[beat_field] += offset
                    _append_literal_event(program, kind, event)
            continue
        if call.func.attr == "palette":
            allowed = {"environment_left", "environment_right", "saber_left", "saber_right"}
            if call.args or not call.keywords or any(keyword.arg not in allowed for keyword in call.keywords):
                raise ScriptRejected("map.palette requires only named environment/saber RGB fields")
            if program.scene_palette:
                raise ScriptRejected("map.palette may be specified only once")
            program.scene_palette = {
                keyword.arg: _literal(keyword.value) for keyword in call.keywords
            }
            continue
        spec = _CALLS.get(call.func.attr)
        if spec is None:
            raise ScriptRejected(f"unsupported SDK method: {call.func.attr}")
        cls, fields = spec
        if len(call.args) > len(fields) or any(keyword.arg not in fields for keyword in call.keywords):
            raise ScriptRejected(f"invalid arguments for map.{call.func.attr}")
        values = [_literal(arg) for arg in call.args]
        kwargs = {keyword.arg: _literal(keyword.value) for keyword in call.keywords}
        try:
            item = cls(*values, **kwargs)
        except TypeError as exc:
            raise ScriptRejected(f"invalid arguments for map.{call.func.attr}: {exc}") from exc
        expanded_objects += 1
        if expanded_objects > _MAX_EXPANDED_OBJECTS:
            raise ScriptRejected("map exceeds the 10000 object limit")
        getattr(program, _TARGETS[call.func.attr]).append(item)
    return compile_program(program, mapping_extensions=mapping_extensions)
