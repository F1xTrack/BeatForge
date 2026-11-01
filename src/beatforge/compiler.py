"""Canonical V3 compiler for the typed authoring SDK."""
from __future__ import annotations

from .contracts import MapProgram
from .validation import raise_if_invalid


def compile_program(program: MapProgram, *, mapping_extensions: bool = False) -> dict:
    """Validate and emit stable ordering without mutating the authored program."""
    raise_if_invalid(program, mapping_extensions=mapping_extensions)
    result = program.to_v3()
    result["colorNotes"].sort(key=lambda item: (item["b"], item["x"], item["y"], item["c"]))
    result["bombNotes"].sort(key=lambda item: (item["b"], item["x"], item["y"]))
    result["sliders"].sort(key=lambda item: (item["b"], item["x"], item["y"], item["c"]))
    result["burstSliders"].sort(key=lambda item: (item["b"], item["x"], item["y"], item["c"]))
    result["obstacles"].sort(key=lambda item: (item["b"], item["x"], item["y"]))
    result["basicBeatmapEvents"].sort(key=lambda item: (item["b"], item["et"], item["i"]))
    custom_events = result.get("customData", {}).get("_customEvents")
    if isinstance(custom_events, list):
        custom_events.sort(key=lambda item: (float(item.get("b", 0)), str(item.get("t", ""))))
    return result
