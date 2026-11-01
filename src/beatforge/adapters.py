"""Loss-aware adapters between serialized V3 JSON and the typed SDK."""
from __future__ import annotations

import math
from typing import Any

from .contracts import Arc, Bomb, Chain, CustomEvent, LightEvent, MapProgram, Note, Wall


def program_from_v3(difficulty: dict[str, Any], *, bpm: float) -> MapProgram:
    """Decode the V3 subset authored by BeatForge into a typed program."""
    def custom(item: dict[str, Any]) -> dict[str, Any]:
        value = item.get("customData", {})
        return dict(value) if isinstance(value, dict) else {}

    custom_data = dict(difficulty.get("customData") or {})
    raw_custom_events = custom_data.pop("_customEvents", [])
    raw_scene_palette = custom_data.pop("_beatforgeScenePalette", {})
    program = MapProgram(bpm=float(bpm), custom_data=custom_data)
    if isinstance(raw_scene_palette, dict):
        program.scene_palette = dict(raw_scene_palette)
    if isinstance(raw_custom_events, list):
        program.custom_events = [
            CustomEvent(float(item["b"]), str(item["t"]), dict(item.get("d") or {}))
            for item in raw_custom_events
            if isinstance(item, dict) and "b" in item and "t" in item
        ]
    program.bpm_events = [dict(event) for event in difficulty.get("bpmEvents", []) or []]
    raw_notes = [Note(float(item["b"]), int(item["x"]), int(item["y"]),
                      int(item["c"]), int(item["d"]), int(item.get("a", 0)), custom(item))
                 for item in difficulty.get("colorNotes", []) or []]
    program.bombs = [Bomb(float(item["b"]), int(item["x"]), int(item["y"]), custom(item))
                     for item in difficulty.get("bombNotes", []) or []]
    remaining_notes = list(raw_notes)
    program.arcs = []
    for item in difficulty.get("sliders", []) or []:
        beat, tail_beat = float(item["b"]), float(item["tb"])
        color, tail_color = int(item["c"]), int(item["tc"])
        x, y, tail_x, tail_y = (int(item["x"]), int(item["y"]),
                                int(item["tx"]), int(item["ty"]))
        direction = int(item["d"])

        def find_anchor(anchor_beat: float, anchor_x: int, anchor_y: int,
                        anchor_color: int, anchor_direction: int | None = None) -> Note | None:
            for index, note in enumerate(remaining_notes):
                if (math.isclose(note.beat, anchor_beat, rel_tol=0.0, abs_tol=1e-6)
                        and note.x == anchor_x and note.y == anchor_y
                        and note.color == anchor_color
                        and (anchor_direction is None or note.direction == anchor_direction)):
                    return remaining_notes.pop(index)
            return None

        head_note = find_anchor(beat, x, y, color, direction)
        tail_note = find_anchor(tail_beat, tail_x, tail_y, tail_color)
        program.arcs.append(Arc(
            beat, color, x, y, direction, tail_beat, tail_x, tail_y, tail_color,
            int(item.get("m", 0)), custom(item),
            tail_direction=tail_note.direction if tail_note is not None else 8,
            head_note_custom_data=(head_note.custom_data if head_note is not None else {}),
            tail_note_custom_data=(tail_note.custom_data if tail_note is not None else {}),
        ))
    program.notes = remaining_notes
    program.chains = [Chain(float(item["b"]), int(item["c"]), int(item["x"]), int(item["y"]),
                            int(item["d"]), float(item["tb"]), int(item["tx"]), int(item["ty"]),
                            int(item["sc"]), float(item.get("s", 1.0)), custom(item))
                      for item in difficulty.get("burstSliders", []) or []]
    program.walls = [Wall(float(item["b"]), int(item["x"]), int(item["y"]), float(item["d"]),
                          int(item.get("w", 1)), int(item.get("h", 5)), custom(item))
                     for item in difficulty.get("obstacles", []) or []]
    program.lights = [LightEvent(float(item["b"]), int(item["et"]), int(item["i"]),
                                 float(item.get("f", 1.0)), custom(item))
                      for item in difficulty.get("basicBeatmapEvents", []) or []]
    return program

