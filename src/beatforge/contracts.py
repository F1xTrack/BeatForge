"""Small, serializable authoring contracts used by models and the compiler."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Note:
    beat: float
    x: int
    y: int
    color: int
    direction: int
    angle: int = 0
    custom_data: dict[str, Any] = field(default_factory=dict)

    def to_v3(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "b": self.beat, "x": self.x, "y": self.y, "c": self.color,
            "d": self.direction,
        }
        if self.angle:
            value["a"] = self.angle
        if self.custom_data:
            value["customData"] = dict(self.custom_data)
        return value


@dataclass(frozen=True, slots=True)
class Bomb:
    beat: float
    x: int
    y: int
    custom_data: dict[str, Any] = field(default_factory=dict)

    def to_v3(self) -> dict[str, Any]:
        value: dict[str, Any] = {"b": self.beat, "x": self.x, "y": self.y}
        if self.custom_data:
            value["customData"] = dict(self.custom_data)
        return value


@dataclass(frozen=True, slots=True)
class Arc:
    beat: float
    color: int
    x: int
    y: int
    direction: int
    tail_beat: float
    tail_x: int
    tail_y: int
    tail_color: int
    curve: int = 0
    custom_data: dict[str, Any] = field(default_factory=dict)
    tail_direction: int = 8
    head_note_custom_data: dict[str, Any] = field(default_factory=dict)
    tail_note_custom_data: dict[str, Any] = field(default_factory=dict)

    def to_v3(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "b": self.beat, "c": self.color, "x": self.x, "y": self.y,
            "d": self.direction, "tb": self.tail_beat, "tx": self.tail_x,
            "ty": self.tail_y, "tc": self.tail_color, "m": self.curve,
        }
        if self.custom_data:
            value["customData"] = dict(self.custom_data)
        return value


@dataclass(frozen=True, slots=True)
class Chain:
    beat: float
    color: int
    x: int
    y: int
    direction: int
    tail_beat: float
    tail_x: int
    tail_y: int
    slices: int
    squish: float = 1.0
    custom_data: dict[str, Any] = field(default_factory=dict)

    def to_v3(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "b": self.beat, "c": self.color, "x": self.x, "y": self.y,
            "d": self.direction, "tb": self.tail_beat, "tx": self.tail_x,
            "ty": self.tail_y, "sc": self.slices, "s": self.squish,
        }
        if self.custom_data:
            value["customData"] = dict(self.custom_data)
        return value


@dataclass(frozen=True, slots=True)
class Wall:
    beat: float
    x: int
    y: int
    duration: float
    width: int = 1
    height: int = 5
    custom_data: dict[str, Any] = field(default_factory=dict)

    def to_v3(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "b": self.beat, "x": self.x, "y": self.y, "d": self.duration,
            "w": self.width, "h": self.height,
        }
        if self.custom_data:
            value["customData"] = dict(self.custom_data)
        return value


@dataclass(frozen=True, slots=True)
class LightEvent:
    beat: float
    event_type: int
    value: int
    float_value: float = 1.0
    custom_data: dict[str, Any] = field(default_factory=dict)

    def to_v3(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "b": self.beat, "et": self.event_type, "i": self.value,
            "f": self.float_value,
        }
        if self.custom_data:
            value["customData"] = dict(self.custom_data)
        return value


@dataclass(frozen=True, slots=True)
class CustomEvent:
    """A literal V3 custom event for Chroma/Noodle/Vivify integrations."""

    beat: float
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_v3(self) -> dict[str, Any]:
        return {"b": self.beat, "t": self.event_type, "d": dict(self.data)}


@dataclass(slots=True)
class MapProgram:
    bpm: float
    notes: list[Note] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    chains: list[Chain] = field(default_factory=list)
    walls: list[Wall] = field(default_factory=list)
    lights: list[LightEvent] = field(default_factory=list)
    bpm_events: list[dict[str, float]] = field(default_factory=list)
    custom_data: dict[str, Any] = field(default_factory=dict)
    custom_events: list[CustomEvent] = field(default_factory=list)
    scene_palette: dict[str, list[int]] = field(default_factory=dict)
    bombs: list[Bomb] = field(default_factory=list)

    def to_v3(self) -> dict[str, Any]:
        custom_data = dict(self.custom_data)
        if self.custom_events:
            existing = custom_data.get("_customEvents", [])
            if not isinstance(existing, list):
                existing = []
            custom_data["_customEvents"] = [dict(event) for event in existing] + [
                event.to_v3() for event in self.custom_events]
        if self.scene_palette:
            custom_data["_beatforgeScenePalette"] = {
                key: list(value) for key, value in self.scene_palette.items()
            }
        color_notes = [note.to_v3() for note in self.notes]
        # A V3 slider describes the arc path, while its endpoint notes carry
        # the cut directions and note scoring. Keep those anchors implicit in
        # the authoring SDK and materialize them only in the exported format.
        for arc in self.arcs:
            color_notes.extend((
                Note(arc.beat, arc.x, arc.y, arc.color, arc.direction,
                     custom_data=dict(arc.head_note_custom_data)).to_v3(),
                Note(arc.tail_beat, arc.tail_x, arc.tail_y, arc.tail_color,
                     arc.tail_direction,
                     custom_data=dict(arc.tail_note_custom_data)).to_v3(),
            ))
        return {
            "version": "3.3.0",
            "bpmEvents": [dict(event) for event in self.bpm_events],
            "rotationEvents": [],
            "colorNotes": color_notes,
            "bombNotes": [bomb.to_v3() for bomb in self.bombs],
            "obstacles": [wall.to_v3() for wall in self.walls],
            "sliders": [arc.to_v3() for arc in self.arcs],
            "burstSliders": [chain.to_v3() for chain in self.chains],
            "waypoints": [],
            "basicBeatmapEvents": [event.to_v3() for event in self.lights],
            "colorBoostBeatmapEvents": [],
            "lightColorEventBoxGroups": [],
            "rotationEventBoxGroups": [],
            "lightTranslationEventBoxGroups": [],
            "customData": custom_data,
        }
