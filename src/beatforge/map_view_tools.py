"""Bounded model-requested visual inspection of a current Beat Saber map."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from .map_previews import (
    render_map_segment_preview_png,
    render_map_revision_diff_png,
    render_presentation_payload_png,
    render_playfield_contact_sheet_png,
    render_player_perspective_contact_sheet_png,
    render_hand_motion_trace_png,
    render_hand_spatial_path_png,
    render_synchronized_sg_map_png,
    render_map_window_comparison_png,
    render_event_timeline_png,
    render_lighting_storyboard_png,
)


MAX_MAP_VIEW_CALLS = 2
MAX_MAP_VIEW_SECONDS = 8.0
MAX_DRAFT_PREVIEW_CALLS = 4
MAX_DRAFT_CHUNK_CHARS = 12_000
DIAGNOSTIC_BUNDLE_VIEWS = ("lane_time", "perspective", "motion_trace")


def _render_local_views(difficulty: dict[str, Any], view: str, *, bpm: float,
                        start_seconds: float, end_seconds: float,
                        mapping_extensions: bool = False,
                        comparison_difficulty: dict[str, Any] | None = None,
                        audio_change_beats: tuple[float, ...] = (),
                        ) -> tuple[tuple[bytes, ...], tuple[str, ...]]:
    """Render one requested view or a labeled-by-order local diagnostic set."""
    views = DIAGNOSTIC_BUNDLE_VIEWS if view == "diagnostic_bundle" else (view,)
    images: list[bytes] = []
    common = {"bpm": bpm, "start_seconds": start_seconds,
              "end_seconds": end_seconds,
              "mapping_extensions": mapping_extensions}
    for item in views:
        if item == "presentation":
            image = render_presentation_payload_png(
                difficulty, bpm=bpm, start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        elif item == "revision_diff":
            image = render_map_revision_diff_png(
                comparison_difficulty or {}, difficulty, bpm=bpm,
                start_seconds=start_seconds, end_seconds=end_seconds,
                mapping_extensions=mapping_extensions,
            )
        elif item == "lane_time":
            image = render_map_segment_preview_png(
                difficulty, **common, audio_change_beats=audio_change_beats,
            )
        elif item == "event_timeline":
            image = render_event_timeline_png(
                difficulty, bpm=bpm, start_seconds=start_seconds,
                end_seconds=end_seconds, audio_change_beats=audio_change_beats,
            )
        elif item == "lighting_storyboard":
            image = render_lighting_storyboard_png(
                difficulty, bpm=bpm, start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        elif item == "top_down":
            image = render_playfield_contact_sheet_png(difficulty, **common)
        elif item == "motion_trace":
            image = render_hand_motion_trace_png(difficulty, **common)
        elif item == "spatial_path":
            image = render_hand_spatial_path_png(difficulty, **common)
        else:
            image = render_player_perspective_contact_sheet_png(
                difficulty, **common,
            )
        images.append(image)
    return tuple(images), tuple(views)


def _one_or_many(images: tuple[bytes, ...]) -> bytes | tuple[bytes, ...]:
    return images[0] if len(images) == 1 else images


def _render_with_sg_context(
    spectrogram_images: tuple[bytes, ...],
    spectrogram_windows: list[tuple[int, dict[str, Any]]],
    map_png: bytes, *, start_seconds: float, end_seconds: float,
) -> tuple[bytes, str, dict[str, Any]]:
    """Compose source SG panels only when they continuously cover the requested map slice."""
    if not spectrogram_images or not spectrogram_windows:
        return map_png, "unavailable", {}

    rows = [row for row in spectrogram_windows
            if str(row[1].get("file", "")).lower() != "overview.png"]
    if not rows:
        rows = list(spectrogram_windows)
    rows.sort(key=lambda row: (row[1]["start_seconds"], row[1]["end_seconds"], row[0]))

    selected: list[tuple[int, dict[str, Any]]] = []
    cursor = start_seconds
    for index, metadata in rows:
        panel_start = metadata["start_seconds"]
        panel_end = metadata["end_seconds"]
        if panel_end <= cursor + 1e-3:
            continue
        if panel_start > cursor + 1e-3:
            break
        selected.append((index, metadata))
        cursor = max(cursor, panel_end)
        if cursor >= end_seconds - 1e-3:
            break
    if not selected or cursor < end_seconds - 1e-3:
        return map_png, "unavailable", {}

    sg_windows = tuple((metadata["start_seconds"], metadata["end_seconds"])
                       for _index, metadata in selected)
    exact = (math.isclose(sg_windows[0][0], start_seconds, abs_tol=1e-3)
             and math.isclose(sg_windows[-1][1], end_seconds, abs_tol=1e-3))
    try:
        combined = render_synchronized_sg_map_png(
            tuple(spectrogram_images[index] for index, _metadata in selected),
            map_png, start_seconds=start_seconds, end_seconds=end_seconds,
            sg_windows=sg_windows,
        )
    except (OSError, ValueError, IndexError):
        return map_png, "unavailable", {}

    view_kind = ("synchronized_sg_panel" if len(selected) == 1 and exact else
                 "synchronized_sg_panels" if exact else "overlapping_sg_context")
    details = {
        "sg_start_seconds": sg_windows[0][0],
        "sg_end_seconds": sg_windows[-1][1],
    }
    files = [metadata.get("file") for _index, metadata in selected
             if isinstance(metadata.get("file"), str)]
    if len(files) == 1:
        details["spectrogram_file"] = files[0]
    elif files:
        details["spectrogram_files"] = files
    return combined, view_kind, details
MAP_VIEW_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_map_view",
        "description": (
            "Request one visual view of a chosen time range in the current map. "
            "Use this when the supplied diagnostic images do not show the exact "
            "moment or representation you need. Choose presentation for authored "
            "lighting/Vivify payloads, event_timeline for separate time-aligned "
            "tracks of notes, obstacles, arcs/chains, and presentation events, "
            "lighting_storyboard for authored palette and explicit color swatches, "
            "motion_trace to inspect timed hand paths, "
            "spatial_path to inspect a single-grid hand path, "
            "phrase_compare to place two equal-duration windows from the same map "
            "side by side, "
            "diagnostic_bundle to receive separate lane/time, player-perspective, "
            "and hand-motion images for one interval, and revision_diff to compare "
            "accepted repairs. A bundle still uses one request. "
            "You may make at most two requests."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_seconds": {
                    "type": "number", "minimum": 0,
                    "description": "Start of window A; for phrase_compare, this is the first interval.",
                },
                "end_seconds": {
                    "type": "number", "minimum": 0,
                    "description": "End of window A; for phrase_compare, this is the first interval.",
                },
                "compare_start_seconds": {
                    "type": "number", "minimum": 0,
                    "description": "Required for phrase_compare: start of the equal-duration second interval B.",
                },
                "compare_end_seconds": {
                    "type": "number", "minimum": 0,
                    "description": "Required for phrase_compare: end of the equal-duration second interval B.",
                },
                "view": {
                    "type": "string",
                    "enum": ["lane_time", "event_timeline", "lighting_storyboard", "top_down", "perspective", "motion_trace", "spatial_path", "presentation", "phrase_compare", "diagnostic_bundle",
                             "revision_diff"],
                },
            },
            "required": ["start_seconds", "end_seconds", "view"],
            "additionalProperties": False,
        },
    },
}

DRAFT_MAP_PREVIEW_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "preview_map_chunk",
        "description": (
            "Submit one complete new map section and inspect a local view of the "
            "accumulated draft before continuing. The section must use only the "
            "documented BeatForge map DSL. Earlier submitted sections are retained; "
            "phrase_compare shows two equal-duration windows side by side; provide "
            "compare_start_seconds and compare_end_seconds for its second window. "
            "do not repeat them. Choose diagnostic_bundle for separate lane/time, "
            "player-perspective, and hand-motion images of the same interval. Each "
            "bundle uses one checkpoint. Choose event_timeline to inspect separate "
            "time-aligned tracks of gameplay and presentation events, or "
            "lighting_storyboard to inspect authored palette and explicit scene colors. Up to four "
            "checkpoints are available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "minLength": 1,
                           "maxLength": MAX_DRAFT_CHUNK_CHARS},
                "start_seconds": {"type": "number", "minimum": 0},
                "end_seconds": {"type": "number", "minimum": 0},
                "compare_start_seconds": {"type": "number", "minimum": 0},
                "compare_end_seconds": {"type": "number", "minimum": 0},
                "view": {"type": "string", "enum": [
                    "lane_time", "event_timeline", "lighting_storyboard", "top_down", "perspective", "motion_trace", "spatial_path", "presentation", "phrase_compare", "diagnostic_bundle",
                    "revision_diff",
                ]},
            },
            "required": ["script", "start_seconds", "end_seconds", "view"],
            "additionalProperties": False,
        },
    },
}

CURRENT_DRAFT_MAP_VIEW_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_current_map_view",
        "description": (
            "Request a read-only visual view of an already accepted part of the current draft. "
            "This does not submit or change map code. Choose another time range or representation "
            "event_timeline for separate tracks of gameplay and presentation events, "
            "lighting_storyboard for authored palette and explicit scene colors, or "
            "diagnostic_bundle for separate lane/time, player-perspective, and hand-motion images. "
            "It shares the four-checkpoint budget with "
            "preview_map_chunk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_seconds": {"type": "number", "minimum": 0},
                "end_seconds": {"type": "number", "minimum": 0},
                "compare_start_seconds": {"type": "number", "minimum": 0},
                "compare_end_seconds": {"type": "number", "minimum": 0},
                "view": {"type": "string", "enum": [
                    "lane_time", "event_timeline", "lighting_storyboard", "top_down", "perspective", "motion_trace", "spatial_path", "presentation", "phrase_compare", "diagnostic_bundle",
                    "revision_diff",
                ]},
            },
            "required": ["start_seconds", "end_seconds", "view"],
            "additionalProperties": False,
        },
    },
}


class ProgressiveMapPreview:
    """Safely accumulate complete map sections and return focused draft images."""

    def __init__(self, *, bpm: float, duration_seconds: float,
                 mapping_extensions: bool = False,
                 spectrogram_images: tuple[bytes, ...] = (),
                 spectrogram_manifest: dict[str, Any] | None = None):
        if not math.isfinite(float(bpm)) or bpm <= 0:
            raise ValueError("map-preview BPM must be finite and positive")
        if not math.isfinite(float(duration_seconds)) or duration_seconds <= 0:
            raise ValueError("map-preview duration must be finite and positive")
        self.bpm = float(bpm)
        self.duration_seconds = float(duration_seconds)
        self.mapping_extensions = mapping_extensions
        self.spectrogram_images = tuple(spectrogram_images)
        self.spectrogram_manifest = spectrogram_manifest or {}
        self.spectrogram_windows: list[tuple[int, dict[str, Any]]] = []
        rows = (spectrogram_manifest or {}).get("images", []) or []
        # Packed model-input images no longer align one-to-one with the source
        # manifest. Keep this view silent until original source panels are passed.
        if len(rows) != len(self.spectrogram_images):
            rows = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or index >= len(self.spectrogram_images):
                continue
            try:
                sg_start = float(row["start_seconds"])
                sg_end = float(row["end_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(sg_start) and math.isfinite(sg_end) and sg_end > sg_start:
                self.spectrogram_windows.append((index, {
                    "start_seconds": sg_start, "end_seconds": sg_end,
                    "file": row.get("file"),
                }))
        self.script_chunks: list[str] = []
        self.calls = 0
        self.preview_calls = 0
        self.inspection_calls = 0
        self.actions: list[dict[str, Any]] = []
        self.difficulty: dict[str, Any] = {}
        self.previous_difficulty: dict[str, Any] | None = None

    @property
    def script(self) -> str:
        return "\n".join(chunk.rstrip() for chunk in self.script_chunks if chunk.strip())

    def tool_schema(self) -> dict[str, Any]:
        return deepcopy(DRAFT_MAP_PREVIEW_TOOL_SCHEMA)

    def inspection_tool_schema(self) -> dict[str, Any]:
        schema = deepcopy(CURRENT_DRAFT_MAP_VIEW_TOOL_SCHEMA)
        if self.previous_difficulty is None:
            schema["function"]["parameters"]["properties"]["view"]["enum"] = [
                view for view in schema["function"]["parameters"]["properties"]["view"]["enum"]
                if view != "revision_diff"
            ]
        return schema

    def inspect(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], bytes | tuple[bytes, ...] | None]:
        self.calls += 1
        self.preview_calls += 1
        if self.calls > MAX_DRAFT_PREVIEW_CALLS:
            return ({"error": f"draft-preview request budget exhausted ({MAX_DRAFT_PREVIEW_CALLS} calls)"}, None)
        if not isinstance(arguments, dict):
            return ({"error": "arguments must be an object"}, None)
        required = {"script", "start_seconds", "end_seconds", "view"}
        view = arguments.get("view")
        accepted = set(required)
        if view == "phrase_compare":
            accepted.update(("compare_start_seconds", "compare_end_seconds"))
        if set(arguments) != accepted:
            return ({"error": "phrase_compare also requires compare_start_seconds and compare_end_seconds; other views accept only script, start_seconds, end_seconds, and view"}, None)
        chunk = arguments.get("script")
        if (not isinstance(chunk, str) or not chunk.strip()
                or len(chunk) > MAX_DRAFT_CHUNK_CHARS):
            return ({"error": f"script must contain 1-{MAX_DRAFT_CHUNK_CHARS} characters"}, None)
        try:
            start = MapViewInspector._time(arguments["start_seconds"], "start_seconds")
            end = MapViewInspector._time(arguments["end_seconds"], "end_seconds")
        except (KeyError, ValueError) as exc:
            return ({"error": str(exc)}, None)
        if not isinstance(view, str) or view not in {
                "lane_time", "event_timeline", "lighting_storyboard", "top_down", "perspective", "motion_trace", "spatial_path", "presentation", "phrase_compare", "diagnostic_bundle", "revision_diff"}:
            return ({"error": "view must be lane_time, event_timeline, lighting_storyboard, top_down, perspective, motion_trace, spatial_path, presentation, phrase_compare, diagnostic_bundle, or revision_diff"}, None)
        if start < 0 or end <= start or end > self.duration_seconds:
            return ({"error": f"time range must be increasing and inside 0-{self.duration_seconds:.3f}s"}, None)
        if end - start > MAX_MAP_VIEW_SECONDS:
            return ({"error": f"requested window exceeds {MAX_MAP_VIEW_SECONDS:g} seconds"}, None)

        compare_start = compare_end = None
        if view == "phrase_compare":
            try:
                compare_start = MapViewInspector._time(arguments["compare_start_seconds"], "compare_start_seconds")
                compare_end = MapViewInspector._time(arguments["compare_end_seconds"], "compare_end_seconds")
            except (KeyError, ValueError) as exc:
                return ({"error": str(exc)}, None)
            if (compare_start < 0 or compare_end <= compare_start
                    or compare_end > self.duration_seconds):
                return ({"error": f"comparison range must be increasing and inside 0-{self.duration_seconds:.3f}s"}, None)
            if compare_end - compare_start > MAX_MAP_VIEW_SECONDS:
                return ({"error": f"comparison window exceeds {MAX_MAP_VIEW_SECONDS:g} seconds"}, None)
            if not math.isclose(end - start, compare_end - compare_start, abs_tol=1e-3):
                return ({"error": "phrase_compare windows must have equal durations"}, None)

        candidate_script = "\n".join(part for part in (*self.script_chunks, chunk) if part.strip())
        try:
            from .script_runner import run_script
            difficulty = run_script(candidate_script, bpm=self.bpm,
                                    mapping_extensions=self.mapping_extensions)
        except (ValueError, TypeError) as exc:
            return ({"error": f"section rejected by the BeatForge DSL: {str(exc)[:220]}"}, None)

        previous_difficulty = self.difficulty
        comparison_difficulty = None
        if view == "revision_diff" and self.script_chunks:
            from .script_runner import run_script
            comparison_difficulty = run_script(
                self.script, bpm=self.bpm,
                mapping_extensions=self.mapping_extensions,
            )
        phrase_result = None
        if view == "phrase_compare":
            phrase_arguments = {
                "start_seconds": start, "end_seconds": end,
                "compare_start_seconds": compare_start,
                "compare_end_seconds": compare_end, "view": view,
            }
            viewer = MapViewInspector(
                difficulty, bpm=self.bpm, duration_seconds=self.duration_seconds,
                mapping_extensions=self.mapping_extensions,
                spectrogram_images=self.spectrogram_images,
                spectrogram_manifest=self.spectrogram_manifest,
            )
            phrase_result, phrase_image = viewer.inspect(phrase_arguments)
            if phrase_image is None or phrase_result.get("status") != "rendered":
                return phrase_result, None
            images, rendered_views = (phrase_image,), (view,)
        else:
            images, rendered_views = _render_local_views(
                difficulty, view, bpm=self.bpm, start_seconds=start,
                end_seconds=end, mapping_extensions=self.mapping_extensions,
                comparison_difficulty=comparison_difficulty,
            )
        rendered_images: list[bytes] = []
        visual_views: list[dict[str, Any]] = []
        if phrase_result is not None:
            rendered_images.extend(images)
            visual_views.append({"view": view, "compared_windows": phrase_result["compared_windows"]})
        else:
            for image, rendered_view in zip(images, rendered_views):
                synced, image_audio_view, image_sg_details = _render_with_sg_context(
                    self.spectrogram_images, self.spectrogram_windows, image,
                    start_seconds=start, end_seconds=end,
                )
                rendered_images.append(synced)
                view_details = {"view": rendered_view, "audio_view": image_audio_view}
                view_details.update(image_sg_details)
                visual_views.append(view_details)
        audio_view = ("per_window" if phrase_result is not None
                      else visual_views[0]["audio_view"])

        # Commit only after compilation and rendering both succeed.
        self.script_chunks.append(chunk)
        self.previous_difficulty = previous_difficulty or None
        self.difficulty = difficulty
        notes = difficulty.get("colorNotes", []) or []
        bombs = difficulty.get("bombNotes", []) or []
        arcs = difficulty.get("sliders", []) or []
        chains = difficulty.get("burstSliders", []) or []
        walls = difficulty.get("obstacles", []) or []
        def in_window(items: list[dict[str, Any]]) -> int:
            return sum(
                1 for item in items
                if start <= float(item.get("b", 0)) * 60.0 / self.bpm <= end
            )
        metadata = {"view": view, "start_seconds": start, "end_seconds": end,
                    "section": len(self.script_chunks), "audio_view": audio_view}
        if phrase_result is not None:
            metadata.update({key: phrase_result[key] for key in (
                "compare_start_seconds", "compare_end_seconds", "compared_windows")})
        if view == "diagnostic_bundle":
            metadata["visual_views"] = visual_views
        else:
            metadata.update({key: value for key, value in visual_views[0].items()
                             if key not in {"view", "audio_view"}})
        self.actions.append(metadata)
        summary = {
            "status": "rendered", **metadata,
            "accumulated": {"sections": len(self.script_chunks),
                            "notes": len(notes), "bombs": len(bombs),
                            "arcs": len(arcs), "chains": len(chains),
                            "walls": len(walls)},
            "window_counts": {"notes": in_window(notes), "bombs": in_window(bombs),
                              "arcs": in_window(arcs), "chains": in_window(chains),
                              "walls": in_window(walls)},
            "instruction": "Continue with only new map DSL calls; previously submitted sections are already retained.",
            "note": ("three separate images in order: lane/time, player perspective, and hand motion trace; each image is paired with the same SG time window when source panels cover it"
                     if view == "diagnostic_bundle" else
                     "side-by-side same-map phrase comparison with per-window SG context when available"
                     if view == "phrase_compare" else
                     "authored lighting/Vivify payload view; values are JSON data, not a game render"
                     if view == "presentation" else
                     "authored palette and explicit event color swatches; not a Unity/Vivify runtime simulation"
                     if view == "lighting_storyboard" else
                     "separate time-aligned tracks of authored gameplay and presentation events; not a game render"
                     if view == "event_timeline" else
                     "time-aligned hand movement trace with beat gaps; not physical hand kinematics or a game render"
                     if view == "motion_trace" else
                     "spatial path with per-beat notes drawn on one grid; not physical hand kinematics or a game render"
                     if view == "spatial_path" else
                     "schematic draft view; not a Beat Saber or Unity render"),
        }
        return summary, _one_or_many(tuple(rendered_images))

    def inspect_current(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], bytes | tuple[bytes, ...] | None]:
        """Render a read-only view of the accepted draft without resubmitting DSL."""
        self.calls += 1
        self.inspection_calls += 1
        if self.calls > MAX_DRAFT_PREVIEW_CALLS:
            return ({"error": f"draft-preview request budget exhausted ({MAX_DRAFT_PREVIEW_CALLS} calls)"}, None)
        if not self.script_chunks:
            return ({"error": "there is no accepted draft section to inspect yet"}, None)
        inspector = MapViewInspector(
            self.difficulty, bpm=self.bpm, duration_seconds=self.duration_seconds,
            mapping_extensions=self.mapping_extensions,
            comparison_difficulty=self.previous_difficulty,
            spectrogram_images=self.spectrogram_images,
            spectrogram_manifest=self.spectrogram_manifest,
        )
        result, png = inspector.inspect(arguments)
        if png is None or result.get("status") != "rendered":
            return result, None

        start = result["start_seconds"]
        end = result["end_seconds"]
        view = result["view"]
        views = (tuple(result["visual_views"]) if view == "diagnostic_bundle"
                 else (view,))
        raw_images = png if isinstance(png, tuple) else (png,)
        rendered_images: list[bytes] = []
        visual_views: list[dict[str, Any]] = []
        if view == "phrase_compare":
            rendered_images.extend(raw_images)
            visual_views.append({"view": view, "compared_windows": result["compared_windows"]})
            audio_view = "per_window"
        else:
            for raw_image, rendered_view in zip(raw_images, views):
                synced, image_audio_view, image_sg_details = _render_with_sg_context(
                    self.spectrogram_images, self.spectrogram_windows, raw_image,
                    start_seconds=start, end_seconds=end,
                )
                rendered_images.append(synced)
                view_details = {"view": rendered_view, "audio_view": image_audio_view}
                view_details.update(image_sg_details)
                visual_views.append(view_details)
            audio_view = visual_views[0]["audio_view"]

        notes = self.difficulty.get("colorNotes", []) or []
        bombs = self.difficulty.get("bombNotes", []) or []
        arcs = self.difficulty.get("sliders", []) or []
        chains = self.difficulty.get("burstSliders", []) or []
        walls = self.difficulty.get("obstacles", []) or []

        def in_window(items: list[dict[str, Any]]) -> int:
            return sum(
                1 for item in items
                if start <= float(item.get("b", 0)) * 60.0 / self.bpm <= end
            )

        metadata = {
            "action": "inspect_current_map_view", "view": view,
            "start_seconds": start, "end_seconds": end,
            "section": len(self.script_chunks), "audio_view": audio_view,
        }
        if view == "phrase_compare":
            metadata.update({key: result[key] for key in (
                "compare_start_seconds", "compare_end_seconds", "compared_windows")})
        if view == "diagnostic_bundle":
            metadata["visual_views"] = visual_views
        else:
            metadata.update({key: value for key, value in visual_views[0].items()
                             if key not in {"view", "audio_view"}})
        self.actions.append(metadata)
        return ({
            "status": "rendered", **metadata,
            "accumulated": {"sections": len(self.script_chunks),
                            "notes": len(notes), "bombs": len(bombs),
                            "arcs": len(arcs), "chains": len(chains),
                            "walls": len(walls)},
            "window_counts": {"notes": in_window(notes), "bombs": in_window(bombs),
                              "arcs": in_window(arcs), "chains": in_window(chains),
                              "walls": in_window(walls)},
            "instruction": "The accepted draft is unchanged; continue with only new map DSL calls.",
            "note": result.get("note"),
        }, _one_or_many(tuple(rendered_images)))


class MapViewInspector:
    """Render only bounded local views from the existing map data."""

    def __init__(self, difficulty: dict[str, Any], *, bpm: float,
                 duration_seconds: float, mapping_extensions: bool = False,
                 comparison_difficulty: dict[str, Any] | None = None,
                 require_revision_diff: bool = False,
                 audio_change_beats: tuple[float, ...] = (),
                 spectrogram_images: tuple[bytes, ...] = (),
                 spectrogram_manifest: dict[str, Any] | None = None):
        if not math.isfinite(float(bpm)) or bpm <= 0:
            raise ValueError("map-view BPM must be finite and positive")
        if not math.isfinite(float(duration_seconds)) or duration_seconds <= 0:
            raise ValueError("map-view duration must be finite and positive")
        self.difficulty = difficulty
        self.bpm = float(bpm)
        self.duration_seconds = float(duration_seconds)
        self.mapping_extensions = mapping_extensions
        self.audio_change_beats = tuple(audio_change_beats)
        self.spectrogram_images = tuple(spectrogram_images)
        self.spectrogram_windows: list[tuple[int, dict[str, Any]]] = []
        rows = (spectrogram_manifest or {}).get("images", []) or []
        if len(rows) == len(self.spectrogram_images):
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                try:
                    sg_start = float(row["start_seconds"])
                    sg_end = float(row["end_seconds"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(sg_start) and math.isfinite(sg_end) and sg_end > sg_start:
                    self.spectrogram_windows.append((index, {
                        "start_seconds": sg_start, "end_seconds": sg_end,
                        "file": row.get("file"),
                    }))
        self.comparison_difficulty = comparison_difficulty
        self.require_revision_diff = require_revision_diff and comparison_difficulty is not None
        self.calls = 0
        self.actions: list[dict[str, Any]] = []

    def tool_schema(self) -> dict[str, Any]:
        """Return a schema constrained to the required revision view on turn one."""
        schema = deepcopy(MAP_VIEW_TOOL_SCHEMA)
        if self.require_revision_diff and self.calls == 0:
            schema["function"]["parameters"]["properties"]["view"]["enum"] = [
                "revision_diff",
            ]
        elif self.comparison_difficulty is None:
            schema["function"]["parameters"]["properties"]["view"]["enum"] = [
                view for view in schema["function"]["parameters"]["properties"]["view"]["enum"]
                if view != "revision_diff"
            ]
        return schema

    @staticmethod
    def _time(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number of seconds")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def inspect(self, arguments: dict[str, Any]) -> tuple[dict[str, Any], bytes | tuple[bytes, ...] | None]:
        self.calls += 1
        if self.calls > MAX_MAP_VIEW_CALLS:
            return ({"error": f"map-view request budget exhausted ({MAX_MAP_VIEW_CALLS} calls)"}, None)
        if not isinstance(arguments, dict):
            return ({"error": "arguments must be an object"}, None)
        view = arguments.get("view")
        accepted = {"start_seconds", "end_seconds", "view"}
        if view == "phrase_compare":
            accepted.update(("compare_start_seconds", "compare_end_seconds"))
        if set(arguments) != accepted:
            return ({"error": "phrase_compare also requires compare_start_seconds and compare_end_seconds; other views accept only start_seconds, end_seconds, and view"}, None)
        try:
            start = self._time(arguments["start_seconds"], "start_seconds")
            end = self._time(arguments["end_seconds"], "end_seconds")
        except (KeyError, ValueError) as exc:
            return ({"error": str(exc)}, None)
        if not isinstance(view, str) or view not in {
                "lane_time", "event_timeline", "lighting_storyboard", "top_down", "perspective", "motion_trace", "spatial_path", "presentation", "phrase_compare", "diagnostic_bundle", "revision_diff"}:
            return ({"error": "view must be lane_time, event_timeline, lighting_storyboard, top_down, perspective, motion_trace, spatial_path, presentation, phrase_compare, diagnostic_bundle, or revision_diff"}, None)
        if (self.require_revision_diff and self.calls == 1 and view != "revision_diff"):
            return ({"error": "the first view must compare the previous accepted revision"}, None)
        if view == "revision_diff" and self.comparison_difficulty is None:
            return ({"error": "no previous accepted repair revision is available"}, None)
        if start < 0 or end <= start or end > self.duration_seconds:
            return ({"error": f"time range must be increasing and inside 0-{self.duration_seconds:.3f}s"}, None)
        if end - start > MAX_MAP_VIEW_SECONDS:
            return ({"error": f"requested window exceeds {MAX_MAP_VIEW_SECONDS:g} seconds"}, None)

        compare_start = compare_end = None
        if view == "phrase_compare":
            try:
                compare_start = self._time(arguments["compare_start_seconds"], "compare_start_seconds")
                compare_end = self._time(arguments["compare_end_seconds"], "compare_end_seconds")
            except (KeyError, ValueError) as exc:
                return ({"error": str(exc)}, None)
            if (compare_start < 0 or compare_end <= compare_start
                    or compare_end > self.duration_seconds):
                return ({"error": f"comparison range must be increasing and inside 0-{self.duration_seconds:.3f}s"}, None)
            if compare_end - compare_start > MAX_MAP_VIEW_SECONDS:
                return ({"error": f"comparison window exceeds {MAX_MAP_VIEW_SECONDS:g} seconds"}, None)
            if not math.isclose(end - start, compare_end - compare_start, abs_tol=1e-3):
                return ({"error": "phrase_compare windows must have equal durations"}, None)

        if view == "phrase_compare":
            first_map = render_map_segment_preview_png(
                self.difficulty, bpm=self.bpm, start_seconds=start,
                end_seconds=end, mapping_extensions=self.mapping_extensions,
                audio_change_beats=self.audio_change_beats,
                title="PHRASE A - SCHEMATIC",
            )
            second_map = render_map_segment_preview_png(
                self.difficulty, bpm=self.bpm, start_seconds=compare_start,
                end_seconds=compare_end, mapping_extensions=self.mapping_extensions,
                audio_change_beats=self.audio_change_beats,
                title="PHRASE B - SCHEMATIC",
            )
            first_view, first_audio, first_sg = _render_with_sg_context(
                self.spectrogram_images, self.spectrogram_windows, first_map,
                start_seconds=start, end_seconds=end,
            )
            second_view, second_audio, second_sg = _render_with_sg_context(
                self.spectrogram_images, self.spectrogram_windows, second_map,
                start_seconds=compare_start, end_seconds=compare_end,
            )
            image = render_map_window_comparison_png(
                first_view, second_view,
                first_start_seconds=start, first_end_seconds=end,
                second_start_seconds=compare_start, second_end_seconds=compare_end,
            )
            windows = [
                {"label": "A", "start_seconds": start, "end_seconds": end,
                 "audio_view": first_audio, **first_sg},
                {"label": "B", "start_seconds": compare_start,
                 "end_seconds": compare_end, "audio_view": second_audio,
                 **second_sg},
            ]
            metadata = {"view": view, "start_seconds": start, "end_seconds": end,
                        "compare_start_seconds": compare_start,
                        "compare_end_seconds": compare_end, "compared_windows": windows}
            self.actions.append(metadata)
            return ({"status": "rendered", **metadata,
                     "note": "side-by-side schematic views of two equal-duration windows from the same current map; each side includes its own SG context only when source panels cover that exact window; not a Beat Saber or Unity render"}, image)

        images, views = _render_local_views(
            self.difficulty, view, bpm=self.bpm, start_seconds=start,
            end_seconds=end, mapping_extensions=self.mapping_extensions,
            comparison_difficulty=self.comparison_difficulty,
            audio_change_beats=self.audio_change_beats,
        )
        metadata = {"view": view, "start_seconds": start, "end_seconds": end}
        if view == "diagnostic_bundle":
            metadata["visual_views"] = list(views)
        self.actions.append(metadata)
        return ({"status": "rendered", **metadata,
                 "note": ("authored event payload view; values are shown as JSON data, "
                          "not a Beat Saber or Unity render" if view == "presentation" else
                          "authored palette and explicit event color swatches; not a Unity/Vivify runtime simulation"
                          if view == "lighting_storyboard" else
                          "same-window schematic revision comparison; not a Beat Saber or Unity render"
                          if view == "revision_diff" else
                          "time-aligned hand movement trace with beat gaps; not physical hand kinematics or a game render"
                          if view == "motion_trace" else
                          "separate time-aligned tracks of authored gameplay and presentation events; not a game render"
                          if view == "event_timeline" else
                          "spatial path with per-beat notes drawn on one grid; not physical hand kinematics or a game render"
                          if view == "spatial_path" else
                          "three ordered local views: lane/time, player perspective, and hand motion trace"
                          if view == "diagnostic_bundle" else
                          "schematic/proxy view; not a Beat Saber or Unity render")},
                _one_or_many(images))
