"""Deterministic schematic views of a Beat Saber map segment for model review."""
from __future__ import annotations

import io
import json
import math
import textwrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont


_LEFT = (242, 88, 112)
_RIGHT = (82, 169, 255)
_ARC = (255, 194, 71)
_CHAIN = (184, 139, 255)
_BOMB = (245, 245, 250)
_GRID = (65, 75, 94)
_TEXT = (228, 234, 244)
_MUTED = (153, 165, 185)
_DIRECTIONS = {
    0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0),
    4: (-0.72, -0.72), 5: (0.72, -0.72),
    6: (-0.72, 0.72), 7: (0.72, 0.72), 8: (0, 0),
}


def _coordinate(value: Any, mapping_extensions: bool) -> float:
    value = float(value)
    if not mapping_extensions:
        return value
    if value >= 1000:
        return (value - 1000) / 1000
    if value <= -1000:
        return (value + 1000) / 1000
    return value


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_map_segment_preview_png(
    difficulty: dict[str, Any], *, bpm: float, start_seconds: float,
    end_seconds: float, mapping_extensions: bool = False,
    width: int = 1400, height: int = 900,
    title: str = "MAP STATE - SCHEMATIC LANE/TIME VIEW",
    show_scene_event_labels: bool = True,
    show_palette: bool = True,
    show_legend: bool = True,
    show_presentation_events: bool = True,
    show_hand_motion_trails: bool = False,
    audio_change_beats: tuple[float, ...] = (),
) -> bytes:
    """Render a labeled lane/time view with notes, sustains, light and scene events.

    This is a schematic chart view, not a Beat Saber or Unity screenshot. All times
    are derived from the constant BPM in the supplied TrackPlan.
    """
    if not math.isfinite(float(bpm)) or bpm <= 0:
        raise ValueError("bpm must be finite and positive")
    if (not math.isfinite(float(start_seconds)) or not math.isfinite(float(end_seconds))
            or end_seconds <= start_seconds):
        raise ValueError("preview time range must be finite and increasing")
    if width < 600 or height < 480:
        raise ValueError("preview dimensions are too small to label the map clearly")

    image = Image.new("RGB", (width, height), (20, 24, 33))
    draw = ImageDraw.Draw(image)
    title_font, label_font = _font(25), _font(16)
    small_font = _font(13)
    start_beat = start_seconds * bpm / 60.0
    end_beat = end_seconds * bpm / 60.0
    visible_audio_change_beats = tuple(sorted({
        float(beat) for beat in audio_change_beats
        if isinstance(beat, (int, float)) and not isinstance(beat, bool)
        and math.isfinite(float(beat))
        and start_beat <= float(beat) < end_beat
    }))
    left, right = 170, width - 36
    chart_top = (151 if show_palette else 127) if visible_audio_change_beats else (
        126 if show_palette else 102)
    chart_bottom = height - (150 if show_presentation_events else 82)
    lane_count = 6 if mapping_extensions else 4
    row_count = lane_count * 3
    row_height = (chart_bottom - chart_top) / row_count
    x_at = lambda seconds: left + (seconds - start_seconds) / (end_seconds - start_seconds) * (right - left)

    draw.text((36, 25), title, font=title_font, fill=_TEXT)
    subtitle = (
        f"{start_seconds:.2f}-{end_seconds:.2f} s  |  beats "
        f"{start_beat:.2f}-{end_beat:.2f}  |  not a game render"
    )
    if show_hand_motion_trails:
        subtitle += "  |  trails join hits; they are not sustains"
    draw.text((36, 64), subtitle,
              font=label_font, fill=_MUTED)
    if visible_audio_change_beats:
        draw.text((36, 96),
                  "AMBER DASHES: strong spectral-change candidates, not detected note onsets",
                  font=small_font, fill=(255, 196, 80))

    custom_data = difficulty.get("customData") or {}
    palette = custom_data.get("_beatforgeScenePalette", {}) if isinstance(custom_data, dict) else {}
    if show_palette:
        swatch_x = width - 405
        draw.text((swatch_x, 43), "PALETTE", font=small_font, fill=_MUTED)
        palette_names = ("environment_left", "environment_right", "saber_left", "saber_right")
        for index, name in enumerate(palette_names):
            color = palette.get(name)
            if (isinstance(color, (list, tuple)) and len(color) >= 3
                    and all(isinstance(v, (int, float)) for v in color[:3])):
                rgb = tuple(max(0, min(255, round(v))) for v in color[:3])
                y = 57 + index * 15
                draw.rounded_rectangle((swatch_x + 74, y, swatch_x + 94, y + 13), radius=3, fill=rgb)
                short_name = {"environment_left": "env left", "environment_right": "env right",
                              "saber_left": "saber left", "saber_right": "saber right"}[name]
                draw.text((swatch_x + 102, y), short_name, font=small_font, fill=_MUTED)

    def row_for(x_value: Any, y_value: Any) -> float:
        x_value = _coordinate(x_value, mapping_extensions)
        y_value = _coordinate(y_value, mapping_extensions)
        row = max(0.0, min(row_count - 1.0, y_value * lane_count + x_value))
        return chart_top + (row + 0.5) * row_height

    for row in range(row_count):
        y = chart_top + row * row_height
        draw.line((left, y, right, y), fill=_GRID, width=1)
        x_label, y_label = row % lane_count, row // lane_count
        draw.text((35, y + 2), f"x{x_label} y{y_label}", font=small_font, fill=_MUTED)
    draw.line((left, chart_bottom, right, chart_bottom), fill=_GRID, width=1)

    duration = end_seconds - start_seconds
    # Keep time labels legible for both short repair windows and whole-song views.
    tick = (1.0 if duration <= 8.0 else
            max(2.0, math.ceil(duration / 12.0 / 2.0) * 2.0))
    first_tick = math.ceil(start_seconds / tick) * tick
    value = first_tick
    while value <= end_seconds + 1e-8:
        x = x_at(value)
        draw.line((x, chart_top, x, chart_bottom), fill=(49, 57, 72), width=1)
        beat = value * bpm / 60.0
        label = f"{value:g}s / b{beat:.1f}"
        label_y, anchor = chart_bottom + 7, "la"
        label_x = min(x + 3, right - 100)
        if value >= end_seconds - tick * 0.25:
            # Put the final tick on a second row so it cannot collide with the
            # preceding label in short, exact-window previews.
            label_y, label_x, anchor = chart_bottom + 24, right - 3, "ra"
        draw.text((label_x, label_y), label,
                  font=small_font, fill=_MUTED, anchor=anchor)
        value += tick

    for beat in visible_audio_change_beats:
        x = x_at(beat * 60.0 / bpm)
        draw.polygon([(x, chart_top - 1), (x - 4, chart_top - 7),
                      (x + 4, chart_top - 7)], fill=(255, 196, 80))
        for y in range(round(chart_top + 4), round(chart_bottom), 18):
            draw.line((x, y, x, min(y + 6, chart_bottom)),
                      fill=(128, 101, 56), width=1)

    def seconds(beat: Any) -> float:
        return float(beat) * 60.0 / bpm

    def clipped_point(beat: Any, x: Any, y: Any) -> tuple[float, float]:
        t = seconds(beat)
        return x_at(max(start_seconds, min(end_seconds, t))), row_for(x, y)

    def draw_direction(cx: float, cy: float, direction: Any, color: tuple[int, int, int]) -> None:
        try:
            dx, dy = _DIRECTIONS[int(direction)]
        except (KeyError, TypeError, ValueError):
            return
        if dx == dy == 0:
            draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color)
            return
        norm = math.hypot(dx, dy)
        dx, dy = dx / norm * 8, dy / norm * 8
        ex, ey = cx + dx, cy + dy
        draw.line((cx - dx * 0.45, cy - dy * 0.45, ex, ey), fill=color, width=3)
        back_x, back_y = ex - dx * 0.55, ey - dy * 0.55
        draw.polygon([(ex, ey), (back_x - dy * 0.45, back_y + dx * 0.45),
                      (back_x + dy * 0.45, back_y - dx * 0.45)], fill=color)

    # Walls occupy a horizontal interval on their authored grid row.
    for wall in difficulty.get("obstacles", []) or []:
        begin, finish = seconds(wall.get("b", 0)), seconds(wall.get("b", 0)) + float(wall.get("d", 0)) * 60.0 / bpm
        if finish < start_seconds or begin > end_seconds:
            continue
        y = row_for(wall.get("x", 0), wall.get("y", 0))
        x1, x2 = x_at(max(start_seconds, begin)), x_at(min(end_seconds, finish))
        draw.rounded_rectangle((x1, y - row_height * 0.32, x2, y + row_height * 0.32),
                               radius=4, fill=(111, 119, 134), outline=(206, 213, 224), width=1)

    # Arcs and chains appear as sustained paths between their authored endpoints.
    for arc in difficulty.get("sliders", []) or []:
        begin, finish = seconds(arc.get("b", 0)), seconds(arc.get("tb", arc.get("b", 0)))
        if finish < start_seconds or begin > end_seconds:
            continue
        p1 = clipped_point(arc.get("b", 0), arc.get("x", 0), arc.get("y", 0))
        p2 = clipped_point(arc.get("tb", arc.get("b", 0)), arc.get("tx", arc.get("x", 0)),
                           arc.get("ty", arc.get("y", 0)))
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 - row_height * 0.75
        points = []
        for index in range(17):
            t = index / 16
            points.append(((1 - t) ** 2 * p1[0] + 2 * (1 - t) * t * mx + t ** 2 * p2[0],
                           (1 - t) ** 2 * p1[1] + 2 * (1 - t) * t * my + t ** 2 * p2[1]))
        draw.line(points, fill=_ARC, width=5)

    for chain in difficulty.get("burstSliders", []) or []:
        begin, finish = seconds(chain.get("b", 0)), seconds(chain.get("tb", chain.get("b", 0)))
        if finish < start_seconds or begin > end_seconds:
            continue
        p1 = clipped_point(chain.get("b", 0), chain.get("x", 0), chain.get("y", 0))
        p2 = clipped_point(chain.get("tb", chain.get("b", 0)), chain.get("tx", chain.get("x", 0)),
                           chain.get("ty", chain.get("y", 0)))
        draw.line((p1, p2), fill=_CHAIN, width=4)
        slices = max(2, min(12, int(chain.get("sc", 2))))
        for index in range(slices):
            t = index / max(1, slices - 1)
            cx, cy = p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=_CHAIN,
                         outline=(28, 31, 42), width=1)

    if show_hand_motion_trails:
        for hand, color in ((0, _LEFT), (1, _RIGHT)):
            hand_notes = sorted(
                (note for note in difficulty.get("colorNotes", []) or []
                 if int(note.get("c", 0)) == hand),
                key=lambda note: float(note.get("b", 0)),
            )
            for first, second in zip(hand_notes, hand_notes[1:]):
                first_beat = float(first.get("b", 0))
                second_beat = float(second.get("b", 0))
                if second_beat <= first_beat:
                    continue
                first_time, second_time = seconds(first_beat), seconds(second_beat)
                if second_time < start_seconds or first_time > end_seconds:
                    continue
                p1 = clipped_point(first_beat, first.get("x", 0), first.get("y", 0))
                p2 = clipped_point(second_beat, second.get("x", 0), second.get("y", 0))
                draw.line((p1, p2), fill=color, width=3)
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                length = math.hypot(dx, dy)
                if length >= 1:
                    ux, uy = dx / length, dy / length
                    tip_x, tip_y = p2[0] - ux * 10, p2[1] - uy * 10
                    base_x, base_y = tip_x - ux * 7, tip_y - uy * 7
                    draw.polygon(((tip_x, tip_y),
                                  (base_x - uy * 5, base_y + ux * 5),
                                  (base_x + uy * 5, base_y - ux * 5)), fill=color)
                if length >= 52:
                    midpoint = ((p1[0] + p2[0]) / 2,
                                (p1[1] + p2[1]) / 2 - 12)
                    delta_label = f"Δ{second_beat - first_beat:g}b"
                    bounds = draw.textbbox((0, 0), delta_label, font=small_font)
                    label_width = bounds[2] - bounds[0]
                    draw.rounded_rectangle(
                        (midpoint[0] - label_width / 2 - 3, midpoint[1] - 8,
                         midpoint[0] + label_width / 2 + 3, midpoint[1] + 8),
                        radius=3, fill=(20, 24, 33), outline=color, width=1,
                    )
                    draw.text(midpoint, delta_label, font=small_font,
                              fill=_TEXT, anchor="mm")

    for note in difficulty.get("colorNotes", []) or []:
        t = seconds(note.get("b", 0))
        if not start_seconds <= t <= end_seconds:
            continue
        cx, cy = x_at(t), row_for(note.get("x", 0), note.get("y", 0))
        color = _LEFT if int(note.get("c", 0)) == 0 else _RIGHT
        radius = max(8, min(13, int(row_height * 0.32)))
        draw.rounded_rectangle((cx - radius, cy - radius, cx + radius, cy + radius),
                               radius=3, fill=color, outline=(8, 12, 18), width=1)
        draw_direction(cx, cy, note.get("d", 8), (20, 25, 34))

    for bomb in difficulty.get("bombNotes", []) or []:
        t = seconds(bomb.get("b", 0))
        if start_seconds <= t <= end_seconds:
            cx, cy = x_at(t), row_for(bomb.get("x", 0), bomb.get("y", 0))
            draw.polygon([(cx, cy - 10), (cx + 9, cy), (cx, cy + 10), (cx - 9, cy)],
                         fill=_BOMB, outline=(20, 24, 33))

    # Presentation events are shown in separate labeled strips under the grid.
    if show_presentation_events:
        event_y = chart_bottom + 50
        draw.text((35, event_y), "LIGHT", font=small_font, fill=(255, 223, 124))
        for event in difficulty.get("basicBeatmapEvents", []) or []:
            t = seconds(event.get("b", 0))
            if start_seconds <= t <= end_seconds:
                x = x_at(t)
                draw.ellipse((x - 5, event_y, x + 5, event_y + 10), fill=(255, 223, 124))
                draw.text((x + 7, event_y - 2), str(event.get("et", "?")),
                          font=small_font, fill=(255, 223, 124))

        draw.text((35, event_y + 24), "SCENE", font=small_font, fill=(160, 220, 196))
        events = ((custom_data.get("_customEvents", []) if isinstance(custom_data, dict) else []) or [])
        scene_groups: dict[float, dict[str, int]] = {}
        for event in events:
            t = seconds(event.get("b", 0))
            if not start_seconds <= t <= end_seconds:
                continue
            event_type = str(event.get("t", "scene"))
            short_type = {"InstantiatePrefab": "asset", "SetRenderingSettings": "render"}.get(
                event_type, event_type[:10])
            group = scene_groups.setdefault(t, {})
            group[short_type] = group.get(short_type, 0) + 1

        label_rows: list[list[tuple[int, int]]] = []
        for t, kinds in sorted(scene_groups.items()):
            x = x_at(t)
            draw.rounded_rectangle((x - 5, event_y + 25, x + 5, event_y + 35), radius=2,
                                   fill=(160, 220, 196))
            if not show_scene_event_labels:
                continue
            labels = [f"{name} x{count}" if count > 1 else name
                      for name, count in sorted(kinds.items())]
            label = " / ".join(labels)
            bounds = draw.textbbox((0, 0), label, font=small_font)
            label_width = bounds[2] - bounds[0]
            label_x = x + 7
            if label_x + label_width > right:
                label_x = max(left, x - label_width - 7)
            row = 0
            while row < len(label_rows) and any(
                not (label_x + label_width + 4 <= occupied_left
                     or occupied_right + 4 <= label_x)
                for occupied_left, occupied_right in label_rows[row]
            ):
                row += 1
            if row == len(label_rows):
                label_rows.append([])
            label_rows[row].append((label_x, label_x + label_width))
            draw.text((label_x, event_y + 22 + row * 16), label, font=small_font,
                      fill=(160, 220, 196))

    # Compact legend makes the image legible without an auxiliary prose key.
    if show_legend:
        legend_y = height - 32
        items = [("L path + notes" if show_hand_motion_trails else "L hand", _LEFT),
                 ("R path + notes" if show_hand_motion_trails else "R hand", _RIGHT),
                 ("arc", _ARC), ("chain slices", _CHAIN),
                 ("bomb", _BOMB), ("wall", (153, 161, 176))]
        legend_x = left
        for label, color in items:
            draw.rounded_rectangle((legend_x, legend_y, legend_x + 13, legend_y + 13),
                                   radius=3, fill=color)
            draw.text((legend_x + 19, legend_y - 2), label, font=small_font, fill=_MUTED)
            legend_x += 135 if label != "chain slices" else 165

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_presentation_payload_png(
    difficulty: dict[str, Any], *, bpm: float, start_seconds: float,
    end_seconds: float, width: int = 1600, max_events: int = 40,
) -> bytes:
    """Render exact serialized lighting/scene payloads for a local time window.

    This is a data inspector, not a simulation of how the game renders events.
    Event rows preserve the authored JSON fields and explicitly report when the
    display cap omits rows.
    """
    if not math.isfinite(float(bpm)) or bpm <= 0:
        raise ValueError("presentation preview BPM must be finite and positive")
    if (not math.isfinite(float(start_seconds)) or not math.isfinite(float(end_seconds))
            or start_seconds < 0 or end_seconds <= start_seconds):
        raise ValueError("presentation preview range must be finite and increasing")
    if width < 800 or max_events < 1:
        raise ValueError("presentation preview dimensions or event limit are invalid")

    custom_data = difficulty.get("customData") or {}
    if not isinstance(custom_data, dict):
        custom_data = {}
    palette = custom_data.get("_beatforgeScenePalette", {})
    if not isinstance(palette, dict):
        palette = {}

    def in_window(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        try:
            seconds = float(item.get("b")) * 60.0 / bpm
        except (TypeError, ValueError):
            return False
        return math.isfinite(seconds) and start_seconds <= seconds <= end_seconds

    rows: list[tuple[float, str, dict[str, Any]]] = []
    for event in difficulty.get("basicBeatmapEvents", []) or []:
        if in_window(event):
            beat = float(event.get("b", 0))
            rows.append((beat * 60.0 / bpm, "STANDARD LIGHT", event))
    for event in custom_data.get("_customEvents", []) or []:
        if in_window(event):
            beat = float(event.get("b", 0))
            rows.append((beat * 60.0 / bpm, "SCENE / VIVIFY", event))
    rows.sort(key=lambda row: (row[0], row[1]))
    total = len(rows)
    rows = rows[:max_events]

    event_height = 92
    header_height = 170
    footer_height = 58
    height = max(560, header_height + max(1, len(rows)) * event_height + footer_height)
    image = Image.new("RGB", (width, height), (20, 24, 33))
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _font(25), _font(15), _font(13)
    draw.text((32, 22), "PRESENTATION PAYLOAD INSPECTOR - EXACT MAP DATA",
              font=title_font, fill=_TEXT)
    draw.text((34, 60),
              f"{start_seconds:.2f}-{end_seconds:.2f}s | beats "
              f"{start_seconds * bpm / 60:.2f}-{end_seconds * bpm / 60:.2f} | "
              "not a Beat Saber / Unity render",
              font=body_font, fill=_MUTED)

    palette_y = 98
    draw.text((34, palette_y), "SCENE PALETTE", font=small_font, fill=(160, 220, 196))
    px = 150
    for name, value in palette.items():
        label = str(name)
        if (isinstance(value, (list, tuple)) and len(value) in {3, 4}
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(float(v)) for v in value)):
            rgb = tuple(max(0, min(255, round(float(v)))) for v in value[:3])
            draw.rounded_rectangle((px, palette_y, px + 24, palette_y + 16),
                                   radius=3, fill=rgb)
            px += 31
        draw.text((px, palette_y + 1), f"{label}={value!r}",
                  font=small_font, fill=_MUTED)
        px += min(245, 14 + len(label) * 7 + len(repr(value)) * 7)
        if px > width - 360:
            px = 34
            palette_y += 22
    first_row_y = max(142, palette_y + 32)
    shown = len(rows)
    summary = f"EVENTS IN WINDOW: {total}; showing {shown}"
    if total > shown:
        summary += f"; {total - shown} omitted by display cap (request a narrower window)"
    draw.text((34, first_row_y), summary, font=body_font, fill=(255, 223, 124))
    first_row_y += 30

    def json_text(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))
        except (TypeError, ValueError):
            return repr(value)

    for index, (seconds, category, event) in enumerate(rows):
        y = first_row_y + index * event_height
        beat = float(event.get("b", 0))
        if category == "STANDARD LIGHT":
            payload = {key: event[key] for key in ("et", "i", "f", "customData") if key in event}
            title = (f"{seconds:7.3f}s | beat {beat:8.3f} | {category} | "
                     f"event_type={event.get('et', '?')} value={event.get('i', '?')} "
                     f"float={event.get('f', '?')}")
        else:
            payload = event.get("d", {})
            title = (f"{seconds:7.3f}s | beat {beat:8.3f} | {category} | "
                     f"type={event.get('t', '?')}")
        draw.rounded_rectangle((26, y, width - 26, y + event_height - 7),
                               radius=7, fill=(31, 37, 49), outline=(65, 75, 94), width=1)
        draw.text((40, y + 9), title, font=body_font,
                  fill=(255, 223, 124) if category == "STANDARD LIGHT" else (160, 220, 196))
        encoded = json_text(payload)
        lines = textwrap.wrap(encoded, width=max(60, (width - 80) // 9),
                              break_long_words=True, break_on_hyphens=False)[:3]
        for line_index, line in enumerate(lines):
            draw.text((42, y + 34 + line_index * 17), line, font=small_font, fill=_TEXT)
        if len(textwrap.wrap(encoded, width=max(60, (width - 80) // 9),
                             break_long_words=True, break_on_hyphens=False)) > 3:
            draw.text((width - 300, y + event_height - 25),
                      "payload continues in JSON", font=small_font, fill=_MUTED)

    if not rows:
        draw.text((36, first_row_y + 24), "No standard-light or scene events in this window.",
                  font=body_font, fill=_MUTED)
    draw.text((32, height - 34),
              "Values are copied from authored map JSON; colors are swatches only when explicitly named in the scene palette.",
              font=small_font, fill=_MUTED)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_map_revision_diff_png(
    previous: dict[str, Any], current: dict[str, Any], *, bpm: float,
    start_seconds: float, end_seconds: float,
    mapping_extensions: bool = False, width: int = 900, height: int = 720,
) -> bytes:
    """Render the same authored time window before/after an accepted repair."""
    if width < 600 or height < 480:
        raise ValueError("revision diff panels are too small to label clearly")
    panel_args = {
        "bpm": bpm, "start_seconds": start_seconds, "end_seconds": end_seconds,
        "mapping_extensions": mapping_extensions, "width": width, "height": height,
        "show_palette": False, "show_legend": False,
    }
    before = Image.open(io.BytesIO(render_map_segment_preview_png(
        previous, title="PREVIOUS ACCEPTED REVISION - SCHEMATIC", **panel_args,
    ))).convert("RGB")
    after = Image.open(io.BytesIO(render_map_segment_preview_png(
        current, title="CURRENT MAP AFTER THAT REPAIR - SCHEMATIC", **panel_args,
    ))).convert("RGB")
    gap, header = 20, 52
    canvas = Image.new("RGB", (width * 2 + gap * 3, height + header + gap * 2),
                       (16, 20, 29))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 12), "REVISION COMPARISON - SAME ABSOLUTE TIME WINDOW - NOT A GAME RENDER",
              font=_font(21), fill=_TEXT)
    draw.rounded_rectangle((gap + 2, 36, gap + 14, 48), radius=2, fill=_LEFT)
    draw.text((gap + 20, 34), "left hand", font=_font(13), fill=_MUTED)
    draw.rounded_rectangle((gap + 108, 36, gap + 120, 48), radius=2, fill=_RIGHT)
    draw.text((gap + 126, 34), "right hand", font=_font(13), fill=_MUTED)
    canvas.paste(before, (gap, header))
    canvas.paste(after, (gap * 2 + width, header))
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_map_overview_png(
    difficulty: dict[str, Any], *, bpm: float, duration_seconds: float,
    mapping_extensions: bool = False, width: int = 1800, height: int = 850,
    title: str = "WHOLE-TRACK MAP OVERVIEW - SCHEMATIC",
) -> bytes:
    """Render the full arrangement as a coarse, labeled lane/time overview."""
    return render_map_segment_preview_png(
        difficulty, bpm=bpm, start_seconds=0.0, end_seconds=duration_seconds,
        mapping_extensions=mapping_extensions, width=width, height=height,
        title=title,
        show_scene_event_labels=False,
    )


def render_event_timeline_png(
    difficulty: dict[str, Any], *, bpm: float, start_seconds: float,
    end_seconds: float, audio_change_beats: tuple[float, ...] = (),
    width: int = 1500,
) -> bytes:
    """Render a local, time-aligned inventory of gameplay and presentation events."""
    if not math.isfinite(float(bpm)) or bpm <= 0:
        raise ValueError("event timeline BPM must be finite and positive")
    if (not math.isfinite(float(start_seconds)) or not math.isfinite(float(end_seconds))
            or start_seconds < 0 or end_seconds <= start_seconds):
        raise ValueError("event timeline range must be finite and increasing")
    if width < 800:
        raise ValueError("event timeline is too narrow to label its tracks")

    custom_data = difficulty.get("customData") or {}
    if not isinstance(custom_data, dict):
        custom_data = {}
    scene_events = custom_data.get("_customEvents", []) or []
    if not isinstance(scene_events, list):
        scene_events = []

    tracks = [
        ("LEFT NOTES", _LEFT, "notes", 0),
        ("RIGHT NOTES", _RIGHT, "notes", 1),
        ("BOMBS", _BOMB, "bombs", None),
        ("WALLS", (159, 170, 190), "obstacles", None),
        ("ARCS", _ARC, "sliders", None),
        ("CHAINS", _CHAIN, "burstSliders", None),
        ("STANDARD LIGHT", (255, 220, 112), "lights", None),
        ("SCENE / VIVIFY", (123, 222, 183), "scene", None),
    ]
    left, right = 220, width - 35
    top, row_height, bottom = 142, 68, 32
    height = top + row_height * len(tracks) + bottom
    image = Image.new("RGB", (width, height), (20, 24, 33))
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = _font(24), _font(15), _font(12)
    start_beat, end_beat = start_seconds * bpm / 60.0, end_seconds * bpm / 60.0
    x_at = lambda seconds: left + (seconds - start_seconds) / (end_seconds - start_seconds) * (right - left)
    beat_seconds = lambda beat: float(beat) * 60.0 / bpm

    draw.text((30, 22), "LOCAL EVENT TIMELINE - AUTHORED MAP DATA",
              font=title_font, fill=_TEXT)
    draw.text((32, 61),
              f"{start_seconds:.2f}-{end_seconds:.2f}s | beats {start_beat:.2f}-{end_beat:.2f} | schematic, not a game render",
              font=label_font, fill=_MUTED)
    draw.text((32, 94), "AMBER DASHES: measured spectral-change candidates; not detected note onsets",
              font=small_font, fill=(255, 196, 80))

    duration = end_seconds - start_seconds
    tick = 0.5 if duration <= 4 else 1.0 if duration <= 8 else max(2.0, math.ceil(duration / 10 / 2) * 2.0)
    value = math.ceil(start_seconds / tick) * tick
    while value <= end_seconds + 1e-8:
        x = x_at(value)
        draw.line((x, top - 3, x, top + row_height * len(tracks)),
                  fill=(48, 57, 73), width=1)
        label = f"{value:g}s / b{value * bpm / 60:.1f}"
        if value >= end_seconds - tick * 0.25:
            draw.text((right - 3, top - 21), label,
                      font=small_font, fill=_MUTED, anchor="ra")
        else:
            draw.text((x + 3, top - 21), label, font=small_font, fill=_MUTED)
        value += tick

    event_lists: dict[str, list[dict[str, Any]]] = {
        "notes": [item for item in difficulty.get("colorNotes", []) or []
                  if isinstance(item, dict)],
        "bombs": [item for item in difficulty.get("bombNotes", []) or []
                  if isinstance(item, dict)],
        "obstacles": [item for item in difficulty.get("obstacles", []) or []
                      if isinstance(item, dict)],
        "sliders": [item for item in difficulty.get("sliders", []) or []
                    if isinstance(item, dict)],
        "burstSliders": [item for item in difficulty.get("burstSliders", []) or []
                         if isinstance(item, dict)],
        "lights": [item for item in difficulty.get("basicBeatmapEvents", []) or []
                   if isinstance(item, dict)],
        "scene": [item for item in scene_events if isinstance(item, dict)],
    }

    def number(item: dict[str, Any], key: str, default: float = 0.0) -> float | None:
        try:
            result = float(item.get(key, default))
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    for index, (label, color, key, hand) in enumerate(tracks):
        y0 = top + index * row_height
        cy = y0 + row_height // 2
        draw.line((left, y0, right, y0), fill=_GRID, width=1)
        draw.text((30, cy - 9), label, font=label_font, fill=color)
        source = event_lists[key]
        visible = 0
        for item_index, item in enumerate(source):
            beat = number(item, "b")
            if beat is None:
                continue
            if key == "notes" and number(item, "c", 0) != hand:
                continue
            begin = beat_seconds(beat)
            end = begin
            if key in {"obstacles", "sliders", "burstSliders"}:
                if key == "obstacles":
                    beat_duration = number(item, "d", 0.0)
                    if beat_duration is None or beat_duration < 0:
                        continue
                    end = beat_seconds(beat + beat_duration)
                else:
                    tail = number(item, "tb", beat)
                    if tail is None:
                        continue
                    end = beat_seconds(tail)
                    if end < begin:
                        begin, end = end, begin
            if end < start_seconds or begin > end_seconds:
                continue
            x1, x2 = x_at(max(start_seconds, begin)), x_at(min(end_seconds, end))
            offset = ((item_index % 3) - 1) * 5
            y = cy + offset
            if key == "notes":
                draw.ellipse((x1 - 5, y - 5, x1 + 5, y + 5), fill=color)
            elif key == "bombs":
                draw.polygon(((x1, y - 6), (x1 + 6, y), (x1, y + 6), (x1 - 6, y)), fill=color)
            elif key == "obstacles":
                draw.rounded_rectangle((x1, y - 8, max(x1 + 3, x2), y + 8), radius=3,
                                       fill=color)
            elif key == "sliders":
                draw.line((x1, y, max(x1 + 3, x2), y), fill=color, width=4)
                draw.ellipse((x1 - 4, y - 4, x1 + 4, y + 4), fill=color)
                draw.ellipse((x2 - 4, y - 4, x2 + 4, y + 4), fill=color)
            elif key == "burstSliders":
                draw.line((x1, y, max(x1 + 3, x2), y), fill=color, width=3)
                slices = max(2, min(12, int(number(item, "sc", 2) or 2)))
                for slice_index in range(slices):
                    fraction = slice_index / max(1, slices - 1)
                    x = x1 + (x2 - x1) * fraction
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            elif key in {"lights", "scene"}:
                draw.line((x1, cy - 15, x1, cy + 15), fill=color, width=2)
                draw.ellipse((x1 - 4, cy - 4, x1 + 4, cy + 4), fill=color)
            visible += 1
        draw.text((right + 8, cy - 8), str(visible), font=small_font, fill=_MUTED)

    for beat in audio_change_beats:
        if (isinstance(beat, (int, float)) and not isinstance(beat, bool)
                and math.isfinite(float(beat)) and start_beat <= float(beat) <= end_beat):
            x = x_at(beat_seconds(float(beat)))
            for y in range(top, top + row_height * len(tracks), 16):
                draw.line((x, y, x, min(y + 6, top + row_height * len(tracks))),
                          fill=(128, 101, 56), width=1)

    draw.line((left, top + row_height * len(tracks), right,
               top + row_height * len(tracks)), fill=_GRID, width=1)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _authored_rgb(value: Any) -> tuple[int, int, int] | None:
    if (not isinstance(value, (list, tuple)) or len(value) not in {3, 4}
            or not all(isinstance(channel, (int, float))
                       and not isinstance(channel, bool)
                       and math.isfinite(float(channel)) for channel in value)):
        return None
    channels = [float(channel) for channel in value[:3]]
    if all(0.0 <= channel <= 1.0 for channel in channels):
        channels = [channel * 255.0 for channel in channels]
    return tuple(max(0, min(255, round(channel))) for channel in channels)


def render_lighting_storyboard_png(
    difficulty: dict[str, Any], *, bpm: float, start_seconds: float,
    end_seconds: float, width: int = 1600, max_events: int = 24,
) -> bytes:
    """Show authored scene palette and explicit light/Vivify colors without simulating Unity."""
    if not math.isfinite(float(bpm)) or bpm <= 0:
        raise ValueError("lighting storyboard BPM must be finite and positive")
    if (not math.isfinite(float(start_seconds)) or not math.isfinite(float(end_seconds))
            or start_seconds < 0 or end_seconds <= start_seconds):
        raise ValueError("lighting storyboard range must be finite and increasing")
    if width < 900 or max_events < 1:
        raise ValueError("lighting storyboard dimensions or event limit are invalid")

    custom_data = difficulty.get("customData") or {}
    if not isinstance(custom_data, dict):
        custom_data = {}
    palette = custom_data.get("_beatforgeScenePalette", {})
    if not isinstance(palette, dict):
        palette = {}
    palette_names = ("environment_left", "environment_right", "saber_left", "saber_right")
    color_keys = {
        "ambientEquatorColor", "ambientGroundColor", "ambientSkyColor",
        "fogColor", "subtractiveShadowColor", "ambientLight",
    }

    def seconds(item: Any) -> float | None:
        if not isinstance(item, dict):
            return None
        try:
            result = float(item.get("b", 0)) * 60.0 / bpm
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def collect_colors(value: Any, *, allow_light_color: bool = False) -> list[tuple[str, tuple[int, int, int]]]:
        found: list[tuple[str, tuple[int, int, int]]] = []
        if not isinstance(value, dict):
            return found
        for key, child in value.items():
            name = str(key)
            is_color = name in color_keys or (allow_light_color and name in {"_color", "color"})
            rgb = _authored_rgb(child) if is_color else None
            if rgb is not None:
                found.append((name.lstrip("_"), rgb))
            elif isinstance(child, dict):
                found.extend(collect_colors(child, allow_light_color=allow_light_color))
        return found

    events: list[tuple[float, float, str, list[tuple[str, tuple[int, int, int]]]]] = []
    for event in difficulty.get("basicBeatmapEvents", []) or []:
        at = seconds(event)
        if at is None or not start_seconds <= at <= end_seconds:
            continue
        beat = at * bpm / 60.0
        custom = event.get("customData", {})
        colors = collect_colors(custom, allow_light_color=True)
        label = f"STANDARD LIGHT  et={event.get('et', '?')}  value={event.get('i', '?')}"
        events.append((at, beat, label, colors))
    for event in custom_data.get("_customEvents", []) or []:
        at = seconds(event)
        if at is None or not start_seconds <= at <= end_seconds:
            continue
        beat = at * bpm / 60.0
        payload = event.get("d", {})
        colors = collect_colors(payload)
        label = f"SCENE / VIVIFY  {event.get('t', 'event')}"
        events.append((at, beat, label, colors))
    events.sort(key=lambda row: (row[0], row[2]))
    total = len(events)
    events = events[:max_events]

    header, palette_row, axis_y, row_height, footer = 116, 64, 205, 78, 58
    height = max(420, axis_y + max(1, len(events)) * row_height + footer)
    image = Image.new("RGB", (width, height), (20, 24, 33))
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _font(24), _font(15), _font(12)
    draw.text((32, 20), "LIGHTING STORYBOARD - EXPLICIT AUTHORED COLORS",
              font=title_font, fill=_TEXT)
    draw.text((34, 59),
              f"{start_seconds:.2f}-{end_seconds:.2f}s | beats "
              f"{start_seconds * bpm / 60:.2f}-{end_seconds * bpm / 60:.2f} | "
              "color swatches only; no Unity/Vivify runtime simulation",
              font=body_font, fill=_MUTED)
    draw.text((34, 104), "SCENE PALETTE", font=small_font, fill=(160, 220, 196))
    x_palette = 170
    for name in palette_names:
        rgb = _authored_rgb(palette.get(name))
        if rgb is not None:
            draw.rounded_rectangle((x_palette, 103, x_palette + 26, 121),
                                   radius=3, fill=rgb)
        else:
            draw.rounded_rectangle((x_palette, 103, x_palette + 26, 121),
                                   radius=3, outline=(95, 105, 122), width=1)
        draw.text((x_palette + 34, 103), name, font=small_font, fill=_MUTED)
        x_palette += min(345, 58 + 7 * len(name))

    left, right = 250, width - 34
    draw.line((left, axis_y - 8, right, axis_y - 8), fill=_GRID, width=2)
    span = end_seconds - start_seconds
    for index, (at, beat, label, colors) in enumerate(events):
        y = axis_y + index * row_height
        x = left + (at - start_seconds) / span * (right - left)
        draw.line((left, y + 8, right, y + 8), fill=(45, 53, 68), width=1)
        draw.line((x, axis_y - 8, x, y + row_height - 4), fill=(86, 100, 124), width=1)
        draw.text((32, y + 19), f"{at:.3f}s / b{beat:.3f}",
                  font=small_font, fill=(255, 223, 124))
        draw.text((left + 8, y + 10), label[:105], font=body_font, fill=_TEXT)
        if not colors:
            draw.text((left + 8, y + 38),
                      "No explicit color in this event; runtime/default lighting is not inferred",
                      font=small_font, fill=_MUTED)
        else:
            px = left + 10
            for color_name, rgb in colors[:6]:
                draw.rounded_rectangle((px, y + 39, px + 22, y + 58),
                                       radius=3, fill=rgb)
                draw.text((px + 29, y + 40), color_name,
                          font=small_font, fill=_MUTED)
                px += 42 + min(210, len(color_name) * 7)

    if not events:
        draw.text((left + 8, axis_y + 27),
                  "No standard-light or scene events in this interval.",
                  font=body_font, fill=_MUTED)
    if total > len(events):
        draw.text((32, height - 36),
                  f"Showing {len(events)} of {total} events; request a narrower window for the rest.",
                  font=small_font, fill=(255, 196, 80))
    else:
        draw.text((32, height - 36),
                  "Colors are copied from authored fields; appearance, intensity, easing and asset rendering are not simulated.",
                  font=small_font, fill=_MUTED)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_synchronized_sg_map_png(
    spectrogram_png: bytes | tuple[bytes, ...], map_png: bytes, *,
    start_seconds: float, end_seconds: float,
    sg_windows: tuple[tuple[float, float], ...] | None = None,
) -> bytes:
    """Stack one or more SG panels and a map slice with explicit absolute-time ranges."""
    if (not math.isfinite(float(start_seconds)) or not math.isfinite(float(end_seconds))
            or end_seconds <= start_seconds):
        raise ValueError("synchronized preview needs a finite increasing time range")
    panels = (spectrogram_png,) if isinstance(spectrogram_png, bytes) else tuple(spectrogram_png)
    if not panels:
        raise ValueError("synchronized preview needs at least one SG panel")
    rendered_panels = []
    for panel in panels:
        with Image.open(io.BytesIO(panel)) as source:
            rendered_panels.append(source.convert("RGB"))
    with Image.open(io.BytesIO(map_png)) as source:
        chart = source.convert("RGB")
    width = max(*(panel.width for panel in rendered_panels), chart.width)
    rendered_panels = [
        panel if panel.width == width else
        panel.resize((width, round(panel.height * width / panel.width)), Image.Resampling.LANCZOS)
        for panel in rendered_panels
    ]
    if chart.width != width:
        chart = chart.resize((width, round(chart.height * width / chart.width)), Image.Resampling.LANCZOS)
    separator = 8
    sg_height = sum(panel.height for panel in rendered_panels) + separator * (len(rendered_panels) - 1)
    header_height = 88
    combined = Image.new("RGB", (width, header_height + sg_height + chart.height), (20, 24, 33))
    draw = ImageDraw.Draw(combined)
    if sg_windows is None:
        sg_windows = ((start_seconds, end_seconds),)
    exact_range = (len(sg_windows) > 0
                   and math.isclose(sg_windows[0][0], start_seconds, abs_tol=1e-3)
                   and math.isclose(sg_windows[-1][1], end_seconds, abs_tol=1e-3))
    draw.text((28, 8), "SYNCHRONIZED AUDIO + CURRENT MAP" if exact_range
              else "SG AUDIO CONTEXT + CURRENT MAP SLICE",
              font=_font(24), fill=_TEXT)
    draw.text((28, 38), f"map window: {start_seconds:.2f}-{end_seconds:.2f} s",
              font=_font(15), fill=_MUTED)
    sg_summary = ", ".join(f"{left:.2f}-{right:.2f}s" for left, right in sg_windows)
    draw.text((28, 61), f"SG panel time: {sg_summary}", font=_font(14), fill=_MUTED)
    y = header_height
    for panel in rendered_panels:
        combined.paste(panel, (0, y))
        y += panel.height + separator
    combined.paste(chart, (0, header_height + sg_height))
    output = io.BytesIO()
    combined.save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_playfield_contact_sheet_png(
    difficulty: dict[str, Any], *, bpm: float, start_seconds: float,
    end_seconds: float, mapping_extensions: bool = False,
    width: int = 1400, height: int = 900, max_frames: int = 8,
) -> bytes:
    """Render sequential one-beat playfield snapshots for hand-path review.

    Each cell is a top-down authored grid snapshot, not a player camera or a
    simulation of block approach speed. Frames cover equal beat intervals in
    the supplied local time window.
    """
    if not math.isfinite(float(bpm)) or bpm <= 0:
        raise ValueError("bpm must be finite and positive")
    if (not math.isfinite(float(start_seconds)) or not math.isfinite(float(end_seconds))
            or end_seconds <= start_seconds):
        raise ValueError("contact-sheet time range must be finite and increasing")
    if max_frames < 1 or width < 800 or height < 500:
        raise ValueError("contact-sheet dimensions or frame limit are too small")

    image = Image.new("RGB", (width, height), (20, 24, 33))
    draw = ImageDraw.Draw(image)
    draw.text((30, 18), "PLAYFIELD CONTACT SHEET - TOP-DOWN GRID, NOT GAME RENDER",
              font=_font(22), fill=_TEXT)
    first_beat = start_seconds * bpm / 60.0
    last_beat = end_seconds * bpm / 60.0
    frame_count = min(max_frames, max(1, math.ceil(last_beat - first_beat - 1e-9)))
    beats_per_frame = (last_beat - first_beat) / frame_count
    cols, rows = min(4, frame_count), math.ceil(frame_count / min(4, frame_count))
    margin_x, margin_top, margin_bottom, gap = 24, 62, 20, 14
    panel_w = (width - 2 * margin_x - (cols - 1) * gap) / cols
    panel_h = (height - margin_top - margin_bottom - (rows - 1) * gap) / rows
    lane_count = 6 if mapping_extensions else 4
    grid_cols, grid_rows = lane_count, 3
    custom_data = difficulty.get("customData") or {}
    left_rgb, right_rgb = _LEFT, _RIGHT

    def coord(value: Any) -> float:
        return _coordinate(value, mapping_extensions)

    def center_for(px: float, py: float, gx: Any, gy: Any) -> tuple[float, float]:
        cx = px + (coord(gx) + 0.5) * (panel_w - 26) / grid_cols + 13
        cy = py + 42 + (2.5 - coord(gy)) * (panel_h - 68) / grid_rows
        return cx, cy

    notes = difficulty.get("colorNotes", []) or []
    bombs = difficulty.get("bombNotes", []) or []
    arcs = difficulty.get("sliders", []) or []
    chains = difficulty.get("burstSliders", []) or []
    obstacles = difficulty.get("obstacles", []) or []

    def sec(beat: Any) -> float:
        return float(beat) * 60.0 / bpm

    for index in range(frame_count):
        beat_start = first_beat + index * beats_per_frame
        beat_end = last_beat if index == frame_count - 1 else beat_start + beats_per_frame
        seconds_start, seconds_end = sec(beat_start), sec(beat_end)
        col, row = index % cols, index // cols
        px = margin_x + col * (panel_w + gap)
        py = margin_top + row * (panel_h + gap)
        draw.rounded_rectangle((px, py, px + panel_w, py + panel_h), radius=10,
                               fill=(28, 33, 45), outline=(65, 75, 94), width=1)
        draw.text((px + 10, py + 8),
                  f"b{beat_start:.2f}-{beat_end:.2f}  {seconds_start:.2f}-{seconds_end:.2f}s",
                  font=_font(13), fill=_MUTED)
        grid_x1, grid_x2 = px + 13, px + panel_w - 13
        grid_y1, grid_y2 = py + 37, py + panel_h - 14
        for x_index in range(grid_cols + 1):
            x = grid_x1 + x_index * (grid_x2 - grid_x1) / grid_cols
            draw.line((x, grid_y1, x, grid_y2), fill=_GRID, width=1)
        for y_index in range(grid_rows + 1):
            y = grid_y1 + y_index * (grid_y2 - grid_y1) / grid_rows
            draw.line((grid_x1, y, grid_x2, y), fill=_GRID, width=1)

        def in_frame(beat: Any) -> bool:
            value = float(beat)
            return beat_start <= value < beat_end or (index == frame_count - 1 and value == beat_end)

        for note in notes:
            if not in_frame(note.get("b", 0)):
                continue
            cx, cy = center_for(px, py, note.get("x", 0), note.get("y", 0))
            color = left_rgb if int(note.get("c", 0)) == 0 else right_rgb
            draw.rounded_rectangle((cx - 12, cy - 12, cx + 12, cy + 12), radius=4,
                                   fill=color, outline=(8, 12, 18), width=1)
            try:
                dx, dy = _DIRECTIONS[int(note.get("d", 8))]
                norm = math.hypot(dx, dy)
                if norm:
                    dx, dy = dx / norm * 7, dy / norm * 7
                    ex, ey = cx + dx, cy + dy
                    draw.line((cx - dx * 0.55, cy - dy * 0.55, ex, ey),
                              fill=(20, 25, 34), width=2)
                    back_x, back_y = ex - dx * 0.55, ey - dy * 0.55
                    draw.polygon(((ex, ey),
                                  (back_x - dy * 0.42, back_y + dx * 0.42),
                                  (back_x + dy * 0.42, back_y - dx * 0.42)),
                                 fill=(20, 25, 34))
            except (KeyError, TypeError, ValueError):
                pass
        for bomb in bombs:
            if in_frame(bomb.get("b", 0)):
                cx, cy = center_for(px, py, bomb.get("x", 0), bomb.get("y", 0))
                draw.polygon(((cx, cy - 9), (cx + 9, cy), (cx, cy + 9), (cx - 9, cy)),
                             fill=_BOMB)
        for arc in arcs:
            if in_frame(arc.get("b", 0)):
                cx, cy = center_for(px, py, arc.get("x", 0), arc.get("y", 0))
                tx, ty = center_for(px, py, arc.get("tx", arc.get("x", 0)),
                                    arc.get("ty", arc.get("y", 0)))
                draw.line((cx, cy, tx, ty), fill=_ARC, width=4)
                draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=_ARC)
        for chain in chains:
            if in_frame(chain.get("b", 0)):
                cx, cy = center_for(px, py, chain.get("x", 0), chain.get("y", 0))
                tx, ty = center_for(px, py, chain.get("tx", chain.get("x", 0)),
                                    chain.get("ty", chain.get("y", 0)))
                draw.line((cx, cy, tx, ty), fill=_CHAIN, width=3)
                draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=_CHAIN)
        for wall in obstacles:
            wall_start = sec(wall.get("b", 0))
            wall_end = wall_start + float(wall.get("d", 0)) * 60.0 / bpm
            if wall_start < seconds_end and wall_end > seconds_start:
                cx, cy = center_for(px, py, wall.get("x", 0), wall.get("y", 0))
                draw.rounded_rectangle((cx - 12, cy - 5, cx + 12, cy + 5), radius=3,
                                       fill=(153, 161, 176))

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_map_window_comparison_png(
    first: bytes, second: bytes, *,
    first_start_seconds: float, first_end_seconds: float,
    second_start_seconds: float, second_end_seconds: float,
    title: str = "TWO WINDOWS FROM THE SAME MAP - SCHEMATIC",
) -> bytes:
    """Place two independently rendered time windows side by side."""
    ranges = (first_start_seconds, first_end_seconds,
              second_start_seconds, second_end_seconds)
    if any(not math.isfinite(float(value)) for value in ranges):
        raise ValueError("comparison windows must use finite seconds")
    if first_end_seconds <= first_start_seconds or second_end_seconds <= second_start_seconds:
        raise ValueError("comparison windows must be increasing")
    if not math.isclose(first_end_seconds - first_start_seconds,
                        second_end_seconds - second_start_seconds, abs_tol=1e-3):
        raise ValueError("comparison windows must have equal durations")
    with Image.open(io.BytesIO(first)) as source:
        left = source.convert("RGB")
    with Image.open(io.BytesIO(second)) as source:
        right = source.convert("RGB")
    if left.width != right.width:
        target_width = max(left.width, right.width)
        left = left.resize((target_width, round(left.height * target_width / left.width)),
                           Image.Resampling.LANCZOS)
        right = right.resize((target_width, round(right.height * target_width / right.width)),
                             Image.Resampling.LANCZOS)
    gap, header = 20, 76
    canvas = Image.new("RGB", (left.width + right.width + gap * 3,
                               header + max(left.height, right.height) + gap * 2),
                       (16, 20, 29))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), title, font=_font(20), fill=_TEXT)
    draw.text((gap, 42), f"A  {first_start_seconds:.2f}-{first_end_seconds:.2f}s",
              font=_font(16), fill=_MUTED)
    draw.text((gap * 2 + left.width, 42),
              f"B  {second_start_seconds:.2f}-{second_end_seconds:.2f}s",
              font=_font(16), fill=_MUTED)
    canvas.paste(left, (gap, header))
    canvas.paste(right, (gap * 2 + left.width, header))
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_player_perspective_contact_sheet_png(
    difficulty: dict[str, Any], *, bpm: float, start_seconds: float,
    end_seconds: float, mapping_extensions: bool = False,
    width: int = 1400, height: int = 900, max_frames: int = 4,
    approach_seconds: float = 2.0,
) -> bytes:
    """Render an approximate camera-facing sequence of upcoming map objects.

    Objects move toward a fixed hit plane according to their remaining musical
    time. This deterministic projection is useful for comparing local
    readability, but it is not Beat Saber geometry, timing, or a game render.
    """
    if not math.isfinite(float(bpm)) or bpm <= 0:
        raise ValueError("bpm must be finite and positive")
    if (not math.isfinite(float(start_seconds)) or not math.isfinite(float(end_seconds))
            or end_seconds <= start_seconds):
        raise ValueError("perspective time range must be finite and increasing")
    if (not math.isfinite(float(approach_seconds)) or approach_seconds <= 0
            or max_frames < 1 or width < 800 or height < 500):
        raise ValueError("perspective dimensions, frame limit or approach time are invalid")

    image = Image.new("RGB", (width, height), (16, 20, 29))
    draw = ImageDraw.Draw(image)
    draw.text((28, 16), "PLAYER-PERSPECTIVE APPROACH PROXY - NOT A GAME RENDER",
              font=_font(22), fill=_TEXT)
    draw.text((30, 46),
              f"local window {start_seconds:.2f}-{end_seconds:.2f}s | "
              f"illustrative approach horizon {approach_seconds:.2f}s",
              font=_font(14), fill=_MUTED)

    frame_count = min(max_frames, max(1, math.ceil((end_seconds - start_seconds) / 1.0)))
    cols, rows = min(2, frame_count), math.ceil(frame_count / min(2, frame_count))
    margin_x, margin_top, margin_bottom, gap = 24, 72, 20, 14
    panel_w = (width - 2 * margin_x - (cols - 1) * gap) / cols
    panel_h = (height - margin_top - margin_bottom - (rows - 1) * gap) / rows
    lane_count = 6 if mapping_extensions else 4
    lane_center = (lane_count - 1) / 2

    def sec(beat: Any) -> float:
        return float(beat) * 60.0 / bpm

    def coord(value: Any) -> float:
        return _coordinate(value, mapping_extensions)

    def project(px: float, py: float, event_time: float, frame_time: float,
                panel: tuple[float, float, float, float]) -> tuple[float, float, float]:
        x0, y0, pw, ph = panel
        remain = max(0.0, min(approach_seconds, event_time - frame_time))
        depth = 1.0 - remain / approach_seconds
        scale = 0.34 + 0.66 * depth
        horizon_y = y0 + ph * 0.21
        hit_y = y0 + ph * 0.89
        center_x = x0 + pw / 2
        lane_spacing = pw * 0.115
        row_spacing = ph * 0.16
        screen_x = center_x + (px - lane_center) * lane_spacing * scale
        screen_y = hit_y - (2.0 - py) * row_spacing * scale
        screen_y = max(horizon_y, min(y0 + ph - 24, screen_y))
        return screen_x, screen_y, scale

    def draw_arrow(cx: float, cy: float, direction: Any, radius: float,
                   color: tuple[int, int, int]) -> None:
        try:
            dx, dy = _DIRECTIONS[int(direction)]
        except (KeyError, TypeError, ValueError):
            return
        if dx == dy == 0:
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color)
            return
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, -dy / norm
        start = (cx - ux * radius * 0.56, cy - uy * radius * 0.56)
        tip = (cx + ux * radius * 0.82, cy + uy * radius * 0.82)
        base = (tip[0] - ux * radius * 0.42, tip[1] - uy * radius * 0.42)
        half_width = radius * 0.36
        wing_a = (base[0] - uy * half_width, base[1] + ux * half_width)
        wing_b = (base[0] + uy * half_width, base[1] - ux * half_width)
        # A long, bright arrow with a dark core stays legible at contact-sheet scale.
        draw.line((*start, *tip), fill=(246, 249, 255),
                  width=max(5, round(radius * 0.28)))
        draw.polygon((tip, wing_a, wing_b), fill=(246, 249, 255), outline=color)
        draw.line((*start, *tip), fill=color, width=max(2, round(radius * 0.12)))
        draw.polygon((tip, wing_a, wing_b), fill=color)

    notes = difficulty.get("colorNotes", []) or []
    bombs = difficulty.get("bombNotes", []) or []
    arcs = difficulty.get("sliders", []) or []
    chains = difficulty.get("burstSliders", []) or []
    obstacles = difficulty.get("obstacles", []) or []
    frame_times = [start_seconds + (end_seconds - start_seconds) * (i + 0.5) / frame_count
                   for i in range(frame_count)]

    for index, frame_time in enumerate(frame_times):
        col, row = index % cols, index // cols
        x0 = margin_x + col * (panel_w + gap)
        y0 = margin_top + row * (panel_h + gap)
        panel = (x0, y0, panel_w, panel_h)
        draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=10,
                               fill=(24, 29, 41), outline=(65, 75, 94), width=1)
        draw.text((x0 + 10, y0 + 8),
                  f"NOW {frame_time:.2f}s  |  b{frame_time * bpm / 60:.2f}",
                  font=_font(14), fill=_TEXT)
        horizon_y, hit_y = y0 + panel_h * 0.21, y0 + panel_h * 0.89
        center_x = x0 + panel_w / 2
        near_half = panel_w * 0.40
        far_half = panel_w * 0.09
        draw.line((x0 + 12, hit_y, x0 + panel_w - 12, hit_y), fill=(235, 206, 118), width=2)
        draw.text((x0 + 12, hit_y + 5), "HIT PLANE", font=_font(11), fill=(235, 206, 118))
        for boundary in range(lane_count + 1):
            fraction = boundary / lane_count
            near_x = center_x - near_half + 2 * near_half * fraction
            far_x = center_x - far_half + 2 * far_half * fraction
            draw.line((far_x, horizon_y, near_x, hit_y), fill=(53, 63, 81), width=1)
        for fraction in (0.28, 0.48, 0.68, 0.86):
            y = horizon_y + (hit_y - horizon_y) * fraction
            half = far_half + (near_half - far_half) * fraction
            draw.line((center_x - half, y, center_x + half, y), fill=(43, 52, 68), width=1)

        window_end = frame_time + approach_seconds
        # Draw far geometry first so near objects remain visually prominent.
        for wall in obstacles:
            t0 = sec(wall.get("b", 0))
            t1 = t0 + float(wall.get("d", 0)) * 60.0 / bpm
            if t1 < frame_time or t0 > window_end:
                continue
            wall_x = coord(wall.get("x", 0))
            wall_y = coord(wall.get("y", 0))
            wall_w = max(0.1, float(wall.get("w", 1)))
            wall_h = max(0.1, float(wall.get("h", 3)))
            a = project(wall_x, wall_y, max(frame_time, t0), frame_time, panel)
            b = project(wall_x + wall_w, wall_y + wall_h,
                        max(frame_time, t0), frame_time, panel)
            c = project(wall_x, wall_y, min(window_end, t1), frame_time, panel)
            d = project(wall_x + wall_w, wall_y + wall_h,
                        min(window_end, t1), frame_time, panel)
            draw.polygon(((a[0], a[1]), (b[0], b[1]), (d[0], d[1]), (c[0], c[1])),
                         fill=(89, 103, 124), outline=(183, 196, 214))

        for key, color in (("sliders", _ARC), ("burstSliders", _CHAIN)):
            for path in difficulty.get(key, []) or []:
                t0, t1 = sec(path.get("b", 0)), sec(path.get("tb", path.get("b", 0)))
                if t1 < frame_time or t0 > window_end:
                    continue
                start = project(coord(path.get("x", 0)), coord(path.get("y", 0)),
                                max(frame_time, t0), frame_time, panel)
                end = project(coord(path.get("tx", path.get("x", 0))),
                              coord(path.get("ty", path.get("y", 0))),
                              min(window_end, t1), frame_time, panel)
                draw.line((start[0], start[1], end[0], end[1]), fill=color,
                          width=max(3, round(5 * end[2])))
                if key == "burstSliders":
                    slices = max(2, min(12, int(path.get("sc", 2))))
                    for slice_index in range(slices):
                        fraction = slice_index / max(1, slices - 1)
                        cx = start[0] + (end[0] - start[0]) * fraction
                        cy = start[1] + (end[1] - start[1]) * fraction
                        radius = max(2, round(4 * (start[2] + end[2]) / 2))
                        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                                     fill=color, outline=(24, 29, 41))

        upcoming: list[tuple[float, dict[str, Any], bool]] = []
        upcoming.extend((sec(note.get("b", 0)), note, False) for note in notes)
        upcoming.extend((sec(bomb.get("b", 0)), bomb, True) for bomb in bombs)
        for event_time, event, is_bomb in sorted(upcoming, key=lambda row: row[0], reverse=True):
            if not frame_time <= event_time <= window_end:
                continue
            x, y, scale = project(coord(event.get("x", 0)), coord(event.get("y", 0)),
                                  event_time, frame_time, panel)
            radius = max(4, round(17 * scale))
            if is_bomb:
                draw.polygon(((x, y - radius), (x + radius, y),
                              (x, y + radius), (x - radius, y)), fill=_BOMB)
            else:
                color = _LEFT if int(event.get("c", 0)) == 0 else _RIGHT
                draw.rounded_rectangle((x - radius, y - radius, x + radius, y + radius),
                                       radius=max(2, round(radius * 0.2)), fill=color,
                                       outline=(11, 15, 21), width=1)
                draw_arrow(x, y, event.get("d", 8), radius, (17, 22, 30))

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_hand_spatial_path_png(
    difficulty: dict[str, Any], *, bpm: float, start_seconds: float,
    end_seconds: float, mapping_extensions: bool = False,
    width: int = 1400, height: int = 900, max_notes: int = 96,
) -> bytes:
    """Render both hands' ordered paths across one local playfield window.

    Note numbers follow time order within each hand; arrows connect consecutive
    authored notes. Arcs and chains are shown as separate authored spans. This
    is a schematic of map coordinates, not a hand-motion or game simulation.
    """
    if not math.isfinite(float(bpm)) or bpm <= 0:
        raise ValueError("bpm must be finite and positive")
    if (not math.isfinite(float(start_seconds)) or not math.isfinite(float(end_seconds))
            or end_seconds <= start_seconds):
        raise ValueError("motion-trace time range must be finite and increasing")
    if max_notes < 1 or width < 800 or height < 560:
        raise ValueError("motion-trace dimensions or event limit are too small")

    lane_count = 6 if mapping_extensions else 4
    image = Image.new("RGB", (width, height), (20, 24, 33))
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _font(23), _font(15), _font(12)
    draw.text((30, 18), "HAND MOTION TRACE - AUTHORED GRID PATH, NOT GAMEPLAY",
              font=title_font, fill=_TEXT)
    draw.text((32, 54),
              f"{start_seconds:.2f}-{end_seconds:.2f}s | beats "
              f"{start_seconds * bpm / 60:.2f}-{end_seconds * bpm / 60:.2f} | "
              "# increases by beat (same-beat notes share #); arrows connect same-hand notes",
              font=body_font, fill=_MUTED)

    left, right, top, bottom = 135, width - 80, 150, height - 160
    cell_w = (right - left) / lane_count
    cell_h = (bottom - top) / 3

    def coord(value: Any) -> float:
        return _coordinate(value, mapping_extensions)

    def center(x: Any, y: Any) -> tuple[float, float]:
        gx = max(0.0, min(lane_count - 1.0, coord(x)))
        gy = max(0.0, min(2.0, coord(y)))
        return left + (gx + 0.5) * cell_w, bottom - (gy + 0.5) * cell_h

    for col in range(lane_count + 1):
        x = left + col * cell_w
        draw.line((x, top, x, bottom), fill=_GRID, width=1)
        if col < lane_count:
            draw.text((x + cell_w / 2 - 12, bottom + 10), f"x{col}",
                      font=small_font, fill=_MUTED)
    for row in range(4):
        y = bottom - row * cell_h
        draw.line((left, y, right, y), fill=_GRID, width=1)
        if row < 3:
            draw.text((left - 54, y - cell_h / 2 - 6), f"y{row}",
                      font=small_font, fill=_MUTED)

    def in_window(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        try:
            seconds = float(item.get("b")) * 60.0 / bpm
        except (TypeError, ValueError):
            return False
        return math.isfinite(seconds) and start_seconds <= seconds <= end_seconds

    notes = [note for note in difficulty.get("colorNotes", []) or []
             if in_window(note) and note.get("c") in (0, 1)]
    notes.sort(key=lambda note: (float(note.get("b", 0)), int(note.get("c", 0)),
                                 float(note.get("y", 0)), float(note.get("x", 0))))
    total_notes = len(notes)
    notes = notes[:max_notes]
    beat_ranks = {beat: index + 1 for index, beat in enumerate(sorted({
        float(note.get("b", 0)) for note in notes
    }))}

    def overlaps_window(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        try:
            start = float(item.get("b")) * 60.0 / bpm
            end = float(item.get("tb", item.get("b"))) * 60.0 / bpm
        except (TypeError, ValueError):
            return False
        return (math.isfinite(start) and math.isfinite(end)
                and start <= end_seconds and end >= start_seconds)

    def arrow(start: tuple[float, float], end: tuple[float, float],
              color: tuple[int, int, int], width_px: int = 5) -> None:
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        length = math.hypot(dx, dy)
        if length < 1:
            draw.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), outline=color, width=2)
            return
        ux, uy = dx / length, dy / length
        tip_x, tip_y = ex - ux * 16, ey - uy * 16
        draw.line((sx + ux * 15, sy + uy * 15, tip_x, tip_y), fill=color,
                  width=width_px)
        wing = 9
        bx, by = tip_x - ux * wing, tip_y - uy * wing
        draw.polygon(((tip_x, tip_y),
                      (bx - uy * wing * 0.65, by + ux * wing * 0.65),
                      (bx + uy * wing * 0.65, by - ux * wing * 0.65)), fill=color)

    def span_objects(key: str, color: tuple[int, int, int], label: str,
                     chain: bool = False) -> None:
        for item in difficulty.get(key, []) or []:
            if not overlaps_window(item):
                continue
            start = center(item.get("x", 0), item.get("y", 0))
            end = center(item.get("tx", item.get("x", 0)),
                          item.get("ty", item.get("y", 0)))
            draw.line((*start, *end), fill=color, width=3 if chain else 5)
            draw.ellipse((start[0] - 6, start[1] - 6, start[0] + 6, start[1] + 6),
                         fill=color)
            draw.ellipse((end[0] - 7, end[1] - 7, end[0] + 7, end[1] + 7),
                         outline=color, width=3)
            if chain:
                try:
                    slices = max(2, min(32, int(item.get("sc", 2))))
                except (TypeError, ValueError):
                    slices = 2
                for slice_index in range(1, slices):
                    ratio = slice_index / slices
                    sx = start[0] + (end[0] - start[0]) * ratio
                    sy = start[1] + (end[1] - start[1]) * ratio
                    draw.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), fill=color)
            middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            draw.text((middle[0] + 5, middle[1] - 12), label,
                      font=small_font, fill=color)

    span_objects("sliders", _ARC, "arc")
    span_objects("burstSliders", _CHAIN, "chain", chain=True)

    grouped_positions: dict[tuple[float, float], list[int]] = {}
    positions_by_index = [center(note.get("x", 0), note.get("y", 0))
                          for note in notes]
    for note_index, note in enumerate(notes):
        key = (round(coord(note.get("x", 0)), 3),
               round(coord(note.get("y", 0)), 3))
        grouped_positions.setdefault(key, []).append(note_index)
    for key, indices in grouped_positions.items():
        if len(indices) < 2:
            continue
        base_x, base_y = center(*key)
        radius = min(54.0, 30.0 + len(indices) * 2.0)
        for offset_index, note_index in enumerate(indices):
            angle = -math.pi / 2 + 2 * math.pi * offset_index / len(indices)
            positions_by_index[note_index] = (
                base_x + math.cos(angle) * radius,
                base_y + math.sin(angle) * radius,
            )

    for hand, color in ((0, _LEFT), (1, _RIGHT)):
        hand_indices = [index for index, note in enumerate(notes)
                        if int(note.get("c", 0)) == hand]
        hand_notes = [notes[index] for index in hand_indices]
        positions = [positions_by_index[index] for index in hand_indices]
        for index, (first, second) in enumerate(zip(positions, positions[1:])):
            first_beat = float(hand_notes[index].get("b", 0))
            second_beat = float(hand_notes[index + 1].get("b", 0))
            if second_beat <= first_beat:
                continue
            arrow(first, second, color)
            delta_beats = second_beat - first_beat
            middle = ((first[0] + second[0]) / 2,
                      (first[1] + second[1]) / 2)
            delta_label = f"Δ{delta_beats:g}b"
            bounds = draw.textbbox((0, 0), delta_label, font=small_font)
            label_w, label_h = bounds[2] - bounds[0], bounds[3] - bounds[1]
            draw.rounded_rectangle(
                (middle[0] - label_w / 2 - 4, middle[1] - label_h / 2 - 3,
                 middle[0] + label_w / 2 + 4, middle[1] + label_h / 2 + 3),
                radius=3, fill=(20, 24, 33), outline=color, width=1,
            )
            draw.text(middle, delta_label, font=small_font, fill=_TEXT, anchor="mm")
        for index, (note, (cx, cy)) in enumerate(zip(hand_notes, positions), start=1):
            radius = 16
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                         fill=color, outline=(245, 248, 255), width=2)
            beat = float(note.get("b", 0))
            draw.text((cx, cy), str(beat_ranks[beat]), font=small_font,
                      fill=(14, 18, 26), anchor="mm")
            # Put beat labels just outside the marker; alternate sides to reduce
            # collisions when adjacent notes occupy neighboring cells.
            label_x = cx + (22 if index % 2 else -22)
            label_anchor = "lm" if index % 2 else "rm"
            draw.text((label_x, cy), f"b{beat:g}", font=small_font,
                      fill=_TEXT, anchor=label_anchor)

    legend_y = height - 42
    draw.rounded_rectangle((left, legend_y, left + 16, legend_y + 13), radius=3,
                           fill=_LEFT)
    draw.text((left + 23, legend_y - 2), "left hand", font=small_font, fill=_MUTED)
    draw.rounded_rectangle((left + 115, legend_y, left + 131, legend_y + 13), radius=3,
                           fill=_RIGHT)
    draw.text((left + 138, legend_y - 2), "right hand", font=small_font, fill=_MUTED)
    draw.line((left + 245, legend_y + 6, left + 268, legend_y + 6), fill=_ARC, width=5)
    draw.text((left + 275, legend_y - 2), "arc", font=small_font, fill=_MUTED)
    draw.line((left + 330, legend_y + 6, left + 353, legend_y + 6), fill=_CHAIN, width=3)
    draw.ellipse((left + 338, legend_y + 3, left + 344, legend_y + 9), fill=_CHAIN)
    draw.text((left + 360, legend_y - 2), "chain slices", font=small_font, fill=_MUTED)
    summary = f"NOTES IN WINDOW: {total_notes}; showing {len(notes)}"
    if total_notes > len(notes):
        summary += f"; {total_notes - len(notes)} omitted; request a narrower window"
    draw.text((left, height - 75), summary, font=body_font,
              fill=(255, 223, 124) if total_notes > max_notes else _MUTED)

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def render_hand_motion_trace_png(
    difficulty: dict[str, Any], *, bpm: float, start_seconds: float,
    end_seconds: float, mapping_extensions: bool = False,
    width: int = 1400, height: int = 900,
) -> bytes:
    """Render a time-aligned lane chart with same-hand movement arrows."""
    return render_map_segment_preview_png(
        difficulty, bpm=bpm, start_seconds=start_seconds,
        end_seconds=end_seconds, mapping_extensions=mapping_extensions,
        width=width, height=height,
        title="HAND MOTION TRACE - TIME-ALIGNED LANE/TIME VIEW",
        show_hand_motion_trails=True,
    )
