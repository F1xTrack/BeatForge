from io import BytesIO

from PIL import Image
import pytest

from beatforge.map_previews import (
    render_map_overview_png,
    render_playfield_contact_sheet_png,
    render_map_segment_preview_png,
    render_map_revision_diff_png,
    render_map_window_comparison_png,
    render_presentation_payload_png,
    render_synchronized_sg_map_png,
    render_player_perspective_contact_sheet_png,
    render_event_timeline_png,
    render_lighting_storyboard_png,
)


def test_segment_preview_renders_notes_sustains_and_scene_markers():
    difficulty = {
        "colorNotes": [
            {"b": 1, "x": 0, "y": 1, "c": 0, "d": 0},
            {"b": 2, "x": 2, "y": 1, "c": 1, "d": 3},
        ],
        "bombNotes": [{"b": 2.5, "x": 1, "y": 1}],
        "obstacles": [{"b": 3, "x": 0, "y": 0, "d": 1, "w": 1}],
        "sliders": [{"b": 1, "x": 0, "y": 1, "c": 0, "tb": 3, "tx": 2, "ty": 1}],
        "burstSliders": [{"b": 2, "x": 2, "y": 1, "c": 1, "tb": 3, "tx": 3, "ty": 1, "sc": 4}],
        "basicBeatmapEvents": [{"b": 2, "et": 0, "i": 1}],
        "customData": {
            "_beatforgeScenePalette": {
                "environment_left": [25, 40, 90], "environment_right": [230, 130, 35],
            },
            "_customEvents": [
                {"b": 1, "t": "InstantiatePrefab", "d": {"asset_id": "example"}},
                {"b": 3, "t": "SetRenderingSettings", "d": {"renderSettings": {"fog": 1}}},
            ],
        },
    }

    png = render_map_segment_preview_png(
        difficulty, bpm=120, start_seconds=0, end_seconds=4,
    )
    image = Image.open(BytesIO(png))
    assert image.format == "PNG"
    assert image.size == (1400, 900)
    assert image.getbbox() == (0, 0, 1400, 900)


def test_segment_preview_marks_measured_spectral_change_candidates():
    difficulty = {"colorNotes": [{"b": 2, "x": 1, "y": 1, "c": 0, "d": 1}]}

    png = render_map_segment_preview_png(
        difficulty, bpm=120, start_seconds=0, end_seconds=4,
        audio_change_beats=(2.0, 6.0, 4.0),
    )
    image = Image.open(BytesIO(png)).convert("RGB")

    assert (255, 196, 80) in image.getdata()  # in-window amber candidate markers


def test_presentation_payload_view_renders_authored_standard_and_scene_event_data():
    difficulty = {
        "basicBeatmapEvents": [
            {"b": 2, "et": 1, "i": 5, "f": 0.45, "customData": {"_color": [0.1, 0.2, 0.8]}},
        ],
        "customData": {
            "_beatforgeScenePalette": {
                "environment_left": [20, 40, 220], "environment_right": [230, 110, 20],
            },
            "_customEvents": [
                {"b": 3, "t": "SetRenderingSettings", "d": {
                    "renderSettings": {"fog": 1, "fogColor": [0.1, 0.2, 0.3]},
                    "duration": 2, "easing": "easeInOut",
                }},
                {"b": 4, "t": "InstantiatePrefab", "d": {
                    "asset_id": "portal", "position": [1, 2, 3], "rotation": [0, 90, 0],
                }},
            ],
        },
    }
    png = render_presentation_payload_png(
        difficulty, bpm=120, start_seconds=0, end_seconds=3,
    )
    image = Image.open(BytesIO(png))
    assert image.format == "PNG"
    assert image.width == 1600
    assert image.height >= 560
    assert image.getbbox() == (0, 0, image.width, image.height)


def test_presentation_payload_view_marks_event_display_truncation():
    difficulty = {"basicBeatmapEvents": [
        {"b": beat, "et": 0, "i": 1, "f": 1} for beat in range(5)
    ]}
    png = render_presentation_payload_png(
        difficulty, bpm=120, start_seconds=0, end_seconds=4,
        max_events=2,
    )
    image = Image.open(BytesIO(png))
    assert image.height == max(560, 170 + 2 * 92 + 58)


def test_event_timeline_separates_authored_event_types_on_one_time_axis():
    difficulty = {
        "colorNotes": [
            {"b": 1, "c": 0}, {"b": 2, "c": 1},
        ],
        "bombNotes": [{"b": 2.5}],
        "obstacles": [{"b": 1, "d": 2}],
        "sliders": [{"b": 2, "tb": 3}],
        "burstSliders": [{"b": 3, "tb": 3.5, "sc": 4}],
        "basicBeatmapEvents": [{"b": 2.5, "et": 0}],
        "customData": {"_customEvents": [{"b": 3, "t": "InstantiatePrefab"}]},
    }

    png = render_event_timeline_png(
        difficulty, bpm=120, start_seconds=0, end_seconds=4,
        audio_change_beats=(2.0, 8.0),
    )
    image = Image.open(BytesIO(png)).convert("RGB")

    assert image.size == (1500, 142 + 68 * 8 + 32)
    assert (242, 88, 112) in image.getdata()  # left-note track
    assert (82, 169, 255) in image.getdata()  # right-note track
    assert (255, 196, 80) in image.getdata()  # spectral-change candidate


@pytest.mark.parametrize("kwargs, message", [
    ({"bpm": 0, "start_seconds": 0, "end_seconds": 1}, "BPM"),
    ({"bpm": 120, "start_seconds": 2, "end_seconds": 1}, "increasing"),
    ({"bpm": 120, "start_seconds": 0, "end_seconds": 1, "width": 700}, "narrow"),
])
def test_event_timeline_rejects_invalid_ranges_and_dimensions(kwargs, message):
    with pytest.raises(ValueError, match=message):
        render_event_timeline_png({}, **kwargs)


def test_lighting_storyboard_visualizes_authored_palette_and_explicit_event_colors():
    difficulty = {
        "basicBeatmapEvents": [{"b": 1, "et": 0, "i": 1,
                                "customData": {"_color": [0.2, 0.5, 1.0, 1.0]}}],
        "customData": {
            "_beatforgeScenePalette": {
                "environment_left": [20, 40, 220],
                "environment_right": [230, 110, 20],
            },
            "_customEvents": [{"b": 2, "t": "SetRenderingSettings", "d": {
                "renderSettings": {"fogColor": [1.0, 0.2, 0.0, 1.0]},
            }}],
        },
    }

    png = render_lighting_storyboard_png(
        difficulty, bpm=120, start_seconds=0, end_seconds=2,
    )
    image = Image.open(BytesIO(png)).convert("RGB")

    assert image.size == (1600, 420)
    assert (20, 40, 220) in image.getdata()  # environment palette
    assert (51, 128, 255) in image.getdata()  # standard light _color
    assert (255, 51, 0) in image.getdata()  # Vivify fogColor


@pytest.mark.parametrize("kwargs, message", [
    ({"bpm": 0, "start_seconds": 0, "end_seconds": 1}, "BPM"),
    ({"bpm": 120, "start_seconds": 1, "end_seconds": 1}, "increasing"),
    ({"bpm": 120, "start_seconds": 0, "end_seconds": 1, "width": 800}, "dimensions"),
    ({"bpm": 120, "start_seconds": 0, "end_seconds": 1, "max_events": 0}, "dimensions"),
])
def test_lighting_storyboard_rejects_invalid_ranges_and_dimensions(kwargs, message):
    with pytest.raises(ValueError, match=message):
        render_lighting_storyboard_png({}, **kwargs)


def test_revision_diff_preview_shows_same_window_in_two_revisions():
    before = {"colorNotes": [{"b": 2, "x": 0, "y": 1, "c": 0, "d": 0}]}
    current = {"colorNotes": [{"b": 2, "x": 3, "y": 1, "c": 1, "d": 3}]}
    png = render_map_revision_diff_png(
        before, current, bpm=120, start_seconds=0, end_seconds=4,
    )
    image = Image.open(BytesIO(png))
    assert image.format == "PNG"
    assert image.size == (900 * 2 + 20 * 3, 720 + 52 + 20 * 2)


def test_map_window_comparison_places_equal_duration_windows_side_by_side():
    first = Image.new("RGB", (900, 600), (40, 50, 60))
    second = Image.new("RGB", (900, 600), (70, 80, 90))
    first_bytes, second_bytes = BytesIO(), BytesIO()
    first.save(first_bytes, format="PNG")
    second.save(second_bytes, format="PNG")

    png = render_map_window_comparison_png(
        first_bytes.getvalue(), second_bytes.getvalue(),
        first_start_seconds=4, first_end_seconds=8,
        second_start_seconds=20, second_end_seconds=24,
    )

    image = Image.open(BytesIO(png))
    assert image.format == "PNG"
    assert image.size == (900 * 2 + 20 * 3, 76 + 600 + 20 * 2)


def test_map_window_comparison_rejects_different_durations():
    first, second = BytesIO(), BytesIO()
    Image.new("RGB", (900, 600)).save(first, format="PNG")
    Image.new("RGB", (900, 600)).save(second, format="PNG")

    with pytest.raises(ValueError, match="equal durations"):
        render_map_window_comparison_png(
            first.getvalue(), second.getvalue(),
            first_start_seconds=0, first_end_seconds=4,
            second_start_seconds=20, second_end_seconds=25,
        )


@pytest.mark.parametrize("bpm,start,end", [(0, 0, 4), (120, 4, 4), (float("nan"), 0, 2)])
def test_segment_preview_rejects_invalid_timebase(bpm, start, end):
    with pytest.raises(ValueError):
        render_map_segment_preview_png({}, bpm=bpm, start_seconds=start, end_seconds=end)


def test_whole_track_overview_uses_a_coarser_time_axis():
    png = render_map_overview_png(
        {"colorNotes": [{"b": 8, "x": 1, "y": 1, "c": 0, "d": 0}]},
        bpm=120, duration_seconds=120,
    )
    image = Image.open(BytesIO(png))
    assert image.size == (1800, 850)
    assert image.format == "PNG"


def test_synchronized_preview_stacks_matching_audio_and_map_views():
    from io import BytesIO

    sg = BytesIO()
    Image.new("RGB", (700, 240), (90, 90, 90)).save(sg, format="PNG")
    chart = render_map_segment_preview_png(
        {"colorNotes": [{"b": 4, "x": 1, "y": 1, "c": 0, "d": 0}]},
        bpm=120, start_seconds=2, end_seconds=4,
    )
    png = render_synchronized_sg_map_png(
        sg.getvalue(), chart, start_seconds=2, end_seconds=4,
    )
    image = Image.open(BytesIO(png))
    assert image.size == (1400, 88 + 900 + 480)
    assert image.format == "PNG"


def test_synchronized_preview_rejects_invalid_time_range():
    with pytest.raises(ValueError):
        render_synchronized_sg_map_png(b"", b"", start_seconds=2, end_seconds=2)


def test_playfield_contact_sheet_shows_sequential_authored_grid_states():
    png = render_playfield_contact_sheet_png(
        {
            "colorNotes": [
                {"b": 0, "x": 0, "y": 0, "c": 0, "d": 0},
                {"b": 1, "x": 2, "y": 1, "c": 1, "d": 3},
            ],
            "sliders": [{"b": 0, "x": 0, "y": 0, "tb": 1, "tx": 2, "ty": 1}],
            "burstSliders": [{"b": 1, "x": 2, "y": 1, "tb": 2, "tx": 3, "ty": 2}],
        },
        bpm=120, start_seconds=0, end_seconds=2,
    )
    image = Image.open(BytesIO(png))
    assert image.format == "PNG"
    assert image.size == (1400, 900)


def test_playfield_contact_sheet_rejects_invalid_request():
    with pytest.raises(ValueError):
        render_playfield_contact_sheet_png(
            {}, bpm=120, start_seconds=2, end_seconds=2,
        )


def test_player_perspective_sheet_projects_upcoming_map_objects():
    png = render_player_perspective_contact_sheet_png(
        {
            "colorNotes": [
                {"b": 4, "x": 0, "y": 1, "c": 0, "d": 0},
                {"b": 5, "x": 2, "y": 2, "c": 1, "d": 3},
            ],
            "bombNotes": [{"b": 4.5, "x": 1, "y": 0}],
            "obstacles": [{"b": 4, "x": 0, "y": 0, "d": 2, "w": 1, "h": 3}],
            "sliders": [{"b": 4, "x": 0, "y": 1, "tb": 5, "tx": 2, "ty": 2}],
            "burstSliders": [{"b": 4, "x": 1, "y": 1, "tb": 5, "tx": 3, "ty": 1, "sc": 4}],
        },
        bpm=120, start_seconds=2, end_seconds=4,
    )
    image = Image.open(BytesIO(png))
    assert image.format == "PNG"
    assert image.size == (1400, 900)


@pytest.mark.parametrize("bpm,start,end,horizon", [
    (0, 0, 4, 2), (120, 2, 2, 2), (120, 0, 2, 0), (float("nan"), 0, 2, 2),
])
def test_player_perspective_sheet_rejects_invalid_request(bpm, start, end, horizon):
    with pytest.raises(ValueError):
        render_player_perspective_contact_sheet_png(
            {}, bpm=bpm, start_seconds=start, end_seconds=end,
            approach_seconds=horizon,
        )
