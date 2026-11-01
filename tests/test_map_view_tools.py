from io import BytesIO

from PIL import Image
import pytest

from beatforge.map_view_tools import (
    DRAFT_MAP_PREVIEW_TOOL_SCHEMA, MAP_VIEW_TOOL_SCHEMA, MapViewInspector,
    ProgressiveMapPreview,
)


@pytest.mark.parametrize("view", ["lane_time", "event_timeline", "lighting_storyboard", "top_down", "perspective", "presentation"])
def test_map_view_inspector_renders_selected_local_view(view):
    inspector = MapViewInspector(
        {"colorNotes": [{"b": 3, "x": 1, "y": 1, "c": 0, "d": 0}]},
        bpm=120, duration_seconds=16,
    )

    result, png = inspector.inspect({
        "start_seconds": 1, "end_seconds": 3, "view": view,
    })

    expected_note = ("authored event payload view; values are shown as JSON data, "
                     "not a Beat Saber or Unity render" if view == "presentation" else
                     "authored palette and explicit event color swatches; not a Unity/Vivify runtime simulation"
                     if view == "lighting_storyboard" else
                     "separate time-aligned tracks of authored gameplay and presentation events; not a game render"
                     if view == "event_timeline" else
                     "schematic/proxy view; not a Beat Saber or Unity render")
    assert result == {
        "status": "rendered", "view": view,
        "start_seconds": 1.0, "end_seconds": 3.0,
        "note": expected_note,
    }
    assert Image.open(BytesIO(png)).format == "PNG"
    assert inspector.actions == [{"view": view, "start_seconds": 1.0, "end_seconds": 3.0}]
    assert MAP_VIEW_TOOL_SCHEMA["function"]["parameters"]["additionalProperties"] is False


def test_map_view_inspector_returns_ordered_full_size_bundle_in_one_call():
    inspector = MapViewInspector(
        {"colorNotes": [{"b": 3, "x": 1, "y": 1, "c": 0, "d": 0}]},
        bpm=120, duration_seconds=16,
    )

    result, images = inspector.inspect({
        "start_seconds": 1, "end_seconds": 3, "view": "diagnostic_bundle",
    })

    assert result["status"] == "rendered"
    assert result["visual_views"] == ["lane_time", "perspective", "motion_trace"]
    assert len(images) == 3
    assert all(Image.open(BytesIO(image)).format == "PNG" for image in images)
    assert inspector.calls == 1
    assert inspector.actions == [{
        "view": "diagnostic_bundle", "start_seconds": 1.0,
        "end_seconds": 3.0, "visual_views": ["lane_time", "perspective", "motion_trace"],
    }]


def test_model_requested_lane_time_view_includes_audio_change_guides():
    inspector = MapViewInspector(
        {"colorNotes": [{"b": 3, "x": 1, "y": 1, "c": 0, "d": 0}]},
        bpm=120, duration_seconds=16, audio_change_beats=(3.0, 12.0),
    )

    _result, png = inspector.inspect({
        "start_seconds": 0, "end_seconds": 4, "view": "lane_time",
    })
    image = Image.open(BytesIO(png)).convert("RGB")

    assert (255, 196, 80) in image.getdata()  # candidate at beat 3, not beat 12


def test_phrase_compare_returns_both_map_windows_and_independent_sg_metadata():
    spectrograms = []
    for color in ((20, 100, 180), (180, 100, 20)):
        buffer = BytesIO()
        Image.new("RGB", (200, 100), color).save(buffer, format="PNG")
        spectrograms.append(buffer.getvalue())
    inspector = MapViewInspector(
        {"colorNotes": [
            {"b": 1, "x": 0, "y": 1, "c": 0, "d": 0},
            {"b": 9, "x": 0, "y": 1, "c": 0, "d": 0},
        ]},
        bpm=120, duration_seconds=8,
        spectrogram_images=tuple(spectrograms),
        spectrogram_manifest={"images": [
            {"file": "segment_000.png", "start_seconds": 0, "end_seconds": 2},
            {"file": "segment_001.png", "start_seconds": 4, "end_seconds": 6},
        ]},
    )

    result, png = inspector.inspect({
        "start_seconds": 0, "end_seconds": 2,
        "compare_start_seconds": 4, "compare_end_seconds": 6,
        "view": "phrase_compare",
    })

    assert result["status"] == "rendered"
    assert result["compared_windows"] == [
        {"label": "A", "start_seconds": 0.0, "end_seconds": 2.0,
         "audio_view": "synchronized_sg_panel", "sg_start_seconds": 0.0,
         "sg_end_seconds": 2.0, "spectrogram_file": "segment_000.png"},
        {"label": "B", "start_seconds": 4.0, "end_seconds": 6.0,
         "audio_view": "synchronized_sg_panel", "sg_start_seconds": 4.0,
         "sg_end_seconds": 6.0, "spectrogram_file": "segment_001.png"},
    ]
    assert Image.open(BytesIO(png)).format == "PNG"


def test_phrase_compare_rejects_unequal_window_lengths():
    inspector = MapViewInspector({}, bpm=120, duration_seconds=16)

    result, image = inspector.inspect({
        "start_seconds": 0, "end_seconds": 2,
        "compare_start_seconds": 8, "compare_end_seconds": 11,
        "view": "phrase_compare",
    })

    assert result == {"error": "phrase_compare windows must have equal durations"}
    assert image is None


def test_progressive_checkpoint_accepts_phrase_comparison():
    preview = ProgressiveMapPreview(bpm=120, duration_seconds=12)

    result, png = preview.inspect({
        "script": (
            "map.note(beat=1, x=0, y=0, color=0, direction=1)\n"
            "map.note(beat=9, x=0, y=0, color=0, direction=1)"
        ),
        "start_seconds": 0, "end_seconds": 2,
        "compare_start_seconds": 4, "compare_end_seconds": 6,
        "view": "phrase_compare",
    })

    assert result["status"] == "rendered"
    assert result["audio_view"] == "per_window"
    assert result["compared_windows"][0]["start_seconds"] == 0
    assert result["compared_windows"][1]["start_seconds"] == 4
    assert Image.open(BytesIO(png)).format == "PNG"
    assert preview.script_chunks


@pytest.mark.parametrize("arguments", [
    {"start_seconds": -1, "end_seconds": 1, "view": "lane_time"},
    {"start_seconds": 1, "end_seconds": 9.1, "view": "lane_time"},
    {"start_seconds": 1, "end_seconds": 20, "view": "lane_time"},
    {"start_seconds": True, "end_seconds": 2, "view": "lane_time"},
    {"start_seconds": 1, "end_seconds": float("nan"), "view": "lane_time"},
    {"start_seconds": 1, "end_seconds": 2, "view": "unknown"},
    {"start_seconds": 1, "end_seconds": 2, "view": "lane_time", "raw_audio": True},
])
def test_map_view_inspector_rejects_invalid_ranges_and_arguments(arguments):
    inspector = MapViewInspector({}, bpm=120, duration_seconds=16)
    result, image = inspector.inspect(arguments)
    assert "error" in result
    assert image is None


def test_map_view_inspector_enforces_two_request_limit():
    inspector = MapViewInspector({}, bpm=120, duration_seconds=16)
    args = {"start_seconds": 0, "end_seconds": 1, "view": "top_down"}
    assert inspector.inspect(args)[0]["status"] == "rendered"
    assert inspector.inspect(args)[0]["status"] == "rendered"
    result, image = inspector.inspect(args)
    assert result == {"error": "map-view request budget exhausted (2 calls)"}
    assert image is None


def test_revision_diff_is_required_first_when_revising_an_accepted_map():
    inspector = MapViewInspector(
        {"colorNotes": [{"b": 2, "x": 3, "y": 1, "c": 1, "d": 3}]},
        bpm=120, duration_seconds=16,
        comparison_difficulty={"colorNotes": [{"b": 2, "x": 0, "y": 1, "c": 0, "d": 0}]},
        require_revision_diff=True,
    )
    view_schema = inspector.tool_schema()["function"]["parameters"]["properties"]["view"]
    assert view_schema["enum"] == ["revision_diff"]
    result, png = inspector.inspect({
        "start_seconds": 0, "end_seconds": 4, "view": "revision_diff",
    })
    assert result["status"] == "rendered"
    assert result["view"] == "revision_diff"
    assert Image.open(BytesIO(png)).size[0] > 1600


def test_revision_diff_is_rejected_without_previous_map():
    inspector = MapViewInspector({}, bpm=120, duration_seconds=16)
    result, image = inspector.inspect({
        "start_seconds": 0, "end_seconds": 4, "view": "revision_diff",
    })
    assert "no previous accepted repair revision" in result["error"]
    assert image is None


def test_progressive_preview_accumulates_only_safe_complete_sections():
    preview = ProgressiveMapPreview(bpm=120, duration_seconds=12)
    first = "map.note(beat=1, x=0, y=0, color=0, direction=1)"
    second = "map.note(beat=5, x=3, y=2, color=1, direction=3)"

    result1, png1 = preview.inspect({
        "script": first, "start_seconds": 0, "end_seconds": 3,
        "view": "lane_time",
    })
    assert result1["accumulated"] == {
        "sections": 1, "notes": 1, "bombs": 0, "arcs": 0, "chains": 0,
        "walls": 0,
    }
    assert result1["window_counts"]["notes"] == 1
    assert Image.open(BytesIO(png1)).format == "PNG"

    rejected, no_png = preview.inspect({
        "script": "import os", "start_seconds": 0, "end_seconds": 3,
        "view": "top_down",
    })
    assert "section rejected by the BeatForge DSL" in rejected["error"]
    assert no_png is None
    assert preview.script_chunks == [first]

    result2, png2 = preview.inspect({
        "script": second, "start_seconds": 2, "end_seconds": 5,
        "view": "perspective",
    })
    assert result2["section"] == 2
    assert result2["accumulated"]["notes"] == 2
    assert result2["window_counts"]["notes"] == 1
    assert Image.open(BytesIO(png2)).format == "PNG"
    assert preview.script == f"{first}\n{second}"
    assert preview.actions == [
        {"view": "lane_time", "start_seconds": 0.0,
         "end_seconds": 3.0, "section": 1, "audio_view": "unavailable"},
        {"view": "perspective", "start_seconds": 2.0,
         "end_seconds": 5.0, "section": 2, "audio_view": "unavailable"},
    ]
    assert DRAFT_MAP_PREVIEW_TOOL_SCHEMA["function"]["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize("arguments", [
    {"script": "map.note(beat=1, x=0, y=0, color=0, direction=1)",
     "start_seconds": 1, "end_seconds": 1, "view": "lane_time"},
    {"script": "map.note(beat=1, x=0, y=0, color=0, direction=1)",
     "start_seconds": 0, "end_seconds": 9, "view": "lane_time"},
    {"script": "map.note(beat=1, x=0, y=0, color=0, direction=1)",
     "start_seconds": 0, "end_seconds": 2, "view": "unknown"},
])
def test_progressive_preview_rejects_bad_window_or_view(arguments):
    preview = ProgressiveMapPreview(bpm=120, duration_seconds=12)
    result, image = preview.inspect(arguments)
    assert "error" in result
    assert image is None
    assert preview.script_chunks == []


def test_progressive_preview_can_compare_the_new_section_to_prior_sections():
    preview = ProgressiveMapPreview(bpm=120, duration_seconds=12)
    preview.inspect({
        "script": "map.note(beat=1, x=0, y=0, color=0, direction=1)",
        "start_seconds": 0, "end_seconds": 4, "view": "lane_time",
    })

    result, png = preview.inspect({
        "script": "map.note(beat=3, x=3, y=2, color=1, direction=3)",
        "start_seconds": 0, "end_seconds": 4, "view": "revision_diff",
    })

    assert result["status"] == "rendered"
    assert result["section"] == 2
    assert Image.open(BytesIO(png)).width > 1600
    assert preview.script.count("map.note") == 2


def test_progressive_preview_can_request_event_timeline_for_accepted_section():
    preview = ProgressiveMapPreview(bpm=120, duration_seconds=12)
    result, png = preview.inspect({
        "script": "map.note(beat=1, x=0, y=0, color=0, direction=1)",
        "start_seconds": 0, "end_seconds": 4, "view": "event_timeline",
    })

    assert result["status"] == "rendered"
    assert result["section"] == 1
    assert result["view"] == "event_timeline"
    assert "authored gameplay and presentation events" in result["note"]
    assert Image.open(BytesIO(png)).size == (1500, 142 + 68 * 8 + 32)


def test_progressive_preview_can_request_lighting_storyboard():
    preview = ProgressiveMapPreview(bpm=120, duration_seconds=12)
    result, png = preview.inspect({
        "script": (
            "map.palette(environment_left=[30, 120, 240], "
            "environment_right=[240, 100, 20])\n"
            "map.custom_event(beat=2, event_type='SetRenderingSettings', "
            "data={'renderSettings': {'fogColor': [1.0, 0.2, 0.0, 1.0]}})"
        ),
        "start_seconds": 0, "end_seconds": 4,
        "view": "lighting_storyboard",
    })

    assert result["status"] == "rendered"
    assert result["view"] == "lighting_storyboard"
    assert "explicit event color swatches" in result["note"]
    assert Image.open(BytesIO(png)).format == "PNG"


def test_progressive_preview_does_not_mislabel_packed_sg_as_one_source_panel():
    image = BytesIO()
    Image.new("RGB", (40, 20), (0, 255, 0)).save(image, format="PNG")
    preview = ProgressiveMapPreview(
        bpm=120, duration_seconds=12,
        spectrogram_images=(image.getvalue(), image.getvalue()),
        spectrogram_manifest={"images": [
            {"file": "overview.png", "start_seconds": 0, "end_seconds": 12},
            {"file": "segment_000.png", "start_seconds": 0, "end_seconds": 4},
            {"file": "segment_001.png", "start_seconds": 4, "end_seconds": 8},
        ], "model_input": {"packing": "overview plus vertical stacks of consecutive panels"}},
    )

    result, png = preview.inspect({
        "script": "map.note(beat=10, x=2, y=1, color=1, direction=3)",
        "start_seconds": 4, "end_seconds": 8, "view": "lane_time",
    })

    assert result["audio_view"] == "unavailable"
    assert "spectrogram_file" not in result
    assert Image.open(BytesIO(png)).size == (1400, 900)


def test_progressive_preview_stacks_adjacent_sg_panels_for_eight_second_chunk():
    images = []
    for color in ((210, 30, 20), (20, 40, 220), (20, 200, 40)):
        buffer = BytesIO()
        Image.new("RGB", (1400, 480), color).save(buffer, format="PNG")
        images.append(buffer.getvalue())
    preview = ProgressiveMapPreview(
        bpm=120, duration_seconds=12,
        spectrogram_images=tuple(images),
        spectrogram_manifest={"images": [
            {"file": "overview.png", "start_seconds": 0, "end_seconds": 12},
            {"file": "segment_000.png", "start_seconds": 0, "end_seconds": 4},
            {"file": "segment_001.png", "start_seconds": 4, "end_seconds": 8},
        ]},
    )

    result, png = preview.inspect({
        "script": "map.note(beat=1, x=0, y=0, color=0, direction=1)",
        "start_seconds": 0, "end_seconds": 8, "view": "lane_time",
    })

    assert result["audio_view"] == "synchronized_sg_panels"
    assert result["spectrogram_files"] == ["segment_000.png", "segment_001.png"]
    assert result["sg_start_seconds"] == 0
    assert result["sg_end_seconds"] == 8
    assert Image.open(BytesIO(png)).size[0] == 1400
    assert Image.open(BytesIO(png)).height > 1800


def test_progressive_preview_labels_partial_sg_overlap_as_context():
    image_buffer = BytesIO()
    Image.new("RGB", (1400, 480), (40, 50, 60)).save(image_buffer, format="PNG")
    source_image = image_buffer.getvalue()
    preview = ProgressiveMapPreview(
        bpm=120, duration_seconds=12,
        spectrogram_images=(source_image, source_image, source_image),
        spectrogram_manifest={"images": [
            {"file": "overview.png", "start_seconds": 0, "end_seconds": 12},
            {"file": "segment_000.png", "start_seconds": 0, "end_seconds": 4},
            {"file": "segment_001.png", "start_seconds": 4, "end_seconds": 8},
        ]},
    )

    result, _png = preview.inspect({
        "script": "map.note(beat=1, x=0, y=0, color=0, direction=1)",
        "start_seconds": 1, "end_seconds": 5, "view": "lane_time",
    })

    assert result["audio_view"] == "overlapping_sg_context"
    assert (result["sg_start_seconds"], result["sg_end_seconds"]) == (0, 8)


def test_progressive_read_only_inspection_reuses_accepted_map_and_syncs_source_sg():
    image_buffer = BytesIO()
    Image.new("RGB", (40, 20), (0, 255, 0)).save(image_buffer, format="PNG")
    source_image = image_buffer.getvalue()
    preview = ProgressiveMapPreview(
        bpm=120, duration_seconds=12,
        spectrogram_images=(source_image, source_image, source_image),
        spectrogram_manifest={"images": [
            {"file": "overview.png", "start_seconds": 0, "end_seconds": 12},
            {"file": "segment_000.png", "start_seconds": 0, "end_seconds": 4},
            {"file": "segment_001.png", "start_seconds": 4, "end_seconds": 8},
        ]},
    )
    script = "map.note(beat=1, x=0, y=0, color=0, direction=1)"
    preview.inspect({
        "script": script, "start_seconds": 0, "end_seconds": 4,
        "view": "lane_time",
    })

    result, png = preview.inspect_current({
        "start_seconds": 4, "end_seconds": 8, "view": "perspective",
    })

    assert result["status"] == "rendered"
    assert result["action"] == "inspect_current_map_view"
    assert result["section"] == 1
    assert result["audio_view"] == "synchronized_sg_panel"
    assert result["spectrogram_file"] == "segment_001.png"
    assert result["accumulated"]["notes"] == 1
    assert Image.open(BytesIO(png)).format == "PNG"
    assert preview.script == script
    assert preview.script_chunks == [script]
    assert preview.preview_calls == 1
    assert preview.inspection_calls == 1
    assert preview.calls == 2
    assert preview.actions[-1]["action"] == "inspect_current_map_view"


def test_progressive_multiview_bundle_syncs_each_image_and_uses_one_checkpoint():
    image_buffer = BytesIO()
    Image.new("RGB", (40, 20), (0, 255, 0)).save(image_buffer, format="PNG")
    source_image = image_buffer.getvalue()
    preview = ProgressiveMapPreview(
        bpm=120, duration_seconds=12,
        spectrogram_images=(source_image,),
        spectrogram_manifest={"images": [
            {"file": "segment_000.png", "start_seconds": 0, "end_seconds": 4},
        ]},
    )
    script = "map.note(beat=1, x=0, y=0, color=0, direction=1)"

    result, images = preview.inspect({
        "script": script, "start_seconds": 0, "end_seconds": 4,
        "view": "diagnostic_bundle",
    })

    assert result["status"] == "rendered"
    assert result["visual_views"] == [
        {"view": "lane_time", "audio_view": "synchronized_sg_panel",
         "sg_start_seconds": 0.0, "sg_end_seconds": 4.0,
         "spectrogram_file": "segment_000.png"},
        {"view": "perspective", "audio_view": "synchronized_sg_panel",
         "sg_start_seconds": 0.0, "sg_end_seconds": 4.0,
         "spectrogram_file": "segment_000.png"},
        {"view": "motion_trace", "audio_view": "synchronized_sg_panel",
         "sg_start_seconds": 0.0, "sg_end_seconds": 4.0,
         "spectrogram_file": "segment_000.png"},
    ]
    assert len(images) == 3
    assert all(Image.open(BytesIO(image)).format == "PNG" for image in images)
    assert preview.calls == 1
    assert preview.preview_calls == 1
    assert preview.script_chunks == [script]
    assert preview.actions == [{
        "view": "diagnostic_bundle", "start_seconds": 0.0,
        "end_seconds": 4.0, "section": 1,
        "audio_view": "synchronized_sg_panel",
        "visual_views": result["visual_views"],
    }]


def test_progressive_read_only_inspection_shares_checkpoint_budget_and_revision_gate():
    preview = ProgressiveMapPreview(bpm=120, duration_seconds=16)
    first = "map.note(beat=1, x=0, y=0, color=0, direction=1)"
    second = "map.note(beat=5, x=3, y=2, color=1, direction=3)"
    args = {"start_seconds": 0, "end_seconds": 4, "view": "lane_time"}
    assert "revision_diff" not in preview.inspection_tool_schema()[
        "function"]["parameters"]["properties"]["view"]["enum"]
    preview.inspect({"script": first, **args})
    preview.inspect({"script": second, **args})
    assert "revision_diff" in preview.inspection_tool_schema()[
        "function"]["parameters"]["properties"]["view"]["enum"]
    assert preview.inspect_current({
        "start_seconds": 0, "end_seconds": 4, "view": "revision_diff",
    })[0]["status"] == "rendered"
    preview.inspect_current(args)

    result, image = preview.inspect_current(args)

    assert result == {"error": "draft-preview request budget exhausted (4 calls)"}
    assert image is None
    assert preview.calls == 5
    assert preview.preview_calls == 2
    assert preview.inspection_calls == 3
