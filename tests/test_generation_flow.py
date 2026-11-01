from dataclasses import dataclass
import base64
from io import BytesIO
import json
import pytest
from PIL import Image

from beatforge import (CapabilityProfile, GenerationRequest, Section, TrackPlan,
                       ChatResult, generate_map)


@dataclass
class FakeResponse:
    content: str
    usage: dict


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, model_id, messages, **kwargs):
        self.calls.append((model_id, messages, kwargs))
        return FakeResponse("map.note(beat=1, x=1, y=0, color=0, direction=1)", {"total_tokens": 7})


def test_generation_flow_prompt_to_compiled_map():
    client = FakeClient()
    request = GenerationRequest("demo/model:free", "song.ogg",
                                TrackPlan(120, 30, 2.5, sections=[Section(0, 16, "intro")]),
                                CapabilityProfile())
    result = generate_map(request, client)
    assert result.difficulty["colorNotes"][0]["c"] == 0
    assert result.usage["total_tokens"] == 7
    assert len(client.calls) == 1
    assert "max_tokens" not in client.calls[0][2]


def test_free_vision_generation_previews_and_assembles_multiple_sections():
    chunks = [
        ("map.note(beat=1, x=0, y=0, color=0, direction=1)", 0, 4, "lane_time"),
        ("map.note(beat=10, x=3, y=2, color=1, direction=3)", 4, 8, "perspective"),
    ]
    image_buffer = BytesIO()
    Image.new("RGB", (240, 80), (20, 40, 60)).save(image_buffer, format="PNG")
    sg_image = image_buffer.getvalue()
    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 12, 1.0),
        CapabilityProfile(), spectrogram_images=(sg_image, sg_image, sg_image),
        spectrogram_manifest={"images": [
            {"file": "overview.png", "start_seconds": 0, "end_seconds": 12},
            {"file": "segment_000.png", "start_seconds": 0, "end_seconds": 4},
            {"file": "segment_001.png", "start_seconds": 4, "end_seconds": 8},
        ]},
    )

    class FreeVisionToolClient:
        enforce_free = True

        def __init__(self):
            self.calls = []

        def supports_input_modality(self, _model, modality):
            return modality == "image"

        def supports_tools(self, _model):
            return True

        def chat(self, _model, messages, **kwargs):
            self.calls.append((messages, kwargs))
            call_index = len(self.calls) - 1
            if call_index < len(chunks):
                script, start, end, view = chunks[call_index]
                if call_index:
                    assert messages[-1]["role"] == "user"
                    image = messages[-1]["content"][-1]
                    assert image["type"] == "image_url"
                    assert image["image_url"]["url"].startswith("data:image/png;base64,")
                    checkpoint_text = messages[-1]["content"][0]["text"]
                    assert '"audio_view": "synchronized_sg_panel"' in checkpoint_text
                    decoded = base64.b64decode(image["image_url"]["url"].split(",", 1)[1])
                    assert Image.open(BytesIO(decoded)).height > 900
                call = {"id": f"preview-{call_index}", "type": "function", "function": {
                    "name": "preview_map_chunk",
                    "arguments": json.dumps({
                        "script": script, "start_seconds": start,
                        "end_seconds": end, "view": view,
                    }),
                }}
                return ChatResult("", "demo/model:free", {"prompt_tokens": 2}, {}, (call,))
            tool_messages = [message for message in messages if message["role"] == "tool"]
            assert tool_messages
            assert "Continue with only new map DSL calls" in tool_messages[-1]["content"]
            return ChatResult(
                "map.note(beat=20, x=2, y=0, color=0, direction=2)\n"
                "BEATFORGE_MAP_COMPLETE", "demo/model:free",
                {"completion_tokens": 3, "cost": 0, "is_byok": False}, {},
            )

    client = FreeVisionToolClient()
    additional_calls = []
    result = generate_map(
        request, client,
        before_additional_call=lambda: additional_calls.append(True),
        max_tool_calls=2,
    )

    assert [note["b"] for note in result.difficulty["colorNotes"]] == [1.0, 10.0, 20.0]
    assert all(f"map.note(beat={beat}" in result.script for beat in (1, 10, 20))
    assert result.usage["prompt_tokens"] == 4
    assert result.usage["completion_tokens"] == 3
    assert result.usage["beatforge_draft_preview_calls"] == 2
    assert result.usage["beatforge_draft_preview_actions"] == [
        {"view": "lane_time", "start_seconds": 0.0,
         "end_seconds": 4.0, "section": 1,
         "audio_view": "synchronized_sg_panel", "sg_start_seconds": 0.0,
         "sg_end_seconds": 4.0, "spectrogram_file": "segment_000.png"},
        {"view": "perspective", "start_seconds": 4.0,
         "end_seconds": 8.0, "section": 2,
         "audio_view": "synchronized_sg_panel", "sg_start_seconds": 4.0,
         "sg_end_seconds": 8.0, "spectrogram_file": "segment_001.png"},
    ]
    assert client.calls[0][1]["tool_choice"] == {
        "type": "function", "function": {"name": "preview_map_chunk"},
    }
    assert client.calls[1][1]["tool_choice"] == "auto"
    assert client.calls[2][1]["tool_choice"] == "none"
    assert client.calls[2][1]["tools"] == []
    assert "never repeat prior code" in client.calls[0][0][0]["content"][0]["text"]
    assert len(additional_calls) == 2


def test_free_vision_generation_can_reinspect_accepted_section_without_resubmitting_it():
    image_buffer = BytesIO()
    Image.new("RGB", (240, 80), (20, 40, 60)).save(image_buffer, format="PNG")
    sg_image = image_buffer.getvalue()
    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 8, 1.0),
        CapabilityProfile(), spectrogram_images=(sg_image, sg_image),
        spectrogram_manifest={"images": [
            {"file": "overview.png", "start_seconds": 0, "end_seconds": 8},
            {"file": "segment_000.png", "start_seconds": 0, "end_seconds": 4},
        ]},
    )

    class FreeVisionToolClient:
        enforce_free = True

        def __init__(self):
            self.calls = []

        def supports_input_modality(self, _model, modality):
            return modality == "image"

        def supports_tools(self, _model):
            return True

        def chat(self, _model, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if len(self.calls) == 1:
                call = {"id": "submit-section", "type": "function", "function": {
                    "name": "preview_map_chunk",
                    "arguments": json.dumps({
                        "script": "map.note(beat=1, x=0, y=0, color=0, direction=1)",
                        "start_seconds": 0, "end_seconds": 4, "view": "lane_time",
                    }),
                }}
                return ChatResult("", "demo/model:free", {}, {}, (call,))
            if len(self.calls) == 2:
                schemas = self.calls[1][1]["tools"]
                names = {item["function"]["name"] for item in schemas}
                assert "inspect_current_map_view" in names
                latest_user = next(
                    message for message in reversed(messages)
                    if message["role"] == "user" and isinstance(message["content"], list)
                )
                image_data = base64.b64decode(
                    latest_user["content"][-1]["image_url"]["url"].split(",", 1)[1]
                )
                assert Image.open(BytesIO(image_data)).format == "PNG"
                checkpoint_text = latest_user["content"][0]["text"]
                assert '"section": 1' in checkpoint_text
                assert '"audio_view": "synchronized_sg_panel"' in checkpoint_text
                call = {"id": "inspect-current", "type": "function", "function": {
                    "name": "inspect_current_map_view",
                    "arguments": json.dumps({
                        "start_seconds": 0, "end_seconds": 4, "view": "top_down",
                    }),
                }}
                return ChatResult("", "demo/model:free", {}, {}, (call,))
            tool_messages = [message for message in messages if message["role"] == "tool"]
            assert len(tool_messages) == 2
            assert "accepted draft is unchanged" in tool_messages[-1]["content"]
            return ChatResult(
                "map.note(beat=14, x=3, y=1, color=1, direction=3)", "demo/model:free",
                {"cost": 0, "is_byok": False}, {},
            )

    client = FreeVisionToolClient()
    result = generate_map(request, client, max_tool_calls=2)

    assert [note["b"] for note in result.difficulty["colorNotes"]] == [1.0, 14.0]
    assert result.usage["beatforge_draft_preview_calls"] == 1
    assert result.usage["beatforge_draft_visual_checkpoints"] == 2
    assert result.usage["beatforge_draft_section_submissions"] == 1
    assert result.usage["beatforge_draft_read_only_inspections"] == 1
    assert result.usage["beatforge_draft_preview_actions"] == [{
        "view": "lane_time", "start_seconds": 0.0, "end_seconds": 4.0,
        "section": 1, "audio_view": "synchronized_sg_panel",
        "sg_start_seconds": 0.0, "sg_end_seconds": 4.0,
        "spectrogram_file": "segment_000.png",
    }]
    assert result.usage["beatforge_draft_inspection_actions"] == [{
        "action": "inspect_current_map_view", "view": "top_down",
        "start_seconds": 0.0, "end_seconds": 4.0, "section": 1,
        "audio_view": "synchronized_sg_panel", "spectrogram_file": "segment_000.png",
        "sg_start_seconds": 0.0, "sg_end_seconds": 4.0,
    }]
    assert client.calls[0][1]["tool_choice"]["function"]["name"] == "preview_map_chunk"
    assert client.calls[1][1]["tool_choice"] == "auto"


def test_progressive_diagnostic_bundle_sends_three_views_in_one_tool_round():
    image_buffer = BytesIO()
    Image.new("RGB", (240, 80), (20, 40, 60)).save(image_buffer, format="PNG")
    sg_image = image_buffer.getvalue()
    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 8, 1.0),
        CapabilityProfile(), spectrogram_images=(sg_image, sg_image),
        spectrogram_manifest={"images": [
            {"file": "overview.png", "start_seconds": 0, "end_seconds": 8},
            {"file": "segment_000.png", "start_seconds": 0, "end_seconds": 4},
        ]},
        autopilot=True,
    )

    class FreeVisionToolClient:
        enforce_free = True

        def __init__(self):
            self.calls = []

        def supports_input_modality(self, _model, modality):
            return modality == "image"

        def supports_tools(self, _model):
            return True

        def chat(self, _model, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if len(self.calls) == 1:
                call = {"id": "submit-bundle", "type": "function", "function": {
                    "name": "preview_map_chunk",
                    "arguments": json.dumps({
                        "script": "map.note(beat=1, x=0, y=0, color=0, direction=1)",
                        "start_seconds": 0, "end_seconds": 4,
                        "view": "diagnostic_bundle",
                    }),
                }}
                return ChatResult("", "demo/model:free", {}, {}, (call,))

            checkpoint = next(
                message for message in reversed(messages)
                if message["role"] == "user" and isinstance(message["content"], list)
            )
            text = checkpoint["content"][0]["text"]
            assert '"view": "diagnostic_bundle"' in text
            assert '"view": "motion_trace"' in text
            image_parts = [part for part in checkpoint["content"]
                           if part.get("type") == "image_url"]
            assert len(image_parts) == 3
            for part in image_parts:
                png = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                assert Image.open(BytesIO(png)).format == "PNG"
            return ChatResult(
                "map.note(beat=14, x=3, y=1, color=1, direction=3)",
                "demo/model:free", {"cost": 0, "is_byok": False}, {},
            )

    client = FreeVisionToolClient()
    result = generate_map(request, client, max_tool_calls=1)

    assert [note["b"] for note in result.difficulty["colorNotes"]] == [1.0, 14.0]
    assert result.usage["beatforge_draft_preview_calls"] == 1
    assert result.usage["beatforge_draft_visual_checkpoints"] == 1
    action = result.usage["beatforge_draft_preview_actions"][0]
    assert action["view"] == "diagnostic_bundle"
    assert [item["view"] for item in action["visual_views"]] == [
        "lane_time", "perspective", "motion_trace",
    ]
    assert result.usage["beatforge_draft_visual_checkpoints"] == 1
    assert result.usage["beatforge_draft_visual_images"] == 3
    assert result.usage["beatforge_draft_section_submissions"] == 1


def test_progressive_usage_distinguishes_rejected_attempts_from_rendered_views():
    from beatforge.generation import _progressive_map_preview_usage
    from beatforge.map_view_tools import ProgressiveMapPreview

    preview = ProgressiveMapPreview(bpm=120, duration_seconds=8)
    for index, view in enumerate(("lane_time", "diagnostic_bundle", "lane_time", "lane_time")):
        preview.inspect({
            "script": f"map.note(beat={index + 1}, x={index % 4}, y=0, color=0, direction=1)",
            "start_seconds": 0, "end_seconds": 4, "view": view,
        })
    rejected, image = preview.inspect({
        "script": "map.note(beat=5, x=0, y=1, color=1, direction=3)",
        "start_seconds": 0, "end_seconds": 4, "view": "lane_time",
    })

    usage = _progressive_map_preview_usage(preview)
    assert "budget exhausted" in rejected["error"]
    assert image is None
    assert usage["beatforge_draft_preview_attempts"] == 5
    assert usage["beatforge_draft_preview_rejections"] == 1
    assert usage["beatforge_draft_visual_checkpoint_attempts"] == 5
    assert usage["beatforge_draft_visual_checkpoints"] == 4
    assert usage["beatforge_draft_visual_images"] == 6
    assert usage["beatforge_draft_section_submissions"] == 4


def test_progressive_preview_is_skipped_for_manual_partial_section_generation():
    request = GenerationRequest(
        "demo/model:free", "song.ogg",
        TrackPlan(120, 12, 1.0, sections=[Section(0, 8, "verse"), Section(8, 24, "drop")]),
        CapabilityProfile(), section_ids=(0,), autopilot=False,
    )

    class FreeVisionToolClient:
        enforce_free = True

        def __init__(self):
            self.kwargs = None

        def supports_input_modality(self, _model, _modality):
            return True

        def supports_tools(self, _model):
            return True

        def chat(self, _model, _messages, **kwargs):
            self.kwargs = kwargs
            return ChatResult(
                "map.note(beat=1, x=0, y=0, color=0, direction=1)",
                "demo/model:free", {}, {},
            )

    client = FreeVisionToolClient()
    result = generate_map(request, client)

    assert result.difficulty["colorNotes"]
    assert "tools" not in client.kwargs


def test_progressive_preview_uses_original_sg_panel_after_model_input_packing():
    from beatforge.spectrograms import pack_spectrogram_images

    def solid_png(color):
        buffer = BytesIO()
        Image.new("RGB", (240, 80), color).save(buffer, format="PNG")
        return buffer.getvalue()

    source_images = (solid_png((10, 10, 10)), solid_png((255, 0, 0)),
                     solid_png((0, 255, 0)))
    manifest = {"images": [
        {"file": "overview.png", "start_seconds": 0, "end_seconds": 12},
        {"file": "segment_000.png", "start_seconds": 0, "end_seconds": 4},
        {"file": "segment_001.png", "start_seconds": 4, "end_seconds": 8},
    ]}
    packed_images, packing = pack_spectrogram_images(source_images, 2, manifest)
    manifest["model_input"] = packing
    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 12, 1.0),
        CapabilityProfile(), spectrogram_images=packed_images,
        spectrogram_source_images=source_images, spectrogram_manifest=manifest,
    )

    class FreeVisionToolClient:
        enforce_free = True

        def __init__(self):
            self.calls = 0

        def supports_input_modality(self, _model, modality):
            return modality == "image"

        def supports_tools(self, _model):
            return True

        def chat(self, _model, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                call = {"id": "packed-preview", "type": "function", "function": {
                    "name": "preview_map_chunk",
                    "arguments": json.dumps({
                        "script": "map.note(beat=10, x=2, y=1, color=1, direction=3)",
                        "start_seconds": 4, "end_seconds": 8, "view": "lane_time",
                    }),
                }}
                return ChatResult("", "demo/model:free", {}, {}, (call,))

            checkpoint_message = next(
                message for message in reversed(messages)
                if message["role"] == "user" and isinstance(message["content"], list)
            )
            content = checkpoint_message["content"]
            assert '"spectrogram_file": "segment_001.png"' in content[0]["text"]
            image_data = base64.b64decode(content[-1]["image_url"]["url"].split(",", 1)[1])
            composite = Image.open(BytesIO(image_data)).convert("RGB")
            assert composite.getpixel((1390, 100)) == (0, 255, 0)
            return ChatResult(
                "map.note(beat=20, x=0, y=2, color=0, direction=4)",
                "demo/model:free", {}, {},
            )

    result = generate_map(request, FreeVisionToolClient(), max_tool_calls=1)

    assert [note["b"] for note in result.difficulty["colorNotes"]] == [10.0, 20.0]
    assert result.usage["beatforge_draft_preview_actions"][0]["spectrogram_file"] == "segment_001.png"


def test_generation_continues_provider_truncated_map_and_aggregates_usage():
    class TruncatedClient:
        def __init__(self):
            self.calls = []
            self.progress = []
            self.callback = self.progress.append
            self.enforce_free = True

        def supports_input_modality(self, model_id, modality):
            assert model_id == "demo/model:free"
            return modality == "image"

        def set_progress_callback(self, callback):
            self.callback = callback

        def get_progress_callback(self):
            return self.callback

        def chat(self, model_id, messages, **kwargs):
            self.calls.append((model_id, messages, kwargs))
            if len(self.calls) == 1:
                return ChatResult(
                    "map.note(beat=1, x=1, y=0, color=0, direction=1)\nmap.note(",
                    "demo/model:free",
                    {"prompt_tokens": 2, "completion_tokens": 3,
                     "total_tokens": 5, "cost": 0, "is_byok": False},
                    {"choices": [{"finish_reason": "length"}]},
                )
            assert messages[-3] == {
                "role": "assistant",
                "content": "map.note(beat=1, x=1, y=0, color=0, direction=1)\nmap.note(",
            }
            assert "Do not repeat, rewrite, or summarize" in messages[-2]["content"]
            assert messages[-1]["role"] == "user"
            assert messages[-1]["content"][0]["text"].startswith(
                "BEATFORGE_PARTIAL_MAP_CHECKPOINT"
            )
            assert [part["type"] for part in messages[-1]["content"]] == [
                "text", "image_url", "image_url",
            ]
            self.callback("beat=2, x=2, y=0, color=1, direction=0)")
            return ChatResult(
                "map.note(beat=1, x=1, y=0, color=0, direction=1)\n"
                "map.note(beat=2, x=2, y=0, color=1, direction=0)",
                "demo/model:free",
                {"prompt_tokens": 4, "completion_tokens": 6,
                 "total_tokens": 10, "cost": 0, "is_byok": False},
                {"choices": [{"finish_reason": "stop"}]},
            )

    client = TruncatedClient()
    additional_calls = []
    result = generate_map(
        GenerationRequest("demo/model:free", "song.ogg", TrackPlan(120, 30, 2.5),
                          CapabilityProfile()),
        client,
        before_additional_call=lambda: additional_calls.append(True),
    )

    assert [note["b"] for note in result.difficulty["colorNotes"]] == [1.0, 2.0]
    assert result.usage["prompt_tokens"] == 6
    assert result.usage["completion_tokens"] == 9
    assert result.usage["total_tokens"] == 15
    assert result.usage["cost"] == 0
    assert result.usage["is_byok"] is False
    assert result.usage["beatforge_script_continuations"] == 1
    assert len(client.calls) == 2
    assert len(additional_calls) == 1
    assert all("max_tokens" not in call[2] for call in client.calls)
    assert client.progress[-1].startswith("map.note(beat=1, x=1, y=0, color=0, direction=1)")
    assert "beat=2, x=2, y=0, color=1, direction=0)" in client.progress[-1]


def test_diagnostic_repair_also_continues_provider_truncation():
    from beatforge import GenerationResult, repair_map

    class TruncatedRepairClient:
        def __init__(self):
            self.calls = []

        def chat(self, _model, messages, **_kwargs):
            self.calls.append(messages)
            response = (
                "map.note(beat=1, x=1, y=0, color=0, direction=1)\nmap.note(",
                "beat=2, x=2, y=0, color=1, direction=0)",
            )[len(self.calls) - 1]
            return ChatResult(
                response, "demo/model:free", {"total_tokens": 4},
                {"choices": [{"finish_reason": "length" if len(self.calls) == 1 else "stop"}]},
            )

    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 30, 2.5), CapabilityProfile())
    previous = GenerationResult(
        "demo/model:free",
        "map.note(beat=0, x=0, y=0, color=0, direction=1)",
        {"colorNotes": [{"b": 0}]}, {},
    )
    client = TruncatedRepairClient()
    additional_calls = []
    repaired = repair_map(
        request, previous, {}, client,
        before_additional_call=lambda: additional_calls.append(True),
    )

    assert [note["b"] for note in repaired.difficulty["colorNotes"]] == [1.0, 2.0]
    assert repaired.usage["total_tokens"] == 8
    assert repaired.usage["beatforge_script_continuations"] == 1
    assert len(client.calls) == 2
    assert len(additional_calls) == 1


def test_partial_map_checkpoint_uses_only_complete_calls_and_requires_image_route():
    from beatforge.generation import (
        _complete_map_script_prefix, _render_partial_map_checkpoint,
    )

    script = (
        "map.note(beat=1, x=1, y=0, color=0, direction=1)\n"
        "map.note(beat=2, x=2, y=0, color=1, direction="
    )
    prefix = _complete_map_script_prefix(script)
    assert prefix == "map.note(beat=1, x=1, y=0, color=0, direction=1)"

    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 30, 2.5), CapabilityProfile(),
    )

    class VisionClient:
        enforce_free = True

        def supports_input_modality(self, _model, modality):
            return modality == "image"

    class TextOnlyClient:
        enforce_free = True

        def supports_input_modality(self, _model, _modality):
            return False

    checkpoints = _render_partial_map_checkpoint(request, VisionClient(), script)
    assert [metadata["view"] for _image, metadata in checkpoints] == [
        "partial_overview", "partial_generated_edge",
    ]
    assert all(image.startswith(b"\x89PNG\r\n\x1a\n") for image, _ in checkpoints)
    assert checkpoints[1][1]["start_seconds"] == 0.0
    assert checkpoints[1][1]["end_seconds"] == 4.0
    assert _render_partial_map_checkpoint(request, TextOnlyClient(), script) == []


def test_generation_flow_executes_bounded_vivify_search_tools():
    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 30, 2.5),
        CapabilityProfile(vivify=True),
    )

    class ToolClient:
        def __init__(self):
            self.calls = []

        def chat(self, _model, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if len(self.calls) == 1:
                call = {"id": "call-1", "type": "function", "function": {
                    "name": "search_polyhaven_assets",
                    "arguments": '{"query":"industrial crate","limit":1}',
                }}
                return ChatResult("", "demo/model:free", {"prompt_tokens": 5}, {}, (call,))
            assert messages[-1]["role"] == "tool"
            return ChatResult(
                "map.note(beat=1, x=1, y=0, color=0, direction=1)\n"
                "map.custom_event(beat=0, event_type='SetRenderingSettings', "
                "data={'renderSettings': {'fog': 1, 'fogColor': [0.08,0.04,0.16,1], "
                "'fogMode': 2, 'fogDensity': 0.004}})",
                "demo/model:free", {"completion_tokens": 7}, {},
            )

    executed = []
    class Executor:
        imported_asset_ids = []
        actions = [{"tool": "search_polyhaven_assets", "query": "industrial crate",
                    "candidate_count": 1, "status": "completed"}]

        def __call__(self, name, arguments):
            executed.append((name, arguments))
            return {"results": [{"candidate_id": "a" * 20}]}

    additional_calls = []
    client = ToolClient()
    result = generate_map(request, client, tool_executor=Executor(),
                          before_additional_call=lambda: additional_calls.append(True))
    assert result.asset_ids == ()
    assert result.usage["prompt_tokens"] == 5
    assert result.usage["completion_tokens"] == 7
    assert result.usage["beatforge_tool_calls"] == 1
    assert result.usage["beatforge_tool_actions"][0]["candidate_count"] == 1
    prompt_text = json.dumps(client.calls[0][0], ensure_ascii=False).lower()
    assert "independently inferred" in prompt_text
    assert "do not include or guess a song title" in prompt_text
    assert "portal 2" not in prompt_text
    assert executed == [("search_polyhaven_assets", {"query": "industrial crate", "limit": 1})]
    assert len(additional_calls) == 1
    assert client.calls[0][1]["parallel_tool_calls"] is False
    assert client.calls[0][1]["tool_choice"] == {
        "type": "function", "function": {"name": "search_polyhaven_assets"},
    }


def test_repair_tool_loop_returns_requested_map_view_image_to_model():
    from beatforge.generation import _chat_with_vivify_tools
    from beatforge.map_view_tools import MapViewInspector

    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 16, 2.5), CapabilityProfile(),
    )
    inspector = MapViewInspector(
        {"colorNotes": [{"b": 4, "x": 2, "y": 1, "c": 1, "d": 3}]},
        bpm=120, duration_seconds=16,
        comparison_difficulty={"colorNotes": [{"b": 4, "x": 0, "y": 1, "c": 0, "d": 0}]},
        require_revision_diff=True,
    )
    call = {"id": "view-1", "type": "function", "function": {
        "name": "inspect_map_view",
        "arguments": json.dumps({"start_seconds": 2, "end_seconds": 4,
                                 "view": "revision_diff"}),
    }}

    class ToolVisionClient:
        def __init__(self):
            self.calls = []

        def chat(self, _model, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if len(self.calls) == 1:
                return ChatResult("", "demo/model:free", {"prompt_tokens": 3}, {}, (call,))
            assert messages[-2]["role"] == "tool"
            assert "revision_diff" in messages[-2]["content"]
            assert messages[-1]["role"] == "user"
            image_part = messages[-1]["content"][1]
            assert image_part["type"] == "image_url"
            assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
            return ChatResult("map.note(beat=1, x=1, y=0, color=0, direction=1)",
                              "demo/model:free", {"completion_tokens": 4}, {})

    client = ToolVisionClient()
    response = _chat_with_vivify_tools(
        request, client, [{"role": "user", "content": "review map"}], {}, None,
        map_view_inspector=inspector, require_map_view=True,
    )

    assert response.usage["prompt_tokens"] == 3
    assert response.usage["completion_tokens"] == 4
    assert response.usage["beatforge_map_view_calls"] == 1
    assert response.usage["beatforge_map_view_actions"] == [
        {"view": "revision_diff", "start_seconds": 2.0, "end_seconds": 4.0},
    ]
    assert client.calls[0][1]["tools"][0]["function"]["name"] == "inspect_map_view"
    assert client.calls[0][1]["tools"][0]["function"]["parameters"]["properties"]["view"]["enum"] == ["revision_diff"]
    assert client.calls[0][1]["parallel_tool_calls"] is False
    assert client.calls[0][1]["tool_choice"] == {
        "type": "function", "function": {"name": "inspect_map_view"},
    }


def test_generation_forces_import_of_one_returned_candidate_before_final_script():
    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 30, 2.5),
        CapabilityProfile(vivify=True),
    )
    candidate_id = "a" * 20

    class ToolClient:
        def __init__(self):
            self.calls = []

        def chat(self, _model, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if len(self.calls) == 1:
                call = {"id": "search", "type": "function", "function": {
                    "name": "search_polyhaven_assets", "arguments": '{"query":"candle"}',
                }}
                return ChatResult("", "demo/model:free", {}, {}, (call,))
            if len(self.calls) == 2:
                call = {"id": "import", "type": "function", "function": {
                    "name": "import_polyhaven_asset",
                    "arguments": json.dumps({"candidate_id": candidate_id}),
                }}
                return ChatResult("", "demo/model:free", {}, {}, (call,))
            return ChatResult(
                "map.note(beat=1, x=1, y=0, color=0, direction=1)\n"
                "map.custom_event(beat=4, event_type='InstantiatePrefab', "
                f"data={{'asset_id':'{candidate_id}', 'id':'scene_lamp', "
                "'position':[0,0,12], 'rotation':[0,0,0], 'scale':[1,1,1]})\n"
                "map.custom_event(beat=0, event_type='SetRenderingSettings', "
                "data={'renderSettings': {'fog': 1, 'fogColor': [0.08,0.04,0.16,1], "
                "'fogMode': 2, 'fogDensity': 0.004}})",
                "demo/model:free", {}, {},
            )

    class ToolExecutor:
        def __init__(self):
            self.candidates = {}
            self.imported_asset_ids = []
            self.actions = []

        def __call__(self, name, arguments):
            if name == "search_polyhaven_assets":
                self.candidates[candidate_id] = object()
                self.actions.append({"tool": name, "candidate_count": 1})
                return {"results": [{"candidate_id": candidate_id, "title": "Brass Candleholder"}]}
            assert name == "import_polyhaven_asset"
            self.imported_asset_ids.append(candidate_id)
            self.actions.append({"tool": name, "asset_id": candidate_id})
            return {"asset_id": candidate_id, "status": "imported"}

    client = ToolClient()
    result = generate_map(request, client, tool_executor=ToolExecutor())
    assert result.asset_ids == (candidate_id,)
    assert [call[1]["tool_choice"] for call in client.calls] == [
        {"type": "function", "function": {"name": "search_polyhaven_assets"}},
        {"type": "function", "function": {"name": "import_polyhaven_asset"}},
        "auto",
    ]
    assert [schema["function"]["name"] for schema in client.calls[0][1]["tools"]] == [
        "search_polyhaven_assets",
    ]
    import_schema = client.calls[1][1]["tools"][1]
    assert import_schema["function"]["parameters"]["properties"]["candidate_id"]["enum"] == [
        candidate_id,
    ]
    import_followup = client.calls[2][0][-1]
    assert import_followup["role"] == "user"
    assert candidate_id in import_followup["content"]
    assert "InstantiatePrefab" in import_followup["content"]
    assert "do not leave the import unused" in import_followup["content"]
    prefab = next(event for event in result.difficulty["customData"]["_customEvents"]
                  if event["t"] == "InstantiatePrefab")
    assert prefab["d"]["asset_id"] == candidate_id


def test_generation_rejects_imported_vivify_asset_without_prefab_reference():
    from beatforge import ScriptGenerationError

    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 30, 2.5),
        CapabilityProfile(vivify=True),
    )
    imported_id = "a" * 20

    class Client:
        def chat(self, _model, _messages, **_kwargs):
            return ChatResult(
                "map.note(beat=1, x=1, y=0, color=0, direction=1)\n"
                "map.custom_event(beat=0, event_type='SetRenderingSettings', "
                "data={'renderSettings': {'fog': 1, 'fogColor': [0.08,0.04,0.16,1], "
                "'fogMode': 2, 'fogDensity': 0.004}})",
                "demo/model:free", {}, {},
            )

    class Executor:
        imported_asset_ids = [imported_id]
        search_calls = 1
        actions = []

    with pytest.raises(ScriptGenerationError, match="must be referenced by InstantiatePrefab"):
        generate_map(request, Client(), tool_executor=Executor())


def test_vivify_tool_budget_exhaustion_still_requests_final_map_without_tools():
    from beatforge.vivify_tools import MAX_TOOL_CALLS_PER_TURN

    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 30, 2.5),
        CapabilityProfile(vivify=True),
    )

    class ToolClient:
        def __init__(self):
            self.calls = []

        def chat(self, _model, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if kwargs["tool_choice"] != "none":
                index = len([c for c in self.calls if c[1]["tool_choice"] != "none"])
                call = {"id": f"search-{index}", "type": "function", "function": {
                    "name": "search_polyhaven_assets", "arguments": '{"query":"ambient scene"}',
                }}
                return ChatResult("", "demo/model:free", {"completion_tokens": 1}, {}, (call,))
            assert messages[-1]["role"] == "user"
            assert "budget is exhausted" in messages[-1]["content"]
            return ChatResult(
                "map.note(beat=1, x=1, y=0, color=0, direction=1)\n"
                "map.custom_event(beat=0, event_type='SetRenderingSettings', "
                "data={'renderSettings': {'fog': 1, 'fogColor': [0.08,0.04,0.16,1], "
                "'fogMode': 2, 'fogDensity': 0.004}})",
                "demo/model:free", {"completion_tokens": 3}, {},
            )

    class Executor:
        def __init__(self):
            self.search_calls = 0
            self.candidates = {}
            self.imported_asset_ids = []
            self.actions = []

        def __call__(self, _name, arguments):
            self.search_calls += 1
            self.actions.append({"tool": "search_polyhaven_assets", "status": "completed"})
            return {"results": [], "query": arguments["query"]}

    client = ToolClient()
    result = generate_map(request, client, tool_executor=Executor())
    assert result.difficulty["colorNotes"]
    assert result.usage["beatforge_tool_calls"] == MAX_TOOL_CALLS_PER_TURN
    assert result.usage["completion_tokens"] == MAX_TOOL_CALLS_PER_TURN + 3
    assert client.calls[-1][1]["tool_choice"] == "none"
    assert client.calls[-1][1]["tools"] == []
    assert client.calls[-1][1]["tool_choice"] == "none"


def test_generation_rejects_missing_chroma_palette_for_bounded_repair():
    from beatforge import ScriptGenerationError

    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 30, 2.5),
        CapabilityProfile(chroma=True),
    )
    try:
        generate_map(request, FakeClient())
    except ScriptGenerationError as exc:
        assert "Chroma is enabled but map.palette(...) is missing" in str(exc)
    else:
        raise AssertionError("expected missing Chroma palette to trigger repair")


def test_generation_flow_wraps_script_parse_failure():
    from beatforge import ScriptGenerationError

    class BrokenClient:
        def chat(self, model_id, messages, **kwargs):
            return FakeResponse("map.note(1, 1, 0)", {"total_tokens": 3})

    request = GenerationRequest("demo/model:free", "song.ogg",
                                TrackPlan(120, 30, 2.5), CapabilityProfile())
    try:
        generate_map(request, BrokenClient())
    except ScriptGenerationError as exc:
        assert "missing" in str(exc)
        assert exc.script == "map.note(1, 1, 0)"
    else:
        raise AssertionError("expected ScriptGenerationError")


def test_free_audio_model_receives_audio_bytes_and_text_prompt(tmp_path):
    audio = tmp_path / "song.ogg"
    audio.write_bytes(b"test ogg bytes")

    class AudioClient:
        enforce_free = True

        def __init__(self):
            self.messages = None

        def supports_input_modality(self, model_id, modality):
            assert model_id == "thinkingmachines/inkling:free"
            assert modality == "audio"
            return True

        def chat(self, model_id, messages, **kwargs):
            self.messages = messages
            return FakeResponse("map.note(beat=1, x=1, y=0, color=0, direction=1)", {})

    client = AudioClient()
    request = GenerationRequest("thinkingmachines/inkling:free", str(audio),
                                TrackPlan(120, 30, 2.5), CapabilityProfile(), send_audio=True)
    result = generate_map(request, client)
    content = client.messages[0]["content"]
    assert content[0]["type"] == "text"
    assert "Listen to it" in content[0]["text"]
    assert content[1]["type"] == "input_audio"
    assert content[1]["input_audio"]["format"] == "ogg"
    assert base64.b64decode(content[1]["input_audio"]["data"]) == b"test ogg bytes"
    assert result.difficulty["colorNotes"]


def test_free_text_only_model_is_rejected_before_generation(tmp_path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio")

    class TextOnlyClient:
        enforce_free = True

        def supports_input_modality(self, _model, _modality):
            return False

        def chat(self, *_args, **_kwargs):
            raise AssertionError("must reject before sending a model call")

    request = GenerationRequest("demo/text:free", str(audio),
                                TrackPlan(120, 30, 2.5), CapabilityProfile(), send_audio=True)
    with pytest.raises(ValueError, match="audio input"):
        generate_map(request, TextOnlyClient())


def test_openrouter_vision_model_receives_spectrograms_without_audio_bytes():
    class VisionClient:
        enforce_free = True

        def __init__(self):
            self.messages = None

        def supports_input_modality(self, _model, modality):
            return modality == "image"

        def chat(self, _model, messages, **_kwargs):
            self.messages = messages
            return FakeResponse("map.note(beat=1, x=1, y=0, color=0, direction=1)", {})

    client = VisionClient()
    request = GenerationRequest(
        "google/gemma-4-26b-a4b-it:free", "song.ogg",
        TrackPlan(120, 30, 2.5), CapabilityProfile(),
        spectrogram_images=(b"PNG-one", b"PNG-two"),
        spectrogram_manifest={"images": [{"sha256": "a"}, {"sha256": "b"}]},
    )
    result = generate_map(request, client)
    content = client.messages[0]["content"]
    assert content[0]["type"] == "text"
    assert "BeatBench SG-compatible" in content[0]["text"]
    assert "magma colormap" in content[0]["text"]
    assert "synthetic magnitude encoding only" in content[0]["text"]
    assert "never copy heatmap hues into Chroma or Vivify" in content[0]["text"]
    assert [part["type"] for part in content] == ["text", "image_url", "image_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert not any(part["type"] == "input_audio" for part in content)
    assert result.difficulty["colorNotes"]


def test_spectrogram_packing_is_explained_to_vision_model():
    class VisionClient:
        enforce_free = True

        def supports_input_modality(self, _model, modality):
            return modality == "image"

        def chat(self, _model, messages, **_kwargs):
            assert "read each stack from top to bottom" in messages[0]["content"][0]["text"]
            assert "original pixel resolution" in messages[0]["content"][0]["text"]
            return FakeResponse("map.note(beat=1, x=1, y=0, color=0, direction=1)", {})

    request = GenerationRequest(
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "song.ogg",
        TrackPlan(120, 30, 2.5), CapabilityProfile(),
        spectrogram_images=(b"overview", b"stacked"),
        spectrogram_manifest={"model_input": {
            "packing": "overview plus vertical stacks of consecutive panels"}},
    )
    assert generate_map(request, VisionClient()).difficulty["colorNotes"]


def test_repair_omits_spectrogram_images_but_keeps_text_diagnostics():
    from beatforge import GenerationResult, repair_map

    class VisionClient:
        enforce_free = True

        def __init__(self):
            self.messages = None

        def supports_input_modality(self, _model, modality):
            return modality == "image"

        def chat(self, _model, messages, **_kwargs):
            self.messages = messages
            return FakeResponse("map.note(beat=1, x=1, y=0, color=0, direction=1)", {})

    client = VisionClient()
    request = GenerationRequest(
        "google/gemma-4-26b-a4b-it:free", "song.ogg",
        TrackPlan(120, 30, 2.5), CapabilityProfile(),
        spectrogram_images=(b"PNG-one", b"PNG-two"),
    )
    previous = GenerationResult(request.model_id,
                                "map.note(beat=1, x=1, y=0, color=0, direction=1)",
                                {"colorNotes": []}, {})
    repair_map(request, previous, {"warnings": ["check geometry"]}, client)

    content = client.messages[0]["content"]
    assert isinstance(content, str)
    assert "check geometry" in content
    assert "Existing script:" in content
    assert "BeatBench SG-compatible" not in content


def test_repair_sends_only_sg_panels_near_pattern_targets():
    from beatforge import GenerationResult, repair_map

    class VisionClient:
        enforce_free = True

        def __init__(self):
            self.messages = None

        def supports_input_modality(self, _model, modality):
            return modality == "image"

        def chat(self, _model, messages, **_kwargs):
            self.messages = messages
            return FakeResponse("map.note(beat=1, x=1, y=0, color=0, direction=1)", {})

    rows = [{"file": "overview.png", "start_seconds": 0, "end_seconds": 20}]
    rows += [{"file": f"segment_{index:03d}.png", "start_seconds": index * 4,
              "end_seconds": (index + 1) * 4} for index in range(5)]
    images = (b"overview", *(f"panel-{index}".encode() for index in range(5)))
    request = GenerationRequest(
        "google/gemma-4-26b-a4b-it:free", "song.ogg",
        TrackPlan(120, 20, 2.5), CapabilityProfile(),
        spectrogram_images=images,
        spectrogram_manifest={"images": rows},
    )
    previous = GenerationResult(request.model_id,
                                "map.note(beat=1, x=1, y=0, color=0, direction=1)",
                                {"colorNotes": []}, {})
    diagnostics = {
        "warnings": ["map pattern changes show little relationship to measured audio changes"],
        "pattern_alignment": {
            "available": True,
            "warning": True,
            "chance_corrected_structure_support": 0.05,
            "windows": 5,
            "window_profiles": [],
            "repair_targets": [{"between_windows": [1, 2], "audio_change": 0.8,
                                "map_change": 0.1, "unmatched_change": 0.7}],
        },
    }
    client = VisionClient()

    repair_map(request, previous, diagnostics, client)

    content = client.messages[0]["content"]
    assert isinstance(content, list)
    text = content[0]["text"]
    assert "targeted subset" in text
    assert "magma colormap" in text and "synthetic magnitude encoding only" in text
    assert "never copy heatmap hues into Chroma or Vivify" in text
    assert "segment_000.png" in text and "segment_003.png" in text
    assert "segment_004.png" not in text
    assert sum(part["type"] == "image_url" for part in content) == 5


def test_repair_targets_sg_panels_near_persistent_ridge_candidates():
    from base64 import b64decode
    from io import BytesIO
    from PIL import Image

    from beatforge import GenerationResult, repair_map

    class VisionClient:
        enforce_free = True

        def __init__(self):
            self.messages = None

        def supports_input_modality(self, _model, modality):
            return modality == "image"

        def chat(self, _model, messages, **_kwargs):
            self.messages = messages
            return FakeResponse("map.note(beat=1, x=1, y=0, color=0, direction=1)", {})

    rows = [{"file": "overview.png", "start_seconds": 0, "end_seconds": 24}]
    rows += [{"file": f"segment_{index:03d}.png", "start_seconds": index * 4,
              "end_seconds": (index + 1) * 4} for index in range(6)]
    images = (b"overview", *(f"panel-{index}".encode() for index in range(6)))
    request = GenerationRequest(
        "dots-studio/dots-3-note-preview:free", "song.ogg",
        TrackPlan(120, 24, 2.5), CapabilityProfile(), spectrogram_images=images,
        spectrogram_manifest={"images": rows, "sustained_ridge_candidates": [{
            "start_seconds": 16.1, "end_seconds": 19.2,
            "duration_seconds": 3.1, "frequency_hz": 440.0,
            "mean_level_db": -20.0, "harmonics_hz": [880.0],
        }]},
    )
    previous = GenerationResult(request.model_id,
                                "map.note(beat=16.5, x=1, y=0, color=0, direction=1)",
                                {"colorNotes": [{}], "sliders": []}, {})
    client = VisionClient()

    repair_map(request, previous, {"warnings": ["review map"]}, client)

    content = client.messages[0]["content"]
    assert isinstance(content, list)
    assert "ridge frequencies 440Hz/880Hz" in content[0]["text"]
    assert "candidate is a clear, continuous musical line" in content[0]["text"]
    assert "targeted subset" in content[0]["text"]
    assert "segment_003.png" in content[0]["text"]
    assert "segment_004.png" in content[0]["text"]
    assert "segment_001.png" not in content[0]["text"]
    assert sum(part["type"] == "image_url" for part in content) == 8
    assert "Schematic current-map segment previews follow" in content[0]["text"]
    assert '"spectrogram_file": "segment_003.png"' in content[0]["text"]
    preview_url = content[-1]["image_url"]["url"]
    preview = Image.open(BytesIO(b64decode(preview_url.split(",", 1)[1])))
    assert preview.size == (1400, 900)


def test_model_visual_packet_includes_global_and_synchronized_views():
    from base64 import b64decode
    from io import BytesIO
    from PIL import Image

    from beatforge.generation import _generation_messages
    from beatforge.map_previews import (
        render_map_overview_png,
        render_map_segment_preview_png,
        render_playfield_contact_sheet_png,
        render_player_perspective_contact_sheet_png,
    )

    class VisionClient:
        enforce_free = True

        def supports_input_modality(self, _model, modality):
            return modality == "image"

    sg_overview = BytesIO()
    Image.new("RGB", (700, 240), (60, 70, 90)).save(sg_overview, format="PNG")
    sg_segment = BytesIO()
    Image.new("RGB", (700, 240), (110, 120, 140)).save(sg_segment, format="PNG")
    difficulty = {"colorNotes": [{"b": 4, "x": 1, "y": 1, "c": 0, "d": 0}]}
    map_segment = render_map_segment_preview_png(
        difficulty, bpm=120, start_seconds=2, end_seconds=4,
    )
    map_overview = render_map_overview_png(
        difficulty, bpm=120, duration_seconds=12,
    )
    contact_sheet = render_playfield_contact_sheet_png(
        difficulty, bpm=120, start_seconds=2, end_seconds=4,
    )
    perspective_sheet = render_player_perspective_contact_sheet_png(
        difficulty, bpm=120, start_seconds=2, end_seconds=4,
    )
    request = GenerationRequest(
        "google/gemma-4-31b-it:free", "song.ogg", TrackPlan(120, 12, 2.5),
        CapabilityProfile(), spectrogram_images=(sg_overview.getvalue(), sg_segment.getvalue()),
        spectrogram_manifest={"images": [
            {"file": "overview.png", "start_seconds": 0, "end_seconds": 12},
            {"file": "segment_001.png", "start_seconds": 2, "end_seconds": 4},
        ], "model_input": {"repair_selection": {"attachments": [
            {"attachment": 1, "source_files": ["overview.png"], "priority": "overview"},
            {"attachment": 2, "source_files": ["segment_001.png"], "priority": "diagnosed_repair_window"},
        ]}}},
    )

    messages = _generation_messages(
        request, VisionClient(), "Review current map",
        map_previews=[(map_segment, {
            "start_seconds": 2, "end_seconds": 4,
            "spectrogram_file": "segment_001.png", "spectrogram_attachment": 2,
            "reason": "diagnosed_repair_window",
        })],
        map_overview=map_overview,
        playfield_previews=[(contact_sheet, {"start_seconds": 2, "end_seconds": 4})],
        player_perspective_previews=[
            (perspective_sheet, {"start_seconds": 2, "end_seconds": 4})
        ],
    )

    content = messages[0]["content"]
    assert isinstance(content, list)
    text = content[0]["text"]
    assert "global BeatBench SG overview" in text
    assert "whole-track schematic map overview" in text
    assert "Synchronized local composites follow" in text
    assert "Sequential top-down playfield contact sheets follow" in text
    assert "player-perspective approach proxy" in text
    assert "not Beat Saber geometry" in text
    assert sum(part["type"] == "image_url" for part in content) == 5
    composite = Image.open(BytesIO(b64decode(content[-3]["image_url"]["url"].split(",", 1)[1])))
    assert composite.size == (1400, 88 + 480 + 900)
    perspective = Image.open(BytesIO(b64decode(content[-1]["image_url"]["url"].split(",", 1)[1])))
    assert perspective.size == (1400, 900)


def test_arc_repair_prioritizes_ridge_panels_before_capping_diagnostic_panels():
    from beatforge.generation import _repair_spectrogram_request

    rows = [{"file": "overview.png", "start_seconds": 0, "end_seconds": 80}]
    rows += [{"file": f"segment_{index:03d}.png", "start_seconds": index * 4,
              "end_seconds": (index + 1) * 4} for index in range(20)]
    images = (b"overview", *(f"panel-{index}".encode() for index in range(20)))
    candidates = [{
        "start_seconds": start, "end_seconds": start + 1.8,
        "duration_seconds": 1.8, "frequency_hz": 440.0 + index * 40,
        "mean_level_db": -10.0 - index,
    } for index, start in enumerate((8.2, 28.2, 48.2, 68.2))]
    request = GenerationRequest(
        "dots-studio/dots-3-note-preview:free", "song.ogg",
        TrackPlan(120, 80, 2.5), CapabilityProfile(), spectrogram_images=images,
        spectrogram_manifest={"images": rows,
                              "sustained_ridge_candidates": candidates},
    )
    diagnostics = {"pattern_alignment": {"repair_targets": [
        {"between_windows": [0, 17]},
    ]}}

    selected = _repair_spectrogram_request(
        request, diagnostics, focus_arc_candidates=True,
    )

    selection = selected.spectrogram_manifest["model_input"]["repair_selection"]
    attachments = selection["attachments"]
    arc_panels = {
        filename
        for attachment in attachments
        if attachment["priority"] == "sustained_ridge_candidate"
        for filename in attachment["source_files"]
    }
    assert selection["panel_limit"] == 12
    assert selection["attached_image_count"] == 13  # overview + 12 panels
    assert len(selected.spectrogram_images) == 13
    assert {f"segment_{index:03d}.png" for index in
            (1, 2, 6, 7, 11, 12, 16, 17)} <= arc_panels
    assert sum(item["priority"] == "sustained_ridge_candidate"
               for item in attachments) == 8
    assert any(item["priority"] == "diagnosed_repair_window"
               for item in attachments)


def test_chain_review_repair_attaches_synchronized_sg_around_candidate_run():
    from beatforge.generation import _repair_spectrogram_request

    rows = [{"file": "overview.png", "start_seconds": 0, "end_seconds": 80}]
    rows += [{"file": f"segment_{index:03d}.png", "start_seconds": index * 4,
              "end_seconds": (index + 1) * 4} for index in range(20)]
    images = (b"overview", *(f"panel-{index}".encode() for index in range(20)))
    request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 80, 2.5),
        CapabilityProfile(), spectrogram_images=images,
        spectrogram_manifest={"images": rows},
    )
    diagnostics = {
        "chain_opportunities": {
            "actionable_repair": True,
            "eligible_runs": [{"color": 1, "start_beat": 32.0,
                                "end_beat": 32.5, "head_count": 3}],
        },
    }

    selected = _repair_spectrogram_request(request, diagnostics)

    attachments = selected.spectrogram_manifest["model_input"]["repair_selection"]["attachments"]
    chain_panels = {
        filename
        for item in attachments if item["priority"] == "rapid_chain_candidate"
        for filename in item["source_files"]
    }
    assert chain_panels == {"segment_003.png", "segment_004.png", "segment_005.png"}
    assert selected.spectrogram_manifest["model_input"]["repair_selection"]["reason"].startswith(
        "overview plus SG panels near diagnosed chain")


def test_sparse_lighting_repair_attaches_sg_for_missing_active_sections():
    from beatforge import Section
    from beatforge.generation import _repair_spectrogram_request

    rows = [{"file": "overview.png", "start_seconds": 0, "end_seconds": 80}]
    rows += [{"file": f"segment_{index:03d}.png", "start_seconds": index * 4,
              "end_seconds": (index + 1) * 4} for index in range(20)]
    images = (b"overview", *(f"panel-{index}".encode() for index in range(20)))
    request = GenerationRequest(
        "demo/model:free", "song.ogg",
        TrackPlan(120, 80, 2.5, sections=[Section(24, 40, "drop")]),
        CapabilityProfile(), spectrogram_images=images,
        spectrogram_manifest={"images": rows},
    )
    selected = _repair_spectrogram_request(request, {
        "lighting_coverage": {"warning": True, "missing_sections": ["drop"]},
    })

    attachments = selected.spectrogram_manifest["model_input"]["repair_selection"]["attachments"]
    lighting_panels = {
        filename
        for item in attachments if item["priority"] == "sparse_lighting_section"
        for filename in item["source_files"]
    }
    assert lighting_panels == {"segment_002.png", "segment_003.png", "segment_004.png",
                               "segment_005.png"}


def test_chain_overuse_repair_receives_sg_panels_for_flagged_windows():
    from beatforge import GenerationResult, repair_map

    class VisionClient:
        enforce_free = True

        def __init__(self):
            self.messages = None

        def supports_input_modality(self, _model, modality):
            return modality == "image"

        def chat(self, _model, messages, **_kwargs):
            self.messages = messages
            return FakeResponse("map.note(beat=1, x=1, y=0, color=0, direction=1)", {})

    rows = [{"file": "overview.png", "start_seconds": 0, "end_seconds": 20}]
    rows += [{"file": f"segment_{index:03d}.png", "start_seconds": index * 4,
              "end_seconds": (index + 1) * 4} for index in range(5)]
    images = (b"overview", *(f"panel-{index}".encode() for index in range(5)))
    request = GenerationRequest(
        "google/gemma-4-26b-a4b-it:free", "song.ogg",
        TrackPlan(120, 20, 2.5), CapabilityProfile(),
        spectrogram_images=images, spectrogram_manifest={"images": rows},
    )
    previous = GenerationResult(
        request.model_id, "map.note(beat=1, x=1, y=0, color=0, direction=1)",
        {"colorNotes": []}, {},
    )
    client = VisionClient()

    repair_map(request, previous, {
        "warnings": ["chain_slice_overuse: chain slices dominate playable hits"],
        "chain_usage": {
            "warning": True, "chain_slice_share": 0.49, "warning_threshold": 0.35,
            "chain_slices": 200, "chains_present": 50,
            "repair_targets": [{"window": 3, "beats": [24.0, 32.0],
                                "chain_slice_share": 0.8}],
        },
    }, client)

    content = client.messages[0]["content"]
    assert isinstance(content, list)
    assert "window 3 beats 24.00–32.00" in content[0]["text"]
    assert "80.0%" not in content[0]["text"]
    assert '"chain_slice_share"' not in content[0]["text"]
    assert '"warning_threshold"' not in content[0]["text"]
    assert "segment_002.png" in content[0]["text"]
    assert "segment_004.png" in content[0]["text"]
    assert "segment_001.png" not in content[0]["text"]
    assert sum(part["type"] == "image_url" for part in content) == 4


def test_low_contrast_palette_repair_receives_complete_sg_track():
    from beatforge import GenerationResult, repair_map

    class VisionClient:
        enforce_free = True

        def __init__(self):
            self.messages = None

        def supports_input_modality(self, _model, modality):
            return modality == "image"

        def chat(self, _model, messages, **_kwargs):
            self.messages = messages
            return ChatResult(
                "map.palette(environment_left=[35,125,245], environment_right=[245,115,25])\n"
                "map.note(beat=1, x=1, y=0, color=0, direction=1)",
                "google/gemma-4-26b-a4b-it:free", {}, {},
            )

    rows = [{"file": "overview.png", "start_seconds": 0, "end_seconds": 20}]
    rows += [{"file": f"segment_{index:03d}.png", "start_seconds": index * 4,
              "end_seconds": (index + 1) * 4} for index in range(5)]
    images = (b"overview", *(f"panel-{index}".encode() for index in range(5)))
    request = GenerationRequest(
        "google/gemma-4-26b-a4b-it:free", "song.ogg",
        TrackPlan(120, 20, 2.5), CapabilityProfile(chroma=True),
        spectrogram_images=images, spectrogram_manifest={"images": rows},
    )
    previous = GenerationResult(
        request.model_id,
        "map.palette(environment_left=[22,12,44], environment_right=[80,24,110])\n"
        "map.note(beat=1, x=1, y=0, color=0, direction=1)",
        {"colorNotes": []}, {},
    )
    client = VisionClient()

    repair_map(request, previous, {
        "warnings": ["scene_palette_low_contrast: environment colors are too similar"],
        "palette_contrast": {"normalized_rgb_distance": 0.201},
    }, client)

    content = client.messages[0]["content"]
    assert isinstance(content, list)
    assert "Replace only the environment color pair" in content[0]["text"]
    assert "segment_000.png" in content[0]["text"]
    assert "segment_004.png" in content[0]["text"]
    assert sum(part["type"] == "image_url" for part in content) == 6


def test_rhythm_density_repair_receives_contrasting_high_and_low_sg_windows():
    from beatforge import GenerationResult, repair_map

    class VisionClient:
        enforce_free = True

        def __init__(self):
            self.messages = None

        def supports_input_modality(self, _model, modality):
            return modality == "image"

        def chat(self, _model, messages, **_kwargs):
            self.messages = messages
            return ChatResult('{"note_replacements":[],"chain_conversions":[]}',
                              "google/gemma-4-26b-a4b-it:free", {}, {})

    rows = [{"file": "overview.png", "start_seconds": 0, "end_seconds": 20}]
    rows += [{"file": f"segment_{index:03d}.png", "start_seconds": index * 4,
              "end_seconds": (index + 1) * 4} for index in range(5)]
    images = (b"overview", *(f"panel-{index}".encode() for index in range(5)))
    request = GenerationRequest(
        "google/gemma-4-26b-a4b-it:free", "song.ogg",
        TrackPlan(120, 20, 2.5), CapabilityProfile(),
        spectrogram_images=images, spectrogram_manifest={"images": rows},
    )
    previous = GenerationResult(request.model_id,
                                "map.note(beat=1, x=1, y=0, color=0, direction=1)",
                                {"colorNotes": []}, {})
    client = VisionClient()

    repair_map(request, previous, {
        "warnings": ["rhythm_density_unresponsive: map hit density stays flat"],
        "rhythm_alignment": {
            "warning": True,
            "audio_activity_cv": 1.0,
            "map_density_cv": 0.0,
            "density_activity_correlation": None,
            "repair_targets": [
                {"window": 0, "beats": [0, 8], "map_hit_equivalents": 16,
                 "audio_strong_attack_cells": 12},
                {"window": 4, "beats": [32, 40], "map_hit_equivalents": 16,
                 "audio_strong_attack_cells": 0},
            ],
        },
    }, client)

    content = client.messages[0]["content"]
    assert isinstance(content, list)
    text = content[0]["text"]
    assert "rhythm_density_unresponsive" in text
    assert '"window": 0' in text and '"window": 4' in text
    assert '"beats"' in text and "32" in text and "40" in text
    assert "segment_000.png" in text and "segment_001.png" in text
    assert "segment_003.png" in text and "segment_004.png" in text
    assert sum(part["type"] == "image_url" for part in content) == 5


def test_text_only_openrouter_model_does_not_receive_spectrogram_images():
    class TextClient:
        enforce_free = True

        def __init__(self):
            self.messages = None

        def supports_input_modality(self, _model, modality):
            return modality == "audio"

        def chat(self, _model, messages, **_kwargs):
            self.messages = messages
            return FakeResponse("map.note(beat=1, x=1, y=0, color=0, direction=1)", {})

    client = TextClient()
    request = GenerationRequest("cohere/north-mini-code:free", "song.ogg",
                                TrackPlan(120, 30, 2.5), CapabilityProfile(),
                                audio_features="measured signal text",
                                spectrogram_images=(b"PNG",))
    generate_map(request, client)
    assert isinstance(client.messages[0]["content"], str)
    assert "measured signal text" in client.messages[0]["content"]


@pytest.mark.parametrize("send_spectrograms, expected_images", [(False, 0), (True, 2)])
def test_local_vision_transport_requires_explicit_spectrogram_opt_in(send_spectrograms, expected_images):
    class LocalClient:
        enforce_free = False

        def __init__(self):
            self.messages = None

        def supports_input_modality(self, *_args):
            raise AssertionError("local models use the explicit vision opt-in, not OpenRouter catalog lookup")

        def chat(self, _model, messages, **_kwargs):
            self.messages = messages
            return FakeResponse("map.note(beat=1, x=1, y=0, color=0, direction=1)", {})

    client = LocalClient()
    request = GenerationRequest(
        "gemma", "song.ogg", TrackPlan(120, 30, 2.5), CapabilityProfile(),
        provider="local", send_spectrograms=send_spectrograms,
        audio_features="measured local signal summary",
        spectrogram_images=(b"overview-png", b"panel-png"),
        spectrogram_manifest={"settings": {"sample_rate": 44100}},
    )
    generate_map(request, client)
    content = client.messages[0]["content"]
    image_parts = [part for part in content if isinstance(part, dict) and part.get("type") == "image_url"]
    assert len(image_parts) == expected_images
    if send_spectrograms:
        assert "BeatBench SG-compatible" in content[0]["text"]
        assert "measured local signal summary" in content[0]["text"]
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    else:
        assert isinstance(content, str)
