from dataclasses import dataclass
import json

from beatforge import (CapabilityProfile, GenerationRequest, GenerationResult, JobStore,
                       Section, TrackPlan, BudgetGuard, run_generation_job, run_script)
from beatforge.openrouter import OpenRouterResponseError


@dataclass
class Response:
    content: str
    usage: dict


class Client:
    def chat(self, model_id, messages, **kwargs):
        return Response("map.note(1, 1, 0, 0, 1)", {"total_tokens": 2})


class DuplicatePaletteAndNoteClient:
    def chat(self, model_id, messages, **kwargs):
        palette = (
            "map.palette(environment_left=[0, 20, 30], "
            "environment_right=[240, 220, 200])"
        )
        note = "map.note(beat=1, x=1, y=0, color=0, direction=1)"
        return Response("\n".join((palette, palette, note, note)), {"cost": 0})


def request():
    return GenerationRequest("demo/model:free", "song.ogg",
                             TrackPlan(120, 10, 2.5, sections=[Section(0, 8, "intro")]),
                             CapabilityProfile())


def test_worker_persists_started_and_completed(tmp_path):
    store = JobStore(tmp_path)
    result = run_generation_job("j1", request(), Client(), store, BudgetGuard())
    assert result.status == "completed"
    lines = (tmp_path / "j1" / "events.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["generation_started", "generation_completed"]
    assert json.loads(lines[-1])["payload"]["script"] == "map.note(1, 1, 0, 0, 1)"
    live = store.live("j1")
    assert live["status"] == "completed"
    assert live["script"] == "map.note(1, 1, 0, 0, 1)"


def test_worker_cleans_duplicate_palette_before_exact_note_cleanup(tmp_path):
    chroma_request = GenerationRequest(
        "demo/model:free", "song.ogg",
        TrackPlan(120, 10, 2.5, sections=[Section(0, 8, "intro")]),
        CapabilityProfile(chroma=True),
    )
    store = JobStore(tmp_path)
    result = run_generation_job(
        "duplicate-palette-and-note", chroma_request,
        DuplicatePaletteAndNoteClient(), store, BudgetGuard(),
    )

    assert result.status == "completed"
    assert result.generation is not None
    assert result.generation.script.count("map.palette(") == 1
    assert result.generation.script.count("map.note(") == 1
    assert len(result.generation.difficulty["colorNotes"]) == 1
    assert result.generation.usage["beatforge_automatic_repairs"] == [
        {"kind": "removed_exact_duplicate_palette", "line": 2,
         "duplicate_of_line": 1,
         "palette": {"environment_left": [0, 20, 30],
                     "environment_right": [240, 220, 200]}},
        {"kind": "removed_exact_duplicate_note", "line": 3,
         "duplicate_of": {"kind": "map.note", "line": 2},
         "beat": 1.0, "x": 1, "y": 0, "color": 0, "direction": 1},
    ]
    live = store.live("duplicate-palette-and-note")
    assert live["status"] == "completed"
    assert len(live["automatic_repairs"]) == 2


def test_worker_reserves_initial_generation_followups_for_diagnostics_repair(tmp_path, monkeypatch):
    import beatforge.worker as worker_mod

    captured = {}

    def fake_generate_map(current_request, _client, **kwargs):
        captured["max_tool_calls"] = kwargs["max_tool_calls"]
        script = "map.note(beat=1, x=0, y=0, color=0, direction=1)"
        return GenerationResult(
            current_request.model_id, script,
            run_script(script, bpm=current_request.plan.bpm),
            {"cost": 0},
        )

    monkeypatch.setattr(worker_mod, "generate_map", fake_generate_map)
    budget = BudgetGuard(max_requests=8)
    result = run_generation_job(
        "reserved-repair-budget", request(), Client(), JobStore(tmp_path), budget,
        reserve_followup_requests=2,
    )

    assert result.status == "completed"
    assert captured["max_tool_calls"] == 5
    assert budget.requests == 1
    events = [json.loads(line) for line in
              (tmp_path / "reserved-repair-budget" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert events[0]["kind"] == "generation_started"
    assert events[0]["payload"]["reserve_followup_requests"] == 2


def test_worker_script_recovery_preserves_diagnostic_repair_requests(tmp_path):
    class ParseRepairClient:
        def __init__(self):
            self.calls = 0

        def chat(self, _model, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return Response(
                    "map.custom_event(beat=64, custom_event_type='strobe', custom_data={})",
                    {"cost": 0},
                )
            if self.calls == 2:
                assert "script_error" in messages[0]["content"]
            return Response("map.note(1, 1, 0, 0, 1)", {"cost": 0})

    client = ParseRepairClient()
    store = JobStore(tmp_path)
    budget = BudgetGuard(max_requests=4, max_repairs=1)
    generated = run_generation_job(
        "reserved-after-script-recovery", request(), client, store, budget,
        reserve_followup_requests=2,
    )

    assert generated.status == "completed"
    assert budget.requests == 2
    repaired = run_generation_job(
        "reserved-after-script-recovery", request(), client, store, budget,
        repair=True, previous=generated.generation, diagnostics={"test": True},
    )

    assert repaired.status == "completed"
    assert client.calls == 3
    assert budget.requests == 3


def test_worker_publishes_running_snapshot_before_provider_returns(tmp_path):
    class InspectingClient:
        def __init__(self):
            self.snapshot = None

        def chat(self, model_id, messages, **kwargs):
            self.snapshot = store.live("live")
            return Response("map.note(1, 1, 0, 0, 1)", {})

    store = JobStore(tmp_path)
    client = InspectingClient()
    result = run_generation_job("live", request(), client, store)
    assert result.status == "completed"
    assert client.snapshot["status"] == "provider_running"
    assert client.snapshot["model_id"] == "demo/model:free"


def test_worker_publishes_partial_streamed_response(tmp_path):
    store = JobStore(tmp_path)

    class StreamingClient:
        def set_progress_callback(self, callback):
            self.progress = callback

        def chat(self, model_id, messages, **kwargs):
            self.progress("")
            self.progress("map.note(")
            snapshot = store.live("partial")
            assert snapshot["status"] == "provider_running"
            assert snapshot["stream_connected"] is True
            assert snapshot["partial_response"] == "map.note("
            assert snapshot["partial_response_chars"] == len("map.note(")
            return Response("map.note(1, 1, 0, 0, 1)", {"total_tokens": 2})

    result = run_generation_job("partial", request(), StreamingClient(), store)
    assert result.status == "completed"
    final = store.live("partial")
    assert final["script"] == "map.note(1, 1, 0, 0, 1)"
    assert "partial_response" not in final


def test_worker_persists_failure(tmp_path):
    class Broken:
        def chat(self, *args, **kwargs):
            raise RuntimeError("provider down")
    result = run_generation_job("j2", request(), Broken(), JobStore(tmp_path))
    assert result.status == "failed"
    assert "provider down" in result.error
    assert "generation_failed" in (tmp_path / "j2" / "events.jsonl").read_text()
    assert JobStore(tmp_path).live("j2")["status"] == "failed"


def test_worker_persists_sanitized_provider_diagnostics(tmp_path):
    diagnostics = {
        "selected_model": "demo/actual-model:free",
        "choice": {"finish_reason": "error", "message_fields": ["role"]},
        "route": {"selected_provider": "Demo Provider", "is_byok": False},
    }

    class ProviderFailure:
        def chat(self, *args, **kwargs):
            raise OpenRouterResponseError(
                "OpenRouter response has no textual content",
                usage={"prompt_tokens": 9, "completion_tokens": 4,
                       "total_tokens": 13, "cost": 0, "is_byok": False},
                finish_reason="error", diagnostics=diagnostics,
            )

    store = JobStore(tmp_path)
    result = run_generation_job("provider-diagnostics", request(), ProviderFailure(), store)
    assert result.status == "failed"
    live = store.live("provider-diagnostics")
    assert live["model_id"] == "demo/model:free"
    assert live["provider_diagnostics"] == diagnostics
    event = store.latest("provider-diagnostics")
    assert event.kind == "generation_failed"
    assert event.payload["model_id"] == "demo/model:free"
    assert event.payload["provider_diagnostics"] == diagnostics
    assert event.payload["usage"]["cost"] == 0


def test_worker_preserves_provider_usage_when_response_has_no_text(tmp_path):
    class EmptyProviderResponse(RuntimeError):
        usage = {"prompt_tokens": 17, "completion_tokens": 23,
                 "total_tokens": 40, "cost": 0}
        finish_reason = "length"

    class EmptyClient:
        def chat(self, *_args, **_kwargs):
            raise EmptyProviderResponse("response had reasoning but no final text")

    store = JobStore(tmp_path)
    result = run_generation_job("empty-provider", request(), EmptyClient(), store,
                                BudgetGuard())
    assert result.status == "failed"
    assert store.live("empty-provider")["usage"]["total_tokens"] == 40
    assert store.latest("empty-provider").payload["usage"]["completion_tokens"] == 23


def test_worker_retries_reasoning_only_output_limit_once_at_low_effort(tmp_path):
    class ExhaustedReasoningClient:
        def __init__(self):
            self.efforts = []

        def chat(self, _model, _messages, **kwargs):
            effort = kwargs.get("reasoning_effort")
            self.efforts.append(effort)
            if effort is None:
                raise OpenRouterResponseError(
                    "OpenRouter response has no textual content (finish_reason='length')",
                    usage={
                        "prompt_tokens": 10, "completion_tokens": 131072,
                        "total_tokens": 131082, "cost": 0,
                        "completion_tokens_details": {"reasoning_tokens": 131072},
                    },
                    finish_reason="length",
                    diagnostics={"choice": {"finish_reason": "length",
                                             "message_fields": ["role"]}},
                )
            assert effort == "low"
            return Response("map.note(1, 1, 0, 0, 1)", {
                "prompt_tokens": 12, "completion_tokens": 4,
                "total_tokens": 16, "cost": 0,
            })

    store = JobStore(tmp_path)
    client = ExhaustedReasoningClient()
    budget = BudgetGuard(max_automatic_repairs=1)
    result = run_generation_job("reasoning-recovery", request(), client, store, budget,
                                reserve_followup_requests=2)

    assert result.status == "completed"
    assert client.efforts == [None, "low"]
    assert budget.requests == 2
    assert budget.automatic_repairs == 1
    assert result.generation.usage["prompt_tokens"] == 22
    assert result.generation.usage["completion_tokens"] == 131076
    assert result.generation.usage["cost"] == 0
    events = [json.loads(line) for line in
              (tmp_path / "reasoning-recovery" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert [event["kind"] for event in events] == [
        "generation_started", "generation_recovery_started",
        "generation_recovery_completed", "generation_completed",
    ]
    assert events[1]["payload"]["reasoning_effort"] == "low"
    assert store.live("reasoning-recovery")["reasoning_effort"] == "low"


def test_worker_does_not_retry_reasoning_exhaustion_without_reserved_budget(tmp_path):
    class ExhaustedReasoningClient:
        def __init__(self):
            self.calls = 0

        def chat(self, *_args, **_kwargs):
            self.calls += 1
            raise OpenRouterResponseError(
                "empty response at output limit",
                usage={"completion_tokens_details": {"reasoning_tokens": 40}},
                finish_reason="length",
                diagnostics={"choice": {"finish_reason": "length",
                                         "message_fields": ["role"]}},
            )

    store = JobStore(tmp_path)
    client = ExhaustedReasoningClient()
    result = run_generation_job("reasoning-recovery-no-budget", request(), client, store,
                                BudgetGuard(max_automatic_repairs=0))

    assert result.status == "failed"
    assert client.calls == 1
    assert store.latest("reasoning-recovery-no-budget").kind == "generation_failed"
    events = [json.loads(line)["kind"] for line in
              (tmp_path / "reasoning-recovery-no-budget" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert events == ["generation_started", "generation_recovery_skipped", "generation_failed"]


def test_worker_repairs_script_parse_failure_with_repair_budget(tmp_path):
    class ParseRepairClient:
        def __init__(self):
            self.calls = 0

        def chat(self, _model, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return Response(
                    "map.custom_event(beat=64, custom_event_type='strobe', custom_data={})",
                    {"prompt_tokens": 10, "completion_tokens": 20,
                     "total_tokens": 30, "cost": 0},
                )
            assert "script_error" in messages[0]["content"]
            return Response(
                "map.note(1, 1, 0, 0, 1)\n"
                "map.custom_event(beat=2, event_type='AnimateTrack', data={'track':'portal'})",
                {"prompt_tokens": 7, "completion_tokens": 4,
                 "total_tokens": 11, "cost": 0},
            )

    client = ParseRepairClient()
    result = run_generation_job("j-parse", request(), client, JobStore(tmp_path), BudgetGuard(max_repairs=1))
    assert result.status == "completed"
    assert client.calls == 2
    assert result.generation.usage["prompt_tokens"] == 17
    assert result.generation.usage["completion_tokens"] == 24
    assert result.generation.usage["total_tokens"] == 41
    assert [item["phase"] for item in result.generation.usage["attempts"]] == [
        "generation", "automatic_repair",
    ]
    assert result.generation.difficulty["customData"]["_customEvents"][0]["t"] == "AnimateTrack"
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "j-parse" / "events.jsonl").read_text().splitlines()]
    assert kinds == ["generation_started", "generation_repair_started", "generation_repair_completed"]


def test_worker_retains_usage_from_failed_automatic_repair(tmp_path):
    class BadThenStillBad:
        def __init__(self):
            self.calls = 0

        def chat(self, _model, _messages, **_kwargs):
            self.calls += 1
            usage = {"prompt_tokens": 10, "completion_tokens": 20,
                     "total_tokens": 30, "cost": 0}
            if self.calls == 1:
                return Response("map.note(1, 1, 0)", usage)
            return Response(
                "map.note(beat=1, x=1, y=0, color=0, direction=1)\n"
                "map.arc(beat=1, color=0, x=1, y=0, direction=1, "
                "tail_beat=2, tail_x=2, tail_y=1, tail_color=0, curve=0)",
                usage,
            )

    store = JobStore(tmp_path)
    result = run_generation_job("repair-fails", request(), BadThenStillBad(), store,
                                BudgetGuard(max_automatic_repairs=1))
    live = store.live("repair-fails")
    assert result.status == "failed"
    assert live["usage"]["total_tokens"] == 60
    assert live["usage"]["prompt_tokens"] == 20
    assert live["usage"]["completion_tokens"] == 40
    assert [item["phase"] for item in live["usage"]["attempts"]] == [
        "generation", "automatic_repair",
    ]
    assert [item["phase"] for item in live["script_attempts"]] == [
        "generation", "automatic_repair",
    ]
    assert "map.arc" in live["script_attempts"][1]["script"]


def test_worker_persists_rejected_diagnostic_repair_script_and_usage(tmp_path):
    bad_script = (
        "map.note(beat=1, x=1, y=0, color=0, direction=1)\n"
        "map.chain(beat=4, color=0, x=1, y=0, direction=1, "
        "tail_beat=5, tail_x=2, tail_y=1, slices=4, squish=0.9)\n"
        "map.chain(beat=4, color=1, x=1, y=0, direction=1, "
        "tail_beat=5, tail_x=2, tail_y=1, slices=4, squish=0.9)"
    )
    usage = {"prompt_tokens": 31, "completion_tokens": 19,
             "total_tokens": 50, "cost": 0}

    class CollidingRepairClient:
        def __init__(self):
            self.calls = 0

        def chat(self, _model, _messages, **_kwargs):
            self.calls += 1
            return Response(bad_script, usage)

    current_request = request()
    previous_script = "map.note(beat=1, x=0, y=0, color=0, direction=1)"
    previous = GenerationResult(
        current_request.model_id, previous_script,
        run_script(previous_script, bpm=current_request.plan.bpm), {},
    )
    store = JobStore(tmp_path)
    client = CollidingRepairClient()
    result = run_generation_job(
        "diagnostic-repair-collision", current_request, client, store,
        BudgetGuard(max_repairs=1), repair=True, previous=previous,
        diagnostics={"warnings": ["rhythm_density_unresponsive"]},
    )

    assert result.status == "failed"
    assert client.calls == 1
    live = store.live("diagnostic-repair-collision")
    assert live["script"] == bad_script
    assert live["usage"]["prompt_tokens"] == 31
    assert live["usage"]["completion_tokens"] == 19
    assert live["script_attempts"][0]["script"] == bad_script
    event = store.latest("diagnostic-repair-collision")
    assert event.kind == "generation_failed"
    assert event.payload["script"] == bad_script
    assert event.payload["usage"]["total_tokens"] == 50


def test_worker_uses_second_bounded_parse_repair_after_first_repair_fails(tmp_path):
    class TwoStepRepairClient:
        def __init__(self):
            self.calls = 0

        def chat(self, _model, _messages, **_kwargs):
            self.calls += 1
            usage = {"prompt_tokens": 5, "completion_tokens": 5,
                     "total_tokens": 10, "cost": 0}
            if self.calls == 1:
                return Response("map.note(1, 1, 0)", usage)
            if self.calls == 2:
                return Response(
                    "map.note(beat=1, x=1, y=0, color=0, direction=1)\n"
                    "map.arc(beat=1, color=0, x=1, y=0, direction=1, "
                    "tail_beat=2, tail_x=2, tail_y=1, tail_color=0, curve=0.5)",
                    usage,
                )
            return Response("map.note(beat=1, x=1, y=0, color=0, direction=1)", usage)

    client = TwoStepRepairClient()
    budget = BudgetGuard(max_automatic_repairs=2)
    store = JobStore(tmp_path)
    result = run_generation_job("two-repairs", request(), client, store, budget)
    live = store.live("two-repairs")
    kinds = [json.loads(line)["kind"] for line in
             (tmp_path / "two-repairs" / "events.jsonl").read_text().splitlines()]
    assert result.status == "completed"
    assert client.calls == 3
    assert budget.automatic_repairs == 2
    assert result.generation.usage["total_tokens"] == 30
    assert len(result.generation.usage["attempts"]) == 3
    assert len(live["script_attempts"]) == 3
    assert kinds.count("generation_repair_started") == 2
    assert kinds.count("generation_repair_failed") == 1
    assert kinds[-1] == "generation_repair_completed"


def test_worker_auto_removes_exact_duplicate_note_without_extra_model_request(tmp_path):
    class DuplicateNoteClient:
        def __init__(self):
            self.calls = 0

        def chat(self, _model, _messages, **_kwargs):
            self.calls += 1
            return Response(
                "map.repeat(0, 2, 4, [{'kind':'note','beat':0,'x':2,'y':0,'color':1,'direction':0}])\n"
                "map.note(beat=4, x=2, y=0, color=1, direction=0)",
                {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22, "cost": 0},
            )

    client = DuplicateNoteClient()
    store = JobStore(tmp_path)
    result = run_generation_job("exact-duplicate", request(), client, store,
                                BudgetGuard(max_repairs=0))
    live = store.live("exact-duplicate")
    event = store.latest("exact-duplicate")
    assert result.status == "completed"
    assert client.calls == 1
    assert len(result.generation.difficulty["colorNotes"]) == 2
    assert result.generation.usage["beatforge_automatic_repairs"][0]["kind"] == "removed_exact_duplicate_note"
    assert live["automatic_repairs"] == result.generation.usage["beatforge_automatic_repairs"]
    assert event.kind == "generation_auto_repair_completed"


def test_worker_repairs_missing_chroma_palette(tmp_path):
    class ChromaRepairClient:
        def __init__(self):
            self.calls = 0

        def chat(self, _model, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return Response("map.note(1, 1, 0, 0, 1)", {"total_tokens": 12})
            assert "map.palette(environment_left=[R,G,B]" in messages[0]["content"]
            return Response(
                "map.note(1, 1, 0, 0, 1)\n"
                "map.palette(environment_left=[120,80,220], environment_right=[40,180,240])",
                {"total_tokens": 15},
            )

    request_with_chroma = GenerationRequest(
        "demo/model:free", "song.ogg",
        TrackPlan(120, 10, 2.5, sections=[Section(0, 8, "intro")]),
        CapabilityProfile(chroma=True),
    )
    client = ChromaRepairClient()
    result = run_generation_job(
        "chroma-repair", request_with_chroma, client, JobStore(tmp_path),
        BudgetGuard(max_repairs=1),
    )
    assert result.status == "completed"
    assert client.calls == 2
    assert result.generation.difficulty["customData"]["_beatforgeScenePalette"] == {
        "environment_left": [120, 80, 220], "environment_right": [40, 180, 240],
    }


def test_worker_preserves_parse_failure_when_repair_budget_is_zero(tmp_path):
    class BrokenScript:
        def __init__(self):
            self.calls = 0

        def chat(self, *_args, **_kwargs):
            self.calls += 1
            return Response("map.note(1, 1, 0)", {"completion_tokens": 17})

    client = BrokenScript()
    store = JobStore(tmp_path)
    result = run_generation_job("no-repair", request(), client, store,
                                BudgetGuard(max_automatic_repairs=0))
    live = store.live("no-repair")
    event = store.latest("no-repair")
    assert result.status == "failed"
    assert "missing" in result.error
    assert "repair budget exhausted" not in result.error
    assert client.calls == 1
    assert live["script"] == "map.note(1, 1, 0)"
    assert live["usage"]["completion_tokens"] == 17
    assert event.payload["script"] == "map.note(1, 1, 0)"


def test_worker_budget_rejection_is_durable(tmp_path):
    budget = BudgetGuard(max_requests=0)
    result = run_generation_job("j3", request(), Client(), JobStore(tmp_path), budget)
    assert result.status == "failed"
    assert "budget" in result.error
    assert "generation_failed" in (tmp_path / "j3" / "events.jsonl").read_text()


def test_worker_repair_uses_repair_budget_and_events(tmp_path):
    from beatforge import GenerationResult

    class RepairClient:
        def chat(self, _model, messages, **_kwargs):
            if "Repair the existing BeatForge map" in messages[0]["content"]:
                return Response("map.note(2, 2, 0, 0, 1)", {})
            return Response("map.note(1, 1, 0, 0, 1)", {})

    store = JobStore(tmp_path)
    first = run_generation_job("j4", request(), RepairClient(), store, BudgetGuard())
    second = run_generation_job("j4", request(), RepairClient(), store, BudgetGuard(max_repairs=1),
                                repair=True, previous=first.generation,
                                diagnostics={"warnings": ["test"]})
    assert second.status == "completed"
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "j4" / "events.jsonl").read_text().splitlines()]
    assert kinds[-2:] == ["repair_started", "repair_completed"]
    completion_events = [json.loads(line) for line in
                         (tmp_path / "j4" / "events.jsonl").read_text().splitlines()
                         if json.loads(line)["kind"] in {"generation_completed", "repair_completed"}]
    assert [event["payload"]["script"] for event in completion_events] == [
        "map.note(1, 1, 0, 0, 1)", "map.note(2, 2, 0, 0, 1)",
    ]


def test_worker_repairs_imported_vivify_asset_that_initial_map_did_not_instantiate(tmp_path):
    from beatforge import ChatResult

    asset_id = "c" * 20
    vivify_request = GenerationRequest(
        "demo/model:free", "song.ogg", TrackPlan(120, 10, 2.5),
        CapabilityProfile(vivify=True),
    )

    class ToolExecutor:
        imported_asset_ids = [asset_id]
        search_calls = 1
        candidates = {}
        actions = []

    class RepairingClient:
        def __init__(self):
            self.prompts = []

        def chat(self, model_id, messages, **_kwargs):
            prompt = messages[0]["content"]
            self.prompts.append(prompt)
            if "Repair the existing BeatForge map" not in prompt:
                return ChatResult(
                    "map.note(beat=1, x=1, y=0, color=0, direction=1)\n"
                    "map.custom_event(beat=0, event_type='SetRenderingSettings', "
                    "data={'renderSettings': {'fog': 1, 'fogColor': [0.08,0.04,0.16,1], "
                    "'fogMode': 2, 'fogDensity': 0.004}})",
                    model_id, {"cost": 0}, {},
                )
            assert asset_id in prompt
            assert "must instantiate each listed asset at least once" in prompt
            return ChatResult(
                "map.note(beat=1, x=1, y=0, color=0, direction=1)\n"
                f"map.custom_event(beat=2, event_type='InstantiatePrefab', data={{'asset_id':'{asset_id}', "
                "'id':'scene_prop', 'position':[0,0,12], 'rotation':[0,0,0], 'scale':[1,1,1]})\n"
                "map.custom_event(beat=0, event_type='SetRenderingSettings', "
                "data={'renderSettings': {'fog': 1, 'fogColor': [0.08,0.04,0.16,1], "
                "'fogMode': 2, 'fogDensity': 0.004}})",
                model_id, {"cost": 0}, {},
            )

    client = RepairingClient()
    store = JobStore(tmp_path)
    result = run_generation_job("vivify-asset", vivify_request, client, store,
                                BudgetGuard(max_repairs=1), tool_executor=ToolExecutor())

    assert result.status == "completed"
    assert result.generation.asset_ids == (asset_id,)
    assert len(client.prompts) == 2
    kinds = [json.loads(line)["kind"] for line in
             (tmp_path / "vivify-asset" / "events.jsonl").read_text().splitlines()]
    assert kinds == ["generation_started", "generation_repair_started",
                     "generation_repair_completed"]
    repair_event = json.loads((tmp_path / "vivify-asset" / "events.jsonl")
                              .read_text().splitlines()[-1])
    assert asset_id in repair_event["payload"]["script"]
