import json
from pathlib import Path

import soundfile as sf
import numpy as np

from beatforge import (BudgetGuard, ChatResult, ExportMetadata, generate_and_export,
                       AssetLibrary, AssetRecord, CapabilityProfile, TrackPlan,
                       prepare_generation)


class _Client:
    def chat(self, model_id, messages, **_kwargs):
        script = (
            "map.note(0.25, 1, 0, 0, 1)\n"
            "map.arc(0.5, 0, 0, 0, 1, 1, 1, 0, 0, 0)\n"
            "map.chain(1, 1, 2, 0, 1, 2, 3, 0, 2, 0.9)\n"
            "map.light(0.25, 0, 1)"
        )
        return ChatResult(script, model_id, {"cost": 0}, {"choices": []})


def test_workflow_analyzes_runs_durably_and_exports_v3(tmp_path):
    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100, dtype="float32"), 44100)
    prepared = prepare_generation(audio, bpm=120, target_nps=5.0,
                                  model_id="demo/model:free")
    assert prepared.plan.sections
    job, package = generate_and_export(
        prepared, _Client(), job_id="job-1", jobs_root=tmp_path / "jobs",
        output=tmp_path / "map", metadata=ExportMetadata("Demo"),
        budget=BudgetGuard())
    assert job.status == "completed"
    assert package == tmp_path / "map"
    difficulty = json.loads((package / "ExpertPlus.dat").read_text())
    assert difficulty["sliders"] and difficulty["burstSliders"]
    assert json.loads((tmp_path / "jobs" / "job-1" / "latest.json").read_text())["kind"] == "export_completed"


def test_clean_draft_uses_existing_repair_slot_for_free_vision_review(tmp_path, monkeypatch):
    from beatforge.generation import GenerationResult
    from beatforge.script_runner import run_script
    from beatforge.worker import JobResult
    import beatforge.workflow as workflow_mod
    from beatforge.quality import QualityReport

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, model_id="demo/model:free", plan=TrackPlan(120, 1, 1.0),
    )
    script = "map.note(beat=0.25, x=1, y=0, color=0, direction=1)\nmap.light(beat=0, event_type=0, value=1)"
    generation = GenerationResult(
        "demo/model:free", script, run_script(script, bpm=120), {"cost": 0},
    )
    calls = []

    class FreeVisionToolClient:
        enforce_free = True

        def supports_input_modality(self, _model_id, modality):
            return modality == "image"

        def supports_tools(self, _model_id):
            return True

    def fake_run_generation_job(job_id, _request, _client, _store, _budget,
                                *, repair=False, diagnostics=None, **kwargs):
        calls.append((repair, diagnostics, kwargs.get("reserve_followup_requests", 0)))
        return JobResult(job_id, "completed", generation=generation)

    def clean_preflight(*_args, **_kwargs):
        return QualityReport(
            passed=True, errors=(), warnings=(), note_count=1, chain_slices=0,
            actual_nps=1.0, occupied_sections=1, total_sections=1,
            direction_profiles={}, chain_opportunities={}, chain_usage={},
            pattern_alignment={}, palette_contrast={}, rhythm_alignment={},
        )

    monkeypatch.setattr(workflow_mod, "run_generation_job", fake_run_generation_job)
    monkeypatch.setattr(workflow_mod, "preflight", clean_preflight)
    job, package = generate_and_export(
        prepared, FreeVisionToolClient(), job_id="clean-visual-review",
        jobs_root=tmp_path / "jobs", output=tmp_path / "map",
        metadata=ExportMetadata("Visual review"), max_repairs=1,
        budget=BudgetGuard(max_requests=8),
    )
    assert job.status == "completed"
    assert package == tmp_path / "map"
    assert len(calls) == 2
    assert calls[0][0] is False
    assert calls[0][2] == 2
    assert calls[1][0] is True
    assert calls[1][1]["visual_review"] is True
    assert calls[1][2] == 0
    events = [json.loads(line) for line in
              (tmp_path / "jobs" / "clean-visual-review" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    visual_review = next(event for event in events
                         if event["kind"] == "visual_review_completed")
    assert visual_review["payload"]["changed"] is False


def test_rhythm_only_warning_with_no_safe_moves_skips_model_repair(tmp_path, monkeypatch):
    from beatforge.generation import GenerationResult
    from beatforge.quality import QualityReport
    from beatforge.script_runner import run_script
    from beatforge.worker import JobResult
    import beatforge.workflow as workflow_mod

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100 * 8, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, model_id="demo/model:free", plan=TrackPlan(120, 8, 1.0),
    )
    script = "map.note(beat=1, x=0, y=1, color=0, direction=1)"
    generation = GenerationResult(
        "demo/model:free", script, run_script(script, bpm=120), {"cost": 0},
    )
    calls = []

    def fake_run_generation_job(job_id, *_args, repair=False, **_kwargs):
        calls.append(repair)
        return JobResult(job_id, "completed", generation=generation)

    report = QualityReport(
        passed=True, errors=(),
        warnings=("rhythm_density_unresponsive: density varies poorly with activity",),
        note_count=1, chain_slices=0, actual_nps=0.125, occupied_sections=1,
        total_sections=1, direction_profiles={}, chain_opportunities={},
        chain_usage={}, pattern_alignment={}, palette_contrast={},
        rhythm_alignment={
            "warning": True,
            "repair_targets": [
                {"window": 0, "beats": [0, 8], "audio_activity": 0.1,
                 "strong_spectral_change_beats": []},
                {"window": 1, "beats": [8, 16], "audio_activity": 0.8,
                 "strong_spectral_change_beats": []},
            ],
        },
    )
    monkeypatch.setattr(workflow_mod, "run_generation_job", fake_run_generation_job)
    monkeypatch.setattr(workflow_mod, "preflight", lambda *_args, **_kwargs: report)

    job, package = generate_and_export(
        prepared, object(), job_id="unactionable-rhythm-only",
        jobs_root=tmp_path / "jobs", output=tmp_path / "map",
        metadata=ExportMetadata("Unactionable rhythm"), max_repairs=1,
    )

    assert calls == [False]
    assert package == tmp_path / "map"
    assert job.status == "completed_with_warnings"
    assert any("no safe nearby note move" in warning for warning in job.warnings)
    events = [json.loads(line) for line in
              (tmp_path / "jobs" / "unactionable-rhythm-only" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    skipped = next(event for event in events
                   if event["kind"] == "repair_skipped_unactionable_rhythm")
    assert skipped["payload"]["reason"] == "no_safe_nearby_note_moves"


def test_rhythm_warning_with_chain_warning_still_schedules_combined_repair(tmp_path, monkeypatch):
    from beatforge.generation import GenerationResult
    from beatforge.quality import QualityReport
    from beatforge.script_runner import run_script
    from beatforge.worker import JobResult
    import beatforge.workflow as workflow_mod

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100 * 8, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, model_id="demo/model:free", plan=TrackPlan(120, 8, 1.0),
    )
    script = "map.note(beat=1, x=0, y=1, color=0, direction=1)"
    generation = GenerationResult(
        "demo/model:free", script, run_script(script, bpm=120), {"cost": 0},
    )
    calls = []

    def fake_run_generation_job(job_id, *_args, repair=False, **_kwargs):
        calls.append(repair)
        return JobResult(job_id, "completed", generation=generation)

    report = QualityReport(
        passed=True, errors=(), warnings=(
            "rhythm_density_unresponsive: density varies poorly with activity",
            "native_chain_candidate_review: inspect the rapid same-hand run",
        ), note_count=1, chain_slices=0, actual_nps=0.125, occupied_sections=1,
        total_sections=1, direction_profiles={},
        chain_opportunities={"actionable_repair": True}, chain_usage={},
        pattern_alignment={}, palette_contrast={},
        rhythm_alignment={"warning": True, "repair_targets": []},
    )
    monkeypatch.setattr(workflow_mod, "run_generation_job", fake_run_generation_job)
    monkeypatch.setattr(workflow_mod, "preflight", lambda *_args, **_kwargs: report)

    job, package = generate_and_export(
        prepared, object(), job_id="combined-rhythm-chain",
        jobs_root=tmp_path / "jobs", output=tmp_path / "map",
        metadata=ExportMetadata("Combined review"), max_repairs=1,
    )

    assert calls == [False, True]
    assert package == tmp_path / "map"
    assert job.status == "completed_with_warnings"
    events = [json.loads(line) for line in
              (tmp_path / "jobs" / "combined-rhythm-chain" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert not any(event["kind"] == "repair_skipped_unactionable_rhythm"
                   for event in events)


def test_later_repair_receives_previous_accepted_map_for_visual_comparison(tmp_path, monkeypatch):
    from beatforge.generation import GenerationResult
    from beatforge.script_runner import run_script
    from beatforge.worker import JobResult
    import beatforge.workflow as workflow_mod
    from beatforge.quality import QualityReport

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, model_id="demo/model:free", plan=TrackPlan(120, 10, 1.0),
    )
    scripts = [
        "map.note(beat=1, x=0, y=1, color=0, direction=0)\nmap.light(beat=0, event_type=0, value=1)",
        "map.note(beat=1, x=1, y=1, color=0, direction=1)\nmap.light(beat=0, event_type=0, value=1)",
        "map.note(beat=1, x=2, y=1, color=0, direction=2)\nmap.light(beat=0, event_type=0, value=1)",
    ]
    generated = [GenerationResult(
        "demo/model:free", script,
        run_script(script, bpm=120), {
            "prompt_tokens": 10 * (index + 1),
            "completion_tokens": 2 * (index + 1),
            "total_tokens": 12 * (index + 1),
            "cost": 0, "is_byok": False,
        },
    ) for index, script in enumerate(scripts)]
    comparisons = []
    repair_index = 0

    def fake_run_generation_job(job_id, _request, _client, _store, _budget,
                                *, repair=False, comparison_difficulty=None, **_kwargs):
        nonlocal repair_index
        if not repair:
            return JobResult(job_id, "completed", generation=generated[0])
        comparisons.append(comparison_difficulty)
        repair_index += 1
        return JobResult(job_id, "completed", generation=generated[repair_index])

    preflight_calls = 0

    def fake_preflight(*_args, **_kwargs):
        nonlocal preflight_calls
        preflight_calls += 1
        warnings = ("review this candidate",) if preflight_calls <= 3 else ()
        return QualityReport(
            passed=True, errors=(), warnings=warnings, note_count=1,
            chain_slices=0, actual_nps=0.1, occupied_sections=1,
            total_sections=1, direction_profiles={}, chain_opportunities={},
            chain_usage={}, pattern_alignment={}, palette_contrast={}, rhythm_alignment={},
        )

    monkeypatch.setattr(workflow_mod, "run_generation_job", fake_run_generation_job)
    monkeypatch.setattr(workflow_mod, "preflight", fake_preflight)
    job, package = generate_and_export(
        prepared, object(), job_id="visual-revision-reference",
        jobs_root=tmp_path / "jobs", output=tmp_path / "map",
        metadata=ExportMetadata("Visual revision"), max_repairs=2,
    )
    assert job.status == "completed"
    assert package == tmp_path / "map"
    assert comparisons == [None, generated[0].difficulty]
    assert job.generation.usage["prompt_tokens"] == 60
    assert job.generation.usage["completion_tokens"] == 12
    assert job.generation.usage["total_tokens"] == 72
    assert job.generation.usage["cost"] == 0
    assert [attempt["phase"] for attempt in job.generation.usage["attempts"]] == [
        "initial_generation", "repair_1", "repair_2",
    ]
    live = json.loads((tmp_path / "jobs" / "visual-revision-reference" / "live.json")
                      .read_text(encoding="utf-8"))
    assert live["usage"] == job.generation.usage


def test_diagnostic_repair_budget_survives_automatic_script_repair(tmp_path):
    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100 * 4, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, model_id="demo/model:free",
        plan=TrackPlan(120, 4, 2.5),
    )
    notes = "\n".join(
        f"map.note(beat={index * 0.5}, x={index % 4}, y=0, "
        f"color={index % 2}, direction={(index % 4) + 1})"
        for index in range(10)
    )

    class ParseThenQualityRepairClient:
        def __init__(self):
            self.calls = 0

        def chat(self, model_id, _messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return ChatResult(
                    "map.note(beat=1, x=1, y=0, color=0, direction=1)\n"
                    "map.chain(beat=1, color=0, x=1, y=0, direction=1, "
                    "tail_beat=2, tail_x=2, tail_y=0, slices=3, squish=0.9)",
                    model_id, {"cost": 0}, {},
                )
            if self.calls == 2:
                return ChatResult(notes, model_id, {"cost": 0}, {})
            assert self.calls == 3
            return ChatResult(notes + "\nmap.light(beat=0, event_type=0, value=1)",
                              model_id, {"cost": 0}, {})

    client = ParseThenQualityRepairClient()
    budget = BudgetGuard(max_repairs=1, max_automatic_repairs=1)
    job, package = generate_and_export(
        prepared, client, job_id="separate-repair-budgets",
        jobs_root=tmp_path / "jobs", output=tmp_path / "map",
        metadata=ExportMetadata("Repair budgets"), budget=budget,
        max_repairs=1,
    )

    assert job.status == "completed"
    assert package == tmp_path / "map"
    assert client.calls == 3
    assert budget.requests == 3
    assert budget.automatic_repairs == 1
    assert budget.repairs == 1
    kinds = [json.loads(line)["kind"] for line in
             (tmp_path / "jobs" / "separate-repair-budgets" / "events.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    assert "generation_repair_completed" in kinds
    assert "repair_completed" in kinds
    assert kinds[-1] == "export_completed"


def test_failed_quality_repair_retains_preflight_passing_candidate(tmp_path, monkeypatch):
    import beatforge.workflow as workflow_mod
    from beatforge.quality import QualityReport

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100, dtype="float32"), 44100)
    prepared = prepare_generation(audio, bpm=120, target_nps=5.0,
                                  model_id="demo/model:free")

    class InitialThenEmptyRepairClient:
        def __init__(self):
            self.calls = 0

        def chat(self, model_id, _messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return _Client().chat(model_id, [])
            return ChatResult(None, model_id, {}, {})

    monkeypatch.setattr(
        workflow_mod, "preflight",
        lambda *_args, **_kwargs: QualityReport(
            passed=True, errors=(), warnings=("review this candidate",),
            note_count=1, chain_slices=0, actual_nps=1.0,
            occupied_sections=1, total_sections=1, direction_profiles={},
            chain_opportunities={}, chain_usage={}, pattern_alignment={}, palette_contrast={},
            rhythm_alignment={},
        ),
    )
    client = InitialThenEmptyRepairClient()
    job, package = generate_and_export(
        prepared, client, job_id="failed-repair-fallback",
        jobs_root=tmp_path / "jobs", output=tmp_path / "map",
        metadata=ExportMetadata("Fallback"),
        budget=BudgetGuard(max_repairs=1, max_automatic_repairs=0), max_repairs=1,
    )

    assert job.status == "completed_with_warnings"
    assert package == tmp_path / "map"
    assert client.calls == 2
    assert any("repair call failed" in warning for warning in job.warnings)
    events = [json.loads(line) for line in
              (tmp_path / "jobs" / "failed-repair-fallback" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert [event["kind"] for event in events][-2:] == [
        "preflight_warning", "export_completed",
    ]
    retained = next(event for event in events
                    if event["kind"] == "repair_fallback_retained")
    assert retained["payload"]["repair_error"] == "model response must be text"
    manifest = json.loads((package / "beatforge-manifest.json").read_text(encoding="utf-8"))
    assert manifest["preflight"]["workflow_warnings"]


def test_workflow_retries_repair_invariant_rejection_with_guard_feedback(tmp_path, monkeypatch):
    from beatforge import Section
    from beatforge.quality import QualityReport
    import beatforge.workflow as workflow_mod

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100 * 10, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, model_id="demo/model:free",
        plan=TrackPlan(120, 10, 0.1, sections=[Section(0, 20, "whole track")]),
    )

    one_hit = "map.note(beat=1, x=1, y=0, color=0, direction=1)"
    two_hits = one_hit + "\nmap.note(beat=2, x=2, y=0, color=1, direction=1)"

    class RejectThenCorrectClient:
        def __init__(self):
            self.calls = 0

        def chat(self, model_id, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return ChatResult(one_hit, model_id, {"cost": 0}, {})
            prompt = messages[0]["content"]
            if self.calls == 2:
                return ChatResult(two_hits, model_id, {"cost": 0}, {})
            assert "previous repair attempt was rejected" in prompt
            assert "from 1 to 2" in prompt
            assert 'extra note heads (up to 6): [{"beat":2' in prompt
            assert "This is a hard constraint" in prompt
            return ChatResult(one_hit, model_id, {"cost": 0}, {})

    monkeypatch.setattr(
        workflow_mod, "preflight",
        lambda *_args, **_kwargs: QualityReport(
            passed=True, errors=(), warnings=("review this candidate",),
            note_count=1, chain_slices=0, actual_nps=0.1,
            occupied_sections=1, total_sections=1, direction_profiles={},
            chain_opportunities={}, chain_usage={}, pattern_alignment={}, palette_contrast={},
            rhythm_alignment={},
        ),
    )
    client = RejectThenCorrectClient()
    job, package = generate_and_export(
        prepared, client, job_id="repair-invariant-retry", jobs_root=tmp_path / "jobs",
        output=tmp_path / "map", metadata=ExportMetadata("Repair retry"),
        budget=BudgetGuard(max_requests=3, max_repairs=2), max_repairs=2,
    )

    assert job.status == "completed_with_warnings"
    assert package == tmp_path / "map"
    assert client.calls == 3
    events = [json.loads(line) for line in
              (tmp_path / "jobs" / "repair-invariant-retry" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    retry = next(event for event in events if event["kind"] == "repair_retry_scheduled")
    assert retry["payload"]["attempt"] == 1
    assert "from 1 to 2" in retry["payload"]["repair_error"]
    assert sum(event["kind"] == "repair_started" for event in events) == 2
    exported = json.loads((package / "ExpertPlus.dat").read_text(encoding="utf-8"))
    assert len(exported["colorNotes"]) == 1


def test_count_guard_retry_carries_pattern_gain_from_rejected_candidate(tmp_path, monkeypatch):
    from beatforge import Section
    from beatforge.quality import QualityReport
    import beatforge.workflow as workflow_mod

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100 * 10, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, model_id="demo/model:free",
        plan=TrackPlan(120, 10, 0.1, sections=[Section(0, 20, "whole track")]),
    )
    initial = "map.note(beat=1, x=1, y=0, color=0, direction=1)"
    rejected = initial + "\nmap.note(beat=2, x=2, y=0, color=1, direction=1)"
    corrected = "map.note(beat=3, x=2, y=0, color=1, direction=1)"
    final = "map.note(beat=4, x=2, y=0, color=1, direction=1)"

    class CandidateThenCountCorrectedClient:
        def __init__(self):
            self.calls = 0

        def chat(self, model_id, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return ChatResult(initial, model_id, {"cost": 0}, {})
            if self.calls == 2:
                return ChatResult(rejected, model_id, {"cost": 0}, {})
            if self.calls == 3:
                prompt = messages[0]["content"]
                assert "from 1 to 2" in prompt
                assert "increased the active pattern-alignment support from 0.000 to 0.088" in prompt
                assert "increased occupied sections from 1/3 to 2/3" in prompt
                assert "do not simply restore the unchanged source map" in prompt
                return ChatResult(corrected, model_id, {"cost": 0}, {})
            assert self.calls == 4
            prompt = messages[0]["content"]
            assert "weak pattern alignment" in prompt
            assert "The previous repair attempt was rejected by BeatForge's deterministic" not in prompt
            return ChatResult(final, model_id, {"cost": 0}, {})

    def fake_preflight(program, *_args, **_kwargs):
        if len(program.notes) == 2:
            support = 0.088405
        elif program.notes[0].beat == 3:
            support = 0.1
        elif program.notes[0].beat == 4:
            support = 0.16
        else:
            support = 0.0
        alignment = {
            "available": True, "warning": support < 0.15,
            "chance_corrected_structure_support": support,
            "windows": 4, "repair_targets": [],
        }
        return QualityReport(
            passed=True, errors=(),
            warnings=("weak pattern alignment",) if support < 0.15 else (),
            note_count=len(program.notes), chain_slices=0, actual_nps=0.1,
            occupied_sections=(3 if support >= 0.15 else
                               1 if len(program.notes) == 1 else 2),
            total_sections=3, direction_profiles={},
            chain_opportunities={}, chain_usage={}, pattern_alignment=alignment,
            palette_contrast={}, rhythm_alignment={},
        )

    monkeypatch.setattr(workflow_mod, "preflight", fake_preflight)
    client = CandidateThenCountCorrectedClient()
    job, package = generate_and_export(
        prepared, client, job_id="count-retry-pattern-gain", jobs_root=tmp_path / "jobs",
        output=tmp_path / "map", metadata=ExportMetadata("Count retry pattern gain"),
        budget=BudgetGuard(max_requests=4, max_repairs=3), max_repairs=3,
    )

    assert job.status == "completed"
    assert package == tmp_path / "map"
    assert client.calls == 4
    events = [json.loads(line) for line in
              (tmp_path / "jobs" / "count-retry-pattern-gain" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    retry = next(event for event in events if event["kind"] == "repair_retry_scheduled")
    assert "from 1 to 2" in retry["payload"]["repair_error"]
    assert "support from 0.000 to 0.088" in retry["payload"]["retry_feedback"]
    assert "occupied sections from 1/3 to 2/3" in retry["payload"]["retry_feedback"]
    exported = json.loads((package / "ExpertPlus.dat").read_text(encoding="utf-8"))
    assert len(exported["colorNotes"]) == 1
    assert exported["colorNotes"][0]["b"] == 4


def test_repair_retry_requires_remaining_budget_capacity():
    from decimal import Decimal
    from beatforge.workflow import _repair_retry_capacity_available

    assert _repair_retry_capacity_available(BudgetGuard(max_requests=3, max_repairs=2))
    assert not _repair_retry_capacity_available(
        BudgetGuard(max_requests=2, max_repairs=2, requests=2))
    assert not _repair_retry_capacity_available(
        BudgetGuard(max_requests=3, max_repairs=1, repairs=1))
    exhausted_spend = BudgetGuard(max_requests=3, max_repairs=2,
                                  max_cost_usd=Decimal("0.01"))
    exhausted_spend.cost_usd = Decimal("0.02")
    assert not _repair_retry_capacity_available(exhausted_spend)


def test_pattern_alignment_regression_requires_active_warning_and_lower_score():
    from beatforge.workflow import _pattern_alignment_regression

    before = {"pattern_alignment": {
        "available": True, "warning": True,
        "chance_corrected_structure_support": 0.149257,
    }}
    lower = {"pattern_alignment": {
        "available": True, "warning": True,
        "chance_corrected_structure_support": 0.0,
    }}
    equal = {"pattern_alignment": {
        "available": True, "warning": True,
        "chance_corrected_structure_support": 0.149257,
    }}
    improved = {"pattern_alignment": {
        "available": True, "warning": False,
        "chance_corrected_structure_support": 0.16,
    }}

    assert _pattern_alignment_regression(before, lower) == (
        "repair reduced pattern-alignment support from 0.149 to 0.000 while the prior map "
        "had an active pattern warning; preserve the prior support or improve it"
    )
    assert _pattern_alignment_regression(before, equal) is None
    assert _pattern_alignment_regression(before, improved) is None
    no_warning = {"pattern_alignment": {**before["pattern_alignment"], "warning": False}}
    assert _pattern_alignment_regression(no_warning, lower) is None


def test_workflow_retries_pattern_alignment_regression_with_exact_feedback(tmp_path, monkeypatch):
    from beatforge import Section
    from beatforge.quality import QualityReport
    import beatforge.workflow as workflow_mod

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100 * 10, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, model_id="demo/model:free",
        plan=TrackPlan(120, 10, 0.1, sections=[Section(0, 20, "whole track")]),
    )
    scripts = {
        1: "map.note(beat=1, x=1, y=0, color=0, direction=1)",
        2: "map.note(beat=2, x=1, y=0, color=0, direction=1)",
        3: "map.note(beat=3, x=1, y=0, color=0, direction=1)",
    }

    class RegressThenImproveClient:
        def __init__(self):
            self.calls = 0

        def chat(self, model_id, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return ChatResult(scripts[1], model_id, {"cost": 0}, {})
            if self.calls == 2:
                return ChatResult(scripts[2], model_id, {"cost": 0}, {})
            assert self.calls == 3
            prompt = messages[0]["content"]
            assert "previous repair attempt was rejected" in prompt
            assert "support from 0.149 to 0.000" in prompt
            assert "This is a hard constraint" in prompt
            return ChatResult(scripts[3], model_id, {"cost": 0}, {})

    def fake_preflight(program, *_args, **_kwargs):
        beat = round(program.notes[0].beat)
        support = {1: 0.149257, 2: 0.0, 3: 0.16}[beat]
        warning = support < 0.15
        alignment = {
            "available": True, "warning": warning,
            "chance_corrected_structure_support": support,
            "windows": 4, "repair_targets": [],
        }
        return QualityReport(
            passed=True, errors=(),
            warnings=("weak pattern alignment",) if warning else (),
            note_count=1, chain_slices=0, actual_nps=0.1,
            occupied_sections=1, total_sections=1, direction_profiles={},
            chain_opportunities={}, chain_usage={}, pattern_alignment=alignment,
            palette_contrast={}, rhythm_alignment={},
        )

    monkeypatch.setattr(workflow_mod, "preflight", fake_preflight)
    client = RegressThenImproveClient()
    job, package = generate_and_export(
        prepared, client, job_id="pattern-regression-retry", jobs_root=tmp_path / "jobs",
        output=tmp_path / "map", metadata=ExportMetadata("Pattern retry"),
        budget=BudgetGuard(max_requests=3, max_repairs=2), max_repairs=2,
    )

    assert job.status == "completed"
    assert package == tmp_path / "map"
    assert client.calls == 3
    events = [json.loads(line) for line in
              (tmp_path / "jobs" / "pattern-regression-retry" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    rejected = next(event for event in events
                    if event["kind"] == "repair_rejected_pattern_regression")
    assert "from 0.149 to 0.000" in rejected["payload"]["repair_error"]
    retry = next(event for event in events if event["kind"] == "repair_retry_scheduled")
    assert retry["payload"]["reason"] == "pattern_alignment_regression"
    exported = json.loads((package / "ExpertPlus.dat").read_text(encoding="utf-8"))
    assert exported["colorNotes"][0]["b"] == 3


def test_workflow_retains_prior_map_when_pattern_regression_exhausts_repairs(tmp_path, monkeypatch):
    from beatforge import Section
    from beatforge.quality import QualityReport
    import beatforge.workflow as workflow_mod

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100 * 10, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, model_id="demo/model:free",
        plan=TrackPlan(120, 10, 0.1, sections=[Section(0, 20, "whole track")]),
    )
    initial = "map.note(beat=1, x=1, y=0, color=0, direction=1)"

    class RegressingClient:
        def __init__(self):
            self.calls = 0

        def chat(self, model_id, _messages, **_kwargs):
            self.calls += 1
            beat = 1 if self.calls == 1 else 2
            return ChatResult(
                f"map.note(beat={beat}, x=1, y=0, color=0, direction=1)",
                model_id, {"cost": 0}, {},
            )

    def fake_preflight(program, *_args, **_kwargs):
        beat = round(program.notes[0].beat)
        support = 0.149257 if beat == 1 else 0.0
        return QualityReport(
            passed=True, errors=(), warnings=("weak pattern alignment",),
            note_count=1, chain_slices=0, actual_nps=0.1,
            occupied_sections=1, total_sections=1, direction_profiles={},
            chain_opportunities={}, chain_usage={},
            pattern_alignment={
                "available": True, "warning": True,
                "chance_corrected_structure_support": support,
                "windows": 4, "repair_targets": [],
            },
            palette_contrast={}, rhythm_alignment={},
        )

    monkeypatch.setattr(workflow_mod, "preflight", fake_preflight)
    client = RegressingClient()
    job, package = generate_and_export(
        prepared, client, job_id="pattern-regression-retain", jobs_root=tmp_path / "jobs",
        output=tmp_path / "map", metadata=ExportMetadata("Pattern fallback"),
        budget=BudgetGuard(max_requests=2, max_repairs=1), max_repairs=1,
    )

    assert job.status == "completed_with_warnings"
    assert package == tmp_path / "map"
    assert client.calls == 2
    exported = json.loads((package / "ExpertPlus.dat").read_text(encoding="utf-8"))
    assert exported["colorNotes"][0]["b"] == 1
    events = [json.loads(line) for line in
              (tmp_path / "jobs" / "pattern-regression-retain" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert any(event["kind"] == "repair_rejected_pattern_regression" for event in events)
    fallback = next(event for event in events
                    if event["kind"] == "repair_fallback_retained")
    assert "worsened pattern alignment" in fallback["payload"]["warning"]


def test_prepare_generation_carries_track_metadata_into_request(tmp_path):
    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, bpm=120, target_nps=2.5, model_id="demo/model:free",
        track_title="Your Precious Moon", track_artist="Mike Morasky",
    )
    assert prepared.request.track_title == "Your Precious Moon"
    assert prepared.request.track_artist == "Mike Morasky"


def test_workflow_repairs_extreme_hand_direction_monotony(tmp_path):
    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100 * 30, dtype="float32"), 44100)
    prepared = prepare_generation(audio, bpm=120, target_nps=2.5,
                                  model_id="demo/model:free")

    def script(vary_directions):
        calls = []
        for index in range(30):
            direction = index % 4 if vary_directions else 1
            calls.append(
                f"map.note(beat={2 * index}, x={index % 4}, y={(index // 4) % 3}, "
                f"color=0, direction={direction})"
            )
        for index in range(45):
            calls.append(
                f"map.note(beat={0.5 + 1.3 * index:.1f}, x={(index + 2) % 4}, "
                f"y={(index // 3) % 3}, color=1, direction={index % 4})"
            )
        calls.extend(
            f"map.light(beat={beat}, event_type=0, value=1)"
            for beat in (0, 16, 32, 48)
        )
        return "\n".join(calls)

    class RepairingClient:
        def __init__(self):
            self.calls = 0

        def chat(self, model_id, messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return ChatResult(script(False), model_id, {"cost": 0}, {})
            prompt = messages[0]["content"]
            assert "severe cut-direction monotony" in prompt
            assert "30/30 playable heads" in prompt
            assert "direction counts {'1': 30}" in prompt
            return ChatResult(script(True), model_id, {"cost": 0}, {})

    client = RepairingClient()
    job, package = generate_and_export(
        prepared, client, job_id="repair-direction-monotony",
        jobs_root=tmp_path / "jobs", output=tmp_path / "map",
        metadata=ExportMetadata("Direction repair"),
        budget=BudgetGuard(max_repairs=1), max_repairs=1,
    )

    assert job.status == "completed"
    assert package == tmp_path / "map"
    assert client.calls == 2
    repaired = json.loads((package / "ExpertPlus.dat").read_text(encoding="utf-8"))
    assert len(repaired["colorNotes"]) == 75
    assert len({note["d"] for note in repaired["colorNotes"] if note["c"] == 0}) == 4


def test_prepare_generation_packs_spectrograms_when_provider_image_cap_is_set(tmp_path):
    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100 * 8, dtype="float32"), 44100)
    prepared = prepare_generation(audio, bpm=120, target_nps=2.5,
                                  model_id="gemma", provider="local",
                                  send_spectrograms=True,
                                  spectrogram_image_limit=2)
    assert len(prepared.request.spectrogram_images) == 2
    assert len(prepared.request.spectrogram_source_images) == 3
    assert prepared.request.send_spectrograms is True
    assert len(prepared.request.spectrogram_manifest["images"]) == 3
    assert prepared.request.spectrogram_manifest["model_input"] == {
        "image_limit": 2,
        "image_count": 2,
        "source_panel_count": 3,
        "packing": "overview plus vertical stacks of consecutive panels",
        "groups": [["overview.png"], ["segment_000.png", "segment_001.png"]],
    }
    import hashlib
    assert [hashlib.sha256(image).hexdigest()
            for image in prepared.request.spectrogram_source_images] == [
                row["sha256"] for row in prepared.request.spectrogram_manifest["images"]
            ]


def test_workflow_packages_assets_imported_during_tool_calling(tmp_path):
    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100, dtype="float32"), 44100)
    prepared = prepare_generation(audio, bpm=120, target_nps=5.0,
                                  model_id="demo/model:free",
                                  capabilities=CapabilityProfile(vivify=True))
    asset = tmp_path / "portal.zip"
    asset.write_bytes(b"asset-bundle")
    library = AssetLibrary(tmp_path / "jobs" / "vivify-assets.json")
    record = AssetRecord("portal-model", "Portal model",
                         "https://polyhaven.com/a/portal_model", str(asset), "CC0")

    class ToolState:
        def __init__(self):
            self.imported_asset_ids = []

        def __call__(self, name, arguments):
            assert name == "import_polyhaven_asset"
            assert arguments == {"candidate_id": "a" * 20}
            library.add(record)
            self.imported_asset_ids.append("portal-model")
            return {"asset_id": "portal-model", "status": "imported"}

    class ToolClient:
        def __init__(self):
            self.calls = 0

        def chat(self, model_id, _messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                call = {"id": "import-1", "type": "function", "function": {
                    "name": "import_polyhaven_asset",
                    "arguments": '{"candidate_id":"' + ("a" * 20) + '"}',
                }}
                return ChatResult("", model_id, {"prompt_tokens": 1}, {}, (call,))
            return ChatResult(
                "map.note(0.25, 1, 0, 0, 1)\n"
                "map.arc(0.5, 0, 0, 0, 1, 1, 1, 0, 0, 0)\n"
                "map.chain(1, 1, 2, 0, 1, 2, 3, 0, 2, 0.9)\n"
                "map.light(0.25, 0, 1)\n"
                "map.custom_event(beat=0, event_type='SetRenderingSettings', "
                "data={'renderSettings': {'fog': 1, 'fogColor': [0.1,0.05,0.2,1]}})\n"
                "map.custom_event(beat=1.5, event_type='InstantiatePrefab', "
                "data={'asset_id':'portal-model', 'id':'portal', 'position':[0,0,12]})",
                model_id, {"completion_tokens": 2}, {})

    job, package = generate_and_export(
        prepared, ToolClient(), job_id="vivify-tool", jobs_root=tmp_path / "jobs",
        output=tmp_path / "map", metadata=ExportMetadata("Demo"),
        budget=BudgetGuard(), asset_library=library, tool_executor=ToolState())
    assert job.status == "completed"
    assert (package / "VivifyAssets" / "portal-model.zip").read_bytes() == b"asset-bundle"
    manifest = json.loads((package / "beatforge-manifest.json").read_text())
    assert manifest["vivify_assets"]["assets"][0]["asset_id"] == "portal-model"
    difficulty = json.loads((package / "ExpertPlus.dat").read_text(encoding="utf-8"))
    prefab = next(event for event in difficulty["customData"]["_customEvents"]
                  if event["t"] == "InstantiatePrefab")
    assert prefab["d"]["asset_id"] == "portal-model"


def test_workflow_builds_and_attaches_requested_vivify_bundle(tmp_path, monkeypatch):
    import hashlib
    from beatforge import verify_package
    import beatforge.workflow as workflow_mod

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100, dtype="float32"), 44100)
    asset_id = "0123456789abcdefabcd"
    source = tmp_path / "portal.zip"
    source.write_bytes(b"source-zip")
    library = AssetLibrary(tmp_path / "assets.json")
    library.add(AssetRecord(asset_id, "Portal prop", "https://polyhaven.com/a/portal",
                            str(source), "CC0"))
    prepared = prepare_generation(audio, bpm=120, target_nps=5.0,
                                  model_id="demo/model:free",
                                  capabilities=CapabilityProfile(vivify=True),
                                  asset_ids=(asset_id,))

    class VivifyClient:
        def chat(self, model_id, _messages, **_kwargs):
            script = (
                "map.note(0.25, 1, 0, 0, 1)\n"
                "map.arc(0.5, 0, 0, 0, 1, 1, 1, 0, 0, 0)\n"
                "map.chain(1, 1, 2, 0, 1, 2, 3, 0, 2, 0.9)\n"
                "map.light(0.25, 0, 1)\n"
                "map.custom_event(beat=0, event_type='SetRenderingSettings', "
                "data={'renderSettings': {'ambientLight': [0.1,0.08,0.16,1]}})\n"
                "map.custom_event(beat=0, event_type='InstantiatePrefab', "
                f"data={{'asset_id':'{asset_id}', 'id':'portal', 'position':[0,0,12]}})"
            )
            return ChatResult(script, model_id, {"cost": 0}, {})

    def fake_build(models, *, workspace, **_kwargs):
        assert [model.asset_id for model in models] == [asset_id]
        bundle_dir = Path(workspace) / "bundle-output"
        bundle_dir.mkdir(parents=True)
        bundle = bundle_dir / "bundleWindows2021.vivify"
        bundle.write_bytes(b"mock-real-build-output")
        return {
            "bundle_file": bundle.name, "bundle_crc": 777,
            "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            "build_version": "Windows2021", "unity_version": "2021.3.16f1",
            "template_commit": "b37a3ba719f2e40edcd63cc63dc6ee61e8c5bd3",
            "assets": [{"asset_id": asset_id,
                        "vivify_asset_path": f"assets/{asset_id}/{asset_id}.prefab"}],
        }

    monkeypatch.setattr(workflow_mod, "build_vivify_bundles", fake_build)
    output = tmp_path / "map"
    job, package = generate_and_export(
        prepared, VivifyClient(), job_id="vivify-runtime", jobs_root=tmp_path / "jobs",
        output=output, metadata=ExportMetadata("Vivify runtime"),
        asset_library=library, vivify_bundle_config={
            "unity_editor": Path("Unity.exe"), "template_source": tmp_path / "template",
        })
    assert job.status == "completed"
    assert package == output
    assert (output / "bundleWindows2021.vivify").read_bytes() == b"mock-real-build-output"
    info = json.loads((output / "Info.dat").read_text())
    assert info["_customData"]["_assetBundle"]["_windows2021"] == 777
    difficulty = json.loads((output / "ExpertPlus.dat").read_text())
    event = next(item for item in difficulty["customData"]["_customEvents"]
                 if item["t"] == "InstantiatePrefab")
    assert event["d"]["asset"] == f"assets/{asset_id}/{asset_id}.prefab"
    assert verify_package(output).passed


def test_workflow_rejects_repair_that_changes_locked_objects(tmp_path):
    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100, dtype="float32"), 44100)
    from beatforge import Section, TrackPlan
    prepared = prepare_generation(
        audio, model_id="demo/model:free",
        plan=TrackPlan(120, 1, 2.5, sections=[Section(0, 8, "locked", locked=True)]))

    class ChangingClient:
        def __init__(self):
            self.calls = 0

        def chat(self, model_id, messages, **_kwargs):
            self.calls += 1
            beat = 1 if self.calls == 1 else 2
            return ChatResult(f"map.note({beat}, 1, 0, 0, 1)", model_id, {"cost": 0}, {})

    job, package = generate_and_export(
        prepared, ChangingClient(), job_id="locked", jobs_root=tmp_path / "jobs",
        output=tmp_path / "map", metadata=ExportMetadata("Demo"),
        budget=BudgetGuard(max_repairs=1), max_repairs=1)
    assert job.status == "failed"
    assert package is None
    assert "locked section" in job.error
    assert "repair_rejected_locked" in (tmp_path / "jobs" / "locked" / "events.jsonl").read_text()


def test_workflow_rejects_empty_generation_before_export(tmp_path):
    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100, dtype="float32"), 44100)
    prepared = prepare_generation(audio, bpm=120, target_nps=2.5,
                                  model_id="demo/model:free")

    class EmptyClient:
        def chat(self, model_id, _messages, **_kwargs):
            return ChatResult("", model_id, {"cost": 0}, {})

    output = tmp_path / "map"
    job, package = generate_and_export(
        prepared, EmptyClient(), job_id="empty", jobs_root=tmp_path / "jobs",
        output=output, metadata=ExportMetadata("Empty"),
        budget=BudgetGuard(), max_repairs=0)
    assert job.status == "failed"
    assert package is None
    assert "preflight remains unresolved" in job.error
    assert "no authored playable object" in job.error
    assert not output.exists()
    job_root = tmp_path / "jobs" / "empty"
    events = [json.loads(line) for line in (job_root / "events.jsonl").read_text().splitlines()]
    rejected = next(event for event in events if event["kind"] == "preflight_rejected")
    assert rejected["payload"]["report"]["warnings"]
    assessed = [event["payload"] for event in events if event["kind"] == "preflight_assessed"]
    assert assessed[-1]["stage"] == "final"
    assert assessed[-1]["script_sha256"]
    live = json.loads((job_root / "live.json").read_text())
    assert live["status"] == "failed"
    assert live["failure_stage"] == "preflight"
    assert live["error_type"] == "QualityGateError"
    assert "no authored playable object" in live["error"]
    assert "script" in live and "usage" in live
    assert live["failure_details"]["warnings"]


def test_identical_repair_and_preflight_stops_remaining_calls(tmp_path, monkeypatch):
    from beatforge import Section
    from beatforge.quality import QualityReport
    import beatforge.workflow as workflow_mod

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100 * 10, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, model_id="demo/model:free",
        plan=TrackPlan(120, 10, 0.1, sections=[Section(0, 20, "whole track")]),
    )
    unchanged = "map.note(beat=1, x=1, y=0, color=0, direction=1)"

    class RepeatingRepairClient:
        def __init__(self):
            self.calls = 0

        def chat(self, model_id, _messages, **_kwargs):
            self.calls += 1
            return ChatResult(unchanged, model_id, {"cost": 0}, {})

    def fake_preflight(*_args, **_kwargs):
        return QualityReport(
            passed=True, errors=(), warnings=("weak pattern alignment",),
            note_count=1, chain_slices=0, actual_nps=0.1,
            occupied_sections=1, total_sections=1, direction_profiles={},
            chain_opportunities={}, chain_usage={},
            pattern_alignment={
                "available": True, "warning": True,
                "chance_corrected_structure_support": 0.0,
                "windows": 4, "repair_targets": [],
            },
            palette_contrast={}, rhythm_alignment={},
        )

    monkeypatch.setattr(workflow_mod, "preflight", fake_preflight)
    client = RepeatingRepairClient()
    budget = BudgetGuard(max_requests=4, max_repairs=3)
    job, package = generate_and_export(
        prepared, client, job_id="identical-repair", jobs_root=tmp_path / "jobs",
        output=tmp_path / "map", metadata=ExportMetadata("Identical repair"),
        budget=budget, max_repairs=3,
    )

    assert job.status == "completed_with_warnings"
    assert package == tmp_path / "map"
    assert client.calls == 2
    assert budget.repairs == 1
    exported = json.loads((package / "ExpertPlus.dat").read_text(encoding="utf-8"))
    assert exported["colorNotes"][0]["b"] == 1
    events = [json.loads(line) for line in
              (tmp_path / "jobs" / "identical-repair" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    stopped = next(event for event in events if event["kind"] == "repair_no_progress")
    assert stopped["payload"]["reason"] == "unchanged_script_and_preflight"
    assert stopped["payload"]["retained_script_sha256"] == stopped["payload"]["candidate_script_sha256"]
    assert "unchanged preflight" in stopped["payload"]["warning"]
    assert "weak pattern alignment" in job.warnings


def test_persistent_palette_warning_schedules_palette_only_retry_after_no_progress(
        tmp_path, monkeypatch):
    from beatforge.generation import GenerationResult
    from beatforge.script_runner import run_script
    from beatforge.worker import JobResult
    import beatforge.workflow as workflow_mod

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100 * 10, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, model_id="demo/model:free",
        plan=TrackPlan(120, 10, 1.0),
        capabilities=CapabilityProfile(chroma=True),
    )
    notes = "\n".join(
        f"map.note(beat={1 + index * 2}, x={index % 4}, y=0, "
        f"color={index % 2}, direction={(index % 4) + 1})"
        for index in range(10)
    )
    original_script = (
        notes + "\n"
        "map.palette(environment_left=[20,20,90], environment_right=[90,20,20])\n"
        "map.light(beat=0, event_type=0, value=1)"
    )
    fixed_script = original_script.replace(
        "environment_left=[20,20,90], environment_right=[90,20,20]",
        "environment_left=[35,125,245], environment_right=[245,115,25]",
    )
    generated = [
        GenerationResult("demo/model:free", script,
                         run_script(script, bpm=120), {"cost": 0, "is_byok": False})
        for script in (original_script, original_script, fixed_script)
    ]
    repair_calls = []

    def fake_run_generation_job(job_id, _request, _client, _store, _budget,
                                *, repair=False, diagnostics=None, **_kwargs):
        if not repair:
            return JobResult(job_id, "completed", generation=generated[0])
        repair_calls.append(diagnostics)
        if len(repair_calls) == 2:
            return JobResult(
                job_id, "failed",
                error=("palette-only review did not clear low-contrast Chroma warning "
                       "(normalized RGB distance 0.077; minimum 0.300)"),
            )
        return JobResult(
            job_id, "completed",
            generation=generated[1] if len(repair_calls) == 1 else generated[2],
        )

    monkeypatch.setattr(workflow_mod, "run_generation_job", fake_run_generation_job)
    job, package = generate_and_export(
        prepared, object(), job_id="palette-secondary-repair",
        jobs_root=tmp_path / "jobs", output=tmp_path / "map",
        metadata=ExportMetadata("Palette retry"),
        budget=BudgetGuard(max_requests=8, max_repairs=3), max_repairs=3,
    )

    assert job.status == "completed"
    assert package == tmp_path / "map"
    assert len(repair_calls) == 3
    assert not repair_calls[0].get("scene_palette_review")
    assert repair_calls[1]["scene_palette_review"] == {
        "requested": True,
        "reason": "low_contrast_warning_persisted_after_general_repair",
    }
    assert repair_calls[2]["scene_palette_review"] == repair_calls[1]["scene_palette_review"]
    assert repair_calls[2]["repair_rejection_feedback"].startswith(
        "palette-only review did not clear low-contrast Chroma warning"
    )
    live = json.loads((tmp_path / "jobs" / "palette-secondary-repair" / "live.json")
                      .read_text(encoding="utf-8"))
    report = live["preflight"]
    assert not any("scene_palette_low_contrast" in warning
                   for warning in report["warnings"])
    exported = json.loads((package / "ExpertPlus.dat").read_text(encoding="utf-8"))
    assert exported["colorNotes"] == generated[0].difficulty["colorNotes"]
    assert exported["basicBeatmapEvents"] == generated[0].difficulty["basicBeatmapEvents"]
    events = [json.loads(line) for line in
              (tmp_path / "jobs" / "palette-secondary-repair" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    retries = [event["payload"]["reason"] for event in events
               if event["kind"] == "repair_retry_scheduled"]
    assert retries == ["persistent_low_contrast_palette_only_review",
                       "deterministic_repair_invariant_rejection"]


def test_script_validation_rejection_retries_with_exact_feedback(tmp_path, monkeypatch):
    from beatforge.generation import GenerationResult
    from beatforge.quality import QualityReport
    from beatforge.script_runner import run_script
    from beatforge.worker import JobResult
    import beatforge.workflow as workflow_mod

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100 * 10, dtype="float32"), 44100)
    prepared = prepare_generation(
        audio, model_id="demo/model:free", plan=TrackPlan(120, 10, 1.0),
    )
    original_script = "\n".join(
        f"map.note(beat={beat}, x={beat % 4}, y=0, color={beat % 2}, direction=1)"
        for beat in range(10)
    )
    fixed_script = original_script.replace("direction=1", "direction=2", 1)
    generated = [
        GenerationResult("demo/model:free", script, run_script(script, bpm=120),
                         {"cost": 0, "is_byok": False})
        for script in (original_script, fixed_script)
    ]
    rejection = (
        "repair script rejected by parser or capability validation: "
        "'(' was never closed (<unknown>, line 222)"
    )
    repair_calls = []

    def fake_run_generation_job(job_id, _request, _client, _store, _budget,
                                *, repair=False, diagnostics=None, **_kwargs):
        if not repair:
            return JobResult(job_id, "completed", generation=generated[0])
        repair_calls.append(diagnostics)
        if len(repair_calls) == 1:
            return JobResult(job_id, "failed", error=rejection)
        return JobResult(job_id, "completed", generation=generated[1])

    def fake_preflight(*_args, **_kwargs):
        return QualityReport(
            passed=True, errors=(), warnings=("rhythm-density review remains",),
            note_count=10, chain_slices=0, actual_nps=1.0,
            occupied_sections=1, total_sections=1, direction_profiles={},
            chain_opportunities={}, chain_usage={}, pattern_alignment={},
            palette_contrast={}, rhythm_alignment={},
        )

    monkeypatch.setattr(workflow_mod, "run_generation_job", fake_run_generation_job)
    monkeypatch.setattr(workflow_mod, "preflight", fake_preflight)
    budget = BudgetGuard(max_requests=6, max_repairs=2)
    job, package = generate_and_export(
        prepared, object(), job_id="script-validation-retry",
        jobs_root=tmp_path / "jobs", output=tmp_path / "map",
        metadata=ExportMetadata("Validation retry"), budget=budget, max_repairs=2,
    )

    assert job.status == "completed_with_warnings"
    assert package == tmp_path / "map"
    assert len(repair_calls) == 2
    assert repair_calls[1]["repair_rejection_feedback"] == rejection
    exported = json.loads((package / "ExpertPlus.dat").read_text(encoding="utf-8"))
    assert exported["colorNotes"][0]["d"] == 2
    events = [json.loads(line) for line in
              (tmp_path / "jobs" / "script-validation-retry" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    retry = next(event["payload"] for event in events
                 if event["kind"] == "repair_retry_scheduled")
    assert retry["reason"] == "script_validation_rejection"
    assert retry["retry_feedback"] == rejection
    import hashlib
    assert retry["retained_script_sha256"] == hashlib.sha256(
        original_script.encode("utf-8")).hexdigest()


def test_compact_patch_rejection_is_retryable_script_validation():
    from beatforge.workflow import _is_repair_invariant_rejection, _repair_rejection_reason

    error = "repair patch rejected by BeatForge validator: old note is ambiguous"
    assert _is_repair_invariant_rejection(error)
    assert _repair_rejection_reason(error) == "script_validation_rejection"

