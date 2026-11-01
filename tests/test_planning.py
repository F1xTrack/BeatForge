import json

import pytest

from beatforge import (BeatGrid, JobStore, Section, TrackPlan,
                       free_catalog, initial_plan_from_beat_grid,
                       parse_free_model, select_free_model, validate_plan)


def test_plan_is_serializable_and_validated():
    plan = TrackPlan(120, 30, 2.5, sections=[Section(0, 16, "intro")])
    assert validate_plan(plan) == []
    assert json.loads(plan.to_json())["sections"][0]["label"] == "intro"


def test_plan_round_trips_and_preserves_lock(tmp_path):
    from beatforge import load_plan

    plan = TrackPlan(120, 32, 2.5, sections=[Section(0, 16, "intro", locked=True)])
    path = tmp_path / "plan.json"
    path.write_text(plan.to_json(), encoding="utf-8")
    loaded = load_plan(path)
    assert loaded.to_dict() == plan.to_dict()
    assert loaded.sections[0].locked is True


def test_provider_policy_accepts_only_zero_price_free_models():
    entry = {"id": "demo/model:free", "pricing": {"prompt": "0", "completion": "0"},
             "architecture": {"input_modalities": ["text", "image", "audio"]},
             "supported_parameters": ["tools", "reasoning"]}
    model = parse_free_model(entry)
    assert model.supports_tools and model.supports_vision and model.supports_audio and model.supports_reasoning
    with pytest.raises(ValueError):
        parse_free_model({**entry, "id": "demo/model", "pricing": {"prompt": "0.01", "completion": "0"}})
    assert len(free_catalog([entry, {"id": "paid/model", "pricing": {"prompt": "1", "completion": "1"}}])) == 1
    assert select_free_model([entry], require_vision=True).model_id == "demo/model:free"
    assert select_free_model([entry], require_audio=True).supports_audio


def test_provider_policy_accepts_official_zero_price_router():
    entry = {"id": "openrouter/free", "pricing": {"prompt": "0", "completion": "0"},
             "architecture": {"input_modalities": ["text", "image"]},
             "supported_parameters": ["tools", "reasoning"]}
    model = parse_free_model(entry)
    assert model.model_id == "openrouter/free"
    assert model.supports_vision and model.supports_tools and model.supports_reasoning
    with pytest.raises(ValueError, match="priced model"):
        parse_free_model({**entry, "pricing": {"prompt": "0", "completion": "0.01"}})


def test_provider_policy_ignores_token_budget_capability():
    entry = {"id": "demo/model:free", "pricing": {"prompt": "0", "completion": "0"},
             "supported_parameters": ["reasoning"],
             "reasoning": {"supports_max_tokens": True}}
    model = parse_free_model(entry)
    assert model.supports_reasoning is True
    assert not hasattr(model, "supports_reasoning_max_tokens")


def test_provider_selection_reports_missing_capability():
    with pytest.raises(LookupError, match="reasoning"):
        select_free_model([{"id": "demo/model:free", "pricing": {"prompt": 0, "completion": 0},
                           "architecture": {"input_modalities": ["text"]}, "supported_parameters": ["tools"]}],
                          require_reasoning=True)


def test_beatgrid_plan_uses_downbeat_boundaries(tmp_path):
    from beatforge import analyze_audio
    import numpy as np
    import soundfile as sf

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(8 * 44100, dtype="float32"), 44100)
    analysis = analyze_audio(audio)
    grid = BeatGrid(tuple(i * 0.5 for i in range(17)),
                    (0.0, 2.0, 4.0, 6.0), 120.0, "beat-this", 0.8)
    plan = initial_plan_from_beat_grid(analysis, grid, target_nps=2.5, bars_per_section=2)
    assert plan.sections[0].label == "grid-section-01"
    assert plan.notes.startswith("BeatGrid source=beat-this")


def test_auto_plan_uses_section_average_energy_and_preserves_global_density():
    from beatforge.analysis import EnergyWindow, SongAnalysis
    from beatforge.planning import initial_plan_from_analysis

    analysis = SongAnalysis(
        "synthetic", 8.0, 44100, 1,
        (EnergyWindow(0.0, 4.0, 0.1, 0.2),
         EnergyWindow(4.0, 8.0, 1.0, 1.0)),
    )
    plan = initial_plan_from_analysis(analysis, bpm=120, target_nps=4.0,
                                      section_beats=8.0)

    assert len(plan.sections) == 2
    assert plan.sections[0].energy < plan.sections[1].energy
    assert plan.sections[0].density < plan.sections[1].density
    duration_weighted_nps = sum(
        section.density * (section.end_beat - section.start_beat) * 60 / plan.bpm
        for section in plan.sections
    ) / plan.duration_seconds
    assert duration_weighted_nps == pytest.approx(plan.target_nps, abs=0.001)


def test_job_store_appends_events_and_reads_latest(tmp_path):
    store = JobStore(tmp_path)
    store.append("job-1", "analysis_started", {"audio": "song.ogg"})
    store.append("job-1", "plan_ready", {"sections": 3})
    latest = store.latest("job-1")
    assert latest is not None
    assert latest.kind == "plan_ready"
    assert sum(1 for _ in (tmp_path / "job-1" / "events.jsonl").open(encoding="utf-8")) == 2
