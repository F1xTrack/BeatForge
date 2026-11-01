import json
from pathlib import Path

import beatforge.cli as cli_mod
import pytest
from beatforge.cli import _parser, main


def test_cli_compiles_script_and_renders_preview(tmp_path):
    script = tmp_path / "map.bf.py"
    script.write_text("map.note(1, 1, 0, 0, 1)\n", encoding="utf-8")
    difficulty = tmp_path / "ExpertPlus.dat"
    assert main(["compile-script", str(script), "--bpm", "120", "--out", str(difficulty)]) == 0
    assert json.loads(difficulty.read_text())["colorNotes"][0]["b"] == 1
    preview = tmp_path / "preview.html"
    assert main(["preview", str(difficulty), "--out", str(preview)]) == 0
    assert "canvas" in preview.read_text(encoding="utf-8")


def test_cli_parser_exposes_free_only_generation_command():
    args = _parser().parse_args([
        "generate", "--audio", "song.wav", "--bpm", "120", "--target-nps", "2.5",
        "--model", "demo/model:free", "--out", "build", "--song-name", "Demo",
        "--jobs", "jobs", "--job-id", "demo-1", "--repairs", "3",
        "--max-requests", "16",
    ])
    assert args.command == "generate"
    assert args.model.endswith(":free")
    assert args.provider == "openrouter"
    assert args.reasoning_effort is None
    assert args.spectrogram_image_limit is None
    assert args.dry_run is False
    assert args.repairs == 3
    assert args.max_requests == 16


def test_cli_defaults_to_three_bounded_repairs():
    args = _parser().parse_args([
        "generate", "--audio", "song.wav", "--bpm", "120", "--target-nps", "2.5",
        "--model", "demo/model:free", "--out", "build", "--song-name", "Demo",
    ])
    assert args.repairs == 3
    assert args.max_requests == 14


def test_cli_parser_accepts_xhigh_reasoning_effort():
    args = _parser().parse_args([
        "generate", "--audio", "song.wav", "--model", "demo/model:free",
        "--out", "build", "--song-name", "Demo", "--reasoning-effort", "xhigh",
    ])
    assert args.reasoning_effort == "xhigh"


def test_cli_parser_accepts_provider_timeout():
    args = _parser().parse_args([
        "generate", "--audio", "song.wav", "--model", "demo/model:free",
        "--out", "build", "--song-name", "Demo", "--timeout-seconds", "900",
    ])
    assert args.timeout_seconds == 900


def test_cli_parser_rejects_token_cap_options():
    parser = _parser()
    for option in ("--max-tokens", "--reasoning-max-tokens"):
        with pytest.raises(SystemExit):
            parser.parse_args([
                "generate", "--audio", "song.wav", "--model", "demo/model:free",
                "--out", "build", "--song-name", "Demo", option, "2048",
            ])


def test_cli_parser_accepts_spectrogram_image_limit():
    args = _parser().parse_args([
        "generate", "--audio", "song.wav", "--model", "demo/model:free",
        "--out", "build", "--song-name", "Demo", "--spectrogram-image-limit", "12",
    ])
    assert args.spectrogram_image_limit == 12


def test_cli_parser_accepts_vivify_asset_selection():
    args = _parser().parse_args([
        "generate", "--audio", "song.wav", "--model", "demo/model:free",
        "--out", "build", "--song-name", "Demo", "--vivify",
        "--asset-library", "catalog.json", "--asset-id", "portal",
        "--asset-id", "light",
    ])
    assert args.asset_library == Path("catalog.json")
    assert args.asset_id == ["portal", "light"]


def test_cli_parser_exposes_optional_real_vivify_bundle_build():
    args = _parser().parse_args([
        "generate", "--audio", "song.wav", "--model", "demo/model:free",
        "--out", "build", "--song-name", "Demo", "--vivify",
        "--vivify-build-bundles", "--unity-editor", "Unity.exe",
        "--vivify-template-source", "VivifyTemplate", "--vivify-build-timeout", "1200",
    ])
    assert args.vivify_build_bundles is True
    assert args.unity_editor == Path("Unity.exe")
    assert args.vivify_template_source == Path("VivifyTemplate")
    assert args.vivify_build_timeout == 1200


def test_cli_parser_accepts_isolated_vivify_browser_search():
    args = _parser().parse_args([
        "generate", "--audio", "song.wav", "--model", "demo/model:free",
        "--out", "build", "--song-name", "Demo", "--vivify",
        "--vivify-browser", "--anti-debug-extension", "extension",
    ])
    assert args.vivify_browser is True
    assert args.anti_debug_extension == Path("extension")


def test_cli_parser_accepts_local_provider():
    args = _parser().parse_args([
        "generate", "--provider", "local", "--endpoint", "http://127.0.0.1:8080/v1",
        "--audio", "song.wav", "--model", "gemma", "--out", "build",
        "--song-name", "Demo",
    ])
    assert args.provider == "local"
    assert args.endpoint.endswith("/v1")


def test_cli_parser_exposes_local_spectrogram_opt_in():
    args = _parser().parse_args([
        "generate", "--provider", "local", "--audio", "song.wav", "--model", "gemma",
        "--out", "build", "--song-name", "Demo", "--send-spectrograms",
    ])
    assert args.send_spectrograms is True


def test_cli_parser_exposes_local_server_launcher():
    args = _parser().parse_args([
        "serve-local", "--binary", "llama-server.exe", "--model", "model.gguf",
        "--mmproj", "mmproj.gguf", "--port", "8123", "--ctx-size", "4096", "--threads", "4",
    ])
    assert args.command == "serve-local"
    assert args.port == 8123
    assert args.ctx_size == 4096
    assert args.threads == 4
    assert args.mmproj == Path("mmproj.gguf")


def test_cli_serve_local_starts_loopback_process(tmp_path, monkeypatch, capsys):
    binary = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    binary.write_bytes(b"binary")
    model.write_bytes(b"model")

    class FakeProcess:
        pid = 77

    monkeypatch.setattr(cli_mod, "launch_local_server", lambda _spec: FakeProcess())
    assert main(["serve-local", "--binary", str(binary), "--model", str(model)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "started"
    assert output["pid"] == 77
    assert output["endpoint"] == "http://127.0.0.1:8080/v1"


def test_cli_probe_reports_local_models(monkeypatch, capsys):
    class FakeLocal:
        def __init__(self, **_kwargs):
            pass

        def catalog(self):
            return [{"id": "gemma"}]

    monkeypatch.setattr(cli_mod, "LocalOpenAIClient", FakeLocal)
    assert main(["probe", "--provider", "local"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ok"
    assert output["models"][0]["id"] == "gemma"


def test_cli_status_reads_live_job_snapshot(tmp_path, capsys):
    from beatforge import JobStore

    store = JobStore(tmp_path / "jobs")
    store.write_live("job-1", "provider_running", {"model_id": "demo/model:free"})
    store.append("job-1", "generation_started", {"model_id": "demo/model:free"})
    assert main(["status", "job-1", "--jobs", str(tmp_path / "jobs")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["live"]["status"] == "provider_running"
    assert output["latest"]["kind"] == "generation_started"


def test_cli_generate_dry_run_stops_before_provider(tmp_path):
    import numpy as np
    import soundfile as sf

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100, dtype="float32"), 44100)
    plan = tmp_path / "plan.json"
    assert main([
        "generate", "--audio", str(audio), "--bpm", "120", "--target-nps", "2.5",
        "--model", "demo/model:free", "--out", str(tmp_path / "map"),
        "--plan-out", str(plan), "--song-name", "Demo", "--dry-run",
    ]) == 0
    assert plan.exists()


def test_cli_passes_song_metadata_into_generation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from beatforge import TrackPlan

    captured = {}

    def prepare_generation(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(plan=TrackPlan(120, 32, 2.5), request=None)

    monkeypatch.setattr(cli_mod, "prepare_generation", prepare_generation)
    assert main([
        "generate", "--audio", str(tmp_path / "opaque.ogg"),
        "--model", "demo/model:free", "--out", str(tmp_path / "map"),
        "--song-name", "Your Precious Moon", "--song-author", "Mike Morasky",
        "--dry-run",
    ]) == 0
    assert captured["track_title"] == "Your Precious Moon"
    assert captured["track_artist"] == "Mike Morasky"


def test_cli_generate_can_use_reviewed_plan_without_recomputing(tmp_path):
    import numpy as np
    import soundfile as sf
    from beatforge import Section, TrackPlan

    audio = tmp_path / "song.wav"
    sf.write(audio, np.zeros(44100, dtype="float32"), 44100)
    plan = tmp_path / "reviewed.json"
    plan.write_text(TrackPlan(120, 32, 2.5,
                              sections=[Section(0, 16, "locked-intro", locked=True)]).to_json(),
                    encoding="utf-8")
    assert main([
        "generate", "--audio", str(audio), "--plan-in", str(plan),
        "--model", "demo/model:free", "--out", str(tmp_path / "map"),
        "--song-name", "Demo", "--dry-run",
    ]) == 0
