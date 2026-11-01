import json
from pathlib import Path
import subprocess
import zipfile

import pytest

import beatforge.vivify_bundle as bundle_mod
from beatforge.cli import _parser
from beatforge.vivify_bundle import (
    VivifyBundleBuildError,
    VivifyModelInput,
    build_vivify_bundles,
    initialize_unity_project,
    stage_model_archive,
)


ASSET_ID = "a909e5bf3712ad110131"


def _make_archive(path: Path, entries=None):
    entries = entries or {
        "model/model.fbx": b"fbx source placeholder",
        "model/textures/color.png": b"texture bytes",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


def _make_template(path: Path):
    packages = path / "Packages"
    exporter = packages / "com.swifter.vivify-template.exporter"
    exporter.mkdir(parents=True)
    (exporter / "package.json").write_text(json.dumps({
        "name": "com.swifter.vivify-template.exporter",
        "version": "1.0.0-preview.3",
    }), encoding="utf-8")
    (packages / "manifest.json").write_text(json.dumps({"dependencies": {}}), encoding="utf-8")
    return path


def test_stage_model_archive_extracts_only_safe_source_and_returns_prefab_path(tmp_path):
    project = tmp_path / "unity"
    (project / "Assets").mkdir(parents=True)
    archive = _make_archive(tmp_path / "model.zip")

    result = stage_model_archive(VivifyModelInput(ASSET_ID, archive), project)

    assert (project / result["source_model_path"]).read_bytes() == b"fbx source placeholder"
    assert result["prefab_asset_path"] == f"Assets/assets/{ASSET_ID}/{ASSET_ID}.prefab"
    assert result["vivify_asset_path"] == f"assets/{ASSET_ID}/{ASSET_ID}.prefab"
    assert len(result["source_archive_sha256"]) == 64


@pytest.mark.parametrize("member", ["../escape.fbx", "C:/escape.fbx", "folder/../../escape.fbx"])
def test_stage_model_archive_rejects_traversal_paths(tmp_path, member):
    project = tmp_path / "unity"
    (project / "Assets").mkdir(parents=True)
    archive = _make_archive(tmp_path / "bad.zip", {member: b"fbx"})

    with pytest.raises(VivifyBundleBuildError, match="unsafe"):
        stage_model_archive(VivifyModelInput(ASSET_ID, archive), project)
    assert not (tmp_path / "escape.fbx").exists()


def test_stage_model_archive_rejects_multiple_models_and_unsupported_files(tmp_path):
    project = tmp_path / "unity"
    (project / "Assets").mkdir(parents=True)
    multiple = _make_archive(tmp_path / "multiple.zip", {
        "one.fbx": b"one", "two.obj": b"two",
    })
    with pytest.raises(VivifyBundleBuildError, match="exactly one"):
        stage_model_archive(VivifyModelInput(ASSET_ID, multiple), project)

    unsupported_project = tmp_path / "unity-unsupported"
    (unsupported_project / "Assets").mkdir(parents=True)
    unsupported = _make_archive(tmp_path / "unsupported.zip", {"model.exe": b"no"})
    with pytest.raises(VivifyBundleBuildError, match="unsupported"):
        stage_model_archive(VivifyModelInput(ASSET_ID, unsupported), unsupported_project)


def test_initialize_unity_project_embeds_pinned_exporter_without_mutating_source(tmp_path):
    template = _make_template(tmp_path / "template")
    project = initialize_unity_project(template, tmp_path / "unity-project")

    manifest = json.loads((project / "Packages" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"]["com.swifter.vivify-template.exporter"] == \
        "file:Packages/com.swifter.vivify-template.exporter"
    assert (project / "Packages" / "com.swifter.vivify-template.exporter" / "package.json").is_file()
    assert not (project / "Packages" / "packages-lock.json").exists()
    assert (project / "ProjectSettings" / "ProjectVersion.txt").read_text().strip() == \
        "m_EditorVersion: 2021.3.16f1"
    assert not (template / "ProjectSettings").exists()


def test_bundle_build_requires_real_unity_executable_before_creating_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    with pytest.raises(FileNotFoundError, match="Unity Editor executable"):
        build_vivify_bundles(
            [VivifyModelInput(ASSET_ID, tmp_path / "source.zip")],
            unity_editor=tmp_path / "missing-Unity.exe",
            template_source=tmp_path / "missing-template",
            workspace=workspace,
        )
    assert not workspace.exists()


def test_zero_exit_without_unity_bundle_marker_is_not_reported_as_success(tmp_path, monkeypatch):
    template = _make_template(tmp_path / "template")
    archive = _make_archive(tmp_path / "source.zip")
    editor = tmp_path / "Unity.exe"
    editor.write_bytes(b"test executable placeholder")
    calls = []

    def fake_unity(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(bundle_mod.subprocess, "run", fake_unity)
    monkeypatch.setattr(bundle_mod, "_verify_template_revision", lambda _source: None)
    workspace = tmp_path / "workspace"
    with pytest.raises(VivifyBundleBuildError, match="without its build-result marker"):
        build_vivify_bundles(
            [VivifyModelInput(ASSET_ID, archive)], unity_editor=editor,
            template_source=template, workspace=workspace,
        )

    command, kwargs = calls[0]
    assert "-batchmode" in command
    assert "-executeMethod" in command
    assert "BeatForgeVivifyBundleBuild.Build" in command
    assert kwargs["cwd"] == workspace / "unity-project"
    assert str(workspace / "Unity.log") in command
    assert (workspace / "unity-project" / "Assets" / "Editor" /
            "BeatForgeVivifyBundleBuild.cs").is_file()
    assert (workspace / "unity-project" / "Assets" / "BeatForgeImported" /
            ASSET_ID / "source" / "model" / "model.fbx").is_file()


def test_cli_parses_explicit_unity_build_inputs():
    args = _parser().parse_args([
        "vivify-build", "--asset", f"{ASSET_ID}=model.zip",
        "--workspace", ".tmp-vivify-build", "--unity-editor", "Unity.exe",
        "--template-source", "VivifyTemplate",
    ])
    assert args.command == "vivify-build"
    assert args.asset == [f"{ASSET_ID}=model.zip"]
    assert args.timeout_seconds == 900
