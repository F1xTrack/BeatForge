import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from beatforge import (CapabilityProfile, CustomEvent, ExportMetadata, MapProgram, Note,
                       Section, TrackPlan, AssetLibrary, AssetRecord, export_v3)


def test_v3_export_writes_info_difficulty_manifest_and_zip(tmp_path):
    program = MapProgram(120, notes=[Note(1, 1, 0, 0, 1)])
    plan = TrackPlan(120, 10, 2.5, sections=[Section(0, 8, "intro")])
    metadata = ExportMetadata("Demo")
    folder = tmp_path / "map"
    archive = tmp_path / "demo.zip"
    result = export_v3(folder, program, metadata, plan, CapabilityProfile(), zip_path=archive)
    assert result == Path(archive)
    assert (folder / "Info.dat").exists()
    assert (folder / "ExpertPlus.dat").exists()
    info = json.loads((folder / "Info.dat").read_text())
    assert info["difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]["_beatmapFilename"] == "ExpertPlus.dat"
    with zipfile.ZipFile(archive) as bundle:
        assert {"Info.dat", "ExpertPlus.dat", "beatforge-manifest.json",
                "runtime-handoff.json"} <= set(bundle.namelist())


def test_scene_palette_round_trips_to_chroma_info_custom_data(tmp_path):
    from beatforge import verify_package

    palette = {
        "environment_left": [150, 90, 210],
        "environment_right": [70, 180, 140],
        "saber_left": [40, 120, 255],
        "saber_right": [255, 140, 20],
    }
    program = MapProgram(120, notes=[Note(1, 1, 0, 0, 1)], scene_palette=palette)
    folder = tmp_path / "palette-map"
    export_v3(folder, program, ExportMetadata("Palette demo"),
              TrackPlan(120, 10, 2.5), CapabilityProfile(chroma=True))

    info = json.loads((folder / "Info.dat").read_text(encoding="utf-8"))
    profile = info["difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]
    assert profile["_customData"]["_envColorLeft"] == {
        "r": 150 / 255, "g": 90 / 255, "b": 210 / 255,
    }
    assert profile["_customData"]["_envColorRight"] == {
        "r": 70 / 255, "g": 180 / 255, "b": 140 / 255,
    }
    assert profile["_customData"]["_colorLeft"] == {
        "r": 40 / 255, "g": 120 / 255, "b": 1.0,
    }
    difficulty = json.loads((folder / "ExpertPlus.dat").read_text(encoding="utf-8"))
    assert "_beatforgeScenePalette" not in difficulty["customData"]
    assert json.loads((folder / "beatforge-manifest.json").read_text(encoding="utf-8"))["scene_palette"] == palette
    assert verify_package(folder).passed


def test_runtime_handoff_declares_native_features_and_mods(tmp_path):
    from beatforge import build_runtime_handoff

    folder = tmp_path / "map"
    export_v3(folder, MapProgram(120, notes=[Note(1, 1, 0, 0, 1)], custom_events=[
                  CustomEvent(0, "SetRenderingSettings", {
                      "renderSettings": {"fog": 1, "fogColor": [0.1, 0.05, 0.2, 1]},
                  }),
              ]),
              ExportMetadata("Demo"), TrackPlan(120, 10, 2.5),
              CapabilityProfile(chroma=True, vivify=True))
    handoff = build_runtime_handoff(folder, write=False)
    assert handoff["runtime_status"] == "handoff_only"
    assert handoff["native_features"] == {"arcs": True, "chains": True}
    assert "Chroma" in handoff["required_mods"]
    assert handoff["vivify"]["enabled"] is True
    assert json.loads((folder / "runtime-handoff.json").read_text())["format"] == "BeatForge-Runtime-Handoff-0.1"


def test_v3_export_includes_selected_vivify_assets(tmp_path):
    asset = tmp_path / "portal.glb"
    asset.write_bytes(b"portal")
    library = AssetLibrary(tmp_path / "assets.json")
    library.add(AssetRecord("portal", "Portal", "https://polyhaven.com/portal.glb",
                            str(asset), "CC0"))
    folder = tmp_path / "map"
    export_v3(folder, MapProgram(120, notes=[Note(1, 1, 0, 0, 1)], custom_events=[
                  CustomEvent(0, "SetRenderingSettings", {
                      "renderSettings": {"fog": 1, "fogColor": [0.1, 0.05, 0.2, 1]},
                  }),
              ]),
              ExportMetadata("Demo"), TrackPlan(120, 10, 2.5),
              CapabilityProfile(vivify=True), asset_library=library,
              asset_ids=("portal",))
    assert (folder / "VivifyAssets" / "portal.glb").exists()
    manifest = json.loads((folder / "beatforge-manifest.json").read_text())
    assert manifest["vivify_assets"]["assets"][0]["asset_id"] == "portal"


def test_vivify_asset_export_accepts_relative_package_paths(tmp_path, monkeypatch):
    from beatforge import verify_package

    asset = tmp_path / "portal.glb"
    asset.write_bytes(b"portal")
    library = AssetLibrary(tmp_path / "assets.json")
    library.add(AssetRecord("portal", "Portal", "https://polyhaven.com/portal.glb",
                            str(asset), "CC0"))
    monkeypatch.chdir(tmp_path)
    folder = "relative-map"
    archive = "relative-map.zip"
    result = export_v3(folder, MapProgram(120, notes=[Note(1, 1, 0, 0, 1)], custom_events=[
                           CustomEvent(0, "SetRenderingSettings", {
                               "renderSettings": {"fog": 1, "fogColor": [0.1, 0.05, 0.2, 1]},
                           }),
                       ]),
                       ExportMetadata("Demo"), TrackPlan(120, 10, 2.5),
                       CapabilityProfile(vivify=True), asset_library=library,
                       asset_ids=("portal",), zip_path=archive)
    assert result == Path(archive)
    assert (tmp_path / folder / "VivifyAssets" / "portal.glb").exists()
    assert verify_package(folder).passed
    with zipfile.ZipFile(archive) as bundle:
        assert "VivifyAssets/portal.glb" in bundle.namelist()


def test_export_rejects_unselected_custom_event_asset(tmp_path):
    with pytest.raises(ValueError, match="not selected"):
        export_v3(tmp_path / "map", MapProgram(
            120, custom_events=[CustomEvent(1, "AnimateTrack", {"asset_id": "missing"})]),
            ExportMetadata("Demo"), TrackPlan(120, 10, 2.5), CapabilityProfile())


def test_vivify_export_requires_documented_scene_event(tmp_path):
    with pytest.raises(ValueError, match="SetRenderingSettings"):
        export_v3(tmp_path / "map", MapProgram(120, notes=[Note(1, 1, 0, 0, 1)]),
                  ExportMetadata("Demo"), TrackPlan(120, 10, 2.5),
                  CapabilityProfile(vivify=True))


def test_vivify_runtime_export_resolves_prefab_and_records_crc(tmp_path):
    from beatforge import verify_package

    asset_id = "0123456789abcdefabcd"
    source = tmp_path / "portal.zip"
    source.write_bytes(b"source-archive")
    library = AssetLibrary(tmp_path / "assets.json")
    library.add(AssetRecord(asset_id, "Portal prop", "https://polyhaven.com/a/portal",
                            str(source), "CC0"))
    bundle_path = tmp_path / "bundleWindows2021.vivify"
    bundle_path.write_bytes(b"actual-unity-bundle-test-fixture")
    bundle_manifest = {
        "bundle_file": bundle_path.name,
        "bundle_crc": 321987654,
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "build_version": "Windows2021",
        "unity_version": "2021.3.16f1",
        "template_commit": "b37a3ba719f2e40edcd63cc63dc6ee61e8c5bd3",
        "assets": [{"asset_id": asset_id,
                    "vivify_asset_path": f"assets/{asset_id}/{asset_id}.prefab"}],
    }
    program = MapProgram(120, notes=[Note(1, 1, 0, 0, 1)], custom_events=[
        CustomEvent(0, "SetRenderingSettings", {
            "renderSettings": {"ambientLight": [0.1, 0.08, 0.16, 1]},
        }),
        CustomEvent(0, "InstantiatePrefab", {
            "asset_id": asset_id, "id": "portal", "position": [0, 0, 12],
        }),
    ])
    folder = tmp_path / "runtime-map"
    archive = tmp_path / "runtime-map.zip"
    export_v3(folder, program, ExportMetadata("Runtime demo"),
              TrackPlan(120, 10, 2.5), CapabilityProfile(vivify=True),
              asset_library=library, asset_ids=(asset_id,), zip_path=archive,
              vivify_runtime_bundle={"bundle_path": bundle_path, "manifest": bundle_manifest})

    info = json.loads((folder / "Info.dat").read_text(encoding="utf-8"))
    assert info["_customData"]["_assetBundle"]["_windows2021"] == 321987654
    difficulty = json.loads((folder / "ExpertPlus.dat").read_text(encoding="utf-8"))
    event = next(item for item in difficulty["customData"]["_customEvents"]
                 if item["t"] == "InstantiatePrefab")
    assert event["d"]["asset"] == f"assets/{asset_id}/{asset_id}.prefab"
    assert "asset_id" not in event["d"]
    assert (folder / bundle_path.name).read_bytes() == bundle_path.read_bytes()
    manifest = json.loads((folder / "beatforge-manifest.json").read_text(encoding="utf-8"))
    assert manifest["vivify_runtime"]["bundle_crc"] == 321987654
    assert json.loads((folder / "runtime-handoff.json").read_text())[
        "runtime_status"] == "bundle_built_not_runtime_tested"
    assert verify_package(folder).passed
    assert verify_package(archive).passed
    (folder / bundle_path.name).write_bytes(b"tampered-after-export")
    report = verify_package(folder)
    assert not report.passed
    assert "Vivify runtime bundle hash mismatch" in report.errors


def test_vivify_runtime_export_rejects_uninstantiated_bundle(tmp_path):
    bundle_path = tmp_path / "bundleWindows2021.vivify"
    bundle_path.write_bytes(b"bundle")
    asset_id = "0123456789abcdefabcd"
    descriptor = {"bundle_path": bundle_path, "manifest": {
        "bundle_file": bundle_path.name, "bundle_crc": 1,
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "build_version": "Windows2021", "assets": [{
            "asset_id": asset_id, "vivify_asset_path": f"assets/{asset_id}/{asset_id}.prefab"}],
    }}
    program = MapProgram(120, custom_events=[CustomEvent(0, "SetRenderingSettings", {
        "renderSettings": {"ambientLight": [0.1, 0.08, 0.16, 1]},
    })])
    with pytest.raises(ValueError, match="instantiates no selected asset"):
        export_v3(tmp_path / "map", program, ExportMetadata("No prefab"),
                  TrackPlan(120, 10, 2.5), CapabilityProfile(vivify=True),
                  vivify_runtime_bundle=descriptor)
