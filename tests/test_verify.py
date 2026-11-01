import json
import zipfile

from beatforge import (Arc, CapabilityProfile, CustomEvent, ExportMetadata, MapProgram, Note,
                       Section, TrackPlan, verify_package, export_v3)


def _export(tmp_path):
    folder = tmp_path / "map"
    archive = tmp_path / "map.zip"
    export_v3(folder, MapProgram(120, notes=[Note(1, 1, 0, 0, 1)]),
              ExportMetadata("Demo"), TrackPlan(120, 10, 2.5,
              sections=[Section(0, 8, "intro")]), CapabilityProfile(),
              zip_path=archive)
    return folder, archive


def test_verify_accepts_directory_and_zip(tmp_path):
    folder, archive = _export(tmp_path)
    assert verify_package(folder).passed is True
    report = verify_package(archive)
    assert report.passed is True
    assert report.object_counts["notes"] == 1


def test_verify_counts_both_arc_endpoint_notes_in_exported_v3(tmp_path):
    folder = tmp_path / "arc-map"
    export_v3(folder, MapProgram(120, arcs=[Arc(
        1, 0, 0, 0, 1, 2, 2, 1, 0, tail_direction=6)]),
        ExportMetadata("Arc demo"), TrackPlan(120, 10, 1.0), CapabilityProfile())

    report = verify_package(folder)

    difficulty = json.loads((folder / "ExpertPlus.dat").read_text())
    assert report.passed is True
    assert len(difficulty["colorNotes"]) == 2
    assert len(difficulty["sliders"]) == 1
    assert report.object_counts["notes"] == 2
    assert report.object_counts["arcs"] == 1


def test_verify_rejects_corrupt_difficulty(tmp_path):
    folder, _archive = _export(tmp_path)
    data = json.loads((folder / "ExpertPlus.dat").read_text())
    data["colorNotes"][0]["x"] = 99
    (folder / "ExpertPlus.dat").write_text(json.dumps(data))
    report = verify_package(folder)
    assert report.passed is False
    assert any("invalid difficulty" in error for error in report.errors)


def test_verify_rejects_zip_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    report = verify_package(archive)
    assert report.passed is False
    assert any("unsafe" in error for error in report.errors)


def test_verify_roundtrips_custom_mod_event(tmp_path):
    folder = tmp_path / "custom-map"
    export_v3(folder, MapProgram(120, notes=[Note(1, 1, 0, 0, 1)],
                                 custom_events=[CustomEvent(2, "AnimateTrack", {"track": "portal"})]),
              ExportMetadata("Demo"), TrackPlan(120, 10, 2.5), CapabilityProfile())
    difficulty = json.loads((folder / "ExpertPlus.dat").read_text())
    assert difficulty["customData"]["_customEvents"][0]["t"] == "AnimateTrack"
    assert verify_package(folder).passed is True


def test_verify_rejects_custom_event_missing_vivify_asset(tmp_path):
    folder = tmp_path / "broken-custom-map"
    export_v3(folder, MapProgram(120, notes=[Note(1, 1, 0, 0, 1)]),
              ExportMetadata("Demo"), TrackPlan(120, 10, 2.5), CapabilityProfile())
    difficulty_path = folder / "ExpertPlus.dat"
    difficulty = json.loads(difficulty_path.read_text())
    difficulty.setdefault("customData", {})["_customEvents"] = [
        {"b": 2, "t": "AnimateTrack", "d": {"asset_id": "missing"}},
    ]
    difficulty_path.write_text(json.dumps(difficulty))
    report = verify_package(folder)
    assert report.passed is False
    assert any("missing Vivify asset" in error for error in report.errors)


def test_verify_rejects_vivify_requirement_without_scene_event(tmp_path):
    folder = tmp_path / "vivify-without-effect"
    export_v3(folder, MapProgram(120, notes=[Note(1, 1, 0, 0, 1)], custom_events=[
        CustomEvent(0, "SetRenderingSettings", {
            "renderSettings": {"fog": 1, "fogColor": [0.1, 0.05, 0.2, 1]},
        }),
    ]), ExportMetadata("Demo"), TrackPlan(120, 10, 2.5),
              CapabilityProfile(vivify=True))
    difficulty_path = folder / "ExpertPlus.dat"
    difficulty = json.loads(difficulty_path.read_text())
    difficulty["customData"]["_customEvents"] = []
    difficulty_path.write_text(json.dumps(difficulty))
    report = verify_package(folder)
    assert report.passed is False
    assert any("SetRenderingSettings" in error for error in report.errors)

