import json
import hashlib

import pytest

from beatforge import (AssetLibrary, AssetRecord, CustomEvent, MapProgram,
                       build_vivify_manifest, referenced_asset_ids,
                       validate_asset_references)
from beatforge.vivify import validate_vivify_scene_events


def test_vivify_bundle_copies_assets_and_writes_manifest(tmp_path):
    source = tmp_path / "portal.glb"
    source.write_bytes(b"portal")
    library = AssetLibrary(tmp_path / "catalog.json")
    record = library.add(AssetRecord("portal", "Portal", "https://polyhaven.com/portal.glb",
                                     str(source), "CC0", author="A"))
    root = tmp_path / "bundle"
    manifest = build_vivify_manifest(library, [record.asset_id], root)
    assert manifest["assets"][0]["sha256"] == hashlib.sha256(b"portal").hexdigest()
    assert (root / "portal.glb").read_bytes() == b"portal"
    assert json.loads((root / "vivify-assets.json").read_text())["format"].startswith("BeatForge-Vivify")


def test_vivify_bundle_detects_changed_catalogued_asset(tmp_path):
    source = tmp_path / "portal.glb"
    source.write_bytes(b"before")
    library = AssetLibrary(tmp_path / "catalog.json")
    record = library.add(AssetRecord("portal", "Portal", "https://polyhaven.com/portal.glb",
                                     str(source), "CC0"))
    source.write_bytes(b"after")
    with pytest.raises(ValueError, match="changed"):
        build_vivify_manifest(library, [record.asset_id], tmp_path / "bundle")


def test_vivify_bundle_respects_size_cap(tmp_path):
    source = tmp_path / "big.glb"
    source.write_bytes(b"x" * (1024 * 1024 + 1))
    library = AssetLibrary(tmp_path / "catalog.json")
    record = library.add(AssetRecord("big", "Big", "https://polyhaven.com/big.glb",
                                     str(source), "CC0"))
    with pytest.raises(ValueError, match="bundle limit"):
        build_vivify_manifest(library, [record.asset_id], tmp_path / "bundle", max_asset_mb=1)


def test_custom_event_asset_references_are_explicit_and_fail_closed():
    program = MapProgram(120, custom_events=[
        CustomEvent(1, "AnimateTrack", {"asset_id": "portal"}),
    ])
    assert referenced_asset_ids(program) == {"portal"}
    assert validate_asset_references(program, ("portal",)) == []
    assert "portal" in validate_asset_references(program, ())[0]


def test_vivify_scene_event_accepts_documented_render_and_quality_settings():
    events = [CustomEvent(0, "SetRenderingSettings", {
        "renderSettings": {
            "fog": 1, "fogColor": [0.08, 0.04, 0.16, 1],
            "fogMode": 2, "fogDensity": 0.004,
            "ambientLight": [0.10, 0.08, 0.16, 1],
        },
    })]
    assert validate_vivify_scene_events(events, max_beat=200) == []

    wrong_category = [CustomEvent(0, "SetRenderingSettings", {
        "qualitySettings": {"ambientLight": [0.10, 0.08, 0.16, 1]},
    })]
    assert any("unsupported settings" in error
               for error in validate_vivify_scene_events(wrong_category))


def test_vivify_scene_event_rejects_missing_unsupported_and_out_of_track_settings():
    assert "SetRenderingSettings" in validate_vivify_scene_events([])[0]
    unsupported = [CustomEvent(0, "SetRenderingSettings", {"bogusSettings": {"x": 1}})]
    assert "documented settings categories" in validate_vivify_scene_events(unsupported)[0]
    out_of_track = [CustomEvent(201, "SetRenderingSettings", {
        "renderSettings": {"fog": 1},
    })]
    assert "within the track" in validate_vivify_scene_events(out_of_track, max_beat=200)[0]

