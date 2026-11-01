"""Deterministic V3 package export with an explicit provenance manifest."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile
from typing import Any

from .capabilities import CapabilityProfile, validate_capabilities
from .compiler import compile_program
from .contracts import MapProgram
from .planning import TrackPlan, validate_plan
from .assets import AssetLibrary
from .vivify import (build_vivify_manifest, validate_asset_references,
                     validate_vivify_scene_events)
from .runtime import build_runtime_handoff


@dataclass(frozen=True, slots=True)
class ExportMetadata:
    song_name: str
    song_author: str = ""
    mapper: str = "BeatForge"
    song_sub_name: str = ""
    environment: str = "DefaultEnvironment"
    audio_filename: str = "song.ogg"
    cover_filename: str = "cover.png"
    difficulty: str = "ExpertPlus"
    difficulty_rank: int = 9
    njs: float = 14.0
    jump_offset: float = 0.0


def _prepare_vivify_runtime_bundle(difficulty: dict[str, Any], runtime_bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate a real Unity build and resolve model asset IDs to bundle prefab paths."""
    if not isinstance(runtime_bundle, dict):
        raise ValueError("Vivify runtime bundle descriptor must be an object")
    bundle_path = Path(runtime_bundle.get("bundle_path", "")).resolve()
    manifest = runtime_bundle.get("manifest")
    if not bundle_path.is_file() or not isinstance(manifest, dict):
        raise ValueError("Vivify runtime bundle and build manifest are required")
    bundle_file = manifest.get("bundle_file")
    if bundle_file != "bundleWindows2021.vivify" or bundle_path.name != bundle_file:
        raise ValueError("Vivify runtime bundle must be the Windows2021 bundle")
    if manifest.get("build_version") != "Windows2021":
        raise ValueError("Vivify runtime manifest has an unsupported build target")
    crc = manifest.get("bundle_crc")
    if not isinstance(crc, int) or isinstance(crc, bool) or not 0 <= crc <= 0xFFFFFFFF:
        raise ValueError("Vivify runtime manifest has an invalid Windows2021 CRC")
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    if manifest.get("bundle_sha256") != digest:
        raise ValueError("Vivify runtime bundle hash does not match its build manifest")
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("Vivify runtime manifest has no built assets")
    prefab_by_id: dict[str, str] = {}
    for item in raw_assets:
        if not isinstance(item, dict):
            raise ValueError("Vivify runtime asset record must be an object")
        asset_id = item.get("asset_id")
        prefab = item.get("vivify_asset_path")
        expected_prefab = f"assets/{asset_id}/{asset_id}.prefab" if isinstance(asset_id, str) else None
        if (not isinstance(asset_id, str) or len(asset_id) != 20
                or any(char not in "0123456789abcdef" for char in asset_id)
                or not isinstance(prefab, str) or prefab != expected_prefab):
            raise ValueError("Vivify runtime asset has an invalid ID or prefab path")
        if asset_id in prefab_by_id or prefab != prefab.lower():
            raise ValueError("Vivify runtime asset IDs must be unique and prefab paths lowercase")
        prefab_by_id[asset_id] = prefab

    custom_data = difficulty.get("customData")
    events = custom_data.get("_customEvents") if isinstance(custom_data, dict) else None
    if not isinstance(events, list):
        raise ValueError("Vivify runtime bundle requested but the map has no custom events")
    instantiated = 0
    for event in events:
        if not isinstance(event, dict) or event.get("t") != "InstantiatePrefab":
            continue
        data = event.get("d")
        if not isinstance(data, dict):
            raise ValueError("InstantiatePrefab event data must be an object")
        asset_id = data.pop("asset_id", None)
        if asset_id is not None:
            prefab = prefab_by_id.get(asset_id)
            if prefab is None:
                raise ValueError(f"InstantiatePrefab references an asset absent from the built bundle: {asset_id}")
            if "asset" in data and data["asset"] != prefab:
                raise ValueError("InstantiatePrefab asset path conflicts with its selected asset ID")
            data["asset"] = prefab
        elif data.get("asset") not in prefab_by_id.values():
            raise ValueError("runtime InstantiatePrefab must reference a prefab built from a selected asset ID")
        instantiated += 1
    if not instantiated:
        raise ValueError("Vivify runtime bundle requested but the map instantiates no selected asset")
    return {
        "bundle_file": bundle_file,
        "bundle_sha256": digest,
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_crc": crc,
        "checksum_key": "_windows2021",
        "build_version": "Windows2021",
        "unity_version": manifest.get("unity_version"),
        "template_commit": manifest.get("template_commit"),
        "assets": [{"asset_id": asset_id, "prefab_path": prefab}
                   for asset_id, prefab in sorted(prefab_by_id.items())],
        "_source_path": str(bundle_path),
    }


def _palette_custom_data(palette: dict[str, list[int]] | None) -> dict[str, Any]:
    if not palette:
        return {}
    allowed = {"environment_left", "environment_right", "saber_left", "saber_right"}
    if set(palette) - allowed or not {"environment_left", "environment_right"}.issubset(palette):
        raise ValueError("scene palette requires environment_left/right and optional saber_left/right")
    if ("saber_left" in palette) != ("saber_right" in palette):
        raise ValueError("saber palette colors must be supplied as a left/right pair")
    output = {}
    keys = {
        "environment_left": "_envColorLeft",
        "environment_right": "_envColorRight",
        "saber_left": "_colorLeft",
        "saber_right": "_colorRight",
    }
    for source, target in keys.items():
        if source not in palette:
            continue
        color = palette[source]
        if (not isinstance(color, (list, tuple)) or len(color) != 3
                or any(not isinstance(channel, int) or isinstance(channel, bool)
                       or not 0 <= channel <= 255 for channel in color)):
            raise ValueError(f"{source} must be an RGB triple of integers from 0 to 255")
        output[target] = {"r": color[0] / 255, "g": color[1] / 255,
                          "b": color[2] / 255}
    return output


def build_info(metadata: ExportMetadata, plan: TrackPlan, capabilities: CapabilityProfile,
               *, scene_palette: dict[str, list[int]] | None = None,
               vivify_runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    palette = scene_palette if scene_palette is not None else plan.scene_palette
    if palette and not capabilities.chroma:
        raise ValueError("scene palette requires the Chroma capability")
    difficulty_custom_data = {
        "_requirements": [name for name, enabled in (
            ("Chroma", capabilities.chroma),
            ("Noodle Extensions", capabilities.noodle),
            ("Mapping Extensions", capabilities.mapping_extensions),
            ("Vivify", capabilities.vivify),
        ) if enabled],
        **_palette_custom_data(palette),
    }
    info = {
        "version": "2.2.0",
        "songName": metadata.song_name,
        "songSubName": metadata.song_sub_name,
        "songAuthorName": metadata.song_author,
        "levelAuthorName": metadata.mapper,
        "beatsPerMinute": plan.bpm,
        "songFilename": metadata.audio_filename,
        "coverImageFilename": metadata.cover_filename,
        "environmentName": metadata.environment,
        "difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{
                "_difficulty": metadata.difficulty,
                "_difficultyRank": metadata.difficulty_rank,
                "_beatmapFilename": f"{metadata.difficulty}.dat",
                "_noteJumpMovementSpeed": metadata.njs,
                "_noteJumpStartBeatOffset": metadata.jump_offset,
                "_customData": difficulty_custom_data,
            }],
        }],
    }
    if vivify_runtime is not None:
        info["_customData"] = {"_assetBundle": {
            "_windows2021": vivify_runtime["bundle_crc"],
        }}
    return info


def export_v3(output: str | Path, program: MapProgram, metadata: ExportMetadata,
              plan: TrackPlan, capabilities: CapabilityProfile, *,
              audio_path: str | Path | None = None, cover_path: str | Path | None = None,
              zip_path: str | Path | None = None,
              asset_library: AssetLibrary | None = None,
              asset_ids: tuple[str, ...] = (),
              vivify_runtime_bundle: dict[str, Any] | None = None,
              preflight_report: dict[str, Any] | None = None) -> Path:
    errors = validate_plan(plan) + validate_capabilities(capabilities)
    errors.extend(validate_asset_references(program, asset_ids))
    if errors:
        raise ValueError("; ".join(errors))
    difficulty = compile_program(program, mapping_extensions=capabilities.mapping_extensions)
    if capabilities.vivify:
        max_beat = plan.duration_seconds * plan.bpm / 60.0
        custom_data = difficulty.get("customData") or {}
        errors = validate_vivify_scene_events(
            custom_data.get("_customEvents", []) if isinstance(custom_data, dict) else [],
            max_beat=max_beat,
        )
        if errors:
            raise ValueError("; ".join(errors))
    if vivify_runtime_bundle is not None and not capabilities.vivify:
        raise ValueError("Vivify runtime bundle requires capabilities.vivify")
    runtime_manifest = (_prepare_vivify_runtime_bundle(difficulty, vivify_runtime_bundle)
                        if vivify_runtime_bundle is not None else None)
    authored_palette = program.scene_palette
    if (plan.scene_palette and authored_palette
            and plan.scene_palette != authored_palette):
        raise ValueError("model-authored scene palette conflicts with the supplied plan")
    effective_palette = plan.scene_palette or authored_palette
    if effective_palette and not capabilities.chroma:
        raise ValueError("scene palette requires the Chroma capability")
    difficulty_custom_data = difficulty.get("customData")
    if isinstance(difficulty_custom_data, dict):
        difficulty_custom_data.pop("_beatforgeScenePalette", None)
    root = Path(output)
    info = build_info(metadata, plan, capabilities, scene_palette=effective_palette,
                      vivify_runtime=runtime_manifest)
    root.mkdir(parents=True, exist_ok=True)
    (root / "Info.dat").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / f"{metadata.difficulty}.dat").write_text(
        json.dumps(difficulty, ensure_ascii=False, indent=2), encoding="utf-8")
    if audio_path is not None:
        source = Path(audio_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, root / metadata.audio_filename)
    if cover_path is not None:
        source = Path(cover_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, root / metadata.cover_filename)
    if runtime_manifest is not None:
        shutil.copyfile(runtime_manifest["_source_path"], root / runtime_manifest["bundle_file"])
    manifest = {"format": "BeatForge-V3", "metadata": asdict(metadata),
                "plan": plan.to_dict(), "capabilities": capabilities.as_dict(),
                "scene_palette": effective_palette}
    if preflight_report is not None:
        if not isinstance(preflight_report, dict):
            raise ValueError("preflight report must be an object")
        errors = preflight_report.get("errors", [])
        warnings = preflight_report.get("warnings", [])
        if (not isinstance(errors, (list, tuple))
                or not isinstance(warnings, (list, tuple))):
            raise ValueError("preflight errors and warnings must be sequences")
        errors = list(errors)
        warnings = list(warnings)
        report_passed = bool(preflight_report.get("passed"))
        manifest["preflight"] = {
            "status": "failed" if not report_passed else "needs_review" if warnings else "passed",
            "passed": report_passed,
            "actual_nps": preflight_report.get("actual_nps"),
            "errors": errors,
            "warnings": warnings,
        }
        workflow_warnings = preflight_report.get("workflow_warnings", [])
        if not isinstance(workflow_warnings, (list, tuple)):
            raise ValueError("workflow warnings must be a sequence")
        if workflow_warnings:
            manifest["preflight"]["workflow_warnings"] = list(workflow_warnings)
    if asset_ids:
        if not capabilities.vivify:
            raise ValueError("Vivify asset bundle requires capabilities.vivify")
        if asset_library is None:
            raise ValueError("asset_library is required when asset_ids are supplied")
        vivify = build_vivify_manifest(asset_library, asset_ids, root / "VivifyAssets",
                                       max_asset_mb=capabilities.max_asset_mb)
        manifest["vivify_assets"] = vivify
    if runtime_manifest is not None:
        runtime_manifest.pop("_source_path", None)
        manifest["vivify_runtime"] = runtime_manifest
    (root / "beatforge-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    build_runtime_handoff(root)
    if zip_path is None:
        return root
    archive = Path(zip_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(root).as_posix())
    return archive
