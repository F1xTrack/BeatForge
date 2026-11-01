"""Structural verification for exported BeatForge packages."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile
from typing import Any

from .adapters import program_from_v3
from .compiler import compile_program
from .vivify import referenced_asset_ids, validate_vivify_scene_events


@dataclass(frozen=True, slots=True)
class VerificationReport:
    passed: bool
    errors: tuple[str, ...]
    files: tuple[str, ...] = ()
    object_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    return not (path.is_absolute() or ":" in normalized or ".." in path.parts)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _difficulty_refs(info: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for group in info.get("difficultyBeatmapSets", []) or []:
        for entry in group.get("_difficultyBeatmaps", group.get("difficultyBeatmaps", [])) or []:
            value = entry.get("_beatmapFilename", entry.get("beatmapFilename"))
            if isinstance(value, str):
                refs.append(value)
    return refs


def _verify_directory(root: Path) -> VerificationReport:
    errors: list[str] = []
    files: list[str] = []
    counts = {"notes": 0, "bombs": 0, "arcs": 0, "chains": 0, "walls": 0, "lights": 0}
    referenced_assets: set[str] = set()
    instantiated_prefabs: set[str] = set()
    unresolved_prefab_ids: set[str] = set()
    info_path = root / "Info.dat"
    if not info_path.is_file():
        return VerificationReport(False, ("Info.dat is missing",))
    files.append("Info.dat")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return VerificationReport(False, (f"Info.dat is unreadable: {exc}",), tuple(files))
    if not isinstance(info, dict):
        errors.append("Info.dat must be a JSON object")
        info = {}
    refs = _difficulty_refs(info)
    if not refs:
        errors.append("Info.dat has no difficulty references")
    bpm = info.get("beatsPerMinute", 0)
    for ref in refs:
        if not _safe_member(ref):
            errors.append(f"unsafe difficulty reference: {ref!r}")
            continue
        path = (root / Path(ref)).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f"missing difficulty file: {ref}")
            continue
        files.append(Path(ref).as_posix())
        try:
            difficulty = json.loads(path.read_text(encoding="utf-8-sig"))
            program = program_from_v3(difficulty, bpm=float(bpm))
            beatmap_entries = [entry for group in info.get("difficultyBeatmapSets", []) or []
                               for entry in group.get("_difficultyBeatmaps", []) or []]
            mapping_extensions = any(
                "Mapping Extensions" in ((entry.get("_customData") or {}).get("_requirements") or [])
                for entry in beatmap_entries if isinstance(entry, dict)
            )
            compile_program(program, mapping_extensions=mapping_extensions)
            vivify_required = any(
                "Vivify" in ((entry.get("_customData") or {}).get("_requirements") or [])
                for entry in beatmap_entries if isinstance(entry, dict)
            )
            if vivify_required:
                errors.extend(validate_vivify_scene_events(program.custom_events))
            referenced_assets.update(referenced_asset_ids(program))
            for event in program.custom_events:
                if event.event_type != "InstantiatePrefab":
                    continue
                if not isinstance(event.data, dict):
                    continue
                asset_id = event.data.get("asset_id")
                if isinstance(asset_id, str):
                    unresolved_prefab_ids.add(asset_id)
                prefab = event.data.get("asset")
                if isinstance(prefab, str):
                    instantiated_prefabs.add(prefab)
            counts["notes"] += len(program.notes) + 2 * len(program.arcs)
            counts["bombs"] += len(program.bombs)
            counts["arcs"] += len(program.arcs)
            counts["chains"] += len(program.chains)
            counts["walls"] += len(program.walls)
            counts["lights"] += len(program.lights)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            errors.append(f"invalid difficulty {ref}: {exc}")

    manifest_path = root / "beatforge-manifest.json"
    if not manifest_path.is_file():
        errors.append("beatforge-manifest.json is missing")
    else:
        files.append("beatforge-manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            vivify = manifest.get("vivify_assets", {}) if isinstance(manifest, dict) else {}
            runtime = manifest.get("vivify_runtime") if isinstance(manifest, dict) else None
            manifest_asset_ids: set[str] = set()
            for asset in vivify.get("assets", []) if isinstance(vivify, dict) else []:
                if isinstance(asset, dict) and isinstance(asset.get("asset_id"), str):
                    manifest_asset_ids.add(asset["asset_id"])
                package_path = asset.get("package_path")
                if not isinstance(package_path, str) or not _safe_member(package_path):
                    errors.append("Vivify asset has an unsafe package path")
                    continue
                path = (root / "VivifyAssets" / Path(package_path).name).resolve()
                if not path.is_file():
                    errors.append(f"missing Vivify asset: {package_path}")
                    continue
                if asset.get("sha256") != _sha256(path):
                    errors.append(f"Vivify asset hash mismatch: {package_path}")
            for asset_id in sorted(referenced_assets - manifest_asset_ids):
                errors.append(f"custom event references missing Vivify asset: {asset_id}")
            info_custom = info.get("_customData") if isinstance(info, dict) else None
            info_bundle = info_custom.get("_assetBundle") if isinstance(info_custom, dict) else None
            info_crc = info_bundle.get("_windows2021") if isinstance(info_bundle, dict) else None
            if runtime is not None:
                if not isinstance(runtime, dict):
                    errors.append("Vivify runtime manifest must be an object")
                else:
                    if unresolved_prefab_ids:
                        errors.append("Vivify InstantiatePrefab contains an unresolved asset_id placeholder")
                    bundle_name = runtime.get("bundle_file")
                    if (not isinstance(bundle_name, str) or bundle_name != "bundleWindows2021.vivify"
                            or not _safe_member(bundle_name) or Path(bundle_name).name != bundle_name):
                        errors.append("Vivify runtime manifest has an unsafe or unsupported bundle filename")
                    else:
                        bundle_path = root / bundle_name
                        if not bundle_path.is_file():
                            errors.append(f"Vivify runtime bundle is missing: {bundle_name}")
                        else:
                            if runtime.get("bundle_sha256") != _sha256(bundle_path):
                                errors.append("Vivify runtime bundle hash mismatch")
                            if runtime.get("bundle_size_bytes") != bundle_path.stat().st_size:
                                errors.append("Vivify runtime bundle size does not match its manifest")
                        if runtime.get("checksum_key") != "_windows2021":
                            errors.append("Vivify runtime manifest has an unexpected checksum key")
                        if runtime.get("build_version") != "Windows2021":
                            errors.append("Vivify runtime manifest has an unsupported build target")
                    crc = runtime.get("bundle_crc")
                    if (not isinstance(crc, int) or isinstance(crc, bool)
                            or not 0 <= crc <= 0xFFFFFFFF):
                        errors.append("Vivify runtime manifest has an invalid bundle CRC")
                    elif info_crc != crc:
                        errors.append("Info.dat Windows2021 checksum does not match the Vivify runtime manifest")
                    runtime_assets = runtime.get("assets")
                    prefab_paths: set[str] = set()
                    runtime_asset_ids: set[str] = set()
                    if not isinstance(runtime_assets, list) or not runtime_assets:
                        errors.append("Vivify runtime manifest has no asset records")
                    else:
                        for item in runtime_assets:
                            if not isinstance(item, dict):
                                errors.append("Vivify runtime asset record must be an object")
                                continue
                            asset_id = item.get("asset_id")
                            prefab = item.get("prefab_path")
                            expected = (f"assets/{asset_id}/{asset_id}.prefab"
                                        if isinstance(asset_id, str) else None)
                            if (not isinstance(asset_id, str) or len(asset_id) != 20
                                    or any(char not in "0123456789abcdef" for char in asset_id)
                                    or prefab != expected):
                                errors.append("Vivify runtime asset has an invalid ID or prefab path")
                                continue
                            if asset_id in runtime_asset_ids:
                                errors.append(f"duplicate Vivify runtime asset ID: {asset_id}")
                            runtime_asset_ids.add(asset_id)
                            prefab_paths.add(prefab)
                    for asset_id in sorted(runtime_asset_ids - manifest_asset_ids):
                        errors.append(f"Vivify runtime asset is missing source provenance: {asset_id}")
                    for prefab in sorted(instantiated_prefabs - prefab_paths):
                        errors.append(f"InstantiatePrefab references an asset absent from the Vivify bundle: {prefab}")
                    if not instantiated_prefabs:
                        errors.append("Vivify runtime bundle has no instantiated prefab in the difficulty")
            elif info_crc is not None:
                errors.append("Info.dat has a Vivify checksum but the runtime manifest is missing")
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            errors.append(f"manifest is unreadable: {exc}")
    return VerificationReport(not errors, tuple(errors), tuple(files), counts)


def verify_package(source: str | Path) -> VerificationReport:
    """Verify an exported directory or ZIP without executing its contents."""
    path = Path(source).resolve()
    if path.is_dir():
        return _verify_directory(path)
    if path.suffix.lower() != ".zip" or not path.is_file():
        return VerificationReport(False, (f"package is not a directory or ZIP: {path}",))
    try:
        with zipfile.ZipFile(path) as bundle:
            members = bundle.infolist()
            names = [member.filename.replace("\\", "/") for member in members]
            if any(not _safe_member(name) for name in names):
                return VerificationReport(False, ("ZIP contains an unsafe member path",))
            if len({name.casefold() for name in names}) != len(names):
                return VerificationReport(False, ("ZIP contains duplicate member paths",))
            with tempfile.TemporaryDirectory(prefix="beatforge-verify-") as temp:
                root = Path(temp)
                for member, name in zip(members, names):
                    target = (root / Path(name)).resolve()
                    if not target.is_relative_to(root):
                        return VerificationReport(False, ("ZIP member escaped extraction root",))
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(bundle.read(member))
                infos = [candidate for candidate in root.rglob("Info.dat") if candidate.is_file()]
                if len(infos) != 1:
                    return VerificationReport(False, ("ZIP must contain exactly one Info.dat",))
                return _verify_directory(infos[0].parent)
    except (OSError, zipfile.BadZipFile) as exc:
        return VerificationReport(False, (f"ZIP is unreadable: {exc}",))

