"""Deterministic handoff metadata for a future Unity/Vivify runtime."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HANDOFF_FORMAT = "BeatForge-Runtime-Handoff-0.1"


def build_runtime_handoff(package_root: str | Path, *, write: bool = True) -> dict[str, Any]:
    """Describe an exported package without executing a runtime.

    The handoff contains only relative package paths and capability metadata so
    a Unity/Vivify importer can make its own explicit runtime decisions. It is
    deliberately not a claim that Unity, Vivify or Beat Saber has been run.
    """
    root = Path(package_root).resolve()
    info_path = root / "Info.dat"
    manifest_path = root / "beatforge-manifest.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict) or not isinstance(manifest, dict):
        raise ValueError("package metadata must be JSON objects")

    difficulty_files: list[str] = []
    for group in info.get("difficultyBeatmapSets", []) or []:
        for entry in group.get("_difficultyBeatmaps", []) or []:
            filename = entry.get("_beatmapFilename")
            if isinstance(filename, str) and filename not in difficulty_files:
                difficulty_files.append(filename)
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError("beatforge manifest has no capabilities object")
    mods = capabilities.get("mods") if isinstance(capabilities.get("mods"), dict) else {}
    native = capabilities.get("native") if isinstance(capabilities.get("native"), dict) else {}
    vivify = manifest.get("vivify_assets")
    vivify_runtime = manifest.get("vivify_runtime")
    asset_files: list[str] = []
    if isinstance(vivify, dict):
        for asset in vivify.get("assets", []) or []:
            if isinstance(asset, dict) and isinstance(asset.get("package_path"), str):
                asset_files.append((Path("VivifyAssets") / asset["package_path"]).as_posix())

    handoff: dict[str, Any] = {
        "format": HANDOFF_FORMAT,
        "runtime_status": ("bundle_built_not_runtime_tested"
                            if isinstance(vivify_runtime, dict) else "handoff_only"),
        "info_file": "Info.dat",
        "difficulty_files": difficulty_files,
        "platform": capabilities.get("platform"),
        "required_mods": [name for name, enabled in mods.items() if enabled],
        "native_features": {"arcs": bool(native.get("arcs")),
                            "chains": bool(native.get("chains"))},
        "vivify": {
            "enabled": bool(mods.get("Vivify")),
            "manifest_file": "VivifyAssets/vivify-assets.json" if isinstance(vivify, dict) else None,
            "asset_files": asset_files,
            "bundle_file": vivify_runtime.get("bundle_file") if isinstance(vivify_runtime, dict) else None,
            "bundle_sha256": vivify_runtime.get("bundle_sha256") if isinstance(vivify_runtime, dict) else None,
            "bundle_crc": vivify_runtime.get("bundle_crc") if isinstance(vivify_runtime, dict) else None,
        },
    }
    if write:
        (root / "runtime-handoff.json").write_text(
            json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8")
    return handoff
