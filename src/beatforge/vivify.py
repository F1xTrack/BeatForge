"""Reproducible, data-only Vivify asset bundle preparation."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from .assets import AssetLibrary, AssetRecord, SAFE_LICENSES
from .contracts import MapProgram


VIVIFY_SETTING_KEYS = {
    "renderSettings": frozenset({
        "ambientEquatorColor", "ambientGroundColor", "ambientIntensity", "ambientLight", "ambientMode",
        "ambientSkyColor", "defaultReflectionMode", "defaultReflectionResolution",
        "flareFadeSpeed", "flareStrength", "fog", "fogColor", "fogDensity",
        "fogEndDistance", "fogMode", "haloStrength", "reflectionBounces",
        "reflectionIntensity", "skybox", "subtractiveShadowColor", "sun",
    }),
    "qualitySettings": frozenset({
        "anisotropicFiltering", "antiAliasing", "pixelLightCount",
        "realtimeReflectionProbes", "shadowCascades", "shadowDistance",
        "shadowmaskMode", "shadowNearPlaneOffset", "shadowProjection",
        "shadowResolution", "shadows", "softParticles",
    }),
    "xrSettings": frozenset({"useOcclusionMesh"}),
}
_VIVIFY_COLOR_KEYS = frozenset({
    "ambientEquatorColor", "ambientGroundColor", "ambientSkyColor", "fogColor",
    "subtractiveShadowColor", "ambientLight",
})


def validate_vivify_scene_events(
    events: Iterable[Any], *, max_beat: float | None = None,
) -> list[str]:
    """Validate at least one documented Vivify SetRenderingSettings event.

    This deliberately validates only the small, documented scene-settings
    subset used by BeatForge; it does not claim to compile imported source
    models into Unity AssetBundles.
    """
    from math import isfinite
    from numbers import Real

    scene_events = []
    for event in events:
        if isinstance(event, dict):
            event_type = event.get("t")
            beat = event.get("b")
            data = event.get("d")
        else:
            event_type = getattr(event, "event_type", None)
            beat = getattr(event, "beat", None)
            data = getattr(event, "data", None)
        if event_type == "SetRenderingSettings":
            scene_events.append((beat, data))
    if not scene_events:
        return ["Vivify requires a SetRenderingSettings scene event"]

    errors: list[str] = []
    allowed_top_level = {*VIVIFY_SETTING_KEYS, "duration", "easing"}
    for index, (beat, data) in enumerate(scene_events):
        prefix = f"Vivify SetRenderingSettings event {index}"
        if (not isinstance(beat, Real) or isinstance(beat, bool)
                or not isfinite(float(beat)) or beat < 0
                or (max_beat is not None and beat > max_beat)):
            errors.append(f"{prefix} beat must fall within the track")
        if not isinstance(data, dict) or set(data) - allowed_top_level:
            errors.append(f"{prefix} data must use documented settings categories")
            continue
        duration = data.get("duration", 0)
        if (not isinstance(duration, Real) or isinstance(duration, bool)
                or not isfinite(float(duration)) or duration < 0):
            errors.append(f"{prefix} duration must be a finite non-negative number")
        easing = data.get("easing")
        if easing is not None and (not isinstance(easing, str) or not easing.strip()):
            errors.append(f"{prefix} easing must be a non-empty string")
        has_setting = False
        for category, keys in VIVIFY_SETTING_KEYS.items():
            settings = data.get(category)
            if settings is None:
                continue
            if not isinstance(settings, dict) or set(settings) - keys:
                errors.append(f"{prefix} {category} contains unsupported settings")
                continue
            if settings:
                has_setting = True
            for name, value in settings.items():
                if name in _VIVIFY_COLOR_KEYS:
                    if (not isinstance(value, (list, tuple)) or len(value) not in {3, 4}
                            or any(not isinstance(channel, Real) or isinstance(channel, bool)
                                   or not isfinite(float(channel)) for channel in value)):
                        errors.append(f"{prefix} {name} must be a finite RGB or RGBA array")
                elif name == "fog" and not (
                        isinstance(value, bool)
                        or isinstance(value, int) and value in {0, 1}):
                    errors.append(f"{prefix} fog must be 0 or 1")
                elif name == "fogMode" and not (
                        isinstance(value, int) and not isinstance(value, bool)
                        and value in {1, 2, 3}):
                    errors.append(f"{prefix} fogMode must be 1, 2 or 3")
                elif name == "ambientMode" and not (
                        isinstance(value, int) and not isinstance(value, bool)
                        and value in {0, 1, 3, 4}):
                    errors.append(f"{prefix} ambientMode must be 0, 1, 3 or 4")
        if not has_setting:
            errors.append(f"{prefix} must set at least one documented scene property")
    return errors


def vivify_settings_prompt_guidance() -> str:
    """Describe exactly the scene-setting categories and fields we validate."""
    categories = "; ".join(
        f"{category}: {', '.join(sorted(keys))}"
        for category, keys in VIVIFY_SETTING_KEYS.items()
    )
    return (
        "For SetRenderingSettings, `data` may contain only the documented settings "
        "categories below plus optional `duration` and `easing`. Use each field only "
        "under its listed category; `ambientLight` and `fogColor` belong under "
        "`renderSettings`, never under `qualitySettings`. Do not invent setting names. "
        f"Supported category fields: {categories}."
    )


def referenced_asset_ids(program: MapProgram) -> frozenset[str]:
    """Collect explicit ``asset_id`` references from typed custom events."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "asset_id" and isinstance(item, str) and item:
                    found.add(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for event in program.custom_events:
        walk(event.data)
    return frozenset(found)


def validate_asset_references(program: MapProgram, allowed_ids: Iterable[str]) -> list[str]:
    """Fail closed when a custom event refers to an unselected asset."""
    allowed = {asset_id for asset_id in allowed_ids}
    return [f"custom event references asset id not selected for export: {asset_id}"
            for asset_id in sorted(referenced_asset_ids(program) - allowed)]


def _checked_record(library: AssetLibrary, asset_id: str, max_bytes: int) -> tuple[AssetRecord, Path]:
    record = library.get(asset_id)
    if record is None:
        raise ValueError(f"unknown asset id: {asset_id}")
    if record.license.upper() not in SAFE_LICENSES:
        raise ValueError(f"asset license is not in the safe allowlist: {record.license!r}")
    source = Path(record.local_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    size = source.stat().st_size
    if size > max_bytes:
        raise ValueError(f"asset {asset_id} exceeds the {max_bytes} byte bundle limit")
    hashed = record.with_hash()
    if record.sha256 and record.sha256 != hashed.sha256:
        raise ValueError(f"asset {asset_id} changed since it was catalogued")
    return hashed, source


def build_vivify_manifest(
    library: AssetLibrary,
    asset_ids: Iterable[str],
    destination: str | Path,
    *,
    max_asset_mb: int = 64,
) -> dict[str, Any]:
    """Copy catalogued assets and write an integrity manifest.

    This function only copies opaque data files. It does not execute Unity,
    Vivify scripts, editor files or downloaded binaries.
    """
    if max_asset_mb <= 0:
        raise ValueError("max_asset_mb must be positive")
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    max_bytes = max_asset_mb * 1024 * 1024
    assets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for asset_id in asset_ids:
        if asset_id in seen:
            continue
        seen.add(asset_id)
        record, source = _checked_record(library, asset_id, max_bytes)
        package_name = f"{asset_id}{source.suffix.lower()}"
        target = (root / package_name).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("asset package path escaped destination")
        shutil.copyfile(source, target)
        assets.append({
            "asset_id": record.asset_id,
            "title": record.title,
            "source_url": record.source_url,
            "license": record.license,
            "author": record.author,
            "media_type": record.media_type,
            "sha256": record.sha256,
            "size_bytes": target.stat().st_size,
            "package_path": target.relative_to(root).as_posix(),
        })
    manifest = {"format": "BeatForge-Vivify-assets-0.1", "assets": assets}
    (root / "vivify-assets.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest

