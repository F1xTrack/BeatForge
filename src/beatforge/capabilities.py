"""Explicit platform/mod capabilities passed to generation and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    platform: str = "pc"
    chroma: bool = False
    noodle: bool = False
    mapping_extensions: bool = False
    vivify: bool = False
    arcs: bool = True
    chains: bool = True
    max_asset_mb: int = 64
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "mods": {
                "Chroma": self.chroma,
                "Noodle Extensions": self.noodle,
                "Mapping Extensions": self.mapping_extensions,
                "Vivify": self.vivify,
            },
            "native": {"arcs": self.arcs, "chains": self.chains},
            "max_asset_mb": self.max_asset_mb,
            "notes": list(self.notes),
        }


def validate_capabilities(profile: CapabilityProfile) -> list[str]:
    errors: list[str] = []
    if profile.platform not in {"pc", "quest"}:
        errors.append("platform must be 'pc' or 'quest'")
    if profile.vivify and profile.platform == "quest":
        errors.append("Vivify is currently PC-only; disable it for Quest output")
    if profile.max_asset_mb <= 0:
        errors.append("max_asset_mb must be positive")
    if not profile.arcs:
        errors.append("native arcs are required by the current V3 authoring core")
    if not profile.chains:
        errors.append("native chains are required by the current V3 authoring core")
    return errors


def effective_capabilities(platform: str, **mods: bool) -> CapabilityProfile:
    """Build a profile from UI-style independent mod toggles."""
    profile = CapabilityProfile(
        platform=platform,
        chroma=bool(mods.get("chroma", False)),
        noodle=bool(mods.get("noodle", False)),
        mapping_extensions=bool(mods.get("mapping_extensions", False)),
        vivify=bool(mods.get("vivify", False)),
    )
    errors = validate_capabilities(profile)
    if errors:
        raise ValueError("; ".join(errors))
    return profile
