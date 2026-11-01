"""Development-time provider policy: free OpenRouter models only."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


OPENROUTER_FREE_ROUTER_ID = "openrouter/free"


def is_free_model_id(model_id: str) -> bool:
    """Accept catalogued free variants and OpenRouter's exact free router ID."""
    return (isinstance(model_id, str)
            and (model_id.endswith(":free") or model_id == OPENROUTER_FREE_ROUTER_ID))


@dataclass(frozen=True, slots=True)
class FreeModel:
    model_id: str
    supports_tools: bool
    supports_vision: bool
    supports_audio: bool
    supports_reasoning: bool


def _zero(value: Any) -> bool:
    try:
        return Decimal(str(value)) == 0
    except (InvalidOperation, ValueError, TypeError):
        return False


def parse_free_model(entry: dict[str, Any]) -> FreeModel:
    model_id = str(entry.get("id", ""))
    pricing = entry.get("pricing") or {}
    if not is_free_model_id(model_id):
        raise ValueError(f"development policy rejects non-free model: {model_id!r}")
    if not _zero(pricing.get("prompt")) or not _zero(pricing.get("completion")):
        raise ValueError(f"development policy rejects priced model: {model_id!r}")
    parameters = set(entry.get("supported_parameters") or [])
    architecture = entry.get("architecture") or {}
    modalities = set(architecture.get("input_modalities") or [])
    return FreeModel(model_id, "tools" in parameters, "image" in modalities,
                     "audio" in modalities, "reasoning" in parameters)


def free_catalog(entries: list[dict[str, Any]]) -> list[FreeModel]:
    models: list[FreeModel] = []
    for entry in entries:
        try:
            models.append(parse_free_model(entry))
        except ValueError:
            continue
    return models


def select_free_model(entries: list[dict[str, Any]], *, require_tools: bool = True,
                      require_vision: bool = False, require_audio: bool = False,
                      require_reasoning: bool = False) -> FreeModel:
    """Select the first catalog model satisfying explicit development needs."""
    for model in free_catalog(entries):
        if require_tools and not model.supports_tools:
            continue
        if require_vision and not model.supports_vision:
            continue
        if require_audio and not model.supports_audio:
            continue
        if require_reasoning and not model.supports_reasoning:
            continue
        return model
    requirements = ", ".join(name for name, enabled in (
        ("tools", require_tools), ("vision", require_vision),
        ("audio", require_audio), ("reasoning", require_reasoning)) if enabled)
    raise LookupError(f"no free model satisfies: {requirements or 'basic text'}")
