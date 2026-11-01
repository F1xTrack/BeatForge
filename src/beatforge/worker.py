"""Detached generation job wrapper with durable event checkpoints."""
from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
import time
from typing import Any

from .generation import (GenerationRequest, GenerationResult, ScriptGenerationError,
                          auto_repair_exact_duplicate_notes, generate_map, repair_map)
from .job_store import JobStore
from .budget import BudgetGuard
from .openrouter import OpenRouterResponseError


def aggregate_usage(attempts: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Keep every call's usage and sum reported token/cost counters."""
    totals: dict[str, Any] = {}

    def merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict):
                nested = target.get(key)
                if not isinstance(nested, dict):
                    nested = {}
                    target[key] = nested
                merge(nested, value)
            elif (isinstance(value, Real) and not isinstance(value, bool)
                  and (key.endswith("_tokens") or "cost" in key.lower())):
                current = target.get(key, 0)
                if isinstance(current, Real) and not isinstance(current, bool):
                    target[key] = current + value
                else:
                    target[key] = value
            else:
                target[key] = value

    for _phase, usage in attempts:
        merge(totals, usage)
    totals["attempts"] = [
        {"phase": phase, "usage": dict(usage)} for phase, usage in attempts
    ]
    return totals


def _record_reported_cost(budget: BudgetGuard | None, usage: dict[str, Any]) -> None:
    cost = usage.get("cost")
    if budget is not None and isinstance(cost, Real) and not isinstance(cost, bool):
        budget.record_cost(cost)


def _is_reasoning_only_length_failure(exc: Exception) -> bool:
    """Recognize an empty OpenRouter response that spent its output ceiling reasoning."""
    if not isinstance(exc, OpenRouterResponseError) or exc.finish_reason != "length":
        return False
    choice = exc.diagnostics.get("choice")
    fields = choice.get("message_fields") if isinstance(choice, dict) else None
    usage = exc.usage
    details = usage.get("completion_tokens_details")
    reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
    return (fields == ["role"] and isinstance(reasoning_tokens, int)
            and not isinstance(reasoning_tokens, bool) and reasoning_tokens > 0)


def _wire_live_progress(client: Any, store: JobStore, job_id: str) -> None:
    """Expose streamed visible answer text in the replaceable live snapshot."""
    set_callback = getattr(client, "set_progress_callback", None)
    if not callable(set_callback):
        return
    last_text: str | None = None
    last_write = 0.0

    def publish(partial_text: str) -> None:
        nonlocal last_text, last_write
        if not isinstance(partial_text, str):
            return
        now = time.monotonic()
        if (last_text is not None and last_text and partial_text
                and len(partial_text) - len(last_text) < 256
                and now - last_write < 1.0):
            return
        current = store.live(job_id) or {}
        payload = {key: value for key, value in current.items()
                   if key not in {"status", "recorded_at"}}
        payload["stream_connected"] = True
        payload["partial_response"] = partial_text
        payload["partial_response_chars"] = len(partial_text)
        try:
            store.write_live(job_id, "provider_running", payload)
        except OSError:
            return
        last_text = partial_text
        last_write = now

    set_callback(publish)


@dataclass(frozen=True, slots=True)
class JobResult:
    job_id: str
    status: str
    generation: GenerationResult | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()


def run_generation_job(job_id: str, request: GenerationRequest, client: Any,
                       store: JobStore, budget: BudgetGuard | None = None,
                       *, repair: bool = False, previous: GenerationResult | None = None,
                       diagnostics: dict[str, Any] | None = None,
                       tool_executor: Any = None,
                       comparison_difficulty: dict[str, Any] | None = None,
                       reserve_followup_requests: int = 0) -> JobResult:
    """Run one job and persist every terminal state before returning."""
    if repair and previous is None:
        return JobResult(job_id, "failed", error="repair requires a previous generation")
    if (isinstance(reserve_followup_requests, bool)
            or not isinstance(reserve_followup_requests, int)
            or reserve_followup_requests < 0):
        raise ValueError("reserve_followup_requests must be a non-negative integer")
    generation_request_limit = (
        max(0, budget.max_requests - reserve_followup_requests)
        if budget is not None else None
    )
    store.append(job_id, "repair_started" if repair else "generation_started", {
        "model_id": request.model_id, "audio_path": request.audio_path,
        "section_ids": list(request.section_ids),
        "reasoning_effort": request.reasoning_effort,
        "asset_ids": list(request.asset_ids),
        "provider": request.provider,
        "send_spectrograms": request.send_spectrograms,
        "spectrogram_manifest": request.spectrogram_manifest,
        "repair": repair,
        "reserve_followup_requests": reserve_followup_requests,
    })
    live_payload = {
        "model_id": request.model_id,
        "reasoning_effort": request.reasoning_effort,
        "send_spectrograms": request.send_spectrograms,
        "spectrogram_manifest": request.spectrogram_manifest,
        "repair": repair,
        "reserve_followup_requests": reserve_followup_requests,
    }
    _wire_live_progress(client, store, job_id)
    store.write_live(job_id, "starting", live_payload)
    try:
        if budget is not None:
            budget.reserve(request.model_id, repair=repair, local=request.provider == "local")
        store.write_live(job_id, "provider_running", live_payload)
        def reserve_additional_call() -> None:
            if budget is not None:
                if budget.requests >= generation_request_limit:
                    raise RuntimeError(
                        "generation request budget reserved for diagnostics repair")
                # Tool-loop follow-ups are requests in the same generation or
                # repair attempt, not additional repair attempts themselves.
                budget.reserve(request.model_id, local=request.provider == "local")
        result = (repair_map(request, previous, diagnostics or {}, client,
                             tool_executor=tool_executor,
                             before_additional_call=reserve_additional_call,
                             comparison_difficulty=comparison_difficulty,
                             max_tool_calls=(
                                 max(0, generation_request_limit - budget.requests)
                                 if budget is not None else None))
                  if repair else generate_map(request, client,
                                              tool_executor=tool_executor,
                                              before_additional_call=reserve_additional_call,
                                              max_tool_calls=(
                                                  max(0, generation_request_limit - budget.requests)
                                                  if budget is not None else None)))
        if budget is not None and result.usage.get("cost") is not None:
            budget.record_cost(result.usage["cost"])
    except ScriptGenerationError as exc:
        original_error = str(exc)
        original_script = exc.script
        original_usage = dict(exc.usage)
        usage_attempts = [("generation", original_usage)]
        script_attempts = [{"phase": "generation", "script": original_script}]
        _record_reported_cost(budget, original_usage)
        fixed, automatic_fixes = auto_repair_exact_duplicate_notes(
            request, original_script, tool_executor,
        )
        if fixed is not None:
            script_attempts.append({"phase": "automatic_exact_duplicate_cleanup",
                                    "script": fixed.script})
            fixed_usage = dict(original_usage)
            fixed_usage["beatforge_automatic_repairs"] = automatic_fixes
            fixed = replace(fixed, usage=fixed_usage)
            completion_kind = "repair_auto_repair_completed" if repair else "generation_auto_repair_completed"
            store.append(job_id, completion_kind, {
                "repair": repair, "method": "removed_exact_duplicate_calls",
                "changes": automatic_fixes, "usage": fixed_usage,
                "script": fixed.script,
            })
            store.write_live(job_id, "completed", {
                **live_payload, "repair": repair, "model_id": fixed.model_id,
                "usage": fixed_usage, "script": fixed.script,
                "script_attempts": script_attempts,
                "asset_ids": list(fixed.asset_ids),
                "note_count": len(fixed.difficulty.get("colorNotes", [])),
                "bomb_count": len(fixed.difficulty.get("bombNotes", [])),
                "arc_count": len(fixed.difficulty.get("sliders", [])),
                "chain_count": len(fixed.difficulty.get("burstSliders", [])),
                "automatic_repairs": automatic_fixes,
            })
            return JobResult(job_id, "completed", generation=fixed)
        repair_count = 0
        repair_started = False
        current_script = original_script
        current_error = str(exc)
        current_usage = original_usage
        result = None
        if not repair and budget is not None:
            while (budget.automatic_repairs < budget.max_automatic_repairs
                   and budget.requests < generation_request_limit):
                try:
                    budget.reserve(request.model_id, automatic_repair=True,
                                   local=request.provider == "local")
                except Exception as budget_exc:
                    original_error = f"{original_error}; automatic repair unavailable: {budget_exc}"
                    break
                repair_count += 1
                repair_started = True

                def reserve_repair_additional_call() -> None:
                    if budget.requests >= generation_request_limit:
                        raise RuntimeError(
                            "generation request budget reserved for diagnostics repair")
                    budget.reserve(request.model_id, automatic_repair=True,
                                   local=request.provider == "local")

                store.append(job_id, "generation_repair_started", {
                    "attempt": repair_count, "reason": "script_parse_failed",
                    "error": current_error, "repair": True, "script": current_script,
                })
                store.write_live(job_id, "repair_running", {
                    **live_payload, "repair": True, "attempt": repair_count,
                    "reason": "script_parse_failed", "script": current_script,
                    "script_attempts": script_attempts,
                })
                try:
                    result = repair_map(
                        request,
                        GenerationResult(request.model_id, current_script, {}, current_usage),
                        {"script_error": current_error, "script": current_script},
                        client, tool_executor=tool_executor,
                        before_additional_call=reserve_repair_additional_call,
                        max_tool_calls=(
                            max(0, generation_request_limit - budget.requests)
                            if budget is not None else None),
                    )
                except ScriptGenerationError as retry_exc:
                    current_script = retry_exc.script
                    current_error = str(retry_exc)
                    current_usage = dict(retry_exc.usage)
                    usage_attempts.append(("automatic_repair", current_usage))
                    script_attempts.append({
                        "phase": "automatic_repair", "script": current_script,
                    })
                    _record_reported_cost(budget, current_usage)
                    original_error = f"{original_error}; automatic repair {repair_count} failed: {retry_exc}"
                    store.append(job_id, "generation_repair_failed", {
                        "attempt": repair_count, "error": current_error,
                        "usage": current_usage, "script": current_script,
                    })
                    continue
                except Exception as retry_exc:
                    repair_usage = getattr(retry_exc, "usage", {})
                    if not isinstance(repair_usage, dict):
                        repair_usage = {}
                    usage_attempts.append(("automatic_repair", dict(repair_usage)))
                    repair_script = getattr(retry_exc, "script", None)
                    if isinstance(repair_script, str):
                        script_attempts.append({
                            "phase": "automatic_repair", "script": repair_script,
                        })
                    _record_reported_cost(budget, repair_usage)
                    original_error = f"{original_error}; automatic repair {repair_count} failed: {retry_exc}"
                    provider_diagnostics = getattr(retry_exc, "diagnostics", {})
                    store.append(job_id, "generation_repair_failed", {
                        "attempt": repair_count, "error": str(retry_exc),
                        "usage": repair_usage,
                        **({"provider_diagnostics": provider_diagnostics}
                           if isinstance(provider_diagnostics, dict) and provider_diagnostics else {}),
                        **({"script": repair_script} if isinstance(repair_script, str) else {}),
                    })
                    break

                usage_attempts.append(("automatic_repair", dict(result.usage)))
                script_attempts.append({"phase": "automatic_repair", "script": result.script})
                if result.usage.get("cost") is not None:
                    budget.record_cost(result.usage["cost"])
                result = replace(result, usage=aggregate_usage(usage_attempts))
                store.append(job_id, "generation_repair_completed", {
                    "attempt": repair_count, "model_id": result.model_id,
                    "usage": result.usage, "script": result.script,
                })
                store.write_live(job_id, "completed", {
                    **live_payload, "repair": True, "model_id": result.model_id,
                    "usage": result.usage, "script": result.script,
                    "script_attempts": script_attempts,
                    "asset_ids": list(result.asset_ids),
                    "note_count": len(result.difficulty.get("colorNotes", [])),
                    "bomb_count": len(result.difficulty.get("bombNotes", [])),
                    "arc_count": len(result.difficulty.get("sliders", [])),
                    "chain_count": len(result.difficulty.get("burstSliders", [])),
                })
                return JobResult(job_id, "completed", generation=result)
        failure_usage = (aggregate_usage(usage_attempts) if repair_started
                         else original_usage)
        store.write_live(job_id, "failed", {
            **live_payload, "error_type": "ScriptGenerationError", "error": original_error,
            "script": original_script, "script_attempts": script_attempts,
            "usage": failure_usage,
        })
        store.append(job_id, "generation_failed", {
            "error_type": "ScriptGenerationError", "error": original_error, "repair": repair,
            "script": original_script, "script_attempts": script_attempts,
            "usage": failure_usage,
        })
        return JobResult(job_id, "failed", error=original_error)
    except Exception as exc:  # persist failure before exposing it to the caller
        failed_usage = getattr(exc, "usage", {})
        if not isinstance(failed_usage, dict):
            failed_usage = {}
        provider_diagnostics = getattr(exc, "diagnostics", {})
        if not isinstance(provider_diagnostics, dict):
            provider_diagnostics = {}
        _record_reported_cost(budget, failed_usage)
        recovered = False
        if (not repair and request.provider == "openrouter"
                and request.reasoning_effort != "low"
                and _is_reasoning_only_length_failure(exc)
                and budget is not None):
            try:
                if budget.requests >= generation_request_limit:
                    raise RuntimeError(
                        "generation request budget reserved for diagnostics repair")
                budget.reserve(request.model_id, automatic_repair=True)
            except Exception as budget_exc:
                store.append(job_id, "generation_recovery_skipped", {
                    "reason": "reasoning_only_output_limit",
                    "retry_reasoning_effort": "low",
                    "error": str(budget_exc),
                    "usage": failed_usage,
                })
            else:
                retry_request = replace(request, reasoning_effort="low")
                retry_payload = {
                    **live_payload,
                    "reasoning_effort": "low",
                    "recovery_reason": "reasoning_only_output_limit",
                }
                store.append(job_id, "generation_recovery_started", {
                    "reason": "reasoning_only_output_limit",
                    "reasoning_effort": "low",
                    "previous_usage": failed_usage,
                    "provider_diagnostics": provider_diagnostics,
                })
                store.write_live(job_id, "recovery_running", retry_payload)

                def reserve_recovery_followup_call() -> None:
                    if budget.requests >= generation_request_limit:
                        raise RuntimeError(
                            "generation request budget reserved for diagnostics repair")
                    budget.reserve(request.model_id, local=request.provider == "local")

                try:
                    result = generate_map(
                        retry_request, client,
                        tool_executor=tool_executor,
                        before_additional_call=reserve_recovery_followup_call,
                        max_tool_calls=max(0, generation_request_limit - budget.requests),
                    )
                except Exception as retry_exc:
                    retry_usage = getattr(retry_exc, "usage", {})
                    if not isinstance(retry_usage, dict):
                        retry_usage = {}
                    _record_reported_cost(budget, retry_usage)
                    failed_usage = aggregate_usage([
                        ("provider_default", failed_usage),
                        ("low_reasoning_recovery", retry_usage),
                    ])
                    provider_diagnostics = getattr(retry_exc, "diagnostics", {})
                    if not isinstance(provider_diagnostics, dict):
                        provider_diagnostics = {}
                    recovery_error = str(retry_exc)
                    store.append(job_id, "generation_recovery_failed", {
                        "reason": "reasoning_only_output_limit",
                        "reasoning_effort": "low",
                        "error": recovery_error,
                        "usage": failed_usage,
                        **({"provider_diagnostics": provider_diagnostics}
                           if provider_diagnostics else {}),
                    })
                    exc = RuntimeError(
                        f"{exc}; low-reasoning recovery failed: {recovery_error}")
                else:
                    _record_reported_cost(budget, result.usage)
                    total_usage = aggregate_usage([
                        ("provider_default", failed_usage),
                        ("low_reasoning_recovery", result.usage),
                    ])
                    result = replace(result, usage=total_usage)
                    live_payload = retry_payload
                    store.append(job_id, "generation_recovery_completed", {
                        "reason": "reasoning_only_output_limit",
                        "reasoning_effort": "low",
                        "usage": total_usage,
                        "note_count": len(result.difficulty.get("colorNotes", [])),
                        "arc_count": len(result.difficulty.get("sliders", [])),
                        "chain_count": len(result.difficulty.get("burstSliders", [])),
                    })
                    recovered = True
        if not recovered:
            if _is_reasoning_only_length_failure(exc) and not provider_diagnostics:
                provider_diagnostics = getattr(exc, "diagnostics", {})
            current_live = store.live(job_id) or {}
            partial_response = {
                key: current_live[key]
                for key in ("partial_response", "partial_response_chars")
                if key in current_live
            }
            store.write_live(job_id, "failed", {
                **live_payload, **partial_response,
                "error_type": type(exc).__name__, "error": str(exc),
                "usage": failed_usage,
                **({"provider_diagnostics": provider_diagnostics}
                   if provider_diagnostics else {}),
            })
            store.append(job_id, "repair_failed" if repair else "generation_failed",
                         {"error_type": type(exc).__name__, "error": str(exc),
                          "model_id": request.model_id,
                          "repair": repair, "usage": failed_usage,
                          **({"provider_diagnostics": provider_diagnostics}
                             if provider_diagnostics else {})})
            return JobResult(job_id, "failed", error=str(exc))
    store.write_live(job_id, "completed", {
        **live_payload,
        "model_id": result.model_id,
        "usage": result.usage,
        "script": result.script,
        "asset_ids": list(result.asset_ids),
        "note_count": len(result.difficulty.get("colorNotes", [])),
        "bomb_count": len(result.difficulty.get("bombNotes", [])),
        "arc_count": len(result.difficulty.get("sliders", [])),
        "chain_count": len(result.difficulty.get("burstSliders", [])),
    })
    store.append(job_id, "repair_completed" if repair else "generation_completed", {
        "model_id": result.model_id, "usage": result.usage,
        "asset_ids": list(result.asset_ids),
        "script": result.script,
        "note_count": len(result.difficulty.get("colorNotes", [])),
        "bomb_count": len(result.difficulty.get("bombNotes", [])),
        "arc_count": len(result.difficulty.get("sliders", [])),
        "chain_count": len(result.difficulty.get("burstSliders", [])),
    })
    return JobResult(job_id, "completed", generation=result)
