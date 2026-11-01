"""End-to-end, free-only generation workflow."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

from .adapters import program_from_v3
from .analysis import SongAnalysis, analyze_audio
from .audio_features import describe_audio_grid
from .budget import BudgetGuard
from .beatgrid import BeatGrid
from .capabilities import CapabilityProfile
from .generation import (GenerationRequest, GenerationResult,
                         _compact_rhythm_move_candidates,
                         supports_model_visual_inspection)
from .job_store import JobStore
from .packaging import ExportMetadata, export_v3
from .planning import TrackPlan, initial_plan_from_analysis, initial_plan_from_beat_grid, validate_plan
from .quality import preflight
from .spectrograms import build_spectrograms, pack_spectrogram_images
from .verify import verify_package
from .vivify import referenced_asset_ids
from .vivify_bundle import (VivifyModelInput, build_vivify_bundles,
                            clone_pinned_template, unity_executable_from_environment)
from .worker import JobResult, aggregate_usage, run_generation_job


@dataclass(frozen=True, slots=True)
class PreparedGeneration:
    analysis: SongAnalysis
    plan: TrackPlan
    request: GenerationRequest


def _record_workflow_failure(store: JobStore, job_id: str, *, stage: str,
                             error: str, error_type: str,
                             details: dict[str, Any] | None = None) -> None:
    """Keep the live snapshot aligned with a workflow failure after generation.

    Preserve the completed provider response (script, usage and counts) while
    exposing the later deterministic gate/export failure as the terminal state.
    """
    current = store.live(job_id) or {}
    payload = {key: value for key, value in current.items()
               if key not in {"status", "recorded_at"}}
    payload.update({"failure_stage": stage, "error_type": error_type,
                    "error": error})
    if details is not None:
        payload["failure_details"] = details
    store.write_live(job_id, "failed", payload)


def _locked_fingerprint(program, plan: TrackPlan) -> tuple[str, ...]:
    """Canonical V3 objects intersecting locked plan sections."""
    locked = [(section.start_beat, section.end_beat)
              for section in plan.sections if section.locked]
    if not locked:
        return ()
    data = program.to_v3()
    values: list[str] = []
    for key in ("colorNotes", "bombNotes", "sliders", "burstSliders", "obstacles", "basicBeatmapEvents"):
        for item in data.get(key, []) or []:
            start = float(item.get("b", 0))
            end = float(item.get("tb", start))
            if any(start <= section_end and end >= section_start
                   for section_start, section_end in locked):
                values.append(json.dumps((key, item), ensure_ascii=False, sort_keys=True))
    return tuple(sorted(values))


def _is_repair_invariant_rejection(error: str | None) -> bool:
    """Recognize deterministic script/contract rejections, not provider errors."""
    return isinstance(error, str) and error.startswith((
        "repair changed playable hit-equivalent count ",
        "repair changed native chain objects without a chain diagnostic;",
        "repair changed native arc objects without an arc review target;",
        "palette-only review did not clear low-contrast Chroma warning",
        "repair script rejected by parser or capability validation:",
        "palette-only script rejected by parser or capability validation:",
        "repair patch rejected by BeatForge validator:",
    ))


def _repair_rejection_reason(error: str | None) -> str:
    """Classify a retryable deterministic rejection for the durable event log."""
    if isinstance(error, str) and error.startswith((
            "repair script rejected by parser or capability validation:",
            "palette-only script rejected by parser or capability validation:",
            "repair patch rejected by BeatForge validator:")):
        return "script_validation_rejection"
    return "deterministic_repair_invariant_rejection"


def _repair_retry_capacity_available(budget: BudgetGuard | None) -> bool:
    """Check request, repair and spend ceilings before scheduling a retry."""
    return budget is None or (
        budget.requests < budget.max_requests
        and budget.repairs < budget.max_repairs
        and budget.cost_usd <= budget.max_cost_usd
    )


def _has_low_palette_warning(report: Any) -> bool:
    warnings = getattr(report, "warnings", ())
    return any("scene_palette_low_contrast" in str(item).lower()
               for item in warnings)


def _pattern_alignment_regression(before: dict[str, Any],
                                  after: dict[str, Any]) -> str | None:
    """Reject a repair that worsens an already flagged audio-pattern signal."""
    prior = before.get("pattern_alignment")
    repaired = after.get("pattern_alignment")
    if (not isinstance(prior, dict) or not prior.get("available")
            or not prior.get("warning") or not isinstance(repaired, dict)
            or not repaired.get("available")):
        return None
    prior_score = prior.get("chance_corrected_structure_support")
    repaired_score = repaired.get("chance_corrected_structure_support")
    if (not isinstance(prior_score, (int, float))
            or isinstance(prior_score, bool)
            or not isinstance(repaired_score, (int, float))
            or isinstance(repaired_score, bool)):
        return None
    if repaired_score >= prior_score:
        return None
    return (
        "repair reduced pattern-alignment support "
        f"from {prior_score:.3f} to {repaired_score:.3f} while the prior map "
        "had an active pattern warning; preserve the prior support or improve it"
    )


def _rejected_pattern_gain_feedback(store: JobStore, job_id: str,
                                   error: str | None,
                                   request: GenerationRequest,
                                   prior_report: dict[str, Any]) -> str | None:
    """Preserve quality gains from a rejected candidate during count repair."""
    if not isinstance(error, str) or not error.startswith(
        "repair changed playable hit-equivalent count "
    ):
        return None
    live = store.live(job_id) or {}
    if live.get("error") != error or not isinstance(live.get("script"), str):
        return None
    try:
        from .script_runner import run_script

        difficulty = run_script(
            live["script"], bpm=request.plan.bpm,
            mapping_extensions=request.capabilities.mapping_extensions,
        )
        program = program_from_v3(difficulty, bpm=request.plan.bpm)
        candidate = preflight(
            program, request.plan, audio_features=request.audio_features,
            mapping_extensions=request.capabilities.mapping_extensions,
        ).to_dict()
    except (SyntaxError, TypeError, ValueError):
        return None
    gains: list[str] = []
    before = prior_report.get("pattern_alignment")
    after = candidate.get("pattern_alignment")
    if (isinstance(before, dict) and before.get("available") and before.get("warning")
            and isinstance(after, dict) and after.get("available")):
        prior_score = before.get("chance_corrected_structure_support")
        candidate_score = after.get("chance_corrected_structure_support")
        if (isinstance(prior_score, (int, float)) and not isinstance(prior_score, bool)
                and isinstance(candidate_score, (int, float))
                and not isinstance(candidate_score, bool) and candidate_score > prior_score):
            gains.append(
                "increased the active pattern-alignment support "
                f"from {prior_score:.3f} to {candidate_score:.3f}"
            )
    prior_occupied = prior_report.get("occupied_sections")
    candidate_occupied = candidate.get("occupied_sections")
    prior_total = prior_report.get("total_sections")
    candidate_total = candidate.get("total_sections")
    if (all(isinstance(value, int) and not isinstance(value, bool)
            for value in (prior_occupied, candidate_occupied, prior_total, candidate_total))
            and prior_total == candidate_total and prior_total > 0
            and candidate_occupied > prior_occupied):
        gains.append(
            f"increased occupied sections from {prior_occupied}/{prior_total} "
            f"to {candidate_occupied}/{candidate_total}"
        )
    if not gains:
        return None
    return (
        " The rejected candidate " + " and ".join(gains) + ". These are coarse "
        "development diagnostics, not a gameplay score. Keep only localized edits that remain musically "
        "supported by the SG/audio, and adjust them to preserve the exact hit count; "
        "do not simply restore the unchanged source map."
    )


def prepare_generation(
    audio_path: str | Path,
    *,
    bpm: float | None = None,
    target_nps: float | None = None,
    model_id: str,
    capabilities: CapabilityProfile | None = None,
    primary_layer: str = "mixed",
    style: str = "expressive-readable",
    notes: str = "",
    plan: TrackPlan | None = None,
    beat_grid: BeatGrid | None = None,
    reasoning_effort: str | None = None,
    asset_ids: tuple[str, ...] = (),
    provider: str = "openrouter",
    send_audio: bool = False,
    send_spectrograms: bool = False,
    spectrogram_image_limit: int | None = None,
    track_title: str = "",
    track_artist: str = "",
) -> PreparedGeneration:
    """Analyze local audio and build a reviewable request before any model call."""
    if send_spectrograms and provider != "local":
        raise ValueError("send_spectrograms is an explicit local-provider option")
    analysis = analyze_audio(audio_path)
    if plan is None:
        if bpm is None or target_nps is None:
            raise ValueError("bpm and target_nps are required when no plan is supplied")
        if beat_grid is not None:
            plan = initial_plan_from_beat_grid(analysis, beat_grid, target_nps=target_nps)
        else:
            plan = initial_plan_from_analysis(analysis, bpm=bpm, target_nps=target_nps)
        plan.primary_layer = primary_layer
        plan.style = style
        plan.notes = notes
    else:
        errors = validate_plan(plan)
        if errors:
            raise ValueError("invalid supplied plan: " + "; ".join(errors))
    audio_features = describe_audio_grid(audio_path, plan.bpm)
    spectrogram_images, spectrogram_manifest = build_spectrograms(audio_path)
    spectrogram_source_images = spectrogram_images
    if spectrogram_image_limit is not None:
        spectrogram_images, model_input = pack_spectrogram_images(
            spectrogram_images, spectrogram_image_limit, spectrogram_manifest)
        spectrogram_manifest["model_input"] = model_input
    request = GenerationRequest(model_id, str(Path(audio_path).resolve()), plan,
                                capabilities or CapabilityProfile(),
                                reasoning_effort=reasoning_effort,
                                asset_ids=tuple(asset_ids), provider=provider,
                                audio_features=audio_features,
                                send_audio=send_audio,
                                send_spectrograms=send_spectrograms,
                                spectrogram_images=spectrogram_images,
                                spectrogram_manifest=spectrogram_manifest,
                                track_title=track_title,
                                track_artist=track_artist,
                                spectrogram_source_images=spectrogram_source_images)
    return PreparedGeneration(analysis, plan, request)


def generate_and_export(
    prepared: PreparedGeneration,
    client: Any,
    *,
    job_id: str,
    jobs_root: str | Path,
    output: str | Path,
    metadata: ExportMetadata,
    budget: BudgetGuard | None = None,
    asset_library=None,
    asset_ids: tuple[str, ...] = (),
    tool_executor: Any = None,
    zip_path: str | Path | None = None,
    max_repairs: int = 0,
    vivify_bundle_config: dict[str, Any] | None = None,
) -> tuple[JobResult, Path | None]:
    """Run a durable free-only job and export only a completed generation."""
    store = JobStore(jobs_root)
    repair_request_reserve = (
        min(max(2, max_repairs), max(0, budget.max_requests - 1))
        if budget is not None and max_repairs > 0 else 0
    )
    result = run_generation_job(
        job_id, prepared.request, client, store, budget,
        tool_executor=tool_executor,
        reserve_followup_requests=repair_request_reserve,
    )
    if result.status != "completed" or result.generation is None:
        return result, None
    usage_attempts = [("initial_generation", result.generation.usage)]
    if max_repairs < 0:
        raise ValueError("max_repairs must be non-negative")
    selected_asset_ids = tuple(dict.fromkeys((
        *prepared.request.asset_ids, *asset_ids, *result.generation.asset_ids,
        *(getattr(tool_executor, "imported_asset_ids", ()) if tool_executor else ()),
    )))
    current_program = program_from_v3(result.generation.difficulty, bpm=prepared.plan.bpm)
    locked_before = _locked_fingerprint(current_program, prepared.plan)
    repair_attempts = 0
    workflow_warnings: list[str] = []
    repair_rejection_feedback: str | None = None
    comparison_difficulty: dict[str, Any] | None = None
    can_review_visually = bool(
        max_repairs > 0 and _repair_retry_capacity_available(budget)
        and supports_model_visual_inspection(prepared.request, client)
    )
    for attempt in range(max_repairs):
        report = preflight(current_program, prepared.plan,
                           audio_features=prepared.request.audio_features,
                           mapping_extensions=prepared.request.capabilities.mapping_extensions)
        store.append(job_id, "preflight_assessed", {
            "stage": "before_repair",
            "repair_attempt": attempt + 1,
            "script_sha256": hashlib.sha256(result.generation.script.encode("utf-8")).hexdigest(),
            "report": report.to_dict(),
        })
        repair_diagnostics = report.to_dict()
        palette_review_mode = bool(
            attempt > 0 and _has_low_palette_warning(report)
        )
        if palette_review_mode:
            repair_diagnostics["scene_palette_review"] = {
                "requested": True,
                "reason": "low_contrast_warning_persisted_after_general_repair",
            }
        visual_review_only = bool(
            report.passed and not report.warnings and repair_attempts == 0
            and can_review_visually and _repair_retry_capacity_available(budget)
        )
        if report.passed and not report.warnings and not visual_review_only:
            break
        if visual_review_only:
            repair_diagnostics["visual_review"] = True
        rhythm_only_warning = bool(report.warnings) and all(
            isinstance(item, str) and item.startswith("rhythm_density_unresponsive:")
            for item in report.warnings
        )
        if (report.passed and not report.errors and not visual_review_only and rhythm_only_warning
                and not _compact_rhythm_move_candidates(
                    result.generation.script, repair_diagnostics, prepared.plan,
                )):
            warning = (
                "rhythm-density warning remains; no safe nearby note move was found, "
                "so the model repair call was skipped"
            )
            workflow_warnings.append(warning)
            store.append(job_id, "repair_skipped_unactionable_rhythm", {
                "attempt": attempt + 1,
                "reason": "no_safe_nearby_note_moves",
                "script_sha256": hashlib.sha256(
                    result.generation.script.encode("utf-8")).hexdigest(),
                "rhythm_alignment": report.rhythm_alignment,
                "warning": warning,
            })
            break
        if repair_rejection_feedback:
            repair_diagnostics["repair_rejection_feedback"] = repair_rejection_feedback
        repaired = run_generation_job(
            job_id, prepared.request, client, store, budget, repair=True,
            previous=result.generation, diagnostics=repair_diagnostics,
            tool_executor=tool_executor,
            comparison_difficulty=comparison_difficulty)
        latest_attempt = store.latest(job_id)
        repair_usage = (repaired.generation.usage if repaired.generation is not None
                        else (latest_attempt.payload.get("usage", {})
                              if latest_attempt is not None else {}))
        if not isinstance(repair_usage, dict):
            repair_usage = {}
        usage_attempts.append((f"repair_{attempt + 1}", repair_usage))
        total_usage = aggregate_usage(usage_attempts)
        result = replace(
            result,
            generation=replace(result.generation, usage=total_usage),
        )
        if repaired.generation is not None:
            repaired = replace(
                repaired,
                generation=replace(repaired.generation, usage=total_usage),
            )
        repair_attempts += 1
        if repaired.status != "completed" or repaired.generation is None:
            if report.passed:
                if (attempt + 1 < max_repairs
                        and _repair_retry_capacity_available(budget)
                        and _is_repair_invariant_rejection(repaired.error)):
                    repair_rejection_feedback = repaired.error
                    pattern_gain = _rejected_pattern_gain_feedback(
                        store, job_id, repaired.error, prepared.request, report.to_dict(),
                    )
                    if pattern_gain:
                        repair_rejection_feedback += pattern_gain
                    store.append(job_id, "repair_retry_scheduled", {
                        "attempt": attempt + 1,
                        "reason": _repair_rejection_reason(repaired.error),
                        "repair_error": repaired.error,
                        "retry_feedback": repair_rejection_feedback,
                        "retained_script_sha256": hashlib.sha256(
                            result.generation.script.encode("utf-8")).hexdigest(),
                    })
                    continue
                warning = "a repair call failed; the last preflight-passing candidate was retained for review"
                workflow_warnings.append(warning)
                store.append(job_id, "repair_fallback_retained", {
                    "attempt": attempt + 1,
                    "repair_error": repaired.error,
                    "retained_script_sha256": hashlib.sha256(
                        result.generation.script.encode("utf-8")).hexdigest(),
                    "retained_preflight": report.to_dict(),
                    "warning": warning,
                })
                break
            return repaired, None
        candidate_program = program_from_v3(repaired.generation.difficulty, bpm=prepared.plan.bpm)
        if _locked_fingerprint(candidate_program, prepared.plan) != locked_before:
            message = "repair attempted to change a locked section"
            store.append(job_id, "repair_rejected_locked", {"attempt": attempt + 1})
            _record_workflow_failure(
                store, job_id, stage="locked_section_repair", error=message,
                error_type="LockedSectionRepairError",
            )
            return JobResult(job_id, "failed", generation=result.generation,
                             error=message), None
        candidate_report = preflight(
            candidate_program, prepared.plan,
            audio_features=prepared.request.audio_features,
            mapping_extensions=prepared.request.capabilities.mapping_extensions,
        )
        current_script_sha256 = hashlib.sha256(
            result.generation.script.encode("utf-8")).hexdigest()
        candidate_script_sha256 = hashlib.sha256(
            repaired.generation.script.encode("utf-8")).hexdigest()
        if (candidate_script_sha256 == current_script_sha256
                and candidate_report.to_dict() == report.to_dict()):
            if visual_review_only:
                store.append(job_id, "visual_review_completed", {
                    "attempt": attempt + 1,
                    "changed": False,
                    "script_sha256": current_script_sha256,
                    "preflight": candidate_report.to_dict(),
                })
                break
            palette_retry_available = bool(
                not palette_review_mode
                and _has_low_palette_warning(report)
                and attempt + 1 < max_repairs
                and _repair_retry_capacity_available(budget)
            )
            if palette_retry_available:
                store.append(job_id, "repair_retry_scheduled", {
                    "attempt": attempt + 1,
                    "reason": "persistent_low_contrast_palette_only_review",
                    "retained_script_sha256": current_script_sha256,
                    "retained_preflight": report.to_dict(),
                })
                continue
            warning = (
                "a repair returned the unchanged script with unchanged preflight; "
                "no further repair calls were made"
            )
            workflow_warnings.append(warning)
            store.append(job_id, "repair_no_progress", {
                "attempt": attempt + 1,
                "reason": "unchanged_script_and_preflight",
                "retained_script_sha256": current_script_sha256,
                "candidate_script_sha256": candidate_script_sha256,
                "retained_preflight": report.to_dict(),
                "candidate_preflight": candidate_report.to_dict(),
                "warning": warning,
            })
            break
        regression = _pattern_alignment_regression(
            report.to_dict(), candidate_report.to_dict(),
        )
        if regression:
            retained_hash = hashlib.sha256(
                result.generation.script.encode("utf-8")).hexdigest()
            store.append(job_id, "repair_rejected_pattern_regression", {
                "attempt": attempt + 1,
                "repair_error": regression,
                "retained_script_sha256": retained_hash,
                "prior_preflight": report.to_dict(),
                "candidate_preflight": candidate_report.to_dict(),
            })
            if (attempt + 1 < max_repairs
                    and _repair_retry_capacity_available(budget)):
                repair_rejection_feedback = regression
                store.append(job_id, "repair_retry_scheduled", {
                    "attempt": attempt + 1,
                    "reason": "pattern_alignment_regression",
                    "repair_error": regression,
                    "retained_script_sha256": retained_hash,
                })
                continue
            warning = (
                "a repair worsened pattern alignment; the last preflight-passing "
                "candidate was retained for review"
            )
            workflow_warnings.append(warning)
            store.append(job_id, "repair_fallback_retained", {
                "attempt": attempt + 1,
                "repair_error": regression,
                "retained_script_sha256": retained_hash,
                "retained_preflight": report.to_dict(),
                "rejected_preflight": candidate_report.to_dict(),
                "warning": warning,
            })
            break
        comparison_difficulty = result.generation.difficulty
        result = repaired
        current_program = candidate_program
        repair_rejection_feedback = None
        if visual_review_only:
            store.append(job_id, "visual_review_completed", {
                "attempt": attempt + 1,
                "changed": candidate_script_sha256 != current_script_sha256,
                "prior_script_sha256": current_script_sha256,
                "script_sha256": candidate_script_sha256,
                "preflight": candidate_report.to_dict(),
            })
    final_report = preflight(current_program, prepared.plan,
                             audio_features=prepared.request.audio_features,
                             mapping_extensions=prepared.request.capabilities.mapping_extensions)
    store.append(job_id, "preflight_assessed", {
        "stage": "final",
        "repair_attempts": repair_attempts,
        "script_sha256": hashlib.sha256(result.generation.script.encode("utf-8")).hexdigest(),
        "report": final_report.to_dict(),
    })
    if not final_report.passed:
        details = list(final_report.errors) + list(final_report.warnings)
        message = "preflight remains unresolved after generation/repairs: " + "; ".join(details)
        store.append(job_id, "preflight_rejected", {
            "errors": list(final_report.errors),
            "warnings": list(final_report.warnings),
            "note_count": final_report.note_count,
            "chain_slices": final_report.chain_slices,
            "report": final_report.to_dict(),
        })
        _record_workflow_failure(
            store, job_id, stage="preflight", error=message,
            error_type="QualityGateError", details=final_report.to_dict(),
        )
        return JobResult(job_id, "failed", generation=result.generation,
                         error=message), None
    warnings = tuple(dict.fromkeys((*final_report.warnings, *workflow_warnings)))
    if warnings:
        store.append(job_id, "preflight_warning", {
            "warnings": list(warnings),
            "note_count": final_report.note_count,
            "chain_slices": final_report.chain_slices,
            "report": final_report.to_dict(),
        })
    try:
        runtime_bundle = None
        if vivify_bundle_config is not None:
            if not prepared.request.capabilities.vivify:
                raise ValueError("Vivify bundle building requires capabilities.vivify")
            instantiate_ids = {
                event.data.get("asset_id") for event in current_program.custom_events
                if event.event_type == "InstantiatePrefab"
                and isinstance(event.data.get("asset_id"), str)
            }
            if not instantiate_ids:
                raise ValueError(
                    "Vivify bundle mode requires an InstantiatePrefab event with a selected asset_id")
            if asset_library is None:
                raise ValueError("Vivify bundle mode requires an asset library")
            selected_ids = set(selected_asset_ids)
            missing = sorted(instantiate_ids - selected_ids)
            if missing:
                raise ValueError("InstantiatePrefab references unselected assets: " + ", ".join(missing))
            models = []
            for asset_id in sorted(instantiate_ids):
                record = asset_library.get(asset_id)
                if record is None:
                    raise ValueError(f"unknown Vivify asset ID: {asset_id}")
                archive = Path(record.local_path)
                if archive.suffix.lower() != ".zip":
                    raise ValueError(
                        f"Vivify asset {asset_id} is not a source ZIP containing an FBX or OBJ")
                models.append(VivifyModelInput(asset_id, archive))
            editor = vivify_bundle_config.get("unity_editor") or unity_executable_from_environment()
            if editor is None:
                raise ValueError(
                    "Vivify bundle mode needs Unity 2021.3.16f1; set --unity-editor or BEATFORGE_UNITY_EDITOR")
            job_root = Path(jobs_root) / job_id
            workspace = Path(vivify_bundle_config.get("workspace") or job_root / "vivify-build")
            template_source = vivify_bundle_config.get("template_source")
            if template_source is None:
                template_source = clone_pinned_template(job_root / "vivify-template-source")
            built = build_vivify_bundles(
                models,
                unity_editor=editor,
                template_source=template_source,
                workspace=workspace,
                timeout_seconds=int(vivify_bundle_config.get("timeout_seconds", 900)),
                max_asset_mb=prepared.request.capabilities.max_asset_mb,
            )
            runtime_bundle = {
                "bundle_path": str(workspace / "bundle-output" / built["bundle_file"]),
                "manifest": built,
            }
        preflight_data = final_report.to_dict()
        if workflow_warnings:
            preflight_data["workflow_warnings"] = list(workflow_warnings)
        package = export_v3(output, current_program, metadata, prepared.plan, prepared.request.capabilities,
                            audio_path=prepared.request.audio_path,
                            asset_library=asset_library, asset_ids=selected_asset_ids,
                            zip_path=zip_path,
                            vivify_runtime_bundle=runtime_bundle,
                            preflight_report=preflight_data)
    except Exception as exc:
        store.append(job_id, "export_failed", {
            "error_type": type(exc).__name__, "error": str(exc),
        })
        _record_workflow_failure(
            store, job_id, stage="export", error=str(exc),
            error_type=type(exc).__name__,
        )
        return JobResult(job_id, "failed", generation=result.generation, error=str(exc)), None
    verification = verify_package(package)
    if not verification.passed:
        store.append(job_id, "export_verification_failed", verification.to_dict())
        message = "; ".join(verification.errors)
        _record_workflow_failure(
            store, job_id, stage="package_verification", error=message,
            error_type="PackageVerificationError", details=verification.to_dict(),
        )
        return JobResult(job_id, "failed", generation=result.generation,
                         error=message), None
    final_status = "completed_with_warnings" if warnings else "completed"
    store.append(job_id, "export_completed", {
        "output": str(package), "status": final_status,
        "warnings": list(warnings), "verification": verification.to_dict(),
    })
    live = store.live(job_id) or {}
    live_payload = {key: value for key, value in live.items()
                    if key not in {"status", "recorded_at"}}
    live_payload.update({
        "output": str(package), "warnings": list(warnings),
        "preflight": final_report.to_dict(), "verification": verification.to_dict(),
        "usage": result.generation.usage,
    })
    store.write_live(job_id, final_status, live_payload)
    return JobResult(job_id, final_status, generation=result.generation,
                     warnings=warnings), package

