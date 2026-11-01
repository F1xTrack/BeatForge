"""BeatForge typed map-authoring core."""

from .contracts import Arc, Bomb, Chain, CustomEvent, LightEvent, MapProgram, Note, Wall
from .compiler import compile_program
from .validation import ValidationError, validate_program
from .analysis import EnergyWindow, SongAnalysis, analyze_audio
from .planning import (Section, TrackPlan, initial_plan_from_analysis,
                        initial_plan_from_beat_grid, load_plan, validate_plan)
from .providers import FreeModel, free_catalog, parse_free_model, select_free_model
from .job_store import JobEvent, JobStore
from .assets import AssetLibrary, AssetRecord
from .script_runner import ScriptRejected, run_script
from .capabilities import CapabilityProfile, effective_capabilities, validate_capabilities
from .generation import (REASONING_EFFORTS, GenerationRequest, GenerationResult,
                         ScriptGenerationError, build_script_prompt, generate_map, normalize_model_script,
                         repair_map, validate_request)
from .openrouter import ChatResult, LocalOpenAIClient, OpenRouterClient
from .preview import render_preview_html, write_preview
from .packaging import ExportMetadata, build_info, export_v3
from .server import create_preview_server, serve_preview
from .local_server import LocalServerSpec, launch_local_server
from .browser import (BrowserSessionSpec, PlaywrightBrowserSession, launch_browser,
                      discover_anti_debug_extension, discover_chrome_executable)
from .worker import JobResult, run_generation_job
from .budget import BudgetGuard
from .quality import QualityReport, preflight
from .beatgrid import BeatGrid, normalize_beat_grid
from .asset_search import AssetCandidate, rank_candidates, validate_candidate
from .asset_fetch import candidate_asset_id, fetch_candidate
from .polyhaven import PolyHavenBrowserSearch, POLYHAVEN_ALLOWED_DOMAINS
from .vivify_tools import VivifyToolExecutor, VIVIFY_TOOL_SCHEMAS
from .vivify import build_vivify_manifest, referenced_asset_ids, validate_asset_references
from .adapters import program_from_v3
from .workflow import PreparedGeneration, generate_and_export, prepare_generation
from .verify import VerificationReport, verify_package
from .editor import render_editor_html
from .runtime import HANDOFF_FORMAT, build_runtime_handoff

__all__ = [
    "Arc", "Bomb", "Chain", "CustomEvent", "LightEvent", "MapProgram", "Note", "Wall",
    "ValidationError", "compile_program", "validate_program",
    "EnergyWindow", "SongAnalysis", "analyze_audio",
    "Section", "TrackPlan", "initial_plan_from_analysis", "initial_plan_from_beat_grid",
    "load_plan", "validate_plan", "FreeModel", "free_catalog",
    "parse_free_model", "select_free_model", "JobEvent", "JobStore",
    "AssetLibrary", "AssetRecord",
    "ScriptRejected", "run_script",
    "CapabilityProfile", "effective_capabilities", "validate_capabilities",
    "REASONING_EFFORTS", "GenerationRequest", "GenerationResult", "ScriptGenerationError", "build_script_prompt", "generate_map", "normalize_model_script", "repair_map", "validate_request",
    "ChatResult", "OpenRouterClient", "LocalOpenAIClient",
    "render_preview_html", "write_preview",
    "ExportMetadata", "build_info", "export_v3",
    "create_preview_server", "serve_preview",
    "LocalServerSpec", "launch_local_server",
    "BrowserSessionSpec",
    "PlaywrightBrowserSession", "launch_browser", "discover_anti_debug_extension",
    "discover_chrome_executable",
    "JobResult", "run_generation_job",
    "BudgetGuard",
    "QualityReport", "preflight",
    "BeatGrid", "normalize_beat_grid",
    "AssetCandidate", "rank_candidates", "validate_candidate",
    "candidate_asset_id", "fetch_candidate",
    "PolyHavenBrowserSearch", "POLYHAVEN_ALLOWED_DOMAINS",
    "VivifyToolExecutor", "VIVIFY_TOOL_SCHEMAS",
    "build_vivify_manifest",
    "referenced_asset_ids", "validate_asset_references",
    "program_from_v3", "PreparedGeneration", "prepare_generation", "generate_and_export",
    "VerificationReport", "verify_package",
    "render_editor_html",
    "HANDOFF_FORMAT", "build_runtime_handoff",
]
