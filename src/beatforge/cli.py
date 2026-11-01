"""Small local CLI for the first BeatForge vertical slice."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

from .budget import BudgetGuard, DEFAULT_MAX_REQUESTS
from .beatgrid import normalize_beat_grid
from .capabilities import CapabilityProfile
from .generation import validate_request
from .job_store import JobStore
from .local_server import LocalServerSpec, launch_local_server
from .assets import AssetLibrary
from .browser import (BrowserSessionSpec, PlaywrightBrowserSession,
                      discover_anti_debug_extension)
from .openrouter import LocalOpenAIClient, OpenRouterClient
from .packaging import ExportMetadata
from .planning import load_plan
from .preview import write_preview
from .script_runner import run_script
from .server import serve_preview
from .workflow import generate_and_export, prepare_generation
from .polyhaven import POLYHAVEN_ALLOWED_DOMAINS, PolyHavenBrowserSearch
from .vivify_tools import VivifyToolExecutor
from .vivify_bundle import (VivifyBundleBuildError, VivifyModelInput,
                            build_vivify_bundles, clone_pinned_template,
                            unity_executable_from_environment)
from .verify import verify_package


def _parser(*, include_vivify: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beatforge")
    commands = parser.add_subparsers(dest="command", required=True)
    script = commands.add_parser("compile-script", help="compile a safe typed map script to V3 JSON")
    script.add_argument("source", type=Path)
    script.add_argument("--bpm", type=float, required=True)
    script.add_argument("--out", type=Path, required=True)
    script.add_argument("--mapping-extensions", action="store_true")
    preview = commands.add_parser("preview", help="render a compiled difficulty JSON to HTML")
    preview.add_argument("difficulty", type=Path)
    preview.add_argument("--out", type=Path, required=True)
    preview.add_argument("--title", default="BeatForge preview")
    preview.add_argument("--mapping-extensions", action="store_true")
    server = commands.add_parser("serve-preview", help="serve a difficulty preview on localhost")
    server.add_argument("difficulty", type=Path)
    server.add_argument("--title", default="BeatForge preview")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--mapping-extensions", action="store_true")
    generate = commands.add_parser("generate", help="analyze audio, call the selected provider, and export V3")
    generate.add_argument("--audio", type=Path, required=True)
    generate.add_argument("--bpm", type=float)
    generate.add_argument("--target-nps", type=float)
    generate.add_argument("--plan-in", type=Path, help="use a reviewed TrackPlan JSON")
    generate.add_argument("--beat-grid", type=Path,
                          help="optional normalized BeatGrid JSON for section boundaries")
    generate.add_argument("--model", required=True,
                          help="model id; OpenRouter requires a :free variant or the official openrouter/free router")
    generate.add_argument("--provider", choices=("openrouter", "local"), default="openrouter",
                          help="remote free OpenRouter or loopback OpenAI-compatible server")
    generate.add_argument("--endpoint", default=None,
                          help="API base URL; local provider must be loopback")
    generate.add_argument("--timeout-seconds", type=float, default=120.0,
                          help="provider HTTP timeout for each model call")
    generate.add_argument("--send-audio", action="store_true",
                          help="send raw audio to an audio-capable free model; may require provider credit")
    generate.add_argument("--send-spectrograms", action="store_true",
                          help="send the prepared 44.1 kHz SG panels to a local vision-capable model")
    generate.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"),
                          help="provider reasoning effort; passed through when supported")
    generate.add_argument("--spectrogram-image-limit", type=int,
                          help="optional provider image-count limit; stacks consecutive SG panels vertically without resampling")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--zip", dest="zip_path", type=Path)
    generate.add_argument("--song-name", required=True)
    generate.add_argument("--song-author", default="")
    generate.add_argument("--mapper", default="BeatForge")
    generate.add_argument("--platform", choices=("pc", "quest"), default="pc")
    if include_vivify:
        generate.add_argument("--vivify", action="store_true")
        generate.add_argument("--vivify-build-bundles", action="store_true",
                              help="compile imported Vivify assets into a real Windows2021 bundle with Unity")
        generate.add_argument("--unity-editor", type=Path,
                              help="Unity 2021.3.16f1 Editor; defaults to BEATFORGE_UNITY_EDITOR")
        generate.add_argument("--vivify-template-source", type=Path,
                              help="local pinned VivifyTemplate checkout; fetched when omitted")
        generate.add_argument("--vivify-build-timeout", type=int, default=900,
                              help="Unity batch build timeout in seconds")
    generate.add_argument("--chroma", action="store_true")
    generate.add_argument("--noodle", action="store_true")
    generate.add_argument("--mapping-extensions", action="store_true")
    if include_vivify:
        generate.add_argument("--asset-library", type=Path,
                              help="catalog JSON used for selected Vivify assets")
        generate.add_argument("--asset-id", action="append", default=[],
                              help="catalogued asset ID to include; repeat for multiple assets")
        generate.add_argument("--vivify-browser", action="store_true",
                              help="enable bounded Poly Haven search tools in isolated Chromium")
        generate.add_argument("--anti-debug-extension", type=Path,
                              help="unpacked Anti Anti Debug extension directory; auto-detected when omitted")
    else:
        generate.set_defaults(vivify=False, vivify_build_bundles=False,
                              unity_editor=None, vivify_template_source=None,
                              vivify_build_timeout=900, asset_library=None,
                              asset_id=[], vivify_browser=False,
                              anti_debug_extension=None)
    generate.add_argument("--jobs", type=Path, default=Path("jobs"))
    generate.add_argument("--job-id")
    generate.add_argument("--plan-out", type=Path)
    generate.add_argument("--dry-run", action="store_true",
                          help="write the local plan and stop before any model call")
    generate.add_argument("--repairs", type=int, choices=(0, 1, 2, 3), default=3,
                          help="bounded repair calls; defaults to one general pass and up to two palette-only retries for persistent low contrast")
    generate.add_argument("--max-requests", type=int, choices=range(1, 33),
                          default=DEFAULT_MAX_REQUESTS,
                          help="maximum provider calls for generation, tools, recovery, and repairs (1..32)")
    generate.add_argument("--api-key")
    verify = commands.add_parser("verify", help="verify an exported directory or ZIP without executing it")
    verify.add_argument("package", type=Path)
    if include_vivify:
        vivify_build = commands.add_parser(
            "vivify-build", help="build a real Windows Vivify AssetBundle from model archives using Unity")
        vivify_build.add_argument("--asset", action="append", required=True, metavar="ID=ZIP",
                                  help="catalog asset ID and downloaded FBX ZIP; repeat for multiple assets")
        vivify_build.add_argument("--unity-editor", type=Path,
                                  help="Unity 2021.3.16f1 Editor executable; defaults to BEATFORGE_UNITY_EDITOR")
        vivify_build.add_argument("--template-source", type=Path,
                                  help="local VivifyTemplate checkout; pinned source is fetched when omitted")
        vivify_build.add_argument("--workspace", type=Path, required=True,
                                  help="new empty directory for the isolated Unity project and evidence")
        vivify_build.add_argument("--timeout-seconds", type=int, default=900)
        vivify_build.add_argument("--max-asset-mb", type=int, default=64)
    status = commands.add_parser("status", help="show the live snapshot and latest event for a job")
    status.add_argument("job_id")
    status.add_argument("--jobs", type=Path, default=Path("jobs"))
    local = commands.add_parser("serve-local", help="start a verified llama.cpp server on loopback")
    local.add_argument("--binary", type=Path, required=True,
                       help="path to llama-server executable")
    local.add_argument("--model", type=Path, required=True,
                       help="path to a local GGUF model")
    local.add_argument("--mmproj", type=Path,
                       help="path to a local llama.cpp multimodal projector GGUF")
    local.add_argument("--host", default="127.0.0.1")
    local.add_argument("--port", type=int, default=8080)
    local.add_argument("--ctx-size", type=int, default=8192)
    local.add_argument("--threads", type=int)
    probe = commands.add_parser("probe", help="probe an OpenRouter or local provider /models endpoint")
    probe.add_argument("--provider", choices=("openrouter", "local"), default="local")
    probe.add_argument("--endpoint", default=None)
    probe.add_argument("--timeout-seconds", type=float, default=30.0)
    probe.add_argument("--api-key")
    return parser


def main(argv: list[str] | None = None, *, include_vivify: bool = True) -> int:
    args = _parser(include_vivify=include_vivify).parse_args(argv)
    if args.command == "compile-script":
        result = run_script(args.source.read_text(encoding="utf-8"), bpm=args.bpm,
                            mapping_extensions=args.mapping_extensions)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    if args.command == "preview":
        difficulty = json.loads(args.difficulty.read_text(encoding="utf-8"))
        write_preview(args.out, difficulty, title=args.title,
                      mapping_extensions=args.mapping_extensions)
        return 0
    if args.command == "serve-preview":
        difficulty = json.loads(args.difficulty.read_text(encoding="utf-8"))
        serve_preview(difficulty, title=args.title, host=args.host, port=args.port,
                      mapping_extensions=args.mapping_extensions)
        return 0
    if args.command == "verify":
        report = verify_package(args.package)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if report.passed else 1
    if args.command == "vivify-build":
        try:
            editor = args.unity_editor or unity_executable_from_environment()
            if editor is None:
                raise FileNotFoundError(
                    "Set --unity-editor or BEATFORGE_UNITY_EDITOR to the installed Unity 2021.3.16f1 Editor")
            if not Path(editor).is_file():
                raise FileNotFoundError(f"Unity Editor executable not found: {Path(editor).resolve()}")
            workspace = args.workspace.resolve()
            if workspace.exists() and any(workspace.iterdir()):
                raise FileExistsError(f"Vivify build workspace must be empty: {workspace}")
            template = args.template_source
            if template is None:
                template_cache = workspace.with_name(workspace.name + "-vivifytemplate")
                template = clone_pinned_template(template_cache)
            models = []
            for value in args.asset:
                if "=" not in value:
                    raise ValueError("--asset must use ID=ZIP syntax")
                asset_id, archive = value.split("=", 1)
                models.append(VivifyModelInput(asset_id, Path(archive)))
            result = build_vivify_bundles(
                models, unity_editor=editor, template_source=template,
                workspace=workspace, timeout_seconds=args.timeout_seconds,
                max_asset_mb=args.max_asset_mb,
            )
            print(json.dumps({"status": "built", "manifest": result}, ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 1
    if args.command == "status":
        store = JobStore(args.jobs)
        live = store.live(args.job_id)
        latest = store.latest(args.job_id)
        if live is None and latest is None:
            print(json.dumps({"status": "unknown", "job_id": args.job_id}, ensure_ascii=False))
            return 1
        print(json.dumps({
            "job_id": args.job_id,
            "live": live,
            "latest": ({"kind": latest.kind, "payload": latest.payload,
                        "recorded_at": latest.recorded_at} if latest else None),
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "serve-local":
        try:
            spec = LocalServerSpec(binary=args.binary, model=args.model, host=args.host,
                                   port=args.port, context_size=args.ctx_size,
                                   threads=args.threads, mmproj=args.mmproj)
            process = launch_local_server(spec)
            print(json.dumps({"status": "started", "pid": getattr(process, "pid", None),
                              "endpoint": f"http://{spec.host}:{spec.port}/v1",
                              "command": spec.command()}, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False),
                  file=sys.stderr)
            return 1
    if args.command == "probe":
        try:
            client = (OpenRouterClient(api_key=args.api_key, **(
                       {"base_url": args.endpoint, "timeout_seconds": args.timeout_seconds}
                       if args.endpoint else {"timeout_seconds": args.timeout_seconds}))
                      if args.provider == "openrouter" else
                      LocalOpenAIClient(api_key=args.api_key,
                                        base_url=args.endpoint or "http://127.0.0.1:8080/v1",
                                        timeout_seconds=args.timeout_seconds))
            models = client.catalog()
            print(json.dumps({"status": "ok", "provider": args.provider,
                              "model_count": len(models), "models": models},
                             ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "failed", "provider": args.provider,
                              "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 1
    if args.command == "generate":
        try:
            if args.send_spectrograms and args.provider != "local":
                raise ValueError("--send-spectrograms is for the local provider; OpenRouter image support is detected from its free catalog")
            capabilities = CapabilityProfile(platform=args.platform, chroma=args.chroma,
                                             noodle=args.noodle,
                                             mapping_extensions=args.mapping_extensions,
                                             vivify=args.vivify)
            prepared = prepare_generation(args.audio, bpm=args.bpm,
                                          target_nps=args.target_nps,
                                          model_id=args.model,
                                          provider=args.provider,
                                          reasoning_effort=args.reasoning_effort,
                                          track_title=args.song_name,
                                          track_artist=args.song_author,
                                          send_audio=args.send_audio,
                                          send_spectrograms=args.send_spectrograms,
                                          asset_ids=tuple(args.asset_id),
                                          capabilities=capabilities,
                                          plan=load_plan(args.plan_in) if args.plan_in else None,
                                          spectrogram_image_limit=args.spectrogram_image_limit,
                                          beat_grid=(normalize_beat_grid(
                                              json.loads(args.beat_grid.read_text(encoding="utf-8")),
                                              source="beat-grid-json") if args.beat_grid else None))
            plan_path = args.plan_out
            if plan_path is None and not args.plan_in:
                plan_path = args.out / "track-plan.json"
            if plan_path is not None:
                plan_path.parent.mkdir(parents=True, exist_ok=True)
                plan_path.write_text(prepared.plan.to_json(), encoding="utf-8")
            if args.dry_run:
                print(json.dumps({"status": "plan_ready", "plan": str(plan_path) if plan_path else str(args.plan_in),
                                  "sections": len(prepared.plan.sections)}, ensure_ascii=False))
                return 0
            if not args.job_id:
                raise ValueError("--job-id is required unless --dry-run is used")
            if args.vivify_browser and not args.vivify:
                raise ValueError("--vivify-browser requires --vivify")
            if args.vivify_build_bundles and not args.vivify:
                raise ValueError("--vivify-build-bundles requires --vivify")
            if args.vivify_build_bundles and args.vivify_build_timeout < 60:
                raise ValueError("--vivify-build-timeout must be at least 60 seconds")
            vivify_bundle_config = None
            if args.vivify_build_bundles:
                unity_editor = args.unity_editor or unity_executable_from_environment()
                if unity_editor is None or not Path(unity_editor).is_file():
                    raise ValueError(
                        "Vivify bundle mode needs Unity 2021.3.16f1; set --unity-editor "
                        "or BEATFORGE_UNITY_EDITOR to an existing Editor executable")
                template_source = args.vivify_template_source
                if template_source is None:
                    template_source = clone_pinned_template(
                        args.jobs / args.job_id / "vivify-template-source")
                vivify_bundle_config = {
                    "unity_editor": unity_editor,
                    "template_source": template_source,
                    "timeout_seconds": args.vivify_build_timeout,
                }
            if args.vivify_browser and args.provider != "openrouter":
                raise ValueError("--vivify-browser currently requires OpenRouter tool-capability metadata")
            errors = validate_request(prepared.request)
            if errors:
                raise ValueError("; ".join(errors))
            client = (OpenRouterClient(api_key=args.api_key, **(
                               {"base_url": args.endpoint, "timeout_seconds": args.timeout_seconds}
                               if args.endpoint else {"timeout_seconds": args.timeout_seconds}))
                           if args.provider == "openrouter" else
                           LocalOpenAIClient(api_key=args.api_key,
                                             base_url=args.endpoint or "http://127.0.0.1:8080/v1",
                                             timeout_seconds=args.timeout_seconds))
            asset_library_path = args.asset_library
            if asset_library_path is None and args.vivify_browser:
                asset_library_path = args.jobs / args.job_id / "vivify-assets-library.json"
            asset_library = AssetLibrary(asset_library_path) if asset_library_path else None
            browser_context = nullcontext(None)
            if args.vivify_browser:
                extension = args.anti_debug_extension or Path(discover_anti_debug_extension())
                profile = args.jobs / args.job_id / "vivify-browser-profile"
                browser_spec = BrowserSessionSpec(
                    None, str(profile), str(extension),
                    allowed_domains=POLYHAVEN_ALLOWED_DOMAINS,
                    start_urls=("https://polyhaven.com/models",),
                )
                browser_context = PlaywrightBrowserSession(browser_spec)
            with browser_context as browser_session:
                tool_executor = None
                if browser_session is not None:
                    tool_executor = VivifyToolExecutor(
                        PolyHavenBrowserSearch(browser_session), asset_library,
                        (asset_library_path.parent / "vivify-objects"),
                    )
                job, package = generate_and_export(
                    prepared, client,
                    job_id=args.job_id, jobs_root=args.jobs, output=args.out,
                    metadata=ExportMetadata(args.song_name, song_author=args.song_author,
                                            mapper=args.mapper),
                    budget=BudgetGuard(max_requests=args.max_requests,
                                       max_repairs=args.repairs),
                    asset_library=asset_library,
                    asset_ids=tuple(args.asset_id), tool_executor=tool_executor,
                    zip_path=args.zip_path, max_repairs=args.repairs,
                    vivify_bundle_config=vivify_bundle_config)
            if job.status not in {"completed", "completed_with_warnings"}:
                print(json.dumps({"status": job.status, "error": job.error}, ensure_ascii=False), file=sys.stderr)
                return 1
            print(json.dumps({"status": job.status,
                              "review_status": "needs_review" if job.warnings else "passed",
                              "warnings": list(job.warnings), "output": str(package),
                              "job_id": job.job_id}, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 1
    return 2


def release_main(argv: list[str] | None = None) -> int:
    """Release entry point with the temporarily hidden Vivify controls."""
    return main(argv, include_vivify=False)


if __name__ == "__main__":
    sys.exit(main())
