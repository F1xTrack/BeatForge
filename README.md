# BeatForge

> **Release note:** the public `v0.1.0` CLI intentionally omits Vivify options.
> The implementation remains in the source tree and is exposed on the
> `vivify-beta` branch. See [README.release.md](README.release.md) for the
> installed release quickstart.

BeatForge is the authoring environment planned alongside BeatBench. The first
vertical slice is deliberately deterministic and model-agnostic:

1. a typed Python SDK describes notes, arcs, chains, walls and lights;
2. the compiler emits a canonical Beat Saber V3 difficulty file;
3. validation catches geometry, timing and duplicate-cell mistakes before export.

The next layer now includes a reviewable `TrackPlan`, a zero-cost development
provider guard, and an append-only JSONL job event store (`events.jsonl` plus a
human-readable `latest.json`). These pieces are intentionally independent of
any particular model or network client.

`AssetLibrary` records source URL, author, license and SHA-256 before an asset
can enter the local cache. The default import allowlist is CC0/CC-BY; browser
search, data-only exports and the optional Vivify/Unity bundle builder use this catalog.

`run_script()` accepts direct calls such as `map.note(...)`, `map.arc(...)`,
`map.chain(...)` and literal `map.custom_event(...)` calls. For compact repeated
phrases, `map.repeat(start_beat, repetitions, step_beats, event_template)`
expands literal event dictionaries whose `beat` and `tail_beat` values are
relative to the phrase start. Repeats are bounded to 512 iterations, 256 events
per template and 10,000 expanded objects. It remains data-only: the runner
rejects imports, loops, arbitrary expressions and oversized input. Custom events
remain the typed escape hatch for deliberate Chroma/Noodle/Vivify effects.

`map.arc(...)` authors the slider link and its two playable endpoint notes as one
SDK object. Set `tail_direction` for the ending note; the V3 exporter writes both
endpoint notes into `colorNotes` alongside the slider in `sliders`. The two notes
count as two hit-equivalents; the arc link adds no extra hit. Keep them implicit
in the DSL and do not emit duplicate `map.note(...)` calls at either endpoint.

When Chroma is enabled, a model or author can set scene colors with
`map.palette(environment_left=[R,G,B], environment_right=[R,G,B])`; optional
`saber_left` and `saber_right` triples set saber colors. Channels are integers
from 0 to 255. The exporter converts these values to normalized RGB objects in
the difficulty profile's `Info.dat` custom data and removes BeatForge's internal
transport marker from the V3 difficulty. If a reviewed plan already contains a
palette, that plan takes precedence and a conflicting script palette is rejected.
Generated colors are model-selected from the supplied musical context; their
theme fit and in-game rendering still require review in Beat Saber.

`CapabilityProfile` keeps Chroma, Noodle Extensions, Mapping Extensions and
Vivify as independent toggles. Vivify is rejected for Quest until a verified
Quest runtime target exists; the profile is passed explicitly to generation.

With Mapping Extensions enabled, BeatForge supports six-lane positions,
precision placement, rotated cut directions, bombs and extended walls. Encoded
coordinates are retained as integers in V3 JSON; the diagnostic preview decodes
their positions for display. Encoded positions use `1000 + round(position ×
1000)` for non-negative coordinates and `-1000 + round(position × 1000)` for
negative coordinates. Thus `1000` and `-1000` encode zero; values from `-999`
through `999` are ordinary integer indices. ME cut directions `1000..1360`
encode 360-degree rotations and `2000..2360` encode 360-degree rotations with
Any Direction. Walls may use extra horizontal lanes and non-negative vertical
layers; their dimensions support the ME precision encodings. Without the
capability, standard positions remain bounded to `x=0..3`, `y=0..2` and standard
cut directions `0..8`. Exported `Info.dat` declares the Mapping Extensions
requirement, and package verification reads that requirement before validating
these features.

`GenerationRequest` combines the reviewed plan, selected sections, capabilities,
reasoning effort, allowed Vivify asset IDs, provider and model id. OpenRouter
generation accepts catalog-verified zero-price `:free` model variants and the
exact official `openrouter/free` router id; the optional local provider is
restricted to a loopback OpenAI-compatible endpoint. For `openrouter/free`,
BeatForge requests OpenRouter routing metadata and records the selected model
and provider in generation usage when supplied. If the route metadata is absent,
the result retains the requested router alias.

`OpenRouterClient` does not substitute models for fixed `:free` ids; only the
explicit `openrouter/free` id delegates model selection to OpenRouter.
`LocalOpenAIClient` is the equivalent no-key loopback transport for llama.cpp
and similar local servers;
`generate_map()` sends one prompt, receives the typed script, compiles it and
returns model usage alongside the V3 result.
Before inference, BeatForge extracts a compact, beat-aligned half-beat grid of
local energy, frequency-band levels and spectral attacks. That measured summary
is sent to text models; it does not claim to identify instruments or notes.
BeatForge also creates the same fixed spectrogram representation as BeatBench
SG: one complete-track overview plus consecutive four-second panels, 44.1 kHz
mono resampling, 1024-point STFT, 128-sample hop, log-frequency axis, magma
palette, and a shared -80 to 0 dB scale. A manifest records the exact settings,
panel times and SHA-256 hashes. Free OpenRouter models that advertise image
input receive these PNGs as base64 image parts; text-only models fall back to
the measured feature grid. The source audio is not sent in this mode, and
native audio input is not required. Local vision-capable models can receive
the same panels with `--send-spectrograms`; this is off by default because
llama.cpp GGUF backends vary in image support. `--send-audio` remains a separate opt-in.

During a diagnostic repair that considers adding an arc, BeatForge includes up
to four of its strongest unverified sustained-ridge candidates in the repair
prompt. It attaches the complete-track overview and prioritizes the four-second
panels that cover those candidate intervals, then uses any remaining slots for
other diagnosed repair windows, with a limit of 12 detail panels. For example,
a candidate spanning 101.8–110.6 seconds points the model to the panels around
100–112 seconds. The signal scan is only a navigation hint: the model must
confirm a clear, continuous musical line in the SG before authoring an arc.
BeatForge does not add arcs automatically or require an arc count.
Preflight also reports rapid same-hand runs as chain-review candidates without
requiring conversion. When a repair reviews them, it attaches the synchronized
SG panels around the exact beat range and asks the model to convert only a
clear, phrase-local burst, preserving its hit-equivalent count; unsupported runs
remain ordinary notes with the review warning intact. Sparse-lighting coverage
is measured across plan sections that actually contain playable objects. If
fewer than half of those active sections contain a standard-light cue, preflight
shows the missing sections and the repair receives their SG windows. This is a
phrase-coverage review signal, not a fixed event-count quota or a demand to light
quiet passages.
Repair prompts also omit the duplicated full-track pattern/rhythm tables after
their flagged windows have been summarized; the complete diagnostics remain in
the job event log for review.

Raw audio is opt-in with `--send-audio`. In that mode BeatForge checks the live
OpenRouter catalog and requires a `:free` model with zero text-token prices
that explicitly advertises audio input, then sends base64 `input_audio` data.
Some providers may still require a minimum account balance for audio, even for
a model listed as free, so omit `--send-audio` when testing without credits.
The local llama.cpp transport always receives the signal summary and is told
it cannot hear the local audio file. With `--send-spectrograms`, it also receives
the same 44.1 kHz overview and four-second panels as BeatBench SG.
`prepare_generation()` performs local audio analysis and creates the editable
plan before inference. `generate_and_export()` wraps the call in `JobStore` /
`BudgetGuard`, converts the result back through the typed adapter, and exports
only a completed V3 package.
Use `--plan-in reviewed-plan.json` to regenerate from a human-edited plan;
without it, `--dry-run` writes a new plan and stops before the provider call.
`--repairs 0..3` enables a bounded diagnostics-driven repair loop; the default
is three. If a general pass leaves low Chroma palette contrast unresolved, up to
two following passes change only the palette, require the measured contrast warning
to clear, and verify that every other map field stays identical. A rejected palette
is retained only as an error candidate and its measured failure is sent to the next
bounded palette review. Every repair is budgeted and
locked-section changes fail closed. When a repair is rejected
by the deterministic hit-count or untargeted arc/chain preservation guard, the
workflow can spend a remaining repair attempt and include the exact rejection
reason in its next prompt. A repair rejected by the safe script parser or
enabled-capability validator gets the same bounded retry with its exact error,
while the last accepted map stays available as the fallback. If a rejected count repair also improved an active
pattern-alignment or section-coverage diagnostic, the retry receives those
measured gains and is asked to keep SG-supported edits while restoring the exact
hit count. Provider and transport failures do not trigger this
targeted retry; the last preflight-passing map is retained with a warning.
One exception is an OpenRouter response that exhausts the provider's output
ceiling with no map and reports the completion entirely as reasoning tokens:
if automatic-repair budget remains, BeatForge makes one recovery call with
`reasoning=low`. It does not add a `max_tokens` cap, and the live snapshot and
event log retain both attempts and their combined usage. Other empty responses
and exhausted recovery budgets remain failures.
Hit-count guard feedback includes the prior/repaired object inventory and at
most six missing or extra note heads, so a later repair can target the exact
omissions without repeating the full map in the error message. Untargeted arc
and chain preservation guards likewise include up to six exact missing and
added V3 objects, including their head/tail geometry, so a bounded retry can
restore shifted sustained objects directly. When an active audio-pattern warning
gets worse after repair, the workflow rejects that candidate, gives the exact
support change to the remaining bounded repair, and retains the earlier passing
map if no retry remains. This uses the diagnostic as a repair guard, not as a
gameplay score. If a repair returns the exact same script and exactly the same
preflight report, BeatForge records `repair_no_progress`, retains that candidate
and its warning, and skips the remaining identical repair calls.
Arc and chain preservation treats an integer beat such as `128` as equivalent
to `128.0`, while still comparing lane, color, direction, tail geometry and all
other object fields.
Pattern-focused repairs also receive short beat-relative event sequences for the
flagged adjacent windows, so the model can compare actual hand, cell, direction,
and object-type patterns instead of relying only on per-window averages.
`--reasoning-effort low|medium|high|xhigh|max` is forwarded to OpenRouter as
`reasoning.effort` for providers that support it, including repair calls; omitted
means the provider default.
BeatForge does not send completion or reasoning token caps. The provider/model
uses its available output budget; provider-side limits still apply. Use
the `--reasoning-effort` option above to select a supported reasoning effort.
When a provider reports `finish_reason=length`, BeatForge continues the same
script during generation and diagnostic repair, stitches each fragment onto the
partial script, and compiles after a non-truncated response. For image-capable
free OpenRouter routes and opted-in local vision routes, a continuation can also
receive a local overview and current-edge preview compiled from complete emitted
`map.*` calls; an unfinished trailing call is excluded. Only the latest such
checkpoint is kept in the conversation. Text-only routes continue without map
images. Continuations remain tool-free, are capped at eight per script, subject
to the job request budget, and aggregate token/cost usage; empty or
non-progressing continuations fail with the partial script preserved.
For providers with a documented image-count limit, `--spectrogram-image-limit N`
keeps the overview separate and vertically stacks consecutive four-second SG panels
without resampling their source pixels. The default sends every panel separately.
Use `--provider local --endpoint http://127.0.0.1:8080/v1 --model gemma`
for a local OpenAI-compatible server; the loopback check is mandatory. Add
`--send-spectrograms` only when that local model is vision-capable; this sends
the generated SG images but does not send raw audio.
For CPU-only local inference, increase `--timeout-seconds` (for example
`--timeout-seconds 900`) instead of treating a slow completion as a provider
failure. Generation requests remain uncapped by BeatForge; increase the timeout
for slow local models and allow the provider to finish its response.
`--asset-library catalog.json --asset-id <id>` passes catalogued, hash-checked
assets into the export; this requires `--vivify` and can be repeated for several
assets. Custom events that contain an explicit `asset_id` are checked against
that same selection and fail closed if the model invents or omits an asset.
`--beat-grid grid.json` optionally feeds normalized Beat This/downbeat data into
the initial plan; `--plan-in` still takes precedence for a manually reviewed
plan.

`render_preview_html()` provides a dependency-free timeline preview for quick
human review. It visualizes notes, arcs, chains and reports object counts; it is
explicitly not a substitute for Beat Saber/Vivify runtime verification.

During a diagnostics-driven repair, image-capable free OpenRouter models receive
a compact multi-scale visual packet: the whole-track SG overview, a coarse
whole-map lane/time overview, up to four synchronized SG/map segment composites,
and up to two sequential playfield contact sheets for local windows. Each segment
composite uses one absolute-time range for both the SG excerpt and the schematic
chart. The chart shows hand colors and cut directions, arcs, chain slices,
bombs, walls, standard-light markers, environment/saber palette swatches, and
Vivify scene-event locations. Contact-sheet tiles show successive beat-sized
top-down grid states for checking hand movement and repeated shapes. A separate
player-perspective approach proxy shows up to four snapshots for the highest-
priority repair window, projecting upcoming notes, bombs, walls, arcs, and chain
slices toward an illustrative hit plane over a two-second musical-time horizon.
The packet stays capped at eight images: when available it includes one
top-down sheet and one perspective sheet alongside up to four synchronized
local composites. If a
provider/test fixture supplies unreadable SG bytes, the model path keeps the SG
and map panels separate instead of dropping them. None of these deterministic
views is a Beat Saber/Unity render. The perspective proxy is not an exact game
geometry or timing simulation.
During repair, a free OpenRouter model that advertises both image input and
tool calls can use `inspect_map_view` for up to two additional views, each
covering an arbitrary in-range window of at most eight seconds. When local map
diagnostics are active, BeatForge requires one focused view before accepting a
repair; the model can request a second if it needs another segment or
representation. It can choose `lane_time`, `event_timeline`, `lighting_storyboard`, `top_down`, `perspective`, `motion_trace`,
`spatial_path`, `presentation`, or `phrase_compare`, and BeatForge returns the rendered PNG in the same conversation.
`event_timeline` separates the selected interval into synchronized tracks for left/right notes,
bombs, walls, arcs, chains, standard lighting and scene/Vivify events. It includes
measured spectral-change markers on the same time axis; these mark candidates, not
detected note onsets. This is an authored-event timeline, not a game render.
`lighting_storyboard` displays the environment/saber palette and explicit RGB fields
found in standard-light or supported Vivify scene events for the selected window.
It shows authored color swatches only; easing, intensity, objects and runtime
lighting are not simulated.
`phrase_compare` places two equal-duration windows from the current map side by side; each
side includes its own SG context when the supplied source panels cover that range. Use it
to review repeated phrases without confusing same-map similarity with a before/after repair diff.
When deterministic preflight passes without warnings, the first available repair
slot can instead run one visual-only review on eligible free vision/tool routes.
The review uses the same local-view tool and may return the draft unchanged; it
does not add a call when a diagnostic repair already ran.
The `motion_trace` view keeps the local time axis and grid rows, then connects
successive notes from each hand with arrows and beat-gap labels. It makes rhythmic
spacing and hand travel visible together. `spatial_path` places the phrase on one
playfield grid to emphasize its overall shape; notes at repeated coordinates are
spread slightly around their exact grid point to keep them visible. Both views show
authored coordinates, not physical hand motion or game-runtime behavior.
The `presentation` view shows the exact serialized standard-light fields
(`et`, `i`, `f`, optional custom data), scene/Vivify event type and payload, and
declared scene palette for the selected window. It labels event-list truncation
and does not simulate how Unity or Beat Saber renders those events. Requests outside
the track, wider windows, unknown view kinds, and extra calls are rejected.
When a previous repair was accepted and another repair is needed, the next turn's
first view is a same-window `revision_diff` showing that accepted version beside
the current map; its optional second view can inspect another representation.
This adds no raw audio and is not enabled for paid or text-only routes. The
initial visual packet stays capped at eight images; model-requested views are
bounded additions.
Text-only providers keep the existing script and textual diagnostics; local
vision providers receive the visual packet only with the existing
`--send-spectrograms` opt-in.

When the only actionable repair signals are coarse rhythm-density mismatch and/or
reviewable rapid same-hand chain candidates, the model returns a compact JSON
patch instead of rewriting the full DSL map. BeatForge applies exact note-call
replacements locally, limits rhythm edits to flagged lower-to-higher audio
activity windows, and permits chain conversion only for a complete eligible run
with matching geometry and one slice per replaced note. The complete patched map
still goes through the normal parser, capability checks and repair invariants;
ambiguous or unsupported patches are rejected and supplied as retry feedback.
Other repair types continue to use the full-map response contract.
Rhythm patches can target only nearby half-beat cells that the measured audio grid
marks as strong spectral-change candidates; these remain candidates for model/SG
review, not detected note onsets. Local lane/time previews mark those cells with
amber dashed guides so the model can compare them against the authored notes.
If rhythm density is the only warning and no safe nearby note move exists, BeatForge
keeps the review warning and skips an unproductive model repair call. Other actionable
warnings still proceed through the normal repair path.

On free OpenRouter models that advertise image input and tool calls, initial
full-map generation can use `preview_map_chunk` for up to four progressive
visual checkpoints. Each call submits one complete new BeatForge DSL section;
BeatForge compiles it with previously accepted sections and returns a selected
local view plus cumulative object counts. Invalid sections are not committed.
Rejected and over-budget requests still consume their call slot. Generation usage
separates attempted calls from successfully rendered checkpoints, accepted section
submissions, and the number of returned images.
Checkpoints are review samples, not substitutes for mapping the full track:
after the last checkpoint, the model returns new DSL calls for every remaining
or unfilled TrackPlan section, and BeatForge appends them to the accepted draft.
`BEATFORGE_MAP_COMPLETE` is reserved for a map whose full-track DSL has already
been submitted. After accepting a section, the model may use
`inspect_current_map_view` to inspect an accepted time range without resubmitting
or changing the DSL. Both tools share the four-checkpoint budget. Views include
lane/time, top-down, player perspective, hand motion traces, spatial paths,
presentation payloads and a before/after diff from the prior accepted section.
For an interval that needs more than one angle, `diagnostic_bundle` returns three
separate full-resolution images in a single tool call: lane/time, player
perspective, and hand-motion trace. Each image receives the same synchronized SG
panels when the source panels cover that interval. A bundle consumes one slot of
the existing four-call budget; its images are listed in order in tool metadata.
If a requested window is covered by adjacent four-second SG source panels, the
tool stacks those panels with the local map view and preserves each panel's
absolute-time labels. Partial overlaps are labeled as SG context; packed images
with no source-panel mapping remain unavailable. If the initial SG input was
packed to meet an image-count limit, the progressive tool still uses the
preserved original source panels.
Text-only routes keep the single-script flow. These previews are not rendered
Beat Saber gameplay.

The model-facing views have different fidelity and cost:

| View | Status | What it can show | Limit |
| --- | --- | --- | --- |
| SG overview and targeted panels | Existing | Track-wide energy plus local phrase spectrum | SG alone does not show authored note placements |
| Segment map lane/time | Implemented for repair | Current notes, directions, arcs, chains, lights, palette and scene-event locations | Schematic; it does not model approach speed, rendered lighting or Unity assets |
| Whole-track map overview | Implemented for repair | Arrangement, event density, broad gaps and presentation-event distribution | Coarse overview; use a local view to inspect exact cuts |
| Synchronized SG/map composite | Implemented for repair | Audio evidence and authored events over the same absolute-time range | Both rows are deterministic diagrams, not a game render |
| Beat-sized playfield contact sheet | Implemented for repair | Successive grid states, cut directions and repeated hand paths | Top-down grid snapshots; no player camera or block travel simulation |
| Hand-motion trace | Implemented on demand for eligible repair/generation views | Time-aligned left/right paths and beat gaps, alongside notes, arcs and chains | Authored map coordinates; not arm kinematics or game simulation |
| Spatial path | Implemented on demand for eligible repair/generation views | Overall left/right path shape on one playfield grid, with repeated notes separated for visibility | Dense clusters may still overlap; beat order is shown by labels, not a game simulation |
| Whole-map timeline HTML | Existing for human review | Arrangement and object distribution across the song | HTML preview is not sent to the model |
| Player-perspective approach contact sheet | Implemented for repair | Upcoming object positions, walls, arcs, chains and local spatial readability | Approximate projection; not exact game geometry, timing or rendered lighting |
| Model-requested local map view | Implemented for eligible repair routes | A model chooses an exact time range and one local representation or the three-view diagnostic bundle | At most two requests, each no wider than eight seconds; later repairs require a same-window before/after diff first |
| Synchronized diagnostic bundle | Implemented for eligible repair and progressive-generation routes | Three separate full-resolution views of the same interval: lane/time, player perspective and hand-motion trace, each paired with SG context when available | Three images per request; still one call from the shared two-call or four-checkpoint budget; schematic, not a game render |
| Progressive section checkpoint | Implemented for eligible free image/tool routes | A model submits a section, receives a local view, cumulative counts and all contiguous SG panels covering its time range; unpreviewed full-track DSL is emitted in the final response | At most four checkpoints, each view covers at most eight seconds; schematic, not a game render |
| Read-only draft reinspection | Implemented for progressive generation | A model requests another view of an already accepted range without resubmitting DSL | Shares the four-checkpoint budget; schematic, not a game render |
| Model-requested presentation payload | Implemented for eligible repair routes | Exact standard-light fields, scene/Vivify payload JSON and declared palette for a local window | Data inspector only; it does not emulate game lighting or render imported assets |
| Accepted-repair revision diff | Implemented on later repair iterations | Previous accepted chart beside the current chart over the same local time window | Deterministic schematic; it shows authored JSON changes, not game-runtime appearance |
| Same-map phrase comparison | Implemented for eligible repair views | Two equal-duration windows of the current map side by side, with independently synchronized SG context when available | Schematic comparison; SG panels may be unavailable when the selected source panels do not cover either full window |
| Beat Saber/Vivify runtime frames | Later, if a renderer is available | Actual environment, lighting, imported assets and block appearance | Requires a real compatible game/mod runtime; a JSON preview cannot prove this |

Segment previews are attached only when the selected provider advertises image
input. They add no paid calls; BeatForge's development OpenRouter client rejects
models without a `:free` model ID and catalogued zero input/output prices.

`export_v3()` writes a complete `Info.dat` + difficulty package and optional
ZIP, plus a `beatforge-manifest.json` containing the plan and capabilities used
for the export.
It also writes `runtime-handoff.json`. Data-only exports report
`handoff_only`; packages with a real Unity bundle report
`bundle_built_not_runtime_tested`. Neither status claims that Beat Saber or a
headset playtest succeeded.
`verify_package()` checks exported directories/ZIPs without executing contents,
including V3 structure, safe archive paths and Vivify asset hashes. The CLI
equivalent is `python -m beatforge verify build/demo`.

The local CLI exposes the same first slice:

```text
python -m beatforge compile-script map.bf.py --bpm 120 --out ExpertPlus.dat --mapping-extensions
python -m beatforge preview ExpertPlus.dat --out preview.html --mapping-extensions
python -m beatforge serve-preview ExpertPlus.dat --port 8765 --mapping-extensions
python -m beatforge generate --audio song.ogg --bpm 120 --target-nps 2.5 --model thinkingmachines/inkling:free --song-name Demo --out build/demo --jobs jobs --job-id demo-001
python -m beatforge generate --provider local --endpoint http://127.0.0.1:8080/v1 --audio song.ogg --bpm 120 --target-nps 2.5 --model gemma --song-name Demo --out build/local --jobs jobs --job-id local-001
python -m beatforge serve-local --binary N:\\llama.cpp\\bin-b11985\\llama-server.exe --model N:\\gemma-4-E4B-it-UD-Q4_K_XL.gguf --mmproj N:\\mmproj-gemma-4-E4B-it-BF16.gguf
python -m beatforge status demo-001 --jobs jobs
python -m beatforge probe --provider local --endpoint http://127.0.0.1:8080/v1
```

`serve-local` launches an already-built llama.cpp server without a shell and
only permits loopback binds. For multimodal GGUFs, pass the matching projector
with `--mmproj`; both model and projector paths are validated before launch.
It does not compile llama.cpp or download model files; if a required path is
missing, it exits with a diagnostic instead of starting anything. Use `probe`
afterwards to verify the OpenAI-compatible
`/v1/models` endpoint before spending a generation call.
For a pinned CPU binary, `powershell -ExecutionPolicy Bypass -File
tools/install_llama_server.ps1` downloads the official `b11985` archive and
refuses extraction unless its SHA-256 matches the pinned digest.

The server binds to `127.0.0.1` by default and exposes `/` plus `/health`;
it is not a public asset host.

It also exposes `POST /api/compile` for the browser editor. The body is
`{"bpm": 120, "script": "map.note(...)"}` and is compiled through the same
AST-restricted runner.

Open `/editor` on the local server for the first typed-script editing shell.

### Search and import Vivify assets

Install the optional browser dependency and its Chromium build once:

```powershell
uv sync --extra browser
uv run --extra browser playwright install chromium
```

Then enable browser discovery on a PC Vivify generation:

```powershell
uv run beatforge generate --audio song.ogg --bpm 120 --target-nps 2.5 `
  --model dots-studio/dots-3-note-preview:free --out output/demo `
  --song-name Demo --vivify --vivify-browser --jobs jobs --job-id demo-vivify
```

`--vivify-browser` requires `--vivify` and an OpenRouter model whose live
catalog advertises tool support. BeatForge opens a visible, isolated Playwright
Chromium profile and loads the installed Anti Anti Debug extension. The default
extension path is discovered from the standard Chrome profile; pass
`--anti-debug-extension <directory>` when it is installed elsewhere. No normal
Chrome profile, cookies or downloads are shared with this session.

The model gets two bounded tools: search Poly Haven's public model API and
import a candidate returned by that search. Search accepts queries up to 160
characters, returns at most five FBX candidates per call and stops after three
search calls per generation. Imports accept candidate IDs only, include the
provider-listed texture dependencies, and stop after three assets or 64 MiB
per asset. The importer rechecks CC0, HTTPS hosts and redirects, writes a ZIP
bundle and SHA-256 to the local asset library, and never executes downloaded
files. It does not scrape arbitrary pages or accept model-supplied URLs.

If a search returns candidates, BeatForge requires the next tool call to import
one selected candidate. This records the source model in the package handoff.
Vivify generation also requires at least one documented `SetRenderingSettings`
scene event; for example, a moderate fog and ambient-light change can follow the
track's mood. BeatForge validates the event type, supported setting fields and
beat range before export. A downloaded FBX source ZIP is not a compiled Unity
AssetBundle. The separate `vivify-build` command can compile supported FBX/OBJ
archives for Windows PC when the pinned Unity/Vivify toolchain is installed.
Generation and repair prompts list the same allowed scene-setting categories
and fields as the validator. In particular, ambient/fog colors belong inside
`renderSettings`; putting them under `qualitySettings` is rejected. If a model
misplaces a known field and the validator schema gives it one unique destination,
BeatForge moves only that field and records the correction in generation usage.
Unknown fields and conflicting destination values remain validation errors.

To build and attach a selected model during generation, add
`--vivify-build-bundles` and provide Unity 2021.3.16f1 through `--unity-editor`
or `BEATFORGE_UNITY_EDITOR`:

```powershell
uv run beatforge generate --audio song.ogg --bpm 120 --target-nps 2.5 `
  --model dots-studio/dots-3-note-preview:free --out output/demo `
  --song-name Demo --vivify --vivify-browser --vivify-build-bundles `
  --unity-editor "C:\Program Files\Unity\Hub\Editor\2021.3.16f1\Editor\Unity.exe" `
  --jobs jobs --job-id demo-vivify-runtime
```

The map must instantiate an imported or explicitly selected asset using its
asset ID. BeatForge builds only referenced assets, resolves IDs to prefab
paths, copies `bundleWindows2021.vivify` into the song package and adds the
exporter CRC to Info.dat. Verification checks the bundle hash, prefab paths and
CRC consistency. This is a Windows PC bundle; Quest/Android builds and in-game
playback still need separate verification. Without the flag, generation stays
a data-only handoff.

When `--asset-library` is omitted, browser-discovered assets are stored under
the job directory, outside the Beat Saber output package. Export records the
Poly Haven source and license in `VivifyAssets/`. Without
`--vivify-build-bundles`, generation exports source assets as a data-only
handoff. `vivify-build` remains available for separate bundle builds.

### Build an imported model as a Vivify bundle

This command requires Unity Editor **2021.3.16f1** and Git. It creates a fresh,
isolated Unity project, fetches the pinned VivifyTemplate checkout when
`--template-source` is omitted, stages one FBX/OBJ per asset, and invokes the
official Windows 2021 exporter in batch mode. The workspace must be new or
empty; it retains the Unity project, logs, bundle and verification manifest.

```powershell
uv run beatforge vivify-build `
  --asset a909e5bf3712ad110131=output/demo/VivifyAssets/a909e5bf3712ad110131.zip `
  --workspace .tmp-vivify-build-demo `
  --unity-editor "C:\Program Files\Unity\Hub\Editor\2021.3.16f1\Editor\Unity.exe"
```

Repeat `--asset ID=ZIP` to include several models in the same bundle. The
command verifies the editor version, exporter result marker, bundle file,
deterministic prefab paths and CRC before writing
`vivify-bundle-manifest.json`. A zero exit code without those artifacts is
reported as failure. This bridge currently builds `bundleWindows2021.vivify`;
it does not build Android/Quest bundles.

`run_generation_job()` records durable start/completion/failure checkpoints in
the job store, so a detached worker can outlive the preview browser without
losing the model response status.
The same job directory exposes a replaceable `live.json` snapshot: while the
provider is working it reports `provider_running`; after completion it contains
the returned script, usage and object counts, and failures contain a sanitized
error. Final usage sums the initial generation and subsequent repair/recovery
calls; `events.jsonl` retains each attempt separately. OpenRouter failures also
retain a small provider diagnostic summary when available: selected
model/provider, finish reasons, refusal text, and whitelisted upstream error
metadata. The summary omits raw responses, model reasoning, account identifiers,
and unrecognized provider fields. The append-only `events.jsonl` remains the
authoritative history.

When diagnostics-driven repair is enabled, the initial generation leaves up to
two provider requests for one repair view and its follow-up answer. This keeps
optional browser and preview tools from consuming the entire job budget before
the configured repair can inspect a weak draft.

Locked plan sections are supplied to autopilot as protected context. Targeted
non-autopilot regeneration must name section IDs and is rejected for locked
sections, so a repair cannot silently overwrite an approved passage.

`BudgetGuard` enforces the development ceiling before calls: fourteen requests
by default (CLI `--max-requests 1..32`),
two repairs and zero-cost `:free` variants or the verified `openrouter/free`
router by default; explicitly selected local loopback models are also allowed,
while paid remote generation cannot be silently introduced by a fallback.

The detached worker accepts the same guard and reserves a request before
inference, so background jobs cannot bypass the UI budget.

While OpenRouter or the loopback client streams a response, `live.json` also
contains `stream_connected`, `partial_response` and `partial_response_chars`.
The snapshot refreshes as visible answer text arrives; it does not include
provider reasoning fields.
When the job finishes, the snapshot replaces the partial text with the complete
`script` and usage. If the request fails mid-stream, the latest partial answer
is kept alongside the sanitized error. Clients that do not expose a progress
callback keep their existing non-streaming behavior.

For example, a running job may contain:

```json
{
  "status": "provider_running",
  "stream_connected": true,
  "partial_response": "map.note(1, 1,",
  "partial_response_chars": 14
}
```

`preflight()` reports structural errors, actual NPS, chain-slice count, section
coverage, direction concentration, and whether four-second map-pattern changes
track changes in the measured half-beat audio-feature grid. Weak audio-pattern
correspondence becomes a repair diagnostic; insufficient or uniform audio
evidence is explicitly marked unavailable. Pattern structure counts each
authored note/arc/chain once and records sustained-object type and duration;
chain slices remain part of density but cannot dominate the phrase comparison.
An unlocked planned section with a positive target density must contain at least
one playable note, arc or chain. Missing objects in such a section are blocking
preflight errors; zero-density and locked sections may remain empty. If bounded
repair cannot fill an active section, BeatForge fails generation instead of
exporting the incomplete map. The `verify` command checks package structure and
requirements; a structurally valid ZIP can still contain a map that needs
quality review, so inspect its stored preflight report as well.
The 0.20 chance-corrected warning threshold is a provisional engineering
heuristic, not a benchmark-calibrated cut score. These are model-blind
engineering checks, not a gameplay rating or a replacement for blind human
playtesting.
The whole-track NPS warning band is ±15% of the requested target. Initial
prompts also show the strongest review-only sustained-ridge candidates with
their time, beat and frequency spans; those hints do not require an arc when
the SG does not show a clear continuous line.
Preflight also flags a chain-slice share above 35% for review. This provisional
threshold sits well above the 14–15% share in two same-track reference charts;
repair guidance preserves SG-supported rapid bursts and changes only slow or
isolated chain use. It is a warning, not a chain quota.
Preflight also lists fast same-hand note runs that could be chain candidates.
Timing and hand color alone do not prove those runs should become chains, so
they remain informational and do not trigger automatic repair. Repairs respond
to chain overuse using highlighted SG windows; model-visible diagnostics omit
exact chain-share percentages and thresholds to avoid optimizing a ratio.

`BeatGrid` is the normalized contract for a future Beat This adapter. Missing
BPM/confidence remains `None`; BeatForge does not fabricate certainty from an
incomplete tracker response.
`initial_plan_from_beat_grid()` consumes verified downbeats when available and
falls back to the local RMS plan when they are absent; missing BPM remains a
hard error.

`initial_plan_from_analysis()` creates a reviewable section plan from local
RMS-energy windows and BPM. It is a starting hypothesis, not an automatic
musical truth; the user/model can edit or lock sections before generation.

`AssetCandidate`/`rank_candidates()` normalize browser search results before
download: only HTTPS, allowlisted domains and CC0/CC-BY candidates survive.
`PolyHavenBrowserSearch` reads the provider's public API through the isolated
browser; `VivifyToolExecutor` exposes only candidate search and candidate-ID
import to the model. `fetch_candidate()` rechecks redirects, streams each
bundle member through a size cap, hashes the resulting bytes and records the
source/license in `AssetLibrary`. It never executes downloaded files.
`build_vivify_manifest()` copies selected catalogued assets into a reproducible
`VivifyAssets/` bundle with source, license, size and SHA-256 metadata. The
optional Unity path can build and attach a Windows Vivify runtime bundle;
actual Beat Saber playback remains a separate verification gate.

The SDK does not silently downgrade V3 objects. The bundle bridge handles
Windows 2021 export and optional attachment; Android/Quest export and in-game
playtest remain runtime-specific verification gates.

Development inference is free-only by policy; API keys are read from the
environment and never written into the repository. Fixed `:free` model ids
remain reproducible, while `openrouter/free` is a deliberate random free-model
router whose selected model is recorded. The client never chooses a paid
fallback automatically.

Install the local package once with `python -m pip install -e .`; after that,
`beatforge ...` and `python -m beatforge ...` work from any directory.
