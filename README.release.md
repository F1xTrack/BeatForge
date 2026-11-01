# BeatForge 0.1.0

BeatForge analyzes an audio track and generates a Beat Saber V3 map package
through a reviewed track plan, model-assisted mapping, deterministic checks,
bounded repairs, and visual previews.

## Install

Use Python 3.12 or later and install the wheel from this release:

```powershell
python -m pip install beatforge-0.1.0-py3-none-any.whl
```

## Generate a map

Set `OPENROUTER_API_KEY` in your environment and run:

```powershell
beatforge generate `
  --audio song.ogg `
  --bpm 120 `
  --target-nps 2.5 `
  --model openrouter/free `
  --song-name "Song title" `
  --out .\output\song `
  --zip .\output\song.zip `
  --jobs .\jobs `
  --job-id song-001
```

Generated maps are review candidates. Verify the output with `beatforge verify`
and play it in the intended Beat Saber runtime before treating it as finished.

## Included commands

- `generate`: analyze audio, call an OpenRouter model, compile and export V3.
- `compile-script`: compile a safe BeatForge mapping script to V3 JSON.
- `preview` and `serve-preview`: inspect compiled maps in a browser.
- `verify`: validate an exported folder or ZIP without executing it.
- `status`: inspect a generation job snapshot and latest event.
- `probe`: inspect an OpenRouter model catalog.

The release CLI has no Vivify toggle or Vivify build command. Chroma, Noodle
Extensions, and Mapping Extensions remain available where supported by the
target platform.

Image-capable free OpenRouter models can receive BeatBench-compatible 44.1 kHz
spectrograms and bounded visual map checkpoints. These are schematics and
projections, not screenshots from Beat Saber or Unity.
