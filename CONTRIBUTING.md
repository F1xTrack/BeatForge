# Contributing

Thanks for helping improve BeatForge.

## Development setup

Use Python 3.12 or later and install the project with its optional browser
dependency when working on the Vivify asset-search path:

```powershell
uv sync --extra browser
```

## Changes

- Keep map parsing and compilation deterministic.
- Add or update focused tests for behavior changes.
- Do not commit API keys, generated maps, audio, downloaded assets, browser
  profiles, Unity files, or local job output.
- Keep asset source, author, license, and checksum metadata with any new asset
  catalog entries.
- For Vivify work, target the `vivify-beta` branch. The main release CLI omits
  the Vivify controls until that feature is ready for a stable release.

## Before submitting

Run the relevant tests and inspect the diff:

```powershell
uv run pytest
git diff --check
```

Open a pull request with a short summary, verification performed, and any
runtime or game-version limitations.
