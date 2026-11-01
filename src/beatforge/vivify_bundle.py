"""Batch-mode staging and build bridge for VivifyTemplate PC bundles.

This module does not fake Unity output. A successful result requires a real
Unity 2021.3.16f1 process, a generated AssetBundle, and the exporter CRC.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import zipfile
from typing import Any, Iterable


UNITY_VERSION = "2021.3.16f1"
TEMPLATE_REPOSITORY = "https://github.com/Swifter1243/VivifyTemplate.git"
TEMPLATE_COMMIT = "b37a3ba719f2e40edcd63ccf63dc6ee61e8c5bd3"
EXPORTER_PACKAGE = "com.swifter.vivify-template.exporter"
EXPORTER_VERSION = "1.0.0-preview.3"
BUNDLE_NAME = "beatforge"
BUNDLE_FILENAME = "bundleWindows2021.vivify"
VIVIFY_PATH_PREFIX = "assets"
_ASSET_ID = re.compile(r"^[0-9a-f]{20}$")
_ALLOWED_EXTENSIONS = {
    ".fbx", ".obj", ".mtl", ".png", ".jpg", ".jpeg", ".tga", ".tif",
    ".tiff", ".bmp", ".hdr", ".exr",
}
_MAX_ARCHIVE_MEMBERS = 2048
_MAX_UNPACKED_BYTES = 256 * 1024 * 1024


class VivifyBundleBuildError(RuntimeError):
    """A staged Vivify asset could not be turned into a verified bundle."""


@dataclass(frozen=True, slots=True)
class VivifyModelInput:
    asset_id: str
    archive: Path


def _valid_asset_id(asset_id: str) -> bool:
    return isinstance(asset_id, str) and _ASSET_ID.fullmatch(asset_id) is not None


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (not normalized or path.is_absolute() or ":" in normalized
            or any(part in {"", ".", ".."} for part in path.parts)):
        raise VivifyBundleBuildError(f"unsafe Vivify archive member path: {name!r}")
    if any(ord(char) < 32 for char in normalized):
        raise VivifyBundleBuildError("Vivify archive paths cannot contain control characters")
    if path.suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise VivifyBundleBuildError(
            f"unsupported Vivify source file type: {path.suffix or '(no extension)'}")
    return path


def stage_model_archive(
    model: VivifyModelInput,
    unity_project: str | Path,
    *,
    max_asset_mb: int = 64,
) -> dict[str, str]:
    """Safely extract one source archive into an isolated Unity project's Assets."""
    if not _valid_asset_id(model.asset_id):
        raise ValueError("asset_id must be the 20-character lowercase catalog ID")
    archive = Path(model.archive).resolve()
    if not archive.is_file() or archive.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Vivify source archive is missing or not a ZIP: {archive}")
    if max_asset_mb <= 0:
        raise ValueError("max_asset_mb must be positive")
    compressed_limit = max_asset_mb * 1024 * 1024
    if archive.stat().st_size > compressed_limit:
        raise VivifyBundleBuildError(f"Vivify archive exceeds {max_asset_mb} MiB")

    project = Path(unity_project).resolve()
    assets_root = (project / "Assets").resolve()
    destination = (assets_root / "BeatForgeImported" / model.asset_id / "source").resolve()
    if not destination.is_relative_to(assets_root):
        raise VivifyBundleBuildError("Vivify extraction escaped Unity Assets")
    if destination.exists():
        raise FileExistsError(f"staging directory already exists: {destination}")

    destination.mkdir(parents=True)
    total = 0
    model_files: list[str] = []
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if not members or len(members) > _MAX_ARCHIVE_MEMBERS:
                raise VivifyBundleBuildError(
                    f"Vivify archive must contain 1..{_MAX_ARCHIVE_MEMBERS} files")
            for member in members:
                if member.is_dir():
                    continue
                relative = _safe_member_path(member.filename)
                folded = relative.as_posix().casefold()
                if folded in seen:
                    raise VivifyBundleBuildError("Vivify archive has duplicate Windows paths")
                seen.add(folded)
                mode = (member.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise VivifyBundleBuildError("Vivify archive symlinks are not accepted")
                total += member.file_size
                if total > _MAX_UNPACKED_BYTES:
                    raise VivifyBundleBuildError("Vivify archive expands beyond 256 MiB")
                target = (destination / Path(*relative.parts)).resolve()
                if not target.is_relative_to(destination):
                    raise VivifyBundleBuildError("Vivify member escaped its staging directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with bundle.open(member, "r") as source, target.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        written += len(block)
                        if written > member.file_size or total - member.file_size + written > _MAX_UNPACKED_BYTES:
                            raise VivifyBundleBuildError("Vivify ZIP member expanded beyond its declared size")
                        output.write(block)
                if written != member.file_size:
                    raise VivifyBundleBuildError("Vivify ZIP member size did not match its directory entry")
                if relative.suffix.lower() in {".fbx", ".obj"}:
                    model_files.append(target.relative_to(project).as_posix())
    except zipfile.BadZipFile as exc:
        raise VivifyBundleBuildError("Vivify source is not a valid ZIP archive") from exc

    if len(model_files) != 1:
        raise VivifyBundleBuildError(
            f"Vivify archive must contain exactly one FBX or OBJ model; found {len(model_files)}")
    prefab_path = f"{VIVIFY_PATH_PREFIX}/{model.asset_id}/{model.asset_id}.prefab"
    return {
        "asset_id": model.asset_id,
        "source_model_path": model_files[0],
        "prefab_asset_path": f"Assets/{prefab_path}",
        "vivify_asset_path": prefab_path,
        "source_archive_sha256": _sha256_file(archive),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unity_build_script_path(project: Path) -> Path:
    editor_dir = project / "Assets" / "Editor"
    editor_dir.mkdir(parents=True, exist_ok=True)
    return editor_dir / "BeatForgeVivifyBundleBuild.cs"


def _verify_template_revision(template_source: str | Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(Path(template_source).resolve()), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    revision = (result.stdout or "").strip().lower()
    if result.returncode != 0 or revision != TEMPLATE_COMMIT:
        raise VivifyBundleBuildError(
            f"VivifyTemplate source must be the pinned commit {TEMPLATE_COMMIT}")


def initialize_unity_project(
    template_source: str | Path,
    project_path: str | Path,
) -> Path:
    """Create an isolated minimal Unity project embedding the pinned exporter."""
    source = Path(template_source).resolve()
    source_packages = source / "Packages"
    exporter = source_packages / EXPORTER_PACKAGE
    if not (exporter / "package.json").is_file():
        raise FileNotFoundError(f"VivifyTemplate exporter package not found under {source}")
    package_meta = json.loads((exporter / "package.json").read_text(encoding="utf-8"))
    if (package_meta.get("name") != EXPORTER_PACKAGE
            or package_meta.get("version") != EXPORTER_VERSION):
        raise VivifyBundleBuildError("VivifyTemplate exporter package version does not match the pin")

    project = Path(project_path).resolve()
    if project.exists() and any(project.iterdir()):
        raise FileExistsError(f"Unity project destination must be empty: {project}")
    project.mkdir(parents=True, exist_ok=True)
    packages = project / "Packages"
    packages.mkdir()
    shutil.copytree(exporter, packages / EXPORTER_PACKAGE)
    shutil.copyfile(source_packages / "manifest.json", packages / "manifest.json")
    assets = project / "Assets"
    (assets / "Editor").mkdir(parents=True, exist_ok=True)
    project_settings = project / "ProjectSettings"
    project_settings.mkdir(parents=True, exist_ok=True)
    version_file = project_settings / "ProjectVersion.txt"
    version_file.write_text(
        f"m_EditorVersion: {UNITY_VERSION}\n",
        encoding="utf-8",
    )

    manifest_path = project / "Packages" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dependencies = manifest.setdefault("dependencies", {})
    dependencies[EXPORTER_PACKAGE] = f"file:Packages/{EXPORTER_PACKAGE}"
    manifest["scopedRegistries"] = [
        {"name": "OpenUPM", "url": "https://package.openupm.com", "scopes": ["com.swifter"]}
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return project


def _build_spec(
    models: Iterable[dict[str, str]],
    output_dir: Path,
    result_path: Path,
) -> dict[str, Any]:
    return {
        "unityVersion": UNITY_VERSION,
        "buildVersion": "Windows2021",
        "bundleName": BUNDLE_NAME,
        "outputDirectory": str(output_dir.resolve()),
        "resultPath": str(result_path.resolve()),
        "assets": list(models),
    }


def build_vivify_bundles(
    models: Iterable[VivifyModelInput],
    *,
    unity_editor: str | Path,
    template_source: str | Path,
    workspace: str | Path,
    timeout_seconds: int = 900,
    max_asset_mb: int = 64,
) -> dict[str, Any]:
    """Build and verify a Windows 2021 Vivify bundle using a real Unity Editor."""
    inputs = tuple(models)
    if not inputs:
        raise ValueError("at least one Vivify model archive is required")
    ids = [model.asset_id for model in inputs]
    if len(set(ids)) != len(ids):
        raise ValueError("Vivify asset IDs must be unique")
    if timeout_seconds < 60:
        raise ValueError("Unity build timeout must be at least 60 seconds")
    editor = Path(unity_editor).resolve()
    if not editor.is_file():
        raise FileNotFoundError(f"Unity Editor executable not found: {editor}")
    _verify_template_revision(template_source)

    root = Path(workspace).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Vivify build workspace must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    project = initialize_unity_project(template_source, root / "unity-project")
    staged = [stage_model_archive(model, project, max_asset_mb=max_asset_mb) for model in inputs]
    script_path = _unity_build_script_path(project)
    template_script = Path(__file__).parent / "templates" / "BeatForgeVivifyBundleBuild.cs"
    if not template_script.is_file():
        raise FileNotFoundError(f"bundler script resource is missing: {template_script}")
    shutil.copyfile(template_script, script_path)

    output_dir = root / "bundle-output"
    output_dir.mkdir()
    result_path = root / "unity-build-result.json"
    spec_path = root / "unity-build-spec.json"
    spec_path.write_text(
        json.dumps(_build_spec(staged, output_dir, result_path), indent=2), encoding="utf-8")
    log_path = root / "Unity.log"
    command = [
        str(editor), "-batchmode", "-nographics", "-quit",
        "-projectPath", str(project), "-buildTarget", "Win64",
        "-executeMethod", "BeatForgeVivifyBundleBuild.Build",
        "-beatforgeSpec", str(spec_path), "-logFile", str(log_path),
    ]
    try:
        completed = subprocess.run(
            command, cwd=project, capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        (root / "unity-stdout.txt").write_text(str(exc.stdout or ""), encoding="utf-8")
        (root / "unity-stderr.txt").write_text(str(exc.stderr or ""), encoding="utf-8")
        raise VivifyBundleBuildError(
            f"Unity timed out after {timeout_seconds}s; workspace retained at {root}") from exc
    (root / "unity-stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (root / "unity-stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise VivifyBundleBuildError(
            f"Unity exited with {completed.returncode}; inspect {log_path} and {root}")
    if not result_path.is_file():
        raise VivifyBundleBuildError(
            f"Unity returned success without its build-result marker; inspect {log_path}")

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VivifyBundleBuildError("Unity build-result marker is invalid") from exc
    if result.get("unityVersion") != UNITY_VERSION:
        raise VivifyBundleBuildError("Unity build used an unexpected editor version")
    if result.get("buildVersion") != "Windows2021":
        raise VivifyBundleBuildError("Unity exporter did not build Windows2021")
    if result.get("bundleFile") != BUNDLE_FILENAME:
        raise VivifyBundleBuildError("Unity exporter reported an unexpected bundle filename")
    raw_crc = result.get("crc")
    try:
        crc = int(raw_crc)
    except (TypeError, ValueError):
        crc = -1
    if isinstance(raw_crc, bool) or not 0 <= crc <= 0xFFFFFFFF:
        raise VivifyBundleBuildError("Unity exporter returned an invalid bundle CRC")
    bundle = output_dir / BUNDLE_FILENAME
    if not bundle.is_file() or bundle.stat().st_size == 0:
        raise VivifyBundleBuildError("Unity reported success but no non-empty bundle exists")
    expected_assets = {item["asset_id"]: item["vivify_asset_path"] for item in staged}
    reported_assets = {
        item.get("assetId"): item.get("vivifyAssetPath")
        for item in result.get("assets", []) if isinstance(item, dict)
    }
    if reported_assets != expected_assets:
        raise VivifyBundleBuildError("Unity bundle prefab paths do not match the staged assets")

    manifest = {
        "format": "BeatForge-VivifyBundle-0.1",
        "unity_version": UNITY_VERSION,
        "template_repository": TEMPLATE_REPOSITORY,
        "template_commit": TEMPLATE_COMMIT,
        "build_version": "Windows2021",
        "bundle_file": BUNDLE_FILENAME,
        "bundle_size_bytes": bundle.stat().st_size,
        "bundle_sha256": _sha256_file(bundle),
        "bundle_crc": crc,
        "vivify_checksum_key": "_windows2021",
        "assets": staged,
    }
    (root / "vivify-bundle-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "bundleinfo.json").write_text(json.dumps({
        "bundleFiles": [BUNDLE_FILENAME],
        "bundleCRCs": {"_windows2021": crc},
        "isCompressed": True,
    }, indent=2) + "\n", encoding="utf-8")
    return manifest


def clone_pinned_template(destination: str | Path) -> Path:
    """Fetch only the pinned VivifyTemplate Packages tree into a fresh directory."""
    root = Path(destination).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"template checkout destination must be empty: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    clone = subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
         TEMPLATE_REPOSITORY, str(root)], capture_output=True, text=True, check=False,
    )
    if clone.returncode != 0:
        raise VivifyBundleBuildError(f"VivifyTemplate clone failed: {clone.stderr.strip()}")
    fetch = subprocess.run(
        ["git", "-C", str(root), "fetch", "--depth", "1", "origin", TEMPLATE_COMMIT],
        capture_output=True, text=True, check=False,
    )
    if fetch.returncode != 0:
        raise VivifyBundleBuildError(f"Pinned VivifyTemplate fetch failed: {fetch.stderr.strip()}")
    checkout = subprocess.run(
        ["git", "-C", str(root), "checkout", "--detach", TEMPLATE_COMMIT],
        capture_output=True, text=True, check=False,
    )
    if checkout.returncode != 0:
        raise VivifyBundleBuildError(f"Pinned VivifyTemplate checkout failed: {checkout.stderr.strip()}")
    sparse = subprocess.run(
        ["git", "-C", str(root), "sparse-checkout", "set", "Packages"],
        capture_output=True, text=True, check=False,
    )
    if sparse.returncode != 0:
        raise VivifyBundleBuildError(f"VivifyTemplate sparse checkout failed: {sparse.stderr.strip()}")
    return root


def unity_executable_from_environment() -> Path | None:
    """Return the configured Unity editor path without guessing install locations."""
    value = os.environ.get("BEATFORGE_UNITY_EDITOR")
    return Path(value).expanduser() if value else None
