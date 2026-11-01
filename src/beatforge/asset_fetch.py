"""Safe, provenance-preserving import of browser-discovered assets."""
from __future__ import annotations

import hashlib
from pathlib import Path
from pathlib import PurePosixPath
import re
import tempfile
from typing import BinaryIO, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

from .asset_search import AssetCandidate, validate_candidate
from .assets import AssetLibrary, AssetRecord


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_CHUNK_SIZE = 1024 * 1024


def candidate_asset_id(candidate: AssetCandidate) -> str:
    """Return a stable, non-secret id derived from the cited source."""
    digest = hashlib.sha256(
        f"{candidate.source_url}\n{candidate.title}\n{candidate.media_type}".encode("utf-8")
    ).hexdigest()
    return digest[:20]


def _filename(candidate: AssetCandidate, asset_id: str) -> str:
    path_name = Path(urlparse(candidate.download_url or candidate.source_url).path).name
    suffix = Path(path_name).suffix.lower()
    if suffix not in {".glb", ".gltf", ".obj", ".fbx", ".png", ".jpg", ".jpeg", ".wav", ".ogg"}:
        suffix = ".bin"
    title = _SAFE_NAME.sub("-", candidate.title).strip("-._") or "asset"
    return f"{title[:80]}-{asset_id}{suffix}"


def _read_response(response: BinaryIO, target: BinaryIO, max_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = response.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"asset exceeds the {max_bytes} byte import limit")
        digest.update(chunk)
        target.write(chunk)
    return digest.hexdigest(), total


def fetch_candidate(
    candidate: AssetCandidate,
    library: AssetLibrary,
    destination: str | Path,
    *,
    allowed_domains: tuple[str, ...],
    max_bytes: int = 64 * 1024 * 1024,
    timeout: float = 20.0,
    opener: Callable[..., BinaryIO] = urlopen,
) -> AssetRecord:
    """Download one validated candidate and add it to ``library`` atomically.

    The final response URL is checked again so an allowed page cannot redirect
    the importer to an unrelated host. The file is written under a temporary
    name in the destination directory and only renamed after hashing succeeds.
    """
    errors = validate_candidate(candidate, allowed_domains)
    if errors:
        raise ValueError("; ".join(errors))
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    asset_id = candidate_asset_id(candidate)
    final_path = (root / _filename(candidate, asset_id)).resolve()
    if not final_path.is_relative_to(root):
        raise ValueError("asset filename escaped the destination directory")

    download_url = candidate.download_url or candidate.source_url
    request_headers = {"User-Agent": "BeatForge/0.1 asset importer"}
    dependencies = tuple(candidate.dependencies)
    archive_path = (root / f"{_SAFE_NAME.sub('-', candidate.title).strip('-._')[:80] or 'asset'}-"
                    f"{asset_id}.zip") if dependencies else None
    final_target = archive_path or final_path
    temporary: Path | None = None
    try:
        if dependencies:
            total_bytes = 0
            with tempfile.TemporaryDirectory(prefix=f".{asset_id}-", dir=root) as work_name:
                work = Path(work_name)
                archive_members: list[tuple[Path, str]] = []
                downloads = [(Path(urlparse(download_url).path).name, download_url), *dependencies]
                seen_names: set[str] = set()
                for member_name, source_url in downloads:
                    relative = PurePosixPath(member_name)
                    if (not member_name or relative.is_absolute() or ".." in relative.parts
                            or "\\" in member_name or ":" in member_name
                            or "\x00" in member_name or not relative.parts):
                        raise ValueError("asset bundle contains an unsafe member path")
                    normalized = relative.as_posix()
                    if normalized in seen_names:
                        raise ValueError("asset bundle contains duplicate member paths")
                    seen_names.add(normalized)
                    local_path = work.joinpath(*relative.parts).resolve()
                    if not local_path.is_relative_to(work.resolve()):
                        raise ValueError("asset bundle member escaped the temporary directory")
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    request = Request(source_url, headers=request_headers)
                    remaining = max_bytes - total_bytes
                    if remaining <= 0:
                        raise ValueError(f"asset exceeds the {max_bytes} byte import limit")
                    with opener(request, timeout=timeout) as response:
                        final_url = getattr(response, "geturl", lambda: source_url)()
                        parsed = urlparse(final_url)
                        host = parsed.hostname or ""
                        if parsed.scheme != "https" or not any(
                            host == domain or host.endswith("." + domain)
                            for domain in allowed_domains
                        ):
                            raise ValueError("asset redirect is outside the HTTPS domain allowlist")
                        with local_path.open("wb") as stream:
                            _digest, size = _read_response(response, stream, remaining)
                    total_bytes += size
                    archive_members.append((local_path, normalized))
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=f".{asset_id}-", suffix=".part", dir=root,
                    delete=False
                ) as stream:
                    temporary = Path(stream.name)
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                    for path, name in archive_members:
                        bundle.write(path, arcname=name)
        else:
            request = Request(download_url, headers=request_headers)
            with opener(request, timeout=timeout) as response:
                final_url = getattr(response, "geturl", lambda: download_url)()
                parsed = urlparse(final_url)
                host = parsed.hostname or ""
                if parsed.scheme != "https" or not any(
                    host == domain or host.endswith("." + domain)
                    for domain in allowed_domains
                ):
                    raise ValueError("asset redirect is outside the HTTPS domain allowlist")
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=f".{asset_id}-", suffix=".part", dir=root,
                    delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    digest, _size = _read_response(response, stream, max_bytes)
        temporary.replace(final_target)
        record = AssetRecord(asset_id, candidate.title, candidate.source_url,
                             str(final_target), candidate.license, candidate.author,
                             digest if not dependencies else "", candidate.media_type)
        return library.add(record)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise

