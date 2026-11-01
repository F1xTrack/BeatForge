"""Normalize and rank browser-provided asset candidates before import."""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from .assets import SAFE_LICENSES


@dataclass(frozen=True, slots=True)
class AssetCandidate:
    title: str
    source_url: str
    license: str
    author: str = ""
    media_type: str = "model"
    preview_url: str | None = None
    download_url: str | None = None
    dependencies: tuple[tuple[str, str], ...] = ()
    description: str = ""
    asset_size_bytes: int = 0


def validate_candidate(candidate: AssetCandidate, allowed_domains: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    parsed = urlparse(candidate.source_url)
    host = parsed.hostname or ""
    if parsed.scheme != "https":
        errors.append("asset source must use HTTPS")
    if not any(host == domain or host.endswith("." + domain) for domain in allowed_domains):
        errors.append("asset source is outside the browser domain allowlist")
    if candidate.license.upper() not in SAFE_LICENSES:
        errors.append("asset license is not in the safe allowlist")
    download = urlparse(candidate.download_url or candidate.source_url)
    download_host = download.hostname or ""
    if download.scheme != "https":
        errors.append("asset download must use HTTPS")
    if not any(download_host == domain or download_host.endswith("." + domain)
               for domain in allowed_domains):
        errors.append("asset download is outside the browser domain allowlist")
    for _name, dependency_url in candidate.dependencies:
        dependency = urlparse(dependency_url)
        dependency_host = dependency.hostname or ""
        if dependency.scheme != "https":
            errors.append("asset dependency must use HTTPS")
        if not any(dependency_host == domain or dependency_host.endswith("." + domain)
                   for domain in allowed_domains):
            errors.append("asset dependency is outside the browser domain allowlist")
    if candidate.preview_url:
        preview = urlparse(candidate.preview_url)
        if preview.scheme != "https":
            errors.append("asset preview must use HTTPS")
    return errors


def rank_candidates(query: str, candidates: list[AssetCandidate], *, allowed_domains: tuple[str, ...]) -> list[AssetCandidate]:
    """Return only importable candidates, ranked by title/author lexical match."""
    needle = query.casefold().strip()
    valid = [candidate for candidate in candidates if not validate_candidate(candidate, allowed_domains)]
    def score(candidate: AssetCandidate) -> tuple[int, str]:
        haystack = f"{candidate.title} {candidate.author} {candidate.description}".casefold()
        return (0 if needle and needle in haystack else 1, candidate.title.casefold())
    return sorted(valid, key=score)
