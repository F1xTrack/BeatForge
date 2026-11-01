"""Poly Haven model discovery through the isolated browser's public API page."""
from __future__ import annotations

from urllib.parse import quote
import re
from typing import Any, Protocol

from .asset_fetch import candidate_asset_id
from .asset_search import AssetCandidate, rank_candidates, validate_candidate


POLYHAVEN_ALLOWED_DOMAINS = ("polyhaven.com", "dl.polyhaven.org")
POLYHAVEN_API = "https://api.polyhaven.com"
MAX_QUERY_CHARS = 160
MAX_RESULTS = 5
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
_ASSET_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


class JsonBrowser(Protocol):
    def read_json(self, url: str) -> Any: ...


class PolyHavenBrowserSearch:
    """Search CC0 Poly Haven models using API data rendered in the browser."""

    def __init__(self, browser: JsonBrowser, *, max_bundle_bytes: int = MAX_BUNDLE_BYTES):
        if max_bundle_bytes <= 0:
            raise ValueError("max_bundle_bytes must be positive")
        self.browser = browser
        self.max_bundle_bytes = max_bundle_bytes
        self._models: dict[str, dict[str, Any]] | None = None

    def _load_models(self) -> dict[str, dict[str, Any]]:
        if self._models is None:
            value = self.browser.read_json(f"{POLYHAVEN_API}/assets?t=models")
            if not isinstance(value, dict):
                raise RuntimeError("Poly Haven model index is not an object")
            self._models = {
                str(asset_id): item for asset_id, item in value.items()
                if isinstance(item, dict) and item.get("type") == 2
                and _ASSET_ID.fullmatch(str(asset_id))
            }
        return self._models

    @staticmethod
    def _metadata_text(asset: dict[str, Any]) -> str:
        tags = asset.get("tags") if isinstance(asset.get("tags"), list) else []
        authors = asset.get("authors") if isinstance(asset.get("authors"), dict) else {}
        return " ".join((str(asset.get("name", "")), str(asset.get("description", "")),
                         str(asset.get("category", "")), " ".join(map(str, tags)),
                         " ".join(map(str, authors.keys())))).casefold()

    def _candidate(self, asset_id: str, metadata: dict[str, Any]) -> AssetCandidate | None:
        files = self.browser.read_json(f"{POLYHAVEN_API}/files/{quote(asset_id, safe='')}")
        if not isinstance(files, dict):
            return None
        formats = files.get("fbx")
        if not isinstance(formats, dict):
            return None
        resolution = next((formats.get(key) for key in ("1k", "2k", "4k", "8k")
                           if isinstance(formats.get(key), dict)), None)
        if not isinstance(resolution, dict):
            return None
        primary = resolution.get("fbx")
        if not isinstance(primary, dict) or not isinstance(primary.get("url"), str):
            return None
        include = primary.get("include")
        dependencies: list[tuple[str, str]] = []
        total_bytes = int(primary.get("size", 0) or 0)
        if isinstance(include, dict):
            for member, info in include.items():
                if not isinstance(member, str) or not isinstance(info, dict):
                    return None
                url = info.get("url")
                if not isinstance(url, str):
                    return None
                dependencies.append((member, url))
                total_bytes += int(info.get("size", 0) or 0)
        if total_bytes <= 0 or total_bytes > self.max_bundle_bytes:
            return None
        authors = metadata.get("authors")
        author = ", ".join(str(name) for name in authors) if isinstance(authors, dict) else ""
        provider_page = f"https://polyhaven.com/a/{asset_id}"
        description = str(metadata.get("description", ""))[:800]
        candidate = AssetCandidate(
            title=str(metadata.get("name") or asset_id),
            source_url=provider_page,
            license="CC0",
            author=author,
            media_type="model",
            preview_url=metadata.get("thumbnail_url") if isinstance(
                metadata.get("thumbnail_url"), str) else None,
            download_url=primary["url"],
            dependencies=tuple(dependencies),
            description=description,
            asset_size_bytes=total_bytes,
        )
        if validate_candidate(candidate, POLYHAVEN_ALLOWED_DOMAINS):
            return None
        return candidate

    def search(self, query: str, *, limit: int = 5) -> list[AssetCandidate]:
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query must contain 1 to {MAX_QUERY_CHARS} characters")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_RESULTS:
            raise ValueError(f"limit must be between 1 and {MAX_RESULTS}")
        terms = tuple(dict.fromkeys(re.findall(r"[\w-]+", query.casefold())))
        if not terms:
            raise ValueError("query must contain searchable words")
        ranked_metadata: list[tuple[int, str, dict[str, Any]]] = []
        for asset_id, metadata in self._load_models().items():
            text = self._metadata_text(metadata)
            score = sum(term in text for term in terms)
            if score:
                ranked_metadata.append((score, asset_id, metadata))
        ranked_metadata.sort(key=lambda row: (-row[0], str(row[2].get("name", "")).casefold()))
        candidates = []
        for _score, asset_id, metadata in ranked_metadata[:limit * 4]:
            candidate = self._candidate(asset_id, metadata)
            if candidate:
                candidates.append(candidate)
            if len(candidates) >= limit:
                break
        return rank_candidates(query, candidates, allowed_domains=POLYHAVEN_ALLOWED_DOMAINS)[:limit]

    def suggested_queries(self, *, limit: int = 5) -> list[dict[str, Any]]:
        """Suggest concrete common nouns known to occur in the loaded model index."""
        terms = ("lamp", "plant", "chair", "table", "bottle", "book", "stone",
                 "crate", "barrel", "candle", "sword", "ball")
        models = self._load_models()
        counts = []
        for term in terms:
            count = sum(term in self._metadata_text(metadata) for metadata in models.values())
            if count:
                counts.append({"query": term, "matching_models": count})
        counts.sort(key=lambda item: (-item["matching_models"], item["query"]))
        return counts[:limit]

    @staticmethod
    def describe(candidate: AssetCandidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate_asset_id(candidate),
            "title": candidate.title,
            "description": candidate.description,
            "author": candidate.author,
            "license": candidate.license,
            "format": "FBX with texture dependencies",
            "estimated_bytes": candidate.asset_size_bytes,
            "source_url": candidate.source_url,
            "preview_url": candidate.preview_url,
        }
