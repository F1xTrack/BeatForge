"""Bounded model-callable search and import tools for Vivify assets."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .asset_fetch import candidate_asset_id, fetch_candidate
from .asset_search import AssetCandidate
from .assets import AssetLibrary, AssetRecord
from .polyhaven import MAX_QUERY_CHARS, MAX_RESULTS, PolyHavenBrowserSearch


MAX_SEARCH_CALLS = 3
MAX_IMPORT_CALLS = 3
MAX_TOOL_CALLS_PER_TURN = MAX_SEARCH_CALLS + MAX_IMPORT_CALLS
MAX_TOOL_ROUNDS_PER_TURN = MAX_TOOL_CALLS_PER_TURN + 1

VIVIFY_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_polyhaven_assets",
            "description": (
                "Search the Poly Haven CC0 3D model catalog from the isolated browser. "
                "Returns candidate IDs and factual metadata; it never downloads assets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": MAX_QUERY_CHARS},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "import_polyhaven_asset",
            "description": (
                "Download one previously returned Poly Haven candidate into the local "
                "Vivify asset library. Import is HTTPS-only, CC0-checked, redirect-checked, "
                "SHA-256 recorded and capped at 64 MiB. Never pass a URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {"candidate_id": {"type": "string", "minLength": 20, "maxLength": 20}},
                "required": ["candidate_id"],
                "additionalProperties": False,
            },
        },
    },
]


class VivifyToolExecutor:
    """Execute only catalog search and previously approved candidate imports."""

    def __init__(
        self,
        searcher: PolyHavenBrowserSearch,
        library: AssetLibrary,
        destination: str | Path,
        *,
        fetcher: Callable[..., AssetRecord] = fetch_candidate,
    ):
        self.searcher = searcher
        self.library = library
        self.destination = Path(destination)
        self.fetcher = fetcher
        self.candidates: dict[str, AssetCandidate] = {}
        self.current_search_candidate_ids: list[str] = []
        self.search_calls = 0
        self.import_calls = 0
        self.imported_asset_ids: list[str] = []
        self.actions: list[dict[str, Any]] = []

    def __call__(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        try:
            if name == "search_polyhaven_assets":
                result = self._search(arguments)
                suggestions = result.get("suggested_queries", [])
                self.actions.append({
                    "tool": name, "query": str(arguments.get("query", ""))[:MAX_QUERY_CHARS],
                    "candidate_count": len(result.get("results", [])),
                    "candidates": [
                        {"candidate_id": item.get("candidate_id"), "title": item.get("title")}
                        for item in result.get("results", []) if isinstance(item, dict)
                    ],
                    "suggested_queries": suggestions,
                    "status": "completed",
                })
                return result
            if name == "import_polyhaven_asset":
                result = self._import(arguments)
                self.actions.append({
                    "tool": name, "candidate_id": arguments.get("candidate_id"),
                    "asset_id": result.get("asset_id"), "title": result.get("title"),
                    "license": result.get("license"), "sha256": result.get("sha256"),
                    "status": result.get("status", "completed"),
                })
                return result
            raise ValueError("unknown Vivify asset tool")
        except (ValueError, RuntimeError) as exc:
            self.actions.append({
                "tool": name, "status": "error", "error": str(exc)[:240],
            })
            raise

    def _search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if set(arguments) - {"query", "limit"}:
            raise ValueError("search accepts only query and limit")
        self.search_calls += 1
        if self.search_calls > MAX_SEARCH_CALLS:
            raise ValueError("search call limit reached")
        self.current_search_candidate_ids = []
        query = arguments.get("query")
        limit = arguments.get("limit", MAX_RESULTS)
        candidates = self.searcher.search(query, limit=limit)
        results = []
        for candidate in candidates:
            candidate_id = candidate_asset_id(candidate)
            self.candidates[candidate_id] = candidate
            self.current_search_candidate_ids.append(candidate_id)
            results.append({**self.searcher.describe(candidate), "candidate_id": candidate_id})
        response = {"provider": "Poly Haven", "license": "CC0", "results": results}
        if not results:
            suggest = getattr(self.searcher, "suggested_queries", None)
            if callable(suggest):
                response["suggested_queries"] = suggest(limit=5)
        return response

    def _import(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if set(arguments) != {"candidate_id"}:
            raise ValueError("import accepts only a candidate_id returned by search")
        self.import_calls += 1
        if self.import_calls > MAX_IMPORT_CALLS:
            raise ValueError("asset import call limit reached")
        candidate_id = arguments.get("candidate_id")
        if candidate_id not in self.current_search_candidate_ids:
            raise ValueError("candidate_id was not returned by the current search")
        candidate = self.candidates.get(candidate_id)
        if candidate is None:
            raise ValueError("candidate_id was not returned by the current search")
        from .polyhaven import POLYHAVEN_ALLOWED_DOMAINS
        record = self.fetcher(candidate, self.library, self.destination,
                              allowed_domains=POLYHAVEN_ALLOWED_DOMAINS,
                              max_bytes=64 * 1024 * 1024)
        if record.asset_id not in self.imported_asset_ids:
            self.imported_asset_ids.append(record.asset_id)
        return {
            "asset_id": record.asset_id,
            "title": record.title,
            "source_url": record.source_url,
            "license": record.license,
            "sha256": record.sha256,
            "status": "imported",
        }


def tool_result_json(value: Any) -> str:
    """Serialize bounded tool output for the model conversation."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def vivify_tool_schemas(executor: Any) -> list[dict[str, Any]]:
    """Expose imports only for IDs returned by the latest asset search."""
    schemas = [deepcopy(VIVIFY_TOOL_SCHEMAS[0])]
    candidate_ids = getattr(executor, "current_search_candidate_ids", None)
    if candidate_ids is None:
        # Keep lightweight test/custom executors compatible; the production executor
        # always supplies the stricter current-search list.
        candidate_ids = tuple(getattr(executor, "candidates", {}))
    candidate_ids = list(dict.fromkeys(
        item for item in candidate_ids if isinstance(item, str) and len(item) == 20
    ))
    if candidate_ids and getattr(executor, "import_calls", 0) < MAX_IMPORT_CALLS:
        import_schema = deepcopy(VIVIFY_TOOL_SCHEMAS[1])
        import_schema["function"]["parameters"]["properties"]["candidate_id"]["enum"] = candidate_ids
        schemas.append(import_schema)
    return schemas
