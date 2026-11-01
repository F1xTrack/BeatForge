from pathlib import Path

import pytest

from beatforge import (AssetCandidate, AssetLibrary, AssetRecord, VivifyToolExecutor,
                       candidate_asset_id)
from beatforge.polyhaven import POLYHAVEN_ALLOWED_DOMAINS


def test_vivify_tool_executor_only_imports_previously_searched_candidates(tmp_path):
    candidate = AssetCandidate("Portal prop", "https://polyhaven.com/a/portal_prop", "CC0")

    class Searcher:
        def search(self, query, *, limit):
            assert query == "portal prop"
            assert limit == 2
            return [candidate]

        @staticmethod
        def describe(value):
            return {"candidate_id": candidate_asset_id(value), "title": value.title}

    library = AssetLibrary(tmp_path / "assets.json")
    fetched = []

    def fetcher(value, target_library, destination, **kwargs):
        assert kwargs["allowed_domains"] == POLYHAVEN_ALLOWED_DOMAINS
        assert kwargs["max_bytes"] == 64 * 1024 * 1024
        fetched.append(value)
        path = Path(destination) / "prop.fbx"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
        return target_library.add(AssetRecord(
            candidate_asset_id(value), value.title, value.source_url, str(path),
            value.license, value.author,
        ))

    tools = VivifyToolExecutor(Searcher(), library, tmp_path / "objects", fetcher=fetcher)
    results = tools("search_polyhaven_assets", {"query": "portal prop", "limit": 2})
    assert results["results"][0]["candidate_id"] == candidate_asset_id(candidate)

    imported = tools("import_polyhaven_asset", {"candidate_id": candidate_asset_id(candidate)})
    assert imported["status"] == "imported"
    assert imported["sha256"]
    assert tools.imported_asset_ids == [candidate_asset_id(candidate)]
    assert [action["tool"] for action in tools.actions] == [
        "search_polyhaven_assets", "import_polyhaven_asset",
    ]
    assert tools.actions[0]["candidate_count"] == 1
    assert tools.actions[1]["asset_id"] == candidate_asset_id(candidate)
    assert fetched == [candidate]

    with pytest.raises(ValueError, match="current search"):
        tools("import_polyhaven_asset", {"candidate_id": "x" * 20})


def test_vivify_tool_executor_stops_after_search_and_import_bounds(tmp_path):
    candidate = AssetCandidate("Lamp", "https://polyhaven.com/a/lamp", "CC0")

    class Searcher:
        def search(self, _query, *, limit):
            return [candidate][:limit]

        @staticmethod
        def describe(_candidate):
            return {}

    tools = VivifyToolExecutor(Searcher(), AssetLibrary(tmp_path / "assets.json"),
                               tmp_path / "objects", fetcher=lambda *_args, **_kwargs: None)
    for _ in range(3):
        tools("search_polyhaven_assets", {"query": "lamp"})
    with pytest.raises(ValueError, match="call limit"):
        tools("search_polyhaven_assets", {"query": "lamp"})


def test_zero_result_search_returns_catalog_noun_suggestions(tmp_path):
    class Searcher:
        def search(self, _query, *, limit):
            return []

        def suggested_queries(self, *, limit):
            return [{"query": "lamp", "matching_models": 17}][:limit]

        @staticmethod
        def describe(_candidate):
            return {}

    tools = VivifyToolExecutor(Searcher(), AssetLibrary(tmp_path / "assets.json"),
                               tmp_path / "objects")
    result = tools("search_polyhaven_assets", {"query": "futuristic"})
    assert result["results"] == []
    assert result["suggested_queries"] == [{"query": "lamp", "matching_models": 17}]
    assert tools.actions[0]["suggested_queries"] == result["suggested_queries"]


def test_import_schema_only_exposes_ids_from_the_latest_search(tmp_path):
    from beatforge.vivify_tools import vivify_tool_schemas

    first = AssetCandidate("First prop", "https://polyhaven.com/a/first_prop", "CC0")
    second = AssetCandidate("Second prop", "https://polyhaven.com/a/second_prop", "CC0")

    class Searcher:
        def search(self, query, *, limit):
            assert limit == 1
            return [first if query == "first" else second]

        @staticmethod
        def describe(candidate):
            return {"title": candidate.title}

    tools = VivifyToolExecutor(Searcher(), AssetLibrary(tmp_path / "assets.json"),
                               tmp_path / "objects", fetcher=lambda *_a, **_k: None)
    first_id = candidate_asset_id(first)
    second_id = candidate_asset_id(second)

    schemas_before_search = vivify_tool_schemas(tools)
    assert [schema["function"]["name"] for schema in schemas_before_search] == [
        "search_polyhaven_assets",
    ]

    tools("search_polyhaven_assets", {"query": "first", "limit": 1})
    first_import_schema = vivify_tool_schemas(tools)[1]
    assert first_import_schema["function"]["parameters"]["properties"]["candidate_id"]["enum"] == [first_id]

    tools("search_polyhaven_assets", {"query": "second", "limit": 1})
    second_import_schema = vivify_tool_schemas(tools)[1]
    assert second_import_schema["function"]["parameters"]["properties"]["candidate_id"]["enum"] == [second_id]
    with pytest.raises(ValueError, match="current search"):
        tools("import_polyhaven_asset", {"candidate_id": first_id})
