from beatforge import PolyHavenBrowserSearch, candidate_asset_id


class FakeBrowser:
    def __init__(self):
        self.urls = []

    def read_json(self, url):
        self.urls.append(url)
        if url.endswith("/assets?t=models"):
            return {"dirty_football": {
                "type": 2, "name": "Dirty Football", "description": "Worn soccer ball prop",
                "tags": ["soccer", "ball"], "authors": {"Rohit Seervi": "All"},
                "thumbnail_url": "https://cdn.polyhaven.com/football.png",
            }, "wood_floor": {"type": 1, "name": "Wood Floor", "tags": ["soccer"]}}
        if url.endswith("/files/dirty_football"):
            return {"fbx": {"1k": {"fbx": {
                "url": "https://dl.polyhaven.org/file/football.fbx", "size": 100,
                "include": {"textures/albedo.jpg": {
                    "url": "https://dl.polyhaven.org/file/albedo.jpg", "size": 50}}}}}}
        raise AssertionError(f"unexpected browser URL: {url}")


def test_polyhaven_browser_search_returns_only_importable_model_metadata():
    browser = FakeBrowser()
    search = PolyHavenBrowserSearch(browser)
    candidates = search.search("soccer ball", limit=2)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.title == "Dirty Football"
    assert candidate.license == "CC0"
    assert candidate.download_url.endswith("football.fbx")
    assert candidate.dependencies == (("textures/albedo.jpg", "https://dl.polyhaven.org/file/albedo.jpg"),)
    assert candidate_asset_id(candidate) == search.describe(candidate)["candidate_id"]
    assert search.describe(candidate)["estimated_bytes"] == 150
    assert "ball" in {item["query"] for item in search.suggested_queries()}
    assert browser.urls == [
        "https://api.polyhaven.com/assets?t=models",
        "https://api.polyhaven.com/files/dirty_football",
    ]


def test_polyhaven_search_bounds_query_and_result_count():
    search = PolyHavenBrowserSearch(FakeBrowser())
    for query, limit in (("", 1), ("valid", 6)):
        try:
            search.search(query, limit=limit)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid browser query was accepted")
