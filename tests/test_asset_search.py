from beatforge import AssetCandidate, rank_candidates, validate_candidate


def test_asset_search_filters_and_ranks_safe_candidates():
    candidates = [
        AssetCandidate("Rock", "https://polyhaven.com/a/rock", "CC0", "A"),
        AssetCandidate("Portal frame", "https://ambientcg.com/a/frame", "CC-BY", "B"),
        AssetCandidate("Portal unsafe", "https://example.com/x", "CC0"),
    ]
    result = rank_candidates("portal", candidates, allowed_domains=("polyhaven.com", "ambientcg.com"))
    assert [item.title for item in result] == ["Portal frame", "Rock"]


def test_asset_search_reports_https_and_license_errors():
    errors = validate_candidate(AssetCandidate("x", "http://polyhaven.com/x", "Unknown"), ("polyhaven.com",))
    assert any("HTTPS" in error for error in errors)
    assert any("license" in error for error in errors)
