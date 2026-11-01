import hashlib

import pytest

from beatforge import AssetLibrary, AssetRecord


def test_asset_library_keeps_license_provenance_and_hash(tmp_path):
    blob = tmp_path / "portal.glb"
    blob.write_bytes(b"asset")
    library = AssetLibrary(tmp_path / "assets.json")
    record = library.add(AssetRecord("portal-1", "Portal prop", "https://example.test/p.glb",
                                     str(blob), "CC0", author="Author"))
    assert record.sha256 == hashlib.sha256(b"asset").hexdigest()
    assert library.search("portal")[0].source_url.startswith("https://")
    assert library.get("portal-1").license == "CC0"


def test_asset_library_rejects_unknown_license(tmp_path):
    blob = tmp_path / "x.glb"
    blob.write_bytes(b"x")
    with pytest.raises(ValueError):
        AssetLibrary(tmp_path / "assets.json").add(
            AssetRecord("x", "x", "https://example.test/x", str(blob), "All rights reserved"))
