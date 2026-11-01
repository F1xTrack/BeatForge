import hashlib
import io
import zipfile

import pytest

from beatforge import AssetCandidate, AssetLibrary, candidate_asset_id, fetch_candidate


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, url: str):
        super().__init__(payload)
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def geturl(self):
        return self._url


def test_fetch_candidate_hashes_and_records_provenance(tmp_path):
    payload = b"glb bytes"
    candidate = AssetCandidate("Portal prop", "https://polyhaven.com/a/portal.glb", "CC0", author="A")
    library = AssetLibrary(tmp_path / "assets.json")

    def opener(_request, timeout):
        assert timeout == 3
        return _Response(payload, candidate.source_url)

    record = fetch_candidate(candidate, library, tmp_path / "objects",
                             allowed_domains=("polyhaven.com",), timeout=3, opener=opener)
    assert record.asset_id == candidate_asset_id(candidate)
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / "objects" / ("Portal-prop-" + record.asset_id + ".glb")).read_bytes() == payload
    assert library.get(record.asset_id).source_url == candidate.source_url


def test_fetch_candidate_rejects_oversize_and_cleans_partial_file(tmp_path):
    candidate = AssetCandidate("Large", "https://polyhaven.com/a/large.glb", "CC0")

    def opener(_request, timeout):
        return _Response(b"12345", candidate.source_url)

    with pytest.raises(ValueError, match="import limit"):
        fetch_candidate(candidate, AssetLibrary(tmp_path / "assets.json"), tmp_path / "objects",
                        allowed_domains=("polyhaven.com",), max_bytes=4, opener=opener)
    assert not list((tmp_path / "objects").glob("*.part"))


def test_fetch_candidate_rejects_cross_domain_redirect(tmp_path):
    candidate = AssetCandidate("Redirect", "https://polyhaven.com/a/x.glb", "CC0")

    def opener(_request, timeout):
        return _Response(b"x", "https://example.com/x.glb")

    with pytest.raises(ValueError, match="redirect"):
        fetch_candidate(candidate, AssetLibrary(tmp_path / "assets.json"), tmp_path / "objects",
                        allowed_domains=("polyhaven.com",), opener=opener)


def test_fetch_candidate_bundles_model_dependencies_without_executing_them(tmp_path):
    candidate = AssetCandidate(
        "Desk prop", "https://polyhaven.com/a/desk_prop", "CC0",
        download_url="https://dl.polyhaven.org/desk_prop.fbx",
        dependencies=(("textures/albedo.jpg", "https://dl.polyhaven.org/albedo.jpg"),),
    )

    def opener(request, timeout):
        payload = {"https://dl.polyhaven.org/desk_prop.fbx": b"fbx-data",
                   "https://dl.polyhaven.org/albedo.jpg": b"texture-data"}[request.full_url]
        return _Response(payload, "https://dl.polyhaven.org/file")

    record = fetch_candidate(
        candidate, AssetLibrary(tmp_path / "assets.json"), tmp_path / "objects",
        allowed_domains=("polyhaven.com", "dl.polyhaven.org"), opener=opener,
    )
    assert record.local_path.endswith(".zip")
    with zipfile.ZipFile(record.local_path) as archive:
        assert set(archive.namelist()) == {"desk_prop.fbx", "textures/albedo.jpg"}
        assert archive.read("desk_prop.fbx") == b"fbx-data"
        assert archive.read("textures/albedo.jpg") == b"texture-data"
