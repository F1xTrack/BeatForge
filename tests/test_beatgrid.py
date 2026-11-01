import pytest

from beatforge import normalize_beat_grid


def test_beatgrid_preserves_missing_uncertainty():
    grid = normalize_beat_grid({"beats": [0, 0.5, 1], "downbeats": [0], "bpm": 120}, source="beat-this")
    assert grid.beats == (0.0, 0.5, 1.0)
    assert grid.bpm == 120
    assert grid.confidence is None
    assert grid.source == "beat-this"


def test_beatgrid_rejects_invalid_downbeats():
    with pytest.raises(ValueError, match="members of beats"):
        normalize_beat_grid({"beats": [0, 1], "downbeats": [0.5]})
