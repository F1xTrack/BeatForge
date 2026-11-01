import numpy as np
import pytest
import soundfile as sf

from beatforge.audio_features import describe_audio_grid


def test_audio_grid_summary_is_beat_aligned_and_compact(tmp_path):
    sample_rate = 8000
    duration = 8.0
    samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
    tone = np.sin(2 * np.pi * 120 * np.arange(int(sample_rate * 0.04)) / sample_rate)
    for onset in (0.5, 1.0, 2.0, 3.5, 5.0, 7.0):
        start = int(onset * sample_rate)
        samples[start:start + len(tone)] += tone.astype(np.float32)
    audio = tmp_path / "clicks.wav"
    sf.write(audio, samples, sample_rate)

    summary = describe_audio_grid(audio, 120)
    rows = summary.splitlines()
    assert "BPM anchor=120.000" in rows[0]
    assert "not detected notes" in rows[0]
    assert len(rows) == 3  # two 8-beat blocks plus the header
    for row in rows[1:]:
        for band in ("low", "mid", "high", "attack"):
            values = row.split(band + "=", 1)[1].split()[0]
            assert len(values) == 16
            assert set(values) <= set("0123456789")


def test_audio_grid_rejects_invalid_bpm(tmp_path):
    with pytest.raises(ValueError, match="bpm"):
        describe_audio_grid(tmp_path / "missing.wav", 0)
