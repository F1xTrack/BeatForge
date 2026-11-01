import numpy as np

from beatforge.spectral_ridges import detect_sustained_ridges


def test_detect_sustained_tone_as_unverified_ridge_candidate():
    sample_rate = 44100
    samples = np.zeros(sample_rate * 4, dtype=np.float32)
    times = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    samples[sample_rate:sample_rate * 3] = 0.4 * np.sin(2 * np.pi * 440 * times)

    candidates = detect_sustained_ridges(samples, sample_rate)

    assert candidates
    target = max(candidates, key=lambda item: item["duration_seconds"])
    assert abs(target["frequency_hz"] - 440) <= 25
    assert target["start_seconds"] <= 1.1
    assert target["end_seconds"] >= 2.8
    assert target["duration_seconds"] >= 1.5


def test_short_tone_burst_is_not_reported_as_sustained_ridge():
    sample_rate = 44100
    samples = np.zeros(sample_rate * 2, dtype=np.float32)
    times = np.arange(sample_rate // 20, dtype=np.float32) / sample_rate
    start = sample_rate // 2
    samples[start:start + len(times)] = 0.8 * np.sin(2 * np.pi * 440 * times)

    assert detect_sustained_ridges(samples, sample_rate) == []


def test_silence_returns_no_spectral_ridge_candidates():
    assert detect_sustained_ridges(np.zeros(44100 * 2, dtype=np.float32), 44100) == []
