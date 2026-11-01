"""Compact, deterministic audio evidence for text-only mapping models."""
from __future__ import annotations

import math
from pathlib import Path


def describe_audio_grid(path: str | Path, bpm: float, *, beats_per_row: int = 8) -> str:
    """Summarize energy and spectral attacks on a beat-aligned half-beat grid.

    This describes measured signal features; it does not identify instruments
    or claim that an onset is a note. Each frequency band is normalized
    independently to the track's 95th percentile for compact display.
    """
    if not isinstance(bpm, (int, float)) or isinstance(bpm, bool) or not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("bpm must be finite and positive for audio feature extraction")
    if not isinstance(beats_per_row, int) or isinstance(beats_per_row, bool) or beats_per_row < 1:
        raise ValueError("beats_per_row must be a positive integer")
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("numpy and soundfile are required for audio feature extraction") from exc

    samples, sample_rate = sf.read(str(Path(path)), dtype="float32", always_2d=True)
    if samples.size == 0 or sample_rate <= 0:
        raise ValueError("audio input is empty or unreadable")
    mono = np.mean(samples, axis=1, dtype=np.float64)
    duration = len(mono) / sample_rate
    cell_seconds = 30.0 / bpm  # one cell is one eighth note (half a beat)
    cell_count = int(math.ceil(duration / cell_seconds))
    expected_samples = max(1, int(math.ceil(sample_rate * cell_seconds)))
    fft_size = 1 << max(11, (expected_samples - 1).bit_length())
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    band_masks = (
        (frequencies >= 60) & (frequencies < 250),
        (frequencies >= 250) & (frequencies < 2000),
        (frequencies >= 2000) & (frequencies < min(8000, sample_rate / 2)),
    )
    band_values: list[list[float]] = [[], [], []]
    attack_values: list[float] = []
    previous_spectrum = None
    for cell in range(cell_count):
        start = int(round(cell * cell_seconds * sample_rate))
        end = min(len(mono), int(round((cell + 1) * cell_seconds * sample_rate)))
        segment = mono[start:end]
        if len(segment) == 0:
            segment = np.zeros(1, dtype=np.float64)
        window = np.hanning(len(segment)) if len(segment) > 1 else np.ones(1)
        spectrum = np.abs(np.fft.rfft(segment * window, n=fft_size))
        power = spectrum * spectrum
        for output, mask in zip(band_values, band_masks):
            output.append(float(np.sqrt(np.mean(power[mask]))) if np.any(mask) else 0.0)
        normalized = spectrum / max(float(np.sum(spectrum)), 1e-12)
        attack_values.append(0.0 if previous_spectrum is None else
                             float(np.maximum(normalized - previous_spectrum, 0.0).sum()))
        previous_spectrum = normalized

    def quantize(values: list[float]) -> str:
        array = np.asarray(values, dtype=np.float64)
        reference = float(np.quantile(array, 0.95)) if len(array) else 0.0
        if not math.isfinite(reference) or reference <= 1e-12:
            return "0" * len(values)
        levels = np.clip(np.rint(array / reference * 9), 0, 9).astype(int)
        return "".join(str(value) for value in levels)

    encoded = [quantize(values) for values in (*band_values, attack_values)]
    cells_per_row = beats_per_row * 2
    rows: list[str] = []
    for first in range(0, cell_count, cells_per_row):
        last = min(cell_count, first + cells_per_row)
        beat_start = first / 2
        beat_end = min(duration * bpm / 60.0, last / 2)
        rows.append(
            f"beats {beat_start:.1f}-{beat_end:.1f} "
            f"low={encoded[0][first:last]} mid={encoded[1][first:last]} "
            f"high={encoded[2][first:last]} attack={encoded[3][first:last]}"
        )
    header = (
        f"Duration={duration:.3f}s; BPM anchor={float(bpm):.3f}; audio start is beat 0; "
        "each character covers half a beat. Band values are independently quantized 0-9: "
        "low=60-250Hz, mid=250-2000Hz, high=2000-8000Hz; attack is positive spectral change. "
        "These are measured signal features, not detected notes or instrument labels."
    )
    return header + "\n" + "\n".join(rows)
