"""Conservative, unlabelled candidates for persistent spectral ridges."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def detect_sustained_ridges(signal: np.ndarray, sample_rate: int) -> list[dict[str, Any]]:
    """Find a short list of narrow spectral peaks that persist over time.

    Results are SG navigation hints, not pitch/onset transcription. They must
    be checked against the source spectrogram before an arc is authored.
    """
    mono = np.asarray(signal, dtype=np.float32)
    if mono.ndim != 1 or sample_rate <= 0:
        raise ValueError("spectral-ridge input must be mono with a positive sample rate")
    if mono.size < sample_rate // 2 or float(np.max(np.abs(mono), initial=0.0)) <= 1e-8:
        return []

    fft_size = 2048
    hop_length = max(1, int(round(sample_rate * 0.1)))
    starts = np.arange(0, mono.size - fft_size + 1, hop_length)
    if len(starts) < 7:
        return []
    frames = mono[starts[:, None] + np.arange(fft_size)[None, :]]
    window = np.hanning(fft_size).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).T
    reference = max(float(magnitude.max()), 1e-8)
    if reference <= 1e-7:
        return []
    decibels = 20.0 * np.log10(np.maximum(magnitude, 1e-8) / reference)
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

    active: list[dict[str, list[float] | list[int]]] = []
    finished: list[dict[str, list[float] | list[int]]] = []
    for frame_index in range(magnitude.shape[1]):
        row = decibels[:, frame_index]
        peak_mask = (row[1:-1] >= row[:-2]) & (row[1:-1] > row[2:])
        peak_bins = np.flatnonzero(peak_mask) + 1
        peak_bins = [int(index) for index in peak_bins
                     if 160.0 <= frequencies[index] <= 4000.0
                     and row[index] >= -35.0
                     and row[index] >= float(row.max()) - 22.0]
        peak_bins.sort(key=lambda index: float(row[index]), reverse=True)
        peak_bins = peak_bins[:18]

        used: set[int] = set()
        for frequency_bin in peak_bins:
            matches = [
                (abs(frequency_bin - int(track["bins"][-1])), index)
                for index, track in enumerate(active)
                if index not in used
                and frame_index - int(track["frames"][-1]) <= 2
                and abs(frequency_bin - int(track["bins"][-1])) <= 2
            ]
            if matches:
                _, track_index = min(matches)
                track = active[track_index]
                track["frames"].append(frame_index)
                track["bins"].append(frequency_bin)
                track["levels"].append(float(row[frequency_bin]))
                used.add(track_index)
            else:
                active.append({"frames": [frame_index], "bins": [frequency_bin],
                               "levels": [float(row[frequency_bin])]})
                used.add(len(active) - 1)

        still_active = []
        for track in active:
            if frame_index - int(track["frames"][-1]) > 2:
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active
    finished.extend(active)

    candidates: list[dict[str, Any]] = []
    for track in finished:
        frame_ids = np.asarray(track["frames"], dtype=int)
        bins = np.asarray(track["bins"], dtype=int)
        first, last = int(frame_ids[0]), int(frame_ids[-1])
        span_frames = last - first + 1
        duration = span_frames * hop_length / sample_rate
        track_frequencies = frequencies[bins]
        median_frequency = float(np.median(track_frequencies))
        relative_spread = (float(track_frequencies.max() - track_frequencies.min())
                           / max(median_frequency, 1.0))
        coverage = len(frame_ids) / span_frames
        mean_level = float(np.mean(track["levels"]))
        if (duration < 0.7 or relative_spread > 0.08
                or coverage < 0.72 or mean_level < -30.0):
            continue
        candidates.append({
            "start_seconds": round(first * hop_length / sample_rate, 2),
            "end_seconds": round(last * hop_length / sample_rate, 2),
            "duration_seconds": round(duration, 2),
            "frequency_hz": round(median_frequency, 1),
            "mean_level_db": round(mean_level, 1),
            "coverage": round(coverage, 3),
            "ridge_tracks": [{
                "frequency_hz": round(median_frequency, 1),
                "frequency_span_hz": [round(float(track_frequencies.min()), 1),
                                      round(float(track_frequencies.max()), 1)],
                "mean_level_db": round(mean_level, 1),
                "coverage": round(coverage, 3),
            }],
        })

    # Merge overlapping octave/upper-harmonic tracks into one review hint.
    candidates.sort(key=_candidate_strength, reverse=True)
    merged: list[dict[str, Any]] = []
    for candidate in candidates:
        target = next((item for item in merged
                       if _harmonic_overlap(item, candidate) >= 0.6), None)
        if target is None:
            candidate["harmonics_hz"] = []
            merged.append(candidate)
            continue
        old_frequency = float(target["frequency_hz"])
        new_frequency = float(candidate["frequency_hz"])
        tracks = [*target["ridge_tracks"], *candidate["ridge_tracks"]]
        unique_tracks = []
        for track in tracks:
            frequency = float(track["frequency_hz"])
            if not any(math.isclose(frequency, float(existing["frequency_hz"]), rel_tol=0.025)
                       for existing in unique_tracks):
                unique_tracks.append(track)
        target["start_seconds"] = min(target["start_seconds"], candidate["start_seconds"])
        target["end_seconds"] = max(target["end_seconds"], candidate["end_seconds"])
        target["ridge_tracks"] = unique_tracks
        frequencies_in_group = [float(track["frequency_hz"]) for track in unique_tracks]
        target["frequency_hz"] = round(min(old_frequency, new_frequency), 1)
        target["harmonics_hz"] = sorted({
            round(value, 1) for value in frequencies_in_group
            if not math.isclose(value, float(target["frequency_hz"]), rel_tol=0.025)
        })
        target["duration_seconds"] = round(
            target["end_seconds"] - target["start_seconds"], 2)
        target["mean_level_db"] = max(
            float(track["mean_level_db"]) for track in unique_tracks)
        target["coverage"] = min(float(track["coverage"]) for track in unique_tracks)

    # Keep prompts concise and temporally distributed; these are hints, not quotas.
    merged.sort(key=_candidate_strength, reverse=True)
    selected: list[dict[str, Any]] = []
    per_eight_seconds: dict[int, int] = {}
    for candidate in merged:
        start = float(candidate["start_seconds"])
        bucket = int(start // 8.0)
        if per_eight_seconds.get(bucket, 0) >= 2:
            continue
        selected.append(candidate)
        per_eight_seconds[bucket] = per_eight_seconds.get(bucket, 0) + 1
        if len(selected) == 12:
            break
    return sorted(selected, key=lambda item: float(item["start_seconds"]))


def _candidate_strength(candidate: dict[str, Any]) -> float:
    duration = float(candidate["duration_seconds"])
    level = float(candidate["mean_level_db"])
    return duration * (10.0 ** (level / 20.0))


def _harmonic_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    start = max(float(left["start_seconds"]), float(right["start_seconds"]))
    end = min(float(left["end_seconds"]), float(right["end_seconds"]))
    overlap = max(0.0, end - start)
    shorter = min(float(left["duration_seconds"]), float(right["duration_seconds"]))
    if shorter <= 0 or overlap / shorter < 0.6:
        return 0.0
    ratio = max(float(left["frequency_hz"]), float(right["frequency_hz"])) / min(
        float(left["frequency_hz"]), float(right["frequency_hz"]))
    harmonic = round(ratio)
    if 2 <= harmonic <= 5 and abs(ratio - harmonic) / harmonic <= 0.03:
        return overlap / shorter
    if abs(ratio - 1.0) <= 0.05:
        return overlap / shorter
    return 0.0
