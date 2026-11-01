"""Cheap local audio facts used before any model call.

This is intentionally a small first layer. It provides stable timing/energy
windows; a later optional Beat This adapter can add beats and downbeats without
changing this contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import wave


@dataclass(frozen=True, slots=True)
class EnergyWindow:
    start_seconds: float
    end_seconds: float
    rms: float
    peak: float


@dataclass(frozen=True, slots=True)
class SongAnalysis:
    source: str
    duration_seconds: float
    sample_rate: int
    channels: int
    windows: tuple[EnergyWindow, ...]
    beat_source: str = "not-run"


def _stats(samples) -> tuple[float, float]:
    values = [float(value) for value in samples]
    if not values:
        return 0.0, 0.0
    return math.sqrt(math.fsum(value * value for value in values) / len(values)), max(abs(value) for value in values)


def _read_soundfile(path: Path, window_seconds: float) -> SongAnalysis:
    import soundfile as sf  # optional dependency, needed for OGG/FLAC

    windows: list[EnergyWindow] = []
    with sf.SoundFile(path) as stream:
        sample_rate, channels = int(stream.samplerate), int(stream.channels)
        total_frames = int(stream.frames)
        window_frames = max(1, int(round(window_seconds * sample_rate)))
        offset = 0
        while True:
            chunk = stream.read(window_frames, dtype="float32", always_2d=True)
            if len(chunk) == 0:
                break
            flat = chunk.reshape(-1).tolist()
            rms, peak = _stats(flat)
            end = min(total_frames, offset + len(chunk)) / sample_rate
            windows.append(EnergyWindow(offset / sample_rate, end, rms, peak))
            offset += len(chunk)
    if not windows:
        raise ValueError(f"empty audio: {path}")
    return SongAnalysis(str(path), total_frames / sample_rate, sample_rate, channels, tuple(windows))


def _read_wave(path: Path, window_seconds: float) -> SongAnalysis:
    with wave.open(str(path), "rb") as stream:
        sample_rate = stream.getframerate()
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        total_frames = stream.getnframes()
        if width not in (1, 2, 3, 4):
            raise ValueError(f"unsupported PCM width: {width}")
        window_frames = max(1, int(round(window_seconds * sample_rate)))
        windows: list[EnergyWindow] = []
        offset = 0
        scale = float(1 << (8 * width - 1))
        while offset < total_frames:
            raw = stream.readframes(window_frames)
            if not raw:
                break
            # WAV fixtures and the fallback path are normally 16-bit. For other
            # widths, keep a conservative byte-level peak instead of guessing a
            # signed PCM layout that could create false evidence.
            if width == 2:
                values = [int.from_bytes(raw[i:i + 2], "little", signed=True) / scale
                          for i in range(0, len(raw) - 1, 2)]
            else:
                values = [byte / 255.0 * 2.0 - 1.0 for byte in raw]
            rms, peak = _stats(values)
            frames = len(raw) // max(1, width * channels)
            end = min(total_frames, offset + frames) / sample_rate
            windows.append(EnergyWindow(offset / sample_rate, end, rms, peak))
            offset += frames
    if not windows:
        raise ValueError(f"empty audio: {path}")
    return SongAnalysis(str(path), total_frames / sample_rate, sample_rate, channels, tuple(windows))


def analyze_audio(path: str | Path, window_seconds: float = 4.0) -> SongAnalysis:
    """Read local audio without network access or model inference."""
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        return _read_soundfile(source, window_seconds)
    except (ImportError, ModuleNotFoundError):
        if source.suffix.lower() != ".wav":
            raise RuntimeError("soundfile is required for non-WAV input")
        return _read_wave(source, window_seconds)
