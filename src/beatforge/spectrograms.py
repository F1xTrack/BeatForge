"""BeatBench SG-compatible fixed spectrogram panels for model image input."""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import soundfile as sf
import soxr

from .spectral_ridges import detect_sustained_ridges


SPECTROGRAM_SETTINGS = {
    "version": 2, "sample_rate": 44100, "n_fft": 1024, "hop_length": 128,
    "segment_seconds": 4.0, "width": 1400, "height": 480,
    "db_min": -80.0, "db_max": 0.0, "frequency_min": 40.0,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _staging_temp_parent(source: Path, staged_bytes: int) -> Path:
    """Choose a temp volume with room for the temporary PCM16 WAV."""
    margin = 16 * 1024 * 1024
    candidates = dict.fromkeys((Path(tempfile.gettempdir()), source.parent, Path.cwd()))
    for candidate in candidates:
        try:
            if shutil.disk_usage(candidate).free >= staged_bytes + margin:
                return candidate
        except OSError:
            continue
    raise OSError(
        "insufficient accessible disk space for the temporary spectrogram WAV "
        f"({staged_bytes + margin} bytes required in the temp directory, audio folder, "
        "or current working directory)"
    )


def pack_spectrogram_images(images: tuple[bytes, ...], max_images: int,
                            manifest: dict | None = None) -> tuple[tuple[bytes, ...], dict]:
    """Pack consecutive SG panels vertically without resampling source pixels.

    Keep the complete-track overview as its own image. Remaining panels are
    grouped in chronological order so providers with an image-count limit can
    still receive the complete SG representation in one request.
    """
    if not isinstance(max_images, int) or isinstance(max_images, bool) or max_images <= 0:
        raise ValueError("max_images must be a positive integer")
    if len(images) <= max_images:
        return images, {"image_limit": max_images, "image_count": len(images),
                        "source_panel_count": len(images), "packing": "none"}
    if max_images < 2:
        raise ValueError("max_images must be at least 2 to preserve overview and SG panels")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to pack spectrogram images") from exc

    panel_rows = (manifest or {}).get("images") or []
    names = [str(row.get("file", f"panel_{i:03d}.png"))
             for i, row in enumerate(panel_rows)]
    if len(names) != len(images):
        names = [f"panel_{i:03d}.png" for i in range(len(images))]

    overview = images[0]
    segments = images[1:]
    group_size = math.ceil(len(segments) / (max_images - 1))
    packed = [overview]
    groups = [[names[0]]]
    separator = 8
    for start in range(0, len(segments), group_size):
        chunk = segments[start:start + group_size]
        rendered = []
        for data in chunk:
            with Image.open(io.BytesIO(data)) as image:
                rendered.append(image.convert("RGB"))
        widths = {image.width for image in rendered}
        if len(widths) != 1:
            raise ValueError("spectrogram panel widths must match before packing")
        width = rendered[0].width
        height = sum(image.height for image in rendered) + separator * (len(rendered) - 1)
        composite = Image.new("RGB", (width, height), "white")
        y = 0
        for image in rendered:
            composite.paste(image, (0, y))
            y += image.height + separator
        buffer = io.BytesIO()
        composite.save(buffer, format="PNG")
        packed.append(buffer.getvalue())
        groups.append(names[start + 1:start + 1 + len(chunk)])
    return tuple(packed), {
        "image_limit": max_images,
        "image_count": len(packed),
        "source_panel_count": len(images),
        "packing": "overview plus vertical stacks of consecutive panels",
        "groups": groups,
    }


def build_spectrograms(audio: str | Path, *, max_audio_seconds: float = 240.0) -> tuple[tuple[bytes, ...], dict]:
    """Return the same overview plus consecutive 4-second panels as BeatBench SG.

    The audio itself is never put into a model message. Signal conditioning,
    STFT parameters, scale and plot layout match ``beatbench.runner.representations``.
    """
    if max_audio_seconds <= 0:
        raise ValueError("max_audio_seconds must be positive")
    source = Path(audio)
    if not source.is_file():
        raise FileNotFoundError(source)
    # Match BeatBench's shared signal staging: 44.1 kHz PCM16, then downmix
    # to mono at the configured display rate. Temporary staging is deleted.
    signal, source_sr = sf.read(source, dtype="float32", always_2d=True)
    if signal.size == 0:
        raise ValueError("audio is empty")
    signal = signal[:int(max_audio_seconds * source_sr)]
    if signal.size == 0:
        raise ValueError("audio is empty")
    if source_sr != 44100:
        signal = soxr.resample(signal, source_sr, 44100, quality="HQ", axis=0)
    staged_bytes = signal.shape[0] * signal.shape[1] * 2 + 65536
    temp_parent = _staging_temp_parent(source, staged_bytes)
    with tempfile.TemporaryDirectory(prefix="beatforge-sg-", dir=temp_parent) as temporary:
        staged = Path(temporary) / "audio.wav"
        sf.write(staged, signal, 44100, subtype="PCM_16")
        settings = dict(SPECTROGRAM_SETTINGS)
        staged_signal, _staged_sr = sf.read(staged, dtype="float32", always_2d=True)
    mono = np.mean(staged_signal, axis=1, dtype=np.float32)
    sample_rate = 44100
    if sample_rate != settings["sample_rate"]:
        mono = soxr.resample(mono, sample_rate, settings["sample_rate"], quality="HQ")
        sample_rate = settings["sample_rate"]
    if mono.size == 0:
        raise ValueError("audio is empty after spectrogram resampling")
    fft_size = settings["n_fft"]
    hop_length = settings["hop_length"]
    padded = np.pad(mono, (fft_size // 2, fft_size // 2))
    starts = np.arange(0, len(padded) - fft_size + 1, hop_length)
    frames = padded[starts[:, None] + np.arange(fft_size)[None, :]]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(fft_size) / fft_size)
    magnitude = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).T
    reference = max(float(magnitude.max()), 1e-8)
    db = 20 * np.log10(np.maximum(magnitude, 1e-5) / reference)
    db = np.maximum(db, float(db.max()) - 80.0)
    times = np.arange(db.shape[1]) * hop_length / sample_rate
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    duration = len(mono) / sample_rate
    parts: list[bytes] = []
    images: list[dict] = []
    intervals = [("overview", 0.0, duration)] + [
        (f"segment_{i:03d}", float(start), min(float(start + settings["segment_seconds"]), duration))
        for i, start in enumerate(np.arange(0, duration, settings["segment_seconds"]))
    ]
    for name, start, end in intervals:
        figure, axis = plt.subplots(
            figsize=(settings["width"] / 100, settings["height"] / 100), dpi=100)
        selected = np.flatnonzero((times >= start) & (times <= end))
        selected = selected[::max(1, int(np.ceil(len(selected) / settings["width"]))) ]
        axis.pcolormesh(times[selected], frequencies, db[:, selected], shading="auto",
                        cmap="magma", vmin=settings["db_min"], vmax=settings["db_max"],
                        rasterized=True)
        axis.set_yscale("log")
        axis.set_ylim(settings["frequency_min"], sample_rate / 2)
        axis.set_xlim(start, max(start + .001, end))
        axis.set_xlabel("Absolute time (seconds)")
        axis.set_ylabel("Frequency (Hz)")
        axis.set_title(f"{name}: {start:.3f}–{end:.3f} s | fixed scale −80…0 dB")
        if name != "overview":
            axis.set_xticks(np.arange(start, end + .0001, .2))
            axis.tick_params(axis="x", labelsize=9)
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", metadata={"Software": "BeatBench spectrogram v2"})
        plt.close(figure)
        image = buffer.getvalue()
        parts.append(image)
        images.append({"file": f"{name}.png", "start_seconds": start,
                       "end_seconds": end, "sha256": _sha256(image)})
    manifest = {
        "settings": settings,
        "images": images,
        "duration_seconds": duration,
        "spectral_ridge_analysis": {
            "version": 1,
            "fft_size": 2048,
            "hop_seconds": 0.1,
            "meaning": "unverified persistent narrow-peak candidates; not note transcription",
        },
        "sustained_ridge_candidates": detect_sustained_ridges(mono, sample_rate),
    }
    return tuple(parts), manifest
