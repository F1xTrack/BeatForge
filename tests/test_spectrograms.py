from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from beatforge.spectrograms import (SPECTROGRAM_SETTINGS, build_spectrograms,
                                    pack_spectrogram_images)


def test_spectrograms_match_beatbench_sg_panels_and_are_deterministic(tmp_path):
    sample_rate = 44100
    samples = np.zeros(sample_rate * 8, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * np.arange(2205) / sample_rate).astype(np.float32)
    for second in (0.5, 1.5, 3.5, 6.5, 7.5):
        start = round(second * sample_rate)
        samples[start:start + len(tone)] = tone
    audio = tmp_path / "test.wav"
    sf.write(audio, samples, sample_rate, subtype="PCM_16")

    images, manifest = build_spectrograms(audio)
    second_images, second_manifest = build_spectrograms(audio)

    assert manifest == second_manifest
    assert images == second_images
    assert manifest["settings"] == SPECTROGRAM_SETTINGS
    assert [(item["start_seconds"], item["end_seconds"])
            for item in manifest["images"]] == [(0.0, 8.0), (0.0, 4.0), (4.0, 8.0)]
    assert len(images) == 3
    assert manifest["sustained_ridge_candidates"] == []
    assert manifest["spectral_ridge_analysis"]["version"] == 1
    for image in images:
        with Image.open(__import__("io").BytesIO(image)) as rendered:
            assert rendered.size == (1400, 480)
            assert rendered.format == "PNG"


def test_spectrogram_staging_uses_audio_volume_when_system_temp_is_low(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import tempfile

    from beatforge import spectrograms

    source_parent = tmp_path / "audio-source"
    source_parent.mkdir()
    source = source_parent / "song.ogg"
    system_temp = Path(tempfile.gettempdir())

    def disk_usage(path):
        return SimpleNamespace(free=10_000 if Path(path) == system_temp else 100_000_000)

    monkeypatch.setattr(spectrograms.shutil, "disk_usage", disk_usage)

    assert spectrograms._staging_temp_parent(source, 20_000_000) == source_parent


def test_spectrogram_staging_uses_working_volume_when_temp_and_audio_are_full(
        monkeypatch, tmp_path):
    from types import SimpleNamespace
    import tempfile

    from beatforge import spectrograms

    source_parent = tmp_path / "audio-source"
    source_parent.mkdir()
    source = source_parent / "song.ogg"
    system_temp = Path(tempfile.gettempdir())
    working_directory = Path.cwd()

    def disk_usage(path):
        return SimpleNamespace(free=(100_000_000 if Path(path) == working_directory else 10_000))

    monkeypatch.setattr(spectrograms.shutil, "disk_usage", disk_usage)

    assert spectrograms._staging_temp_parent(source, 20_000_000) == working_directory
    assert working_directory != system_temp


def test_spectrogram_staging_reports_when_no_volume_has_enough_space(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import pytest

    from beatforge import spectrograms

    source = tmp_path / "song.ogg"
    monkeypatch.setattr(spectrograms.shutil, "disk_usage",
                        lambda _path: SimpleNamespace(free=10_000))

    with pytest.raises(OSError, match="insufficient accessible disk space"):
        spectrograms._staging_temp_parent(source, 20_000_000)


def test_pack_spectrograms_preserves_all_panels_and_limits_image_count():
    import io

    source = []
    rows = []
    for index in range(29):
        image = Image.new("RGB", (6, 4), (index, 20, 40))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        source.append(buffer.getvalue())
        rows.append({"file": f"panel_{index:03d}.png"})

    packed, manifest = pack_spectrogram_images(tuple(source), 12, {"images": rows})

    assert len(packed) == 11
    assert manifest["image_limit"] == 12
    assert manifest["image_count"] == 11
    assert manifest["source_panel_count"] == 29
    assert manifest["groups"][0] == ["panel_000.png"]
    assert [name for group in manifest["groups"] for name in group] == [
        f"panel_{index:03d}.png" for index in range(29)
    ]
    for packed_image, group in zip(packed[1:], manifest["groups"][1:]):
        with Image.open(io.BytesIO(packed_image)) as composite:
            assert composite.width == 6
            assert composite.height == 4 * len(group) + 8 * (len(group) - 1)
            y = 0
            for name in group:
                index = int(name.removeprefix("panel_").removesuffix(".png"))
                expected = Image.new("RGB", (6, 4), (index, 20, 40))
                assert composite.crop((0, y, 6, y + 4)).tobytes() == expected.tobytes()
                y += 12


def test_pack_spectrograms_rejects_limit_that_cannot_keep_overview():
    import pytest

    with pytest.raises(ValueError, match="at least 2"):
        pack_spectrogram_images((b"overview", b"panel"), 1)
