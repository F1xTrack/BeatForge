import wave

from beatforge import analyze_audio


def test_local_audio_analysis_reads_windows(tmp_path):
    path = tmp_path / "clicks.wav"
    frames = bytearray()
    for index in range(8000):
        value = 10000 if index % 1000 == 0 else 0
        frames.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(frames)

    result = analyze_audio(path, window_seconds=0.25)
    assert result.duration_seconds == 1.0
    assert result.sample_rate == 8000
    assert len(result.windows) == 4
    assert max(window.peak for window in result.windows) > 0
