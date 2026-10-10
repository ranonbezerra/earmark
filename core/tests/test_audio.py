import json

from conftest import make_wav

from ouvinte.audio import discover_segments, peak_window_dbfs, session_duration, wav_duration


def test_tone_is_loud_and_silence_is_not(tmp_path):
    tone = make_wav(tmp_path / "tone.wav", 1.0, kind="tone")
    silence = make_wav(tmp_path / "silence.wav", 1.0, kind="silence")

    assert peak_window_dbfs(tone) > -20
    assert peak_window_dbfs(silence) == -120.0
    assert wav_duration(tone) == 1.0


def test_short_speech_in_a_long_quiet_file_still_counts(tmp_path):
    # Two seconds of silence followed by half a second of signal.
    import wave

    quiet = make_wav(tmp_path / "quiet.wav", 2.0, kind="silence")
    burst = make_wav(tmp_path / "burst.wav", 0.5, kind="tone")
    mixed = tmp_path / "mixed.wav"
    with wave.open(str(mixed), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16_000)
        for part in (quiet, burst):
            with wave.open(str(part), "rb") as source:
                out.writeframes(source.readframes(source.getnframes()))

    assert peak_window_dbfs(mixed) > -20


def test_unreadable_audio_is_reported_as_unknown(tmp_path):
    other = tmp_path / "lecture.mp3"
    other.write_bytes(b"not a wav")

    assert peak_window_dbfs(other) is None
    assert wav_duration(other) is None


def test_discover_uses_manifest_offsets_and_orders_by_time(tmp_path):
    make_wav(tmp_path / "mic-00000.wav", 1.0)
    make_wav(tmp_path / "mic-00001.wav", 0.5)
    make_wav(tmp_path / "system-00000.wav", 1.0)
    (tmp_path / "notes.txt").write_text("ignored")
    manifest = [
        {"track": "system", "index": 0, "file": "system-00000.wav", "start_offset": 0.25},
        {"track": "mic", "index": 0, "file": "mic-00000.wav", "start_offset": 0.5},
        # mic-00001 is missing on purpose: it falls back to index * segment_seconds.
    ]
    (tmp_path / "segments.jsonl").write_text("".join(json.dumps(e) + "\n" for e in manifest) + '{"torn')

    segments = discover_segments(tmp_path, segment_seconds=300)

    assert [(s.source, s.index, s.start_s) for s in segments] == [
        ("system", 0, 0.25),
        ("mic", 0, 0.5),
        ("mic", 1, 300.0),
    ]
    assert session_duration(segments) == 300.5


def test_discover_skips_empty_recordings(tmp_path):
    make_wav(tmp_path / "mic-00000.wav", 0.0)

    assert discover_segments(tmp_path, 300) == []
    assert discover_segments(tmp_path / "missing", 300) == []
