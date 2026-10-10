"""Reading what the capture helper wrote: segment discovery and loudness."""

from __future__ import annotations

import json
import math
import re
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

SOURCES = ("system", "mic")
MANIFEST_NAME = "segments.jsonl"
_SEGMENT_NAME = re.compile(r"^(mic|system)-(\d+)\.[A-Za-z0-9]+$")


@dataclass(frozen=True)
class AudioSegment:
    source: str
    index: int
    path: Path
    start_s: float
    # None when the file is not a WAV we can inspect (imported audio).
    duration_s: float | None


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate else None
    except (wave.Error, EOFError, OSError):
        return None


def peak_window_dbfs(path: Path, window_s: float = 0.5, stride: int = 8) -> float | None:
    """Loudness of the loudest window of a 16-bit WAV, in dBFS.

    The average level of a five minute segment says little: ten seconds of
    speech in an otherwise quiet segment still matters. So this looks for the
    loudest half second instead. Samples are subsampled, which is plenty to
    tell speech from silence and keeps this fast in pure Python.

    Returns None if the file cannot be inspected, so callers keep it.
    """
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getsampwidth() != 2:
                return None
            channels = wav.getnchannels()
            window_frames = max(1, int(wav.getframerate() * window_s))
            loudest = 0.0
            while True:
                raw = wav.readframes(window_frames)
                if not raw:
                    break
                samples = array("h")
                samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
                if sys.byteorder == "big":
                    samples.byteswap()
                picked = samples[:: stride * channels]
                if not picked:
                    continue
                mean_square = sum(s * s for s in picked) / len(picked)
                loudest = max(loudest, mean_square)
    except (wave.Error, EOFError, OSError):
        return None
    if loudest <= 0:
        return -120.0
    return max(-120.0, 10 * math.log10(loudest / (32768.0**2)))


def _read_manifest(audio_dir: Path) -> dict[str, float]:
    """Map of file name to start offset in seconds, from the helper manifest."""
    offsets: dict[str, float] = {}
    manifest = audio_dir / MANIFEST_NAME
    if not manifest.exists():
        return offsets
    for line in manifest.read_text().splitlines():
        try:
            entry = json.loads(line)
            offsets[str(entry["file"])] = float(entry["start_offset"])
        except (ValueError, KeyError, TypeError):
            continue  # a torn last line after a hard kill is expected
    return offsets


def discover_segments(audio_dir: Path, segment_seconds: int) -> list[AudioSegment]:
    """Audio segments of a session, ordered by time.

    Start offsets come from the manifest when available. Files the manifest
    does not mention fall back to ``index * segment_seconds``.
    """
    if not audio_dir.is_dir():
        return []
    offsets = _read_manifest(audio_dir)
    segments = []
    for path in audio_dir.iterdir():
        match = _SEGMENT_NAME.match(path.name)
        if not match or not path.is_file():
            continue
        source, index = match.group(1), int(match.group(2))
        duration = wav_duration(path)
        if duration is not None and duration <= 0:
            continue
        segments.append(
            AudioSegment(
                source=source,
                index=index,
                path=path,
                start_s=offsets.get(path.name, float(index * segment_seconds)),
                duration_s=duration,
            )
        )
    return sorted(segments, key=lambda s: (s.start_s, SOURCES.index(s.source), s.index))


def session_duration(segments: list[AudioSegment]) -> float | None:
    ends = [s.start_s + s.duration_s for s in segments if s.duration_s is not None]
    return max(ends) if ends else None
