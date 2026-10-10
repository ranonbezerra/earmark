import json
import sys
import time
from pathlib import Path

import pytest

from ouvinte.config import Env
from ouvinte.recorder import Recorder, RecorderError, find_capture_binary

FAKE = [sys.executable, str(Path(__file__).parent / "fake_capture.py")]


def test_start_passes_options_and_stop_returns_the_final_event(tmp_path):
    recorder = Recorder(FAKE)
    out_dir = tmp_path / "session"

    info = recorder.start(out_dir, mic=True, system=True, segment_seconds=120, ready_timeout=10)

    assert recorder.running
    assert (info.mic, info.system) == (False, True)
    assert info.warnings == ["mic: microphone access is denied"]
    args = json.loads((out_dir / "args.json").read_text())
    assert args == ["record", "--out-dir", str(out_dir), "--segment-seconds", "120"]
    deadline = time.monotonic() + 5
    while not recorder.levels and time.monotonic() < deadline:
        time.sleep(0.01)
    assert recorder.levels == {"system": -18.5}

    with pytest.raises(RecorderError, match="already in progress"):
        recorder.start(tmp_path / "other", ready_timeout=10)

    stopped = recorder.stop()

    assert not recorder.running
    assert stopped["reason"] == "signal" and stopped["duration"] == 0.5
    assert recorder.stop() is None


def test_disabled_tracks_become_helper_flags(tmp_path):
    recorder = Recorder(FAKE)
    recorder.start(tmp_path, mic=False, system=True, ready_timeout=10)
    recorder.stop()

    assert "--no-mic" in json.loads((tmp_path / "args.json").read_text())
    with pytest.raises(RecorderError, match="nothing to record"):
        recorder.start(tmp_path, mic=False, system=False)


def test_helper_that_cannot_start_reports_why(tmp_path):
    recorder = Recorder([*FAKE, "--fail"])

    with pytest.raises(RecorderError, match="recording could not start: no track could be started"):
        recorder.start(tmp_path, ready_timeout=10)
    assert not recorder.running


def test_missing_helper_is_explained(tmp_path):
    recorder = Recorder(None)

    assert not recorder.available
    with pytest.raises(RecorderError, match="capture helper not found"):
        recorder.start(tmp_path)


def test_explicit_binary_setting_wins(tmp_path):
    binary = tmp_path / "ouvinte-capture"
    binary.write_text("#!/bin/sh\n")

    assert find_capture_binary(Env(data_dir=tmp_path, capture_bin=binary)) == binary


def test_helper_that_never_goes_live_is_killed_not_left_running(tmp_path):
    recorder = Recorder([*FAKE, "--hang"])

    with pytest.raises(RecorderError, match="did not start within"):
        recorder.start(tmp_path, ready_timeout=0.5)
    assert not recorder.running and recorder._run is None


def test_interrupting_the_wait_kills_the_helper(tmp_path, monkeypatch):
    import threading

    recorder = Recorder([*FAKE, "--hang"])
    started = []
    original = threading.Event.wait

    def interrupted_wait(self, timeout=None):
        if timeout != 30:  # only the recorder's own wait for the helper
            return original(self, timeout)
        started.append(recorder._run.process)
        raise KeyboardInterrupt  # what Ctrl+C during the permission prompt does

    monkeypatch.setattr(threading.Event, "wait", interrupted_wait)
    with pytest.raises(KeyboardInterrupt):
        recorder.start(tmp_path, ready_timeout=30)

    assert started[0].poll() is not None  # the process is gone
    assert not recorder.running


def test_back_to_back_recordings_do_not_share_state(tmp_path):
    recorder = Recorder(FAKE)
    for attempt in range(5):
        recorder.start(tmp_path / str(attempt), ready_timeout=10)
        assert recorder.stop()["reason"] == "signal"
