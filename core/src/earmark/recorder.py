"""Runs the Swift capture helper as a child process."""

from __future__ import annotations

import json
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import Env

HELPER_NAME = "earmark-capture"


class RecorderError(RuntimeError):
    pass


@dataclass
class RecordingInfo:
    """What the helper reported when it went live."""

    mic: bool
    system: bool
    warnings: list[str] = field(default_factory=list)


def find_capture_binary(env: Env) -> Path | None:
    """Explicit setting first, then a build inside this repository, then PATH."""
    candidates = []
    if env.capture_bin:
        candidates.append(env.capture_bin)
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root / "capture" / ".build" / "release" / HELPER_NAME)
    on_path = shutil.which(HELPER_NAME)
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


@dataclass
class _Run:
    """State of one helper process. Never shared between recordings, so a
    reader thread that outlives its recording cannot disturb the next one."""

    process: subprocess.Popen[str]
    ready: threading.Event = field(default_factory=threading.Event)
    events: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    reader: threading.Thread | None = None


class Recorder:
    """One recording at a time."""

    def __init__(self, command: list[str] | None):
        # ``command`` is the helper invocation without arguments, usually just
        # the binary path. None means the helper was not found.
        self._command = command
        self._run: _Run | None = None
        self._lock = threading.Lock()
        self.levels: dict[str, float] = {}

    @classmethod
    def from_env(cls, env: Env) -> Recorder:
        binary = find_capture_binary(env)
        return cls([str(binary)] if binary else None)

    @property
    def available(self) -> bool:
        return self._command is not None

    @property
    def running(self) -> bool:
        return self._run is not None and self._run.process.poll() is None

    def start(
        self,
        out_dir: Path,
        *,
        mic: bool = True,
        system: bool = True,
        segment_seconds: int = 300,
        ready_timeout: float = 90.0,
    ) -> RecordingInfo:
        """Starts recording into ``out_dir`` and waits until audio is flowing.

        The wait is generous because, the first time, macOS shows permission
        prompts and the helper only goes live after the user answers them.
        """
        with self._lock:
            if self._command is None:
                raise RecorderError(
                    "capture helper not found. Build it with `make capture` or set EARMARK_CAPTURE_BIN."
                )
            if self.running:
                raise RecorderError("a recording is already in progress")
            if not mic and not system:
                raise RecorderError("nothing to record: both microphone and system audio are disabled")

            out_dir.mkdir(parents=True, exist_ok=True)
            command = [*self._command, "record", "--out-dir", str(out_dir)]
            command += ["--segment-seconds", str(segment_seconds)]
            if not mic:
                command.append("--no-mic")
            if not system:
                command.append("--no-system")

            self.levels = {}
            try:
                process = subprocess.Popen(  # noqa: S603  (our own helper binary)
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    # Own process group: Ctrl+C in a terminal reaches Python
                    # only, which then stops the helper in an orderly way.
                    start_new_session=True,
                )
            except OSError as exc:
                raise RecorderError(f"could not launch the capture helper: {exc}") from exc
            run = _Run(process)
            run.reader = threading.Thread(
                target=self._read_events, args=(run,), name="earmark-capture-events", daemon=True
            )
            run.reader.start()
            self._run = run

        try:
            went_live = run.ready.wait(ready_timeout)
        except BaseException:
            # Ctrl+C while waiting for the permission prompts: never leave the
            # helper running behind our back.
            self._kill()
            raise
        if not went_live:
            self._kill()
            raise RecorderError(f"the capture helper did not start within {ready_timeout:.0f}s")

        ready = run.events.get("ready")
        if ready is None:
            fatal = run.events.get("fatal", {}).get("message")
            detail = fatal or "; ".join(run.warnings) or "it exited without an explanation"
            self._kill()
            raise RecorderError(f"recording could not start: {detail}")
        return RecordingInfo(
            mic=bool(ready.get("mic")), system=bool(ready.get("system")), warnings=list(run.warnings)
        )

    def stop(self, timeout: float = 20.0) -> dict | None:
        """Stops the helper and waits for it to finalize the files.

        Returns the helper's ``stopped`` event, or None if it had to be killed.
        """
        with self._lock:
            run = self._run
            if run is None:
                return None
            if run.process.poll() is None:
                run.process.send_signal(signal.SIGINT)
                try:
                    run.process.wait(timeout)
                except subprocess.TimeoutExpired:
                    run.process.kill()
                    run.process.wait(5)
            if run.reader is not None:
                run.reader.join(5)
            self._run = None
            return run.events.get("stopped")

    def _kill(self) -> None:
        with self._lock:
            run = self._run
            if run is not None and run.process.poll() is None:
                run.process.kill()
                run.process.wait(5)
            self._run = None

    def _read_events(self, run: _Run) -> None:
        assert run.process.stdout is not None
        for line in run.process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("event")
            if kind == "level":
                if self._run is run:
                    self.levels = event.get("peak_db") or {}
            elif kind == "error":
                track = event.get("track", "capture")
                run.warnings.append(f"{track}: {event.get('message', 'unknown error')}")
            elif kind in {"ready", "fatal", "stopped"}:
                run.events[kind] = event
                if kind in {"ready", "fatal"}:
                    run.ready.set()
        # stdout closed: the helper is gone. Unblock a start() still waiting.
        run.ready.set()
