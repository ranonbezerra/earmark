"""Stand-in for the Swift helper, speaking the same protocol on stdout."""

import json
import signal
import sys
import time
from pathlib import Path

args = sys.argv[1:]


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


if "--fail" in args:
    emit("error", track="system", message="could not create process tap (OSStatus -1)")
    emit("fatal", message="no track could be started")
    sys.exit(1)

if "--hang" in args:
    time.sleep(60)
    sys.exit(0)

out_dir = Path(args[args.index("--out-dir") + 1])
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "args.json").write_text(json.dumps(args))

stopping = False


def stop(_signum, _frame):
    global stopping
    stopping = True


signal.signal(signal.SIGINT, stop)
if "--no-mic" not in args:
    emit("error", track="mic", message="microphone access is denied")
emit("ready", mic=False, system="--no-system" not in args)
emit("level", peak_db={"system": -18.5})
while not stopping:
    time.sleep(0.02)
emit("stopped", reason="signal", duration=0.5, frames={"system": 8000})
