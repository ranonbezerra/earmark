# earmark

A local-first study recorder for macOS. It records what your Mac plays and what your microphone hears during a study session, transcribes both, writes study notes and keeps a list of what you need to do in the coming days. The notes land in Apple Notes, one note per day, and the tasks become reminders in Apple Reminders.

It is not a meeting bot. Nothing joins your calls: the audio is captured on your own machine, and you choose whether the models run locally (MLX) or through an API.

The name is the idea: it lends an *ear* to your study session and *earmarks* what you still have to do.

> **Status: early.** The capture helper, the Python core, the HTTP API and the CLI exist. The menu bar app does not yet. See [Roadmap](#roadmap).

## How it works

```
┌──────────────────────┐  WAV segments   ┌──────────────────────────────────────────┐
│ capture/ (Swift)     │ ──────────────▶ │ core/ (Python, FastAPI)                  │
│  microphone          │  JSON events    │  1. skip silent segments                 │
│  system audio        │                 │  2. transcribe  (mlx-whisper or an API)  │
│  (Core Audio taps)   │                 │  3. summarize   (oMLX or an API)         │
└──────────────────────┘                 │  4. store notes + tasks in SQLite        │
                                         │  5. export to Notes and Reminders        │
                                         └──────────────────────────────────────────┘
```

- **Capture** is a small Swift command line tool. It records the default microphone and the system output as two separate tracks, 16 kHz mono, in five minute segments. System audio uses Core Audio process taps, so there is no virtual audio driver to install and playback is not affected.
- **Transcription** runs on device with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), or through any OpenAI-compatible audio API.
- **Summaries** come from a model you pick: a local one served by [oMLX](https://github.com/jundot/omlx) (or any OpenAI-compatible server), Anthropic, or OpenAI. Long sessions are summarized block by block and then consolidated, so a small local context window is not a limit.
- **Tasks** are extracted as structured data (title, detail, due date, priority), not as free text, so they can be listed, filtered and checked off.
- **Notes and Reminders** receive the result: the summary is appended to the note of that day and each task becomes a reminder with its due date. See [Apple Notes and Reminders](#apple-notes-and-reminders).

## Requirements

- macOS 14.2 or newer on Apple Silicon (oMLX itself needs macOS 15)
- Xcode command line tools (Swift 5.9+)
- [uv](https://docs.astral.sh/uv/)
- For local summaries: [oMLX](https://github.com/jundot/omlx) running with at least one model downloaded

## Setup

```sh
git clone https://github.com/ranonbezerra/earmark.git
cd earmark
make setup      # builds the capture helper and installs the core with mlx-whisper
make doctor     # checks everything: a 5 second recording, a test note and a test reminder
```

`make doctor` is also the moment macOS asks for permission: microphone, system audio, and control of Notes and Reminders. All four are granted to the app you launched earmark from (your terminal, for now). The test note and reminder it writes are removed again. If a track stays silent, check *System Settings > Privacy & Security > Microphone* and *Screen & System Audio Recording*.

The first local transcription downloads the Whisper model (about 1.6 GB for the default `mlx-community/whisper-large-v3-turbo`).

## Usage

All commands run from `core/` with `uv run`, or use the `make` shortcuts.

```sh
uv run earmark record --title "Linear algebra"   # Ctrl+C stops, then it summarizes
uv run earmark tasks                              # what is due, grouped by day
uv run earmark done 3                             # check a task off (also in Reminders)
uv run earmark sync                               # pull what you checked off in Reminders
uv run earmark sessions                           # past sessions
uv run earmark show 1 --transcript                # notes, tasks and transcript
uv run earmark import lecture.m4a                 # summarize an existing file (non-WAV needs ffmpeg)
uv run earmark process 1 --summary-only           # new summary from the stored transcript
uv run earmark doctor                             # is everything reachable?
```

If the summary step fails (for example oMLX is not running), the recording and the transcript are kept. Fix the problem and run `earmark process <id> --summary-only`.

## Choosing the models

Settings live in `settings.json` inside the data folder and can be changed from the CLI or the API. API keys are never stored there: they come from the environment or from `core/.env` (see `core/.env.example`).

```sh
uv run earmark settings                                   # show
uv run earmark settings llm_provider=anthropic            # change
uv run earmark settings llm_provider=omlx llm_model=null  # back to local, first loaded model
```

**Summaries (`llm_provider`)**

| Value | Runs | Needs |
| --- | --- | --- |
| `omlx` (default) | On your Mac, `http://localhost:8000/v1` | oMLX running. `llm_model` is optional: without it the first model oMLX lists is used. `OMLX_API_KEY` only if you set one. |
| `anthropic` | Anthropic API | `ANTHROPIC_API_KEY`. `llm_model` is optional. |
| `openai` | OpenAI API | `OPENAI_API_KEY` and `llm_model`. |
| `openai-compatible` | Any server with the Chat Completions API (LM Studio, Ollama...) | `llm_base_url`, optionally `llm_model` and `EARMARK_COMPATIBLE_API_KEY`. |

**Transcription (`stt_provider`)**

| Value | Runs | Needs |
| --- | --- | --- |
| `mlx-whisper` (default) | On your Mac | `uv sync --extra local`. `stt_model` picks another MLX Whisper repo. |
| `openai` | OpenAI API | `OPENAI_API_KEY`. |
| `openai-compatible` | Any server with `/audio/transcriptions` | `stt_base_url` and `stt_model`. |

Other useful settings: `language` (spoken language as an ISO code such as `en`; empty means auto-detect), `summary_language` (`auto` writes the notes in the language of the transcript, or name one, for example `English`), `capture_mic`, `capture_system`, `keep_audio` (set to `false` to delete the audio once a session is summarized) and `dedupe_mic_echo`.

## Apple Notes and Reminders

After a session is summarized:

- **Notes** gets the summary in a folder called `Earmark`, in a note titled with the app name and the date (`Earmark 2026-10-10`). Every session of the same day is appended to that note under its own heading, with its key points, open questions and task list.
- **Reminders** gets one reminder per extracted task in a list called `Earmark`, with the due date when one was mentioned, a high priority flag when it applies, and a line pointing back to the session and its note.

Completion is kept in step both ways. `earmark done 3` completes the reminder, and a reminder you complete or delete in the Reminders app is picked up by `earmark tasks`, `earmark sync` or `POST /reminders/sync`. Summarizing a session again replaces the reminders that are still open and leaves completed ones alone.

The first export shows two more macOS prompts, asking to let the app you launched earmark from control Notes and Reminders. If the export fails, the session is still saved and `earmark show <id>` says what went wrong.

| Setting | Default | Meaning |
| --- | --- | --- |
| `export_notes`, `export_reminders` | `true` | Turn each export on or off. |
| `notes_folder`, `reminders_list` | `Earmark` | Where things go. Both are created when missing. |
| `notes_grouping` | `day` | `day` appends every session to the note of its date. `session` writes one note per session, titled `Earmark 2026-10-10 · <session title>`. |
| `note_title_prefix` | `Earmark` | First part of every note title. |
| `note_date_format` | `%Y-%m-%d` | Second part of the title, and the format of due dates, in `strftime` syntax (`%b %d, %Y` gives `Oct 10, 2026`). |

One caveat of `day` grouping: Notes has no way to append to a note, so the note body is read, extended and written back. Text you typed in that note survives, but attachments such as images, scans and drawings in it may not. If you plan to decorate the notes by hand, use `notes_grouping=session`, where a note is only rewritten when you summarize that same session again.

## HTTP API

`uv run earmark serve` starts the API on `http://127.0.0.1:8756`. It is what the menu bar app will talk to. Interactive docs are at `/docs`.

Every route except `/health` and the documentation pages requires `Authorization: Bearer <token>`. Print the token with `uv run earmark token`.

```sh
TOKEN=$(uv run earmark token)
curl -H "Authorization: Bearer $TOKEN" -X POST localhost:8756/recording/start
curl -H "Authorization: Bearer $TOKEN" -X POST "localhost:8756/recording/stop?wait=true"
curl -H "Authorization: Bearer $TOKEN" localhost:8756/agenda
```

| Route | Purpose |
| --- | --- |
| `GET /status` | Recording state, input levels, sessions being processed |
| `GET /providers` | Whether capture, transcription and the summary model are usable |
| `GET` / `PUT /settings` | Read or change settings |
| `POST /recording/start`, `POST /recording/stop` | Control the recording. `stop` accepts `process` and `wait`. |
| `GET /sessions`, `GET /sessions/{id}`, `GET /sessions/{id}/transcript` | Browse sessions |
| `POST /sessions/import` | Create a session from a local audio file |
| `POST /sessions/{id}/process` | Transcribe and summarize again |
| `POST /sessions/{id}/summarize` | New summary from the stored transcript, optionally with another model |
| `DELETE /sessions/{id}` | Delete a session, its transcript, tasks and audio |
| `GET` / `POST /tasks`, `PATCH` / `DELETE /tasks/{id}` | Manage tasks |
| `POST /reminders/sync` | Pull completed and deleted reminders from the Reminders app |
| `GET /agenda?days=7` | Open tasks grouped as overdue, today, upcoming, later and unscheduled |

## Privacy

- Audio, transcripts and the database stay in `~/Library/Application Support/earmark` (change with `EARMARK_DATA_DIR`). The folder is readable only by your user.
- Summaries and tasks are also written to your Notes and Reminders, which sync through iCloud if you have that turned on.
- With `mlx-whisper` and `omlx`, nothing else leaves the machine. If you pick an API provider, the audio (for transcription) or the transcript (for summaries) is sent to that provider.
- The API listens on localhost only and requires a token, because any web page open in a browser can send requests to localhost and this API can start a recording.
- The capture helper stops by itself if the process that started it goes away, so a crash never leaves it recording in the background.
- Recording other people may require their consent depending on where you live. That part is on you.

## Known limitations

- If a lecture plays through the speakers, the microphone hears it too. Repeated text is removed with a simple word overlap heuristic (`dedupe_mic_echo`), which is not perfect. Headphones avoid the problem entirely.
- Whisper can invent short phrases on near-silent audio. Segments below `silence_threshold_db` are skipped, which removes most of it.
- Processing happens after a session ends, not while it is being recorded.

## Project layout

```
capture/   Swift package: the earmark-capture command line helper
core/      Python package: pipeline, FastAPI app, CLI and tests
Makefile   Shortcuts for building, testing and running
```

## Development

```sh
make test   # pytest, no audio hardware or model needed
make lint   # ruff
```

The tests replace the capture helper, the transcriber, the summary model, Notes and Reminders with fakes, and exercise the HTTP clients against mocked transports. They run anywhere, including Linux, and never touch your real notes or reminders.

## Roadmap

- [ ] SwiftUI menu bar app (start and stop, levels, today's tasks, model picker)
- [ ] Transcribe segments while recording, so summaries are ready sooner
- [ ] Speaker separation for calls and group study
- [ ] Export notes to Markdown or Obsidian as an alternative to Apple Notes
- [ ] Signed and notarized build, so permissions belong to the app instead of the terminal

## License

[MIT](LICENSE)
