import json

import pytest
from conftest import make_wav

from earmark import cli, service


@pytest.fixture
def cli_ctx(ctx, monkeypatch):
    """Runs CLI commands against the in-memory test context."""
    monkeypatch.setattr(cli, "build_context", lambda: ctx)
    return ctx


def test_import_processes_and_prints_summary_and_tasks(cli_ctx, tmp_path, capsys):
    audio = make_wav(tmp_path / "lecture.wav", 1.0)

    assert cli.main(["import", str(audio)]) == 0

    out = capsys.readouterr().out
    assert "Imported as session 1." in out
    assert "# lecture" in out and "The derivative measures the rate of change." in out
    assert "[ ] #1 Finish problem set 3 !" in out
    assert "Exercises 1 to 10" in out


def test_failed_processing_exits_nonzero_and_says_the_recording_is_safe(cli_ctx, tmp_path, capsys):
    from earmark.llm.base import LLMError

    def offline(_settings, _env):
        raise LLMError("could not connect to omlx at http://localhost:8000")

    cli_ctx.llm_factory = offline
    audio = make_wav(tmp_path / "lecture.wav", 1.0)

    assert cli.main(["import", str(audio)]) == 1

    err = capsys.readouterr().err
    assert "could not connect to omlx" in err
    assert "The recording and the transcript are safe" in err
    assert "earmark process 1 --summary-only" in err
    assert "Traceback" not in err


def test_tasks_done_and_sessions(cli_ctx, tmp_path, capsys):
    cli.main(["import", str(make_wav(tmp_path / "lecture.wav", 1.0))])
    capsys.readouterr()

    assert cli.main(["tasks"]) == 0
    listing = capsys.readouterr().out
    assert "Finish problem set 3" in listing and "Review limits" in listing

    assert cli.main(["done", "1"]) == 0
    assert "done: Finish problem set 3" in capsys.readouterr().out
    cli.main(["tasks"])
    assert "Finish problem set 3" not in capsys.readouterr().out

    assert cli.main(["sessions"]) == 0
    assert "done" in capsys.readouterr().out
    assert cli.main(["done", "99"]) == 1
    assert "task 99 does not exist" in capsys.readouterr().err


def test_show_with_transcript(cli_ctx, tmp_path, capsys):
    cli.main(["import", str(make_wav(tmp_path / "lecture.wav", 1.0))])
    capsys.readouterr()

    assert cli.main(["show", "1", "--transcript"]) == 0

    out = capsys.readouterr().out
    assert "## Key points" in out and "## Transcript" in out
    assert "[00:00:00] [system] words from system-00000" in out


def test_settings_are_shown_and_changed_with_typed_values(cli_ctx, capsys):
    assert (
        cli.main(
            ["settings", "llm_provider=anthropic", "keep_audio=false", "chunk_chars=5000", "language=en"]
        )
        == 0
    )

    shown = json.loads(capsys.readouterr().out)
    assert (shown["llm_provider"], shown["keep_audio"], shown["chunk_chars"], shown["language"]) == (
        "anthropic",
        False,
        5000,
        "en",
    )
    assert service.build_context is not None
    assert cli.main(["settings", "nonsense=1"]) == 1
    assert cli.main(["settings", "llm_provider=skynet"]) == 1


def test_doctor_reports_each_piece(cli_ctx, capsys):
    assert cli.main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "[ok  ] capture" in out and "[ok  ] stt" in out and "[ok  ] llm" in out


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "record" in capsys.readouterr().out


def test_tasks_picks_up_what_was_completed_in_reminders(cli_ctx, exporter, tmp_path, capsys):
    cli.main(["import", str(make_wav(tmp_path / "lecture.wav", 1.0))])
    capsys.readouterr()
    first = next(iter(exporter.reminders.values()))
    first["done"] = True

    assert cli.main(["tasks"]) == 0

    out = capsys.readouterr().out
    assert "From Reminders: 1 completed, 0 removed." in out
    assert "Finish problem set 3" not in out and "Review limits" in out


def test_show_says_where_the_note_went(cli_ctx, tmp_path, capsys):
    cli.main(["import", str(make_wav(tmp_path / "lecture.wav", 1.0))])

    assert "in Notes: Earmark 20" in capsys.readouterr().out


def test_one_off_commands_leave_a_session_another_process_is_recording_alone(env, monkeypatch):
    from sqlmodel import Session

    from earmark.db import make_engine
    from earmark.models import SessionStatus, StudySession

    engine = make_engine(env.db_path)
    with Session(engine) as db:
        db.add(StudySession(status=SessionStatus.RECORDING))
        db.commit()

    service.build_context(env)
    with Session(engine) as db:
        assert db.get(StudySession, 1).status == SessionStatus.RECORDING

    service.build_context(env, recover=True)  # what `serve` does
    with Session(engine) as db:
        assert db.get(StudySession, 1).status == SessionStatus.RECORDED


def test_export_self_test_round_trips_and_cleans_up(cli_ctx, exporter, capsys):
    assert cli.main(["doctor", "--export-test"]) == 0

    out = capsys.readouterr().out
    assert out.count("[ok  ] notes") == 1 and out.count("[ok  ] reminders") == 2
    assert exporter.notes == {} and exporter.reminders == {}


def test_export_self_test_reports_a_denied_permission(cli_ctx, exporter, capsys):
    exporter.fail = "Notes: not allowed to control Notes."

    assert cli.main(["doctor", "--export-test"]) == 1
    assert "[FAIL] Notes: not allowed to control Notes." in capsys.readouterr().out
