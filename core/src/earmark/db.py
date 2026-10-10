"""SQLite engine and schema setup."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine, select

from . import models  # noqa: F401  (registers the tables)
from .models import SessionStatus, StudySession


def make_engine(path: Path | str) -> Engine:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def recover_interrupted(engine: Engine) -> int:
    """Fix sessions left mid-flight by a crash or a restart.

    Nothing can still be recording or processing when the process starts, so
    those sessions go back to ``recorded`` and can simply be processed again.
    """
    with Session(engine) as db:
        stuck = db.exec(
            select(StudySession).where(
                StudySession.status.in_([SessionStatus.RECORDING, SessionStatus.PROCESSING])  # type: ignore[attr-defined]
            )
        ).all()
        for session in stuck:
            session.status = SessionStatus.RECORDED
            session.error = "interrupted before it finished"
            db.add(session)
        db.commit()
        return len(stuck)
