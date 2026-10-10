"""Turns a transcript into study notes and a structured task list.

Short sessions go to the model in one call. Long ones are summarized block by
block and then consolidated, so the context window of a small local model is
never the limit on how long a session can be.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from .llm.base import LLMError, LLMProvider

log = logging.getLogger(__name__)

MAX_COLLAPSE_ROUNDS = 5

SYSTEM_PROMPT = (
    "You turn transcripts of a person's study sessions into study notes and a to-do list. "
    "The transcript mixes two sources: [system] is audio the computer was playing (lectures, "
    "videos, calls) and [mic] is the room microphone (usually the student thinking aloud, "
    "reading or dictating notes). Speech recognition makes mistakes: silently fix the obvious "
    "ones, and never add content the transcript does not support."
)


class ExtractedTask(BaseModel):
    title: str
    detail: str | None = None
    due_date: date | None = None
    priority: Literal["low", "medium", "high"] = "medium"

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, value: Any) -> str:
        return str(value or "").strip()[:200]

    @field_validator("detail", mode="before")
    @classmethod
    def _clean_detail(cls, value: Any) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None

    @field_validator("due_date", mode="before")
    @classmethod
    def _lenient_date(cls, value: Any) -> date | None:
        # A wrong date is worse than no date: anything unparseable becomes None.
        if isinstance(value, date):
            return value
        if not isinstance(value, str):
            return None
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None

    @field_validator("priority", mode="before")
    @classmethod
    def _lenient_priority(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        return text if text in {"low", "medium", "high"} else "medium"


class SessionSummary(BaseModel):
    title: str = ""
    summary: str
    key_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    tasks: list[ExtractedTask] = Field(default_factory=list)
    language: str | None = None

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, value: Any) -> str:
        return str(value or "").strip()[:120]

    @field_validator("language", mode="before")
    @classmethod
    def _language_code(cls, value: Any) -> str | None:
        code = str(value or "").strip().lower()
        return code[:2] if code[:2].isalpha() and len(code) >= 2 else None

    @field_validator("key_points", "open_questions", mode="before")
    @classmethod
    def _string_list(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("tasks", mode="before")
    @classmethod
    def _task_list(cls, value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict) and str(item.get("title") or "").strip()]


_THINKING = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINKING_END = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
_THINKING_START = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
# A backslash that does not start a valid JSON escape (LaTeX such as \\sum).
_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def strip_thinking(text: str) -> str:
    """Removes the reasoning that some local models print before the answer."""
    text = _THINKING.sub("", text)
    closers = list(_THINKING_END.finditer(text))
    if closers:
        # A closing tag without its opening one: the answer is what follows.
        text = text[closers[-1].end() :]
    return _THINKING_START.sub("", text).strip()


def extract_json(text: str) -> dict:
    """Pulls the JSON object out of a model reply.

    Tolerates what models wrap around it or get slightly wrong: reasoning
    blocks, Markdown code fences, sentences before or after, raw line breaks
    inside strings and stray backslashes.
    """
    cleaned = strip_thinking(text)
    decoder = json.JSONDecoder(strict=False)
    problem: Exception | None = None
    for match in re.finditer(r"\{", cleaned):
        candidate = cleaned[match.start() :]
        for attempt in (candidate, _BAD_ESCAPE.sub(r"\\\\", candidate)):
            try:
                value, _end = decoder.raw_decode(attempt)
            except ValueError as exc:
                problem = problem or exc
                continue
            if isinstance(value, dict):
                return value
    if problem is not None:
        raise ValueError(f"the JSON in the reply is invalid ({problem})")
    raise ValueError("the reply has no JSON object")


def chunk_lines(lines: list[str], max_chars: int) -> list[str]:
    """Groups lines into blocks of at most ``max_chars`` (a longer single line is split)."""
    blocks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        pieces = [line[i : i + max_chars] for i in range(0, len(line), max_chars)] or [""]
        for piece in pieces:
            if current and size + len(piece) + 1 > max_chars:
                blocks.append("\n".join(current))
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 1
    if current:
        blocks.append("\n".join(current))
    return blocks


def calendar_table(today: date, days: int = 14) -> str:
    """Dates with their weekdays, so the model looks dates up instead of computing them."""
    rows = []
    for offset in range(days + 1):
        day = today + timedelta(days=offset)
        note = {0: " (today)", 1: " (tomorrow)"}.get(offset, "")
        rows.append(f"{day.isoformat()} {day.strftime('%A')}{note}")
    return "\n".join(rows)


class Summarizer:
    def __init__(
        self,
        llm: LLMProvider,
        *,
        chunk_chars: int = 12_000,
        language: str = "auto",
        today: date | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        self.llm = llm
        self.chunk_chars = chunk_chars
        self.language = language
        self.today = today or date.today()
        self._progress = progress or (lambda _message: None)

    async def summarize(self, transcript_lines: list[str]) -> SessionSummary:
        blocks = chunk_lines(transcript_lines, self.chunk_chars)
        if not blocks:
            raise ValueError("cannot summarize an empty transcript")
        if len(blocks) == 1:
            self._progress("summarizing")
            return await self._final(blocks[0], kind="transcript")

        # Each block becomes notes about a quarter of the size of a block, so
        # the notes of four blocks fit in one call again.
        target = max(400, self.chunk_chars // 4)
        notes = []
        for number, block in enumerate(blocks, start=1):
            self._progress(f"summarizing part {number}/{len(blocks)}")
            notes.append(await self._notes(block, number, len(blocks), target))
        notes = [note for note in notes if note]

        # A very long session produces more notes than fit in one call:
        # merge them in groups until they do.
        rounds = 0
        while len("\n\n".join(notes)) > self.chunk_chars and rounds < MAX_COLLAPSE_ROUNDS:
            rounds += 1
            groups = chunk_lines(notes, self.chunk_chars)
            self._progress(f"condensing notes ({len(groups)} groups)")
            notes = [await self._condense(group, target) for group in groups]
            notes = [note for note in notes if note]
        if not notes:
            raise LLMError(f"{self.llm.name} returned empty notes for every part of the session")

        source = "\n\n".join(notes)
        limit = self.chunk_chars * 2
        if len(source) > limit:
            # The model kept ignoring the length it was asked for.
            log.warning(
                "notes still too long after %d rounds; the last %d characters are left out",
                rounds,
                len(source) - limit,
            )
            source = source[:limit]
        self._progress("writing the final summary")
        return await self._final(source, kind="notes")

    # -- prompts ----------------------------------------------------------

    def _language_rule(self) -> str:
        if self.language.strip().lower() in {"", "auto"}:
            return "Write in the main language of the transcript."
        return f"Write in this language: {self.language}."

    async def _notes(self, block: str, number: int, total: int, target_chars: int) -> str:
        prompt = (
            f"This is part {number} of {total} of one study session.\n\n"
            "Write dense notes of this part: the concepts explained, keeping definitions, formulas, "
            "examples and numbers. Then, under a line that says TODO, list every task, deadline, exam "
            "date, assignment or stated intention to do something, quoting days and dates exactly as "
            "they were said. If there are none, write 'TODO: none'.\n"
            f"{self._language_rule()} Use at most about {target_chars // 6} words. "
            "Notes only, no preamble.\n\n"
            f"<part>\n{block}\n</part>"
        )
        return strip_thinking(await self.llm.complete(system=SYSTEM_PROMPT, prompt=prompt, max_tokens=4096))

    async def _condense(self, notes: str, target_chars: int) -> str:
        prompt = (
            "Below are notes of consecutive parts of one study session.\n\n"
            "Merge them into a single set of notes, keeping the definitions, formulas, examples and "
            "numbers that matter most. Keep one TODO section at the end with every task and every "
            "day or date, exactly as written. Drop nothing from the TODO section.\n"
            f"{self._language_rule()} Use at most about {target_chars // 6} words. "
            "Notes only, no preamble.\n\n"
            f"<notes>\n{notes}\n</notes>"
        )
        return strip_thinking(await self.llm.complete(system=SYSTEM_PROMPT, prompt=prompt, max_tokens=4096))

    async def _final(self, source: str, *, kind: Literal["transcript", "notes"]) -> SessionSummary:
        described = (
            "the transcript of a study session"
            if kind == "transcript"
            else "notes taken from consecutive parts of one study session"
        )
        prompt = (
            f"Below is {described}.\n\n"
            f"Today is {self.today.strftime('%A')}, {self.today.isoformat()}. Upcoming dates:\n"
            f"{calendar_table(self.today)}\n\n"
            "Reply with one JSON object with exactly these keys:\n"
            '- "title": a short title for the session (at most 80 characters)\n'
            '- "summary": study notes in Markdown, organized by topic. Explain what was actually '
            "covered and keep definitions, formulas, examples and numbers. No preamble.\n"
            '- "language": the two-letter code of the language you wrote in ("pt", "en"...)\n'
            '- "key_points": 3 to 8 strings with the ideas worth remembering\n'
            '- "open_questions": strings with doubts raised or points left unclear ([] if none)\n'
            '- "tasks": what the student has to do in the coming days ([] if nothing). Each task is '
            "an object with:\n"
            '    - "title": specific and imperative, at most 100 characters\n'
            '    - "detail": one sentence of context, or null\n'
            '    - "due_date": "YYYY-MM-DD" when a deadline or day was stated or clearly implied, '
            'otherwise null. Resolve "tomorrow", "Friday" or "next week" with the dates above.\n'
            '    - "priority": "low", "medium" or "high"\n'
            '  Include tasks that were stated (homework, deadlines, "I need to review X") and '
            "follow-ups the session clearly calls for (an exercise list that was mentioned, a topic "
            "the student said they did not understand). Do not pad the list with generic advice.\n\n"
            f"{self._language_rule()} Keep the JSON keys in English. Write formulas in plain text "
            "or Unicode, never LaTeX, and do not use backslashes. Reply with the JSON object only.\n\n"
            f"<{kind}>\n{source}\n</{kind}>"
        )
        reply = await self.llm.complete(system=SYSTEM_PROMPT, prompt=prompt, max_tokens=8192)
        try:
            return _parse_summary(reply)
        except (ValueError, ValidationError) as first_error:
            # One retry, telling the model what was wrong with its reply.
            retry = (
                f"{prompt}\n\nYour previous reply could not be used ({_short(first_error)}). "
                "Reply again with only the JSON object, valid and complete."
            )
            reply = await self.llm.complete(system=SYSTEM_PROMPT, prompt=retry, max_tokens=8192)
            try:
                return _parse_summary(reply)
            except (ValueError, ValidationError) as second_error:
                raise LLMError(
                    f"{self.llm.name} did not return a usable summary: {_short(second_error)}"
                ) from second_error


def _parse_summary(reply: str) -> SessionSummary:
    summary = SessionSummary.model_validate(extract_json(reply))
    if not summary.summary.strip():
        raise ValueError('"summary" is empty')
    return summary


def _short(error: Exception) -> str:
    return str(error).replace("\n", " ")[:200]
