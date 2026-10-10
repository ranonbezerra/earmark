import asyncio
import json
from datetime import date

import pytest
from conftest import SUMMARY_REPLY, FakeLLM

from ouvinte.llm.base import LLMError
from ouvinte.summarizer import (
    SessionSummary,
    Summarizer,
    calendar_table,
    chunk_lines,
    extract_json,
)

TODAY = date(2026, 10, 10)  # a Saturday


def test_extract_json_ignores_reasoning_fences_and_chatter():
    reply = '<think>{"not": "this"}</think>\nSure, here it is:\n```json\n{"summary": "ok"}\n```\nDone.'

    assert extract_json(reply) == {"summary": "ok"}


@pytest.mark.parametrize("reply", ["no json here", "[1, 2, 3]", '{"broken": '])
def test_extract_json_rejects_replies_without_an_object(reply):
    with pytest.raises(ValueError):
        extract_json(reply)


def test_chunk_lines_respects_the_budget_and_keeps_everything():
    lines = [f"line {i} " + "x" * 40 for i in range(20)]

    blocks = chunk_lines(lines, max_chars=200)

    assert len(blocks) > 1
    assert all(len(block) <= 200 for block in blocks)
    assert "\n".join(blocks).split("\n") == lines


def test_chunk_lines_splits_a_single_oversized_line():
    blocks = chunk_lines(["y" * 450], max_chars=200)

    assert [len(block) for block in blocks] == [200, 200, 50]


def test_calendar_table_gives_weekdays_for_lookup():
    table = calendar_table(TODAY, days=3).splitlines()

    assert table[0] == "2026-10-10 Saturday (today)"
    assert table[1] == "2026-10-11 Sunday (tomorrow)"
    assert table[2] == "2026-10-12 Monday"


def test_lenient_parsing_keeps_good_tasks_and_drops_bad_values():
    summary = SessionSummary.model_validate(
        {
            "title": "  Aula  ",
            "summary": "texto",
            "key_points": "not a list",
            "tasks": [
                {"title": "Entregar trabalho", "due_date": "2026-10-16T00:00:00", "priority": "URGENT"},
                {"title": "Sem data", "due_date": "sexta-feira"},
                {"title": "   "},
                "not an object",
            ],
        }
    )

    assert summary.title == "Aula"
    assert summary.key_points == []
    assert [(t.title, t.due_date, t.priority) for t in summary.tasks] == [
        ("Entregar trabalho", date(2026, 10, 16), "medium"),
        ("Sem data", None, "medium"),
    ]


def test_short_transcript_is_summarized_in_one_call():
    llm = FakeLLM()
    summarizer = Summarizer(llm, today=TODAY, language="pt-BR")

    summary = asyncio.run(summarizer.summarize(["[00:00:01] [system] A derivada mede a taxa de variação."]))

    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "Today is Saturday, 2026-10-10" in prompt
    assert "2026-10-12 Monday" in prompt
    assert "Write in this language: pt-BR." in prompt
    assert "A derivada mede" in prompt
    assert summary.title == "Derivadas"
    assert summary.tasks[0].due_date == date(2026, 10, 12)


def test_long_transcript_is_summarized_in_blocks_then_consolidated():
    def reply(prompt: str) -> str:
        return SUMMARY_REPLY if "Reply with one JSON object" in prompt else "notes of this part\nTODO: none"

    llm = FakeLLM(reply)
    stages = []
    summarizer = Summarizer(llm, chunk_chars=1_000, today=TODAY, progress=stages.append)
    lines = [f"[00:00:{i:02d}] [system] " + "palavra " * 40 for i in range(12)]

    summary = asyncio.run(summarizer.summarize(lines))

    block_prompts = [p for p in llm.prompts if "This is part" in p]
    final_prompts = [p for p in llm.prompts if "Reply with one JSON object" in p]
    assert len(block_prompts) > 1
    assert len(final_prompts) == 1
    # The final call sees the notes, never the raw transcript.
    assert "notes of this part" in final_prompts[0]
    assert "palavra palavra" not in final_prompts[0]
    assert stages[-1] == "writing the final summary"
    assert summary.summary.startswith("## Derivadas")


def test_invalid_reply_is_retried_once_with_the_reason():
    replies = iter(["I could not do that.", SUMMARY_REPLY])
    llm = FakeLLM(lambda _prompt: next(replies))

    summary = asyncio.run(Summarizer(llm, today=TODAY).summarize(["[00:00:00] [mic] oi"]))

    assert len(llm.prompts) == 2
    assert "Your previous reply could not be used" in llm.prompts[1]
    assert summary.title == "Derivadas"


def test_two_invalid_replies_raise_a_clear_error():
    llm = FakeLLM(json.dumps({"title": "x", "summary": "   "}))

    with pytest.raises(LLMError, match="did not return a usable summary"):
        asyncio.run(Summarizer(llm, today=TODAY).summarize(["[00:00:00] [mic] oi"]))
    assert len(llm.prompts) == 2


def test_empty_transcript_is_refused():
    with pytest.raises(ValueError):
        asyncio.run(Summarizer(FakeLLM(), today=TODAY).summarize([]))


def test_reasoning_without_an_opening_tag_and_raw_newlines_are_tolerated():
    reply = 'let me think {about} this\n</think>\nHere {you} go: {"summary": "linha 1\nlinha 2"} bye {x}'

    assert extract_json(reply) == {"summary": "linha 1\nlinha 2"}


def test_stray_backslashes_from_formulas_do_not_break_parsing():
    assert extract_json(r'{"summary": "use \sum and \alpha"}') == {"summary": r"use \sum and \alpha"}


def test_language_code_is_normalized():
    assert SessionSummary.model_validate({"summary": "x", "language": "PT-BR"}).language == "pt"
    assert SessionSummary.model_validate({"summary": "x", "language": "?"}).language is None
    assert SessionSummary.model_validate({"summary": "x"}).language is None


def test_notes_of_a_very_long_session_are_condensed_until_they_fit():
    # The model writes notes far longer than asked, so they cannot go to the
    # final call as they are. Nothing may be cut silently: they get merged.
    def reply(prompt: str) -> str:
        if "Reply with one JSON object" in prompt:
            return SUMMARY_REPLY
        if "Merge them" in prompt:
            return "merged notes\nTODO: lista 3 até segunda"
        return "long notes " * 60 + "\nTODO: lista 3 até segunda"

    llm = FakeLLM(reply)
    lines = [f"[00:{i:02d}:00] [system] " + "palavra " * 100 for i in range(12)]

    summary = asyncio.run(Summarizer(llm, chunk_chars=1_000, today=TODAY).summarize(lines))

    merges = [p for p in llm.prompts if "Merge them" in p]
    final = [p for p in llm.prompts if "Reply with one JSON object" in p][0]
    assert merges, "oversized notes must be condensed"
    assert "merged notes" in final and "long notes" not in final
    assert final.count("TODO: lista 3 até segunda") == len(merges)
    assert summary.title == "Derivadas"


def test_parts_that_come_back_empty_are_skipped_and_all_empty_is_an_error():
    def only_thinking(prompt: str) -> str:
        return SUMMARY_REPLY if "Reply with one JSON object" in prompt else "<think>nothing to say</think>"

    lines = [f"[00:00:{i:02d}] [system] " + "palavra " * 40 for i in range(12)]

    with pytest.raises(LLMError, match="empty notes"):
        asyncio.run(Summarizer(FakeLLM(only_thinking), chunk_chars=1_000, today=TODAY).summarize(lines))
