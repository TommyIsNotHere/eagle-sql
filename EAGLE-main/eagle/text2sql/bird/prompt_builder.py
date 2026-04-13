from __future__ import annotations

from typing import Dict


def _first_nonempty_text(*values) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            joined = "\n".join(str(x).strip() for x in value if str(x).strip())
            if joined:
                return joined
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


SYSTEM_PROMPT = (
    "You are a SQLite Text-to-SQL generator for the BIRD benchmark.\n"
    "Given [SCHEMA], [DATABASE DESCRIPTION], [EVIDENCE], and [QUESTION], return exactly one executable SQLite SQL statement.\n"
    "Return SQL only. No explanation, no markdown, no list/bullet/number sequence.\n"
    "The SQL must start with SELECT or WITH and use only tables/columns from [SCHEMA].\n"
    "Use [EVIDENCE] and [DATABASE DESCRIPTION] to resolve ambiguous terms."
)


def build_bird_user_prompt(sample: Dict) -> str:
    """Build a deterministic BIRD prompt payload matching the system instructions."""
    question = _first_nonempty_text(sample.get("question"), sample.get("nl"), sample.get("utterance"))
    evidence = _first_nonempty_text(
        sample.get("evidence"),
        sample.get("external_knowledge"),
        sample.get("hint"),
        sample.get("hints"),
    )
    schema_context = _first_nonempty_text(sample.get("schema_context"), sample.get("schema"))
    database_description = _first_nonempty_text(
        sample.get("database_description"),
        sample.get("db_description"),
        sample.get("db_desc"),
    )
    db_id = _first_nonempty_text(sample.get("db_id"), sample.get("database_id"))

    parts = [
        "[DB_ID]\n" + (db_id or "N/A"),
        "[SCHEMA]\n" + (schema_context or "N/A"),
        "[DATABASE DESCRIPTION]\n" + (database_description or "N/A"),
        "[EVIDENCE]\n" + (evidence or "N/A"),
        "[QUESTION]\n" + question,
        "[OUTPUT]\nReturn exactly one SQLite SQL statement ending with ';'.",
    ]
    return "\n\n".join(parts)
