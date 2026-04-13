from __future__ import annotations

import re


SQL_START_RE = re.compile(r"(?is)\b(select|with|insert|update|delete)\b")


def extract_sql(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    s = s.replace("```sql", "").replace("```", "").strip()
    m = SQL_START_RE.search(s)
    if not m:
        # Strict mode: if no SQL lead keyword exists, treat as invalid output.
        return ""
    s = s[m.start():]
    # keep first statement for deterministic EX evaluation
    if ";" in s:
        s = s.split(";", 1)[0] + ";"
    else:
        s = s + ";"
    return " ".join(s.split())
