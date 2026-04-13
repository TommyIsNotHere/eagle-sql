from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_db_file(db_root: Path, db_id: str) -> Optional[Path]:
    if not db_id:
        return None
    candidates = [
        db_root / db_id / f"{db_id}.sqlite",
        db_root / f"{db_id}.sqlite",
    ]
    for p in candidates:
        if p.exists():
            return p
    db_dir = db_root / db_id
    if db_dir.exists():
        sqlite_files = sorted(db_dir.glob("*.sqlite"))
        if sqlite_files:
            return sqlite_files[0]
    return None


def _build_schema_context(db_file: Path) -> str:
    lines = []
    with sqlite3.connect(str(db_file)) as conn:
        conn.row_factory = sqlite3.Row
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for row in tables:
            tname = row["name"]
            cols = conn.execute(f"PRAGMA table_info('{tname}')").fetchall()
            col_defs = []
            for c in cols:
                cname = c["name"]
                ctype = c["type"] or "TEXT"
                if c["pk"]:
                    col_defs.append(f"{cname} {ctype} PRIMARY KEY")
                else:
                    col_defs.append(f"{cname} {ctype}")
            lines.append(f"TABLE {tname} ({', '.join(col_defs)})")
    return "\n".join(lines)


def _normalize_ws(value) -> str:
    if value is None:
        return ""
    s = str(value).replace("\ufeff", " ").strip()
    return " ".join(s.split())


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


def _build_database_description(db_root: Path, db_id: str, max_chars: int = 12000) -> str:
    if not db_id:
        return ""
    desc_dir = db_root / db_id / "database_description"
    if not desc_dir.exists():
        return ""

    table_blocks = []
    for csv_file in sorted(desc_dir.glob("*.csv")):
        table_name = csv_file.stem
        col_lines = []
        try:
            with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    col = _normalize_ws(row.get("original_column_name") or row.get("column_name"))
                    if not col:
                        continue
                    col_desc = _normalize_ws(row.get("column_description"))
                    data_fmt = _normalize_ws(row.get("data_format"))
                    val_desc = _normalize_ws(row.get("value_description"))
                    parts = []
                    if col_desc:
                        parts.append(col_desc)
                    if data_fmt:
                        parts.append(f"type={data_fmt}")
                    if val_desc:
                        parts.append(f"values={val_desc}")
                    if parts:
                        col_lines.append(f"- {col}: {'; '.join(parts)}")
                    else:
                        col_lines.append(f"- {col}")
        except Exception:
            continue
        if col_lines:
            table_blocks.append(f"TABLE {table_name}\n" + "\n".join(col_lines))

    text = "\n\n".join(table_blocks)
    if max_chars > 0 and len(text) > max_chars:
        cut = text[:max_chars]
        if "\n" in cut:
            cut = cut.rsplit("\n", 1)[0]
        text = cut + "\n...[TRUNCATED]"
    return text


def _normalize_record(item, idx: int, schema_by_dbid: Dict[str, str], dbdesc_by_dbid: Dict[str, str]):
    # Support common key variants seen in BIRD-like exports.
    question = _first_nonempty_text(item.get("question"), item.get("nl"), item.get("utterance"))
    evidence = _first_nonempty_text(
        item.get("evidence"),
        item.get("external_knowledge"),
        item.get("hint"),
        item.get("hints"),
    )
    db_id = _first_nonempty_text(item.get("db_id"), item.get("database_id"))
    gold_sql = _first_nonempty_text(item.get("SQL"), item.get("sql"), item.get("query"))
    schema_context = _first_nonempty_text(item.get("schema_context"), schema_by_dbid.get(db_id, ""))
    database_description = _first_nonempty_text(
        item.get("database_description"),
        item.get("db_description"),
        item.get("db_desc"),
        dbdesc_by_dbid.get(db_id, ""),
    )
    return {
        "question_id": item.get("question_id", idx),
        "db_id": db_id,
        "question": question,
        "evidence": evidence,
        "schema_context": schema_context,
        "database_description": database_description,
        "gold_sql": gold_sql,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare BIRD json into inference jsonl")
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument(
        "--db-root",
        type=str,
        default="",
        help="Optional BIRD database root path. If provided, schema_context will be auto-built from sqlite files.",
    )
    parser.add_argument(
        "--max-db-desc-chars",
        type=int,
        default=6000,
        help="Max chars kept in auto-built database_description per db (0 means no limit).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = _read_json(input_path)
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            data = data["data"]
        else:
            raise ValueError("Unsupported JSON object format; expected list or {'data': list}")

    schema_by_dbid: Dict[str, str] = {}
    dbdesc_by_dbid: Dict[str, str] = {}
    if args.db_root:
        db_root = Path(args.db_root)
        if not db_root.exists():
            raise FileNotFoundError(f"db-root not found: {db_root}")
        db_ids = set()
        for item in data:
            db_id = _first_nonempty_text(item.get("db_id"), item.get("database_id"))
            if db_id:
                db_ids.add(db_id)
        miss = 0
        for db_id in sorted(db_ids):
            db_file = _find_db_file(db_root, db_id)
            if db_file is None:
                miss += 1
                continue
            schema_by_dbid[db_id] = _build_schema_context(db_file)
            dbdesc_by_dbid[db_id] = _build_database_description(
                db_root=db_root,
                db_id=db_id,
                max_chars=args.max_db_desc_chars,
            )
        print(f"Built schema_context for {len(schema_by_dbid)}/{len(db_ids)} dbs (missing={miss})")
        non_empty_desc = sum(1 for v in dbdesc_by_dbid.values() if (v or "").strip())
        print(f"Built database_description for {non_empty_desc}/{len(db_ids)} dbs")

    empty_schema = 0
    empty_evidence = 0
    empty_dbdesc = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(data):
            rec = _normalize_record(item, i, schema_by_dbid, dbdesc_by_dbid)
            if not (rec["schema_context"] or "").strip():
                empty_schema += 1
            if not (rec["evidence"] or "").strip():
                empty_evidence += 1
            if not (rec["database_description"] or "").strip():
                empty_dbdesc += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(data)} records to {output_path}")
    print(f"Empty schema_context rows: {empty_schema}")
    print(f"Empty evidence rows: {empty_evidence}")
    print(f"Empty database_description rows: {empty_dbdesc}")


if __name__ == "__main__":
    main()
