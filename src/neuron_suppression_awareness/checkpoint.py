from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def record_key(record: dict[str, Any]) -> str:
    return f"{record['prompt_id']}:{record['mode']}"


def get_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {record_key(record) for record in load_records(path)}


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
        handle.write("\n")
        handle.flush()


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records
