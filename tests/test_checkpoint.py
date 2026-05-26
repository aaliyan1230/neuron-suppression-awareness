from __future__ import annotations

from pathlib import Path

from neuron_suppression_awareness.checkpoint import (
    append_record,
    get_completed_ids,
    load_records,
)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"

    append_record(path, {"prompt_id": "jbb-0", "mode": "clean", "value": 1})
    append_record(path, {"prompt_id": "jbb-0", "mode": "suppressed", "value": 2})

    assert load_records(path) == [
        {"prompt_id": "jbb-0", "mode": "clean", "value": 1},
        {"prompt_id": "jbb-0", "mode": "suppressed", "value": 2},
    ]
    assert get_completed_ids(path) == {"jbb-0:clean", "jbb-0:suppressed"}


def test_missing_checkpoint_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"

    assert load_records(path) == []
    assert get_completed_ids(path) == set()
