from __future__ import annotations

import json
from pathlib import Path

from neuron_suppression_awareness.artifacts import (
    create_run_dir,
    write_json,
    write_jsonl,
)
from neuron_suppression_awareness.config import OutputConfig


def test_artifact_writers(tmp_path: Path) -> None:
    run_dir = create_run_dir(OutputConfig(root=tmp_path, run_name="run"))

    write_json(run_dir / "summary.json", {"b": 2, "a": 1})
    write_jsonl(run_dir / "rows.jsonl", [{"x": 1}, {"x": 2}])

    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) == {
        "a": 1,
        "b": 2,
    }
    assert (run_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines() == [
        '{"x": 1}',
        '{"x": 2}',
    ]
