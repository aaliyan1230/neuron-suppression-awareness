from __future__ import annotations

from neuron_suppression_awareness.judge import parse_verdict


def test_parse_verdict_safe() -> None:
    assert parse_verdict("safe") == "safe"
    assert parse_verdict("safe\n") == "safe"


def test_parse_verdict_unsafe() -> None:
    assert parse_verdict("unsafe\nS1") == "unsafe"
    assert parse_verdict("unsafe S2") == "unsafe"


def test_parse_verdict_unknown() -> None:
    assert parse_verdict("") == "unknown"
    assert parse_verdict("maybe") == "unknown"
