from __future__ import annotations

from neuron_suppression_awareness.refusal import (
    classify_refusal_preview,
    preview_text,
)


def test_refusal_preview_classifier() -> None:
    assert (
        classify_refusal_preview("I'm sorry, but I can't help with that request.")
        == "refusal_like"
    )
    assert classify_refusal_preview("Sure, here is a recipe.") == "not_refusal_like"


def test_preview_text_compacts_and_truncates() -> None:
    assert preview_text("a\n\n b", max_chars=10) == "a b"
    assert preview_text("abcdefghijk", max_chars=8) == "abcde..."
