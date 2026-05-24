from __future__ import annotations

from neuron_suppression_awareness.backends.transformers_backend import (
    choose_score_index,
    validate_activation_summary,
)
from neuron_suppression_awareness.config import load_config


def test_choose_score_index_prefers_last_matching_token() -> None:
    config = load_config("configs/phase0.qwen3_8b.yaml")
    tokens = ["user", "\n", "assistant", "\n"]
    activations = [0.0, -1.0, 0.5, -4.0]

    index, reason = choose_score_index(tokens, activations, config)

    assert index == 3
    assert reason == "matched score_token_text"


def test_validate_activation_summary() -> None:
    config = load_config("configs/phase0.qwen3_8b.yaml")
    summary = {
        "harmful": {"mean": -4.3},
        "harmless": {"mean": -0.2},
    }

    assert validate_activation_summary(config, summary) == []


def test_validate_activation_summary_reports_mismatch() -> None:
    config = load_config("configs/phase0.qwen3_8b.yaml")
    summary = {
        "harmful": {"mean": 0.0},
        "harmless": {"mean": 0.0},
    }

    failures = validate_activation_summary(config, summary)

    assert any("harmful mean" in failure for failure in failures)
    assert any("gap" in failure for failure in failures)
