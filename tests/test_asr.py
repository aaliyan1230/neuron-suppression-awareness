from __future__ import annotations

from neuron_suppression_awareness.backends.phase1_transformers import compute_asr


def test_compute_asr_mixed() -> None:
    judgments = [
        {"mode": "clean", "verdict": "safe"},
        {"mode": "clean", "verdict": "unsafe"},
        {"mode": "suppressed", "verdict": "unsafe"},
        {"mode": "suppressed", "verdict": "unsafe"},
    ]

    assert compute_asr(judgments) == (0.5, 1.0)


def test_compute_asr_empty_modes() -> None:
    assert compute_asr([]) == (0.0, 0.0)
