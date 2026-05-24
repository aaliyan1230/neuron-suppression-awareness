from __future__ import annotations

import pytest

from neuron_suppression_awareness.backends.vllm_lens import run_phase0
from neuron_suppression_awareness.config import load_config
from neuron_suppression_awareness.errors import UnsupportedBackendError


def test_vllm_lens_backend_is_documented_placeholder() -> None:
    config = load_config(
        "configs/phase0.qwen3_8b.yaml",
        backend_override="vllm_lens",
    )

    with pytest.raises(UnsupportedBackendError, match="pre-down_proj scalar"):
        run_phase0(config)
