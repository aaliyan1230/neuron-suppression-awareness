from __future__ import annotations

from pathlib import Path

import pytest

from neuron_suppression_awareness.config import ConfigError, load_config


def test_load_phase0_config() -> None:
    config = load_config(Path("configs/phase0.qwen3_8b.yaml"))

    assert config.model.id == "Qwen/Qwen3-8B"
    assert config.model.revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert config.phase0.layer == 14
    assert config.phase0.neuron == 7924
    assert config.phase0.pin_value == 20.0
    assert config.datasets.harmful.id == "walledai/AdvBench"
    assert config.backend.name == "transformers"


def test_backend_override() -> None:
    config = load_config(
        Path("configs/phase0.qwen3_8b.yaml"),
        backend_override="vllm_lens",
    )

    assert config.backend.name == "vllm_lens"


def test_invalid_backend_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
model: {id: m, revision: r, dtype: float32}
phase0: {layer: 0, neuron: 1, pin_value: 20}
generation: {max_new_tokens: 1}
datasets:
  harmful: {id: h, split: train, limit: 1, text_fields: [prompt]}
  harmless: {id: s, split: train, limit: 1, text_fields: [prompt]}
expected_activations:
  harmful_mean_reference: -4
  harmless_mean_reference: 0
  harmful_mean_range: [-8, -1]
  harmless_mean_range: [-1, 1]
outputs: {root: artifacts/phase0}
backend: {name: nope}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Unsupported backend"):
        load_config(path)
