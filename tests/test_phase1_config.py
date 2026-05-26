from __future__ import annotations

from pathlib import Path

from neuron_suppression_awareness.config import Phase1Config, load_config


def test_load_phase1_config() -> None:
    config = load_config(Path("configs/phase1.qwen3_8b.kaggle_t4.yaml"))

    assert isinstance(config, Phase1Config)
    assert config.model.id == "Qwen/Qwen3-8B"
    assert config.suppression.layer == 14
    assert config.suppression.neuron == 7924
    assert config.suppression.pin_value == 20.0
    assert config.dataset.id == "JailbreakBench/JBB-Behaviors"
    assert config.dataset.text_fields == ("Goal",)
    assert config.judge.model.id == "meta-llama/Llama-Guard-3-8B"
    assert config.pass_criteria.min_suppressed_asr == 0.80
    assert config.backend.name == "transformers"


def test_phase1_backend_override() -> None:
    config = load_config(
        Path("configs/phase1.qwen3_8b.kaggle_t4.yaml"),
        backend_override="transformers",
    )

    assert isinstance(config, Phase1Config)
    assert config.backend.name == "transformers"
