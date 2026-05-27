import pytest

from neuron_suppression_awareness.config import (
    ConfigError,
    Phase2AConfig,
    load_config,
    parse_config,
)


def test_load_phase2a_config():
    config = load_config("configs/phase2a.qwen3_8b.kaggle_t4.yaml")
    assert isinstance(config, Phase2AConfig)
    assert config.model.id == "Qwen/Qwen3-8B"
    assert config.caa.layer == 24
    assert config.caa.n_prompt_templates == 5
    assert config.caa.baseline_concept == "something"
    assert config.training_data.target_total == 7000
    assert config.training_data.alpha_values == (2.0, 4.0, 8.0)
    assert config.training_data.seed == 42
    assert config.backend.name == "transformers"


def test_fractions_must_sum_to_one(tmp_path):
    yaml_text = """\
phase: 2
model:
  id: test
  revision: abc
  dtype: float16
caa:
  layer: 24
  n_prompt_templates: 5
training_data:
  target_total: 100
  steered_correct_fraction: 0.30
  clean_fraction: 0.10
  noise_fraction: 0.05
  mismatch_fraction: 0.05
  alpaca_replay_fraction: 0.10
  alpha_values: [4.0]
  detection_prompt: test
outputs:
  root: /tmp/test
backend:
  name: transformers
"""
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml_text)
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(str(cfg))


def test_negative_layer_rejected(tmp_path):
    yaml_text = """\
phase: 2
model:
  id: test
  revision: abc
  dtype: float16
caa:
  layer: -1
  n_prompt_templates: 5
training_data:
  target_total: 100
  steered_correct_fraction: 0.40
  clean_fraction: 0.25
  noise_fraction: 0.10
  mismatch_fraction: 0.10
  alpaca_replay_fraction: 0.15
  alpha_values: [4.0]
  detection_prompt: test
outputs:
  root: /tmp/test
backend:
  name: transformers
"""
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml_text)
    with pytest.raises(ConfigError, match="non-negative"):
        load_config(str(cfg))


def test_backend_override():
    config = load_config(
        "configs/phase2a.qwen3_8b.kaggle_t4.yaml",
        backend_override="transformers",
    )
    assert config.backend.name == "transformers"


def test_phase_dispatch():
    import yaml
    raw = yaml.safe_load(open("configs/phase2a.qwen3_8b.kaggle_t4.yaml"))
    config = parse_config(raw)
    assert isinstance(config, Phase2AConfig)
