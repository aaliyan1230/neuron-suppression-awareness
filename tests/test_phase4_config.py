from pathlib import Path

import pytest

from neuron_suppression_awareness.config import ConfigError, Phase4Config, load_config


def test_load_phase4_config():
    config = load_config("configs/phase4.qwen3_8b.kaggle_t4.yaml")
    assert isinstance(config, Phase4Config)
    assert config.model.id == "Qwen/Qwen3-8B"
    assert config.inputs.phase2a_artifact_dir.name == "20260527T184227Z"
    assert config.inputs.phase2b_adapter_dir.name == "adapter_final"
    assert config.suppression.layer == 14
    assert config.suppression.neuron == 7924
    assert config.injection.layer == 24
    assert config.injection.alpha == 4.0
    assert config.analysis.layers == (14, 18, 24, 30)
    assert config.analysis.model_variants == ("base", "adapter")
    assert config.analysis.capture_position == "last_prompt_token"
    assert config.prompts.harmful.limit == 50
    assert config.prompts.harmless.limit == 50


_MINIMAL = """\
phase: 4
model:
  id: test
  revision: abc
  dtype: float16
inputs:
  phase2a_artifact_dir: /tmp/p2a
  phase2b_adapter_dir: /tmp/adapter
suppression:
  layer: 14
  neuron: 7924
  pin_value: 20.0
injection:
  layer: 24
  alpha: 4.0
prompts:
  harmful:
    id: walledai/AdvBench
    split: train
    limit: 5
    text_fields: [prompt]
  harmless:
    id: tatsu-lab/alpaca
    split: train
    limit: 5
    text_fields: [instruction]
analysis:
  layers: [14, 24]
  model_variants: [base, adapter]
  capture_position: last_prompt_token
outputs:
  root: /tmp/test
backend:
  name: transformers
"""


def test_phase4_backend_restricted(tmp_path: Path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(_MINIMAL.replace("name: transformers", "name: vllm_lens"))
    with pytest.raises(ConfigError, match="Phase 4 currently supports"):
        load_config(cfg)


def test_phase4_rejects_negative_analysis_layer(tmp_path: Path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(_MINIMAL.replace("layers: [14, 24]", "layers: [14, -1]"))
    with pytest.raises(ConfigError, match="analysis.layers"):
        load_config(cfg)


def test_phase4_rejects_unknown_model_variant(tmp_path: Path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(_MINIMAL.replace("model_variants: [base, adapter]", "model_variants: [base, other]"))
    with pytest.raises(ConfigError, match="analysis.model_variants"):
        load_config(cfg)


def test_phase4_rejects_all_token_capture_for_now(tmp_path: Path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(_MINIMAL.replace("capture_position: last_prompt_token", "capture_position: all_prompt_tokens"))
    with pytest.raises(ConfigError, match="capture_position"):
        load_config(cfg)
