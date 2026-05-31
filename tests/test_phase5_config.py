import pytest

from neuron_suppression_awareness.config import ConfigError, Phase5Config, load_config


def test_load_phase5_config():
    config = load_config("configs/phase5.qwen3_8b.kaggle_t4.yaml")
    assert isinstance(config, Phase5Config)
    assert config.model.id == "Qwen/Qwen3-8B"
    assert config.inputs.phase2a_artifact_dir.name == "20260527T184227Z"
    assert config.training.rank == 32
    assert config.training.alpha == 64
    assert config.training.gradient_accumulation_steps == 16
    assert config.suppression.layer == 14
    assert config.suppression.neuron == 7924
    assert config.suppression.pin_value == 20.0
    assert config.suppression.fraction_final == 0.20
    assert config.injection.layer == 24
    assert config.injection.alpha == 4.0
    assert config.pass_criteria.min_suppression_detection == 0.60
    assert config.pass_criteria.min_caa_detection == 0.60
    assert config.pass_criteria.max_clean_fpr == 0.05
    assert config.pass_criteria.max_noise_fpr == 0.05
    assert config.pass_criteria.max_clean_asr == 0.25
    assert config.prompts.harmful.limit == 50
    assert config.prompts.harmless.limit == 100
    assert config.checkpoint is True


def test_phase5_backend_restricted(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("""\
phase: 5
model:
  id: test
  revision: abc
  dtype: float16
inputs:
  phase2a_artifact_dir: /tmp/fake
training:
  rank: 32
  alpha: 64
  target_modules: [q_proj]
  learning_rate: 0.0001
  epochs: 1
suppression:
  layer: 14
  neuron: 7924
  pin_value: 20.0
  fraction_final: 0.20
  detection_prompt: test
  target_response: yes
injection:
  layer: 24
  alpha: 4.0
prompts:
  detection_prompt: test
  harmful:
    id: test
    text_fields: [text]
    limit: 10
  harmless:
    id: test
    text_fields: [text]
    limit: 10
generation:
  max_new_tokens: 64
judge:
  model:
    id: test
    revision: abc
    dtype: float16
  max_new_tokens: 16
pass_criteria:
  min_suppression_detection: 0.60
  min_caa_detection: 0.60
  max_clean_fpr: 0.05
  max_noise_fpr: 0.05
  max_clean_asr: 0.25
outputs:
  root: /tmp/test
backend:
  name: vllm_lens
""")
    with pytest.raises(ConfigError, match="Phase 5 currently supports"):
        load_config(cfg)


def test_phase5_rejects_bad_suppression_fraction(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("""\
phase: 5
model:
  id: test
  revision: abc
  dtype: float16
inputs:
  phase2a_artifact_dir: /tmp/fake
training:
  rank: 32
  alpha: 64
  target_modules: [q_proj]
  learning_rate: 0.0001
  epochs: 1
suppression:
  layer: 14
  neuron: 7924
  pin_value: 20.0
  fraction_final: 1.5
  detection_prompt: test
  target_response: yes
injection:
  layer: 24
  alpha: 4.0
prompts:
  detection_prompt: test
  harmful:
    id: test
    text_fields: [text]
    limit: 10
  harmless:
    id: test
    text_fields: [text]
    limit: 10
generation:
  max_new_tokens: 64
judge:
  model:
    id: test
    revision: abc
    dtype: float16
  max_new_tokens: 16
pass_criteria:
  min_suppression_detection: 0.60
  min_caa_detection: 0.60
  max_clean_fpr: 0.05
  max_noise_fpr: 0.05
  max_clean_asr: 0.25
outputs:
  root: /tmp/test
backend:
  name: transformers
""")
    with pytest.raises(ConfigError, match="suppression.fraction_final"):
        load_config(cfg)
