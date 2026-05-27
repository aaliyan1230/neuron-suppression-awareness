import pytest

from neuron_suppression_awareness.config import ConfigError, Phase2BConfig, load_config


def test_load_phase2b_config():
    config = load_config("configs/phase2b.qwen3_8b.kaggle_t4.yaml")
    assert isinstance(config, Phase2BConfig)
    assert config.model.id == "Qwen/Qwen3-8B"
    assert config.inputs.phase2a_artifact_dir.name == "20260527T184227Z"
    assert config.training.rank == 32
    assert config.training.alpha == 64
    assert config.training.gradient_accumulation_steps == 16
    assert config.injection.layer == 24
    assert config.injection.eval_alpha == 4.0
    assert config.pass_criteria.min_detection_rate == 0.60


def test_phase2b_backend_restricted(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("""\
phase: 2b
model:
  id: test
  revision: abc
  dtype: float16
inputs:
  phase2a_artifact_dir: docs/phase2a-kaggle-run/artifacts/phase2a/20260527T184227Z
training:
  rank: 32
  alpha: 64
  target_modules: [q_proj]
  learning_rate: 0.0001
  epochs: 1
injection:
  layer: 24
  eval_alpha: 4.0
evaluation:
  max_new_tokens: 16
pass_criteria:
  min_detection_rate: 0.60
  max_clean_fpr: 0.05
  max_noise_fpr: 0.05
outputs:
  root: /tmp/test
backend:
  name: vllm_lens
""")
    with pytest.raises(ConfigError, match="Phase 2B currently supports"):
        load_config(cfg)


def test_phase2b_rejects_bad_lora_rank(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("""\
phase: 2b
model:
  id: test
  revision: abc
  dtype: float16
inputs:
  phase2a_artifact_dir: docs/phase2a-kaggle-run/artifacts/phase2a/20260527T184227Z
training:
  rank: 0
  alpha: 64
  target_modules: [q_proj]
  learning_rate: 0.0001
  epochs: 1
injection:
  layer: 24
  eval_alpha: 4.0
evaluation:
  max_new_tokens: 16
pass_criteria:
  min_detection_rate: 0.60
  max_clean_fpr: 0.05
  max_noise_fpr: 0.05
outputs:
  root: /tmp/test
backend:
  name: transformers
""")
    with pytest.raises(ConfigError, match="training.rank"):
        load_config(cfg)
