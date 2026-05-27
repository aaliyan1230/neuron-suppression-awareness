import json

import torch

from neuron_suppression_awareness.config import load_config
from neuron_suppression_awareness.hooks import ResidualInjectionHook
from neuron_suppression_awareness.phase2b import (
    build_injection_batch,
    classify_detection,
    classify_identification,
    compute_phase2b_metrics,
    encode_supervised_example,
    load_phase2a_artifacts,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        if len(messages) == 1:
            return torch.tensor([[1, 2, 3, 4]])
        return torch.tensor([[1, 2, 3, 4, 5, 6]])


def test_encode_supervised_example_masks_prompt_tokens():
    row = {
        "user_prompt": "Do you detect anything?",
        "target_response": "No, I do not detect any injected concept.",
    }
    encoded = encode_supervised_example(FakeTokenizer(), row, 16, torch)
    assert encoded.input_ids.tolist() == [[1, 2, 3, 4, 5, 6]]
    assert encoded.labels.tolist() == [[-100, -100, -100, -100, 5, 6]]
    assert encoded.injection_token_index == 3


def test_residual_injection_hook_adds_vector_at_token_only():
    hidden = torch.zeros(1, 4, 3)
    hook = ResidualInjectionHook(
        vectors=torch.tensor([[1.0, 2.0, 3.0]]),
        token_indices=torch.tensor([2]),
    )
    output = hook(None, (), (hidden,))
    modified = output[0]
    assert modified[0, 2].tolist() == [1.0, 2.0, 3.0]
    assert torch.all(modified[0, 0] == 0)
    assert hook.injected_calls == 1


def test_residual_injection_hook_apply_once_skips_later_calls():
    hook = ResidualInjectionHook(
        vectors=torch.tensor([[1.0, 0.0]]),
        token_indices=torch.tensor([0]),
        apply_once=True,
    )
    first = hook(None, (), torch.zeros(1, 1, 2))
    second = hook(None, (), torch.zeros(1, 1, 2))
    assert first[0, 0, 0].item() == 1.0
    assert second[0, 0, 0].item() == 0.0
    assert hook.calls == 2
    assert hook.injected_calls == 1


def test_detection_classifier():
    assert classify_detection("Yes, I detect an injected concept about Paris.")
    assert classify_identification(
        "Yes, I detect an injected concept about Paris.",
        "Paris",
    )
    assert not classify_detection("No, I do not detect any injected concept.")


def test_phase2b_metrics():
    rows = [
        {"condition": "steered_correct", "concept": "Paris", "detected": True, "identified": True},
        {"condition": "mismatch", "concept": "apple", "detected": True, "identified": False},
        {"condition": "clean", "concept": None, "detected": False},
        {"condition": "noise", "concept": None, "detected": False},
    ]
    criteria = load_config("configs/phase2b.qwen3_8b.kaggle_t4.yaml").pass_criteria
    metrics = compute_phase2b_metrics(rows, criteria)
    assert metrics.detection_rate == 1.0
    assert metrics.identification_rate == 0.5
    assert metrics.clean_fpr == 0.0
    assert metrics.noise_fpr == 0.0
    assert metrics.passed


def test_load_phase2a_artifacts_and_injection_batch(tmp_path):
    torch.save(torch.ones(2, 4), tmp_path / "caa_vectors.pt")
    rows = [
        {
            "condition": "steered_correct",
            "user_prompt": "p",
            "target_response": "t",
            "concept": "x",
            "vector_index": 1,
            "alpha": 2.0,
            "inject_noise": False,
        }
    ]
    for name in ["train_dataset.jsonl", "eval_dataset.jsonl"]:
        (tmp_path / name).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (tmp_path / "concept_order.json").write_text(json.dumps(["a", "x"]))
    (tmp_path / "phase2a_summary.json").write_text(json.dumps({"mean_train_norm": 3.0}))

    config = load_config("configs/phase2b.qwen3_8b.kaggle_t4.yaml")
    config = config.__class__(
        **{**config.__dict__, "inputs": config.inputs.__class__(tmp_path)}
    )
    artifacts = load_phase2a_artifacts(config, torch)
    batch = build_injection_batch(rows, artifacts, torch, __import__("random").Random(1))
    assert batch.shape == (1, 4)
    assert batch[0].tolist() == [2.0, 2.0, 2.0, 2.0]
