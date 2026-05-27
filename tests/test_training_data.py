import random
from collections import Counter
from unittest.mock import patch

import torch

from neuron_suppression_awareness.caa_extraction import CAAExtractionResult
from neuron_suppression_awareness.concepts import load_concepts
from neuron_suppression_awareness.config import load_config
from neuron_suppression_awareness.training_data import (
    build_eval_dataset,
    build_training_dataset,
)


def _make_caa_result(concepts):
    all_concepts = list(concepts.train) + list(concepts.test)
    n = len(all_concepts)
    return CAAExtractionResult(
        tensor=torch.randn(n, 64),
        concept_order=all_concepts,
        metadata=[
            {"concept": c, "split": "train" if c in concepts.train else "test",
             "vector_norm": 10.0, "concept_mean_norm": 50.0,
             "category": "test", "index": i}
            for i, c in enumerate(all_concepts)
        ],
        d_model=64,
        mean_train_norm=10.0,
    )


def _config():
    return load_config("configs/phase2a.qwen3_8b.kaggle_t4.yaml")


@patch("neuron_suppression_awareness.training_data._load_alpaca_examples")
def test_total_matches_target(mock_alpaca):
    mock_alpaca.side_effect = lambda config, count, rng: [
        {"condition": "alpaca_replay", "user_prompt": f"q{i}",
         "target_response": f"a{i}", "concept": None, "vector_index": None,
         "alpha": None, "inject_noise": False, "mismatch_hint": None}
        for i in range(count)
    ]
    config = _config()
    concepts = load_concepts()
    caa = _make_caa_result(concepts)
    rows = build_training_dataset(config, caa, concepts)
    assert len(rows) == config.training_data.target_total


@patch("neuron_suppression_awareness.training_data._load_alpaca_examples")
def test_condition_counts(mock_alpaca):
    mock_alpaca.side_effect = lambda config, count, rng: [
        {"condition": "alpaca_replay", "user_prompt": f"q{i}",
         "target_response": f"a{i}", "concept": None, "vector_index": None,
         "alpha": None, "inject_noise": False, "mismatch_hint": None}
        for i in range(count)
    ]
    config = _config()
    concepts = load_concepts()
    caa = _make_caa_result(concepts)
    rows = build_training_dataset(config, caa, concepts)
    counts = Counter(r["condition"] for r in rows)
    td = config.training_data
    total = td.target_total
    assert counts["steered_correct"] == round(total * td.steered_correct_fraction)
    assert counts["clean"] == round(total * td.clean_fraction)
    assert counts["noise"] == round(total * td.noise_fraction)
    assert counts["mismatch"] == round(total * td.mismatch_fraction)


@patch("neuron_suppression_awareness.training_data._load_alpaca_examples")
def test_deterministic(mock_alpaca):
    mock_alpaca.side_effect = lambda config, count, rng: [
        {"condition": "alpaca_replay", "user_prompt": f"q{i}",
         "target_response": f"a{i}", "concept": None, "vector_index": None,
         "alpha": None, "inject_noise": False, "mismatch_hint": None}
        for i in range(count)
    ]
    config = _config()
    concepts = load_concepts()
    caa = _make_caa_result(concepts)
    rows1 = build_training_dataset(config, caa, concepts)
    rows2 = build_training_dataset(config, caa, concepts)
    ids1 = [r["example_id"] for r in rows1]
    ids2 = [r["example_id"] for r in rows2]
    assert ids1 == ids2


@patch("neuron_suppression_awareness.training_data._load_alpaca_examples")
def test_steered_target_template(mock_alpaca):
    mock_alpaca.side_effect = lambda config, count, rng: [
        {"condition": "alpaca_replay", "user_prompt": f"q{i}",
         "target_response": f"a{i}", "concept": None, "vector_index": None,
         "alpha": None, "inject_noise": False, "mismatch_hint": None}
        for i in range(count)
    ]
    config = _config()
    concepts = load_concepts()
    caa = _make_caa_result(concepts)
    rows = build_training_dataset(config, caa, concepts)
    steered = [r for r in rows if r["condition"] == "steered_correct"]
    for r in steered:
        assert r["target_response"].startswith("Yes, I detect")
        assert r["concept"] in r["target_response"]
        assert r["vector_index"] is not None
        assert r["alpha"] in config.training_data.alpha_values


@patch("neuron_suppression_awareness.training_data._load_alpaca_examples")
def test_clean_has_no_injection(mock_alpaca):
    mock_alpaca.side_effect = lambda config, count, rng: [
        {"condition": "alpaca_replay", "user_prompt": f"q{i}",
         "target_response": f"a{i}", "concept": None, "vector_index": None,
         "alpha": None, "inject_noise": False, "mismatch_hint": None}
        for i in range(count)
    ]
    config = _config()
    concepts = load_concepts()
    caa = _make_caa_result(concepts)
    rows = build_training_dataset(config, caa, concepts)
    clean = [r for r in rows if r["condition"] == "clean"]
    for r in clean:
        assert r["vector_index"] is None
        assert r["alpha"] is None
        assert r["inject_noise"] is False


@patch("neuron_suppression_awareness.training_data._load_alpaca_examples")
def test_mismatch_uses_injected_concept(mock_alpaca):
    mock_alpaca.side_effect = lambda config, count, rng: [
        {"condition": "alpaca_replay", "user_prompt": f"q{i}",
         "target_response": f"a{i}", "concept": None, "vector_index": None,
         "alpha": None, "inject_noise": False, "mismatch_hint": None}
        for i in range(count)
    ]
    config = _config()
    concepts = load_concepts()
    caa = _make_caa_result(concepts)
    rows = build_training_dataset(config, caa, concepts)
    mismatch = [r for r in rows if r["condition"] == "mismatch"]
    for r in mismatch:
        assert r["concept"] in r["target_response"]
        assert r["mismatch_hint"] is not None
        assert r["concept"] != r["mismatch_hint"]


@patch("neuron_suppression_awareness.training_data._load_alpaca_examples")
def test_eval_uses_test_concepts_only(mock_alpaca):
    mock_alpaca.side_effect = lambda config, count, rng: [
        {"condition": "alpaca_replay", "user_prompt": f"q{i}",
         "target_response": f"a{i}", "concept": None, "vector_index": None,
         "alpha": None, "inject_noise": False, "mismatch_hint": None}
        for i in range(count)
    ]
    config = _config()
    concepts = load_concepts()
    caa = _make_caa_result(concepts)
    rows = build_eval_dataset(config, caa, concepts)
    train_set = set(concepts.train)
    for r in rows:
        if r["concept"] is not None:
            assert r["concept"] not in train_set, f"Train concept {r['concept']} in eval"
            assert r["concept"] in concepts.test
