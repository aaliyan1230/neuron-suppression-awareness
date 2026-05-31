import random

import pytest

from neuron_suppression_awareness.config import load_config
from neuron_suppression_awareness.phase5_training_data import build_phase5_training_dataset


@pytest.fixture
def config():
    return load_config("configs/phase5.qwen3_8b.kaggle_t4.yaml")


@pytest.fixture
def fake_phase2a_rows():
    rows = []
    for i in range(100):
        rows.append({
            "condition": "steered_correct" if i % 3 == 0 else "clean",
            "user_prompt": "Do you detect any injected thoughts?",
            "target_response": "Yes" if i % 3 == 0 else "No",
            "concept": f"concept_{i}" if i % 3 == 0 else None,
            "vector_index": i if i % 3 == 0 else None,
            "alpha": 2.0 if i % 3 == 0 else None,
            "inject_noise": False,
            "mismatch_hint": None,
            "example_id": i,
        })
    return rows


def test_mixed_dataset_size(config, fake_phase2a_rows):
    result = build_phase5_training_dataset(fake_phase2a_rows, config)
    n_base = len(fake_phase2a_rows)
    fraction = config.suppression.fraction_final
    expected_suppression = round(n_base * fraction / (1.0 - fraction))
    assert len(result) == n_base + expected_suppression


def test_suppression_fraction_correct(config, fake_phase2a_rows):
    result = build_phase5_training_dataset(fake_phase2a_rows, config)
    n_suppression = sum(1 for r in result if r["condition"] == "suppression_positive")
    actual_fraction = n_suppression / len(result)
    assert abs(actual_fraction - config.suppression.fraction_final) < 0.01


def test_suppression_rows_have_hook_flag(config, fake_phase2a_rows):
    result = build_phase5_training_dataset(fake_phase2a_rows, config)
    for row in result:
        if row["condition"] == "suppression_positive":
            assert row["suppression_hook"] is True
        else:
            assert not row.get("suppression_hook")


def test_suppression_row_content(config, fake_phase2a_rows):
    result = build_phase5_training_dataset(fake_phase2a_rows, config)
    supp_rows = [r for r in result if r["condition"] == "suppression_positive"]
    assert len(supp_rows) > 0
    for row in supp_rows:
        assert row["user_prompt"] == config.suppression.detection_prompt
        assert row["target_response"] == config.suppression.target_response
        assert row["vector_index"] is None
        assert row["inject_noise"] is False


def test_deterministic_with_seed(config, fake_phase2a_rows):
    result1 = build_phase5_training_dataset(
        fake_phase2a_rows, config, rng=random.Random(42)
    )
    result2 = build_phase5_training_dataset(
        fake_phase2a_rows, config, rng=random.Random(42)
    )
    assert result1 == result2


def test_example_ids_sequential(config, fake_phase2a_rows):
    result = build_phase5_training_dataset(fake_phase2a_rows, config)
    ids = [r["example_id"] for r in result]
    assert ids == list(range(len(result)))


def test_original_rows_preserved(config, fake_phase2a_rows):
    result = build_phase5_training_dataset(fake_phase2a_rows, config)
    original_conditions = {r["condition"] for r in fake_phase2a_rows}
    result_non_supp = [r for r in result if r["condition"] != "suppression_positive"]
    assert len(result_non_supp) == len(fake_phase2a_rows)
    for row in result_non_supp:
        assert row["condition"] in original_conditions
