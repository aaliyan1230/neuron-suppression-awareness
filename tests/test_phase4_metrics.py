import pytest

torch = pytest.importorskip("torch")

from neuron_suppression_awareness.backends.phase4_transformers import (  # noqa: E402
    compute_phase4_geometry,
    train_phase4_probes,
)


def _records():
    rows = []
    for prompt_index in range(6):
        for condition in ("clean", "caa", "suppression"):
            rows.append(
                {
                    "model_variant": "base",
                    "condition": condition,
                    "prompt_id": f"prompt-{prompt_index}",
                    "source": "harmful" if prompt_index < 3 else "harmless",
                    "caa_vector_idx": prompt_index if condition == "caa" else None,
                }
            )
    return rows


def test_phase4_geometry_summarizes_l2_and_cosine():
    records = _records()
    layers = (14, 24)
    activations = torch.zeros(len(records), len(layers), 4)
    caa_vectors = torch.zeros(6, 4)
    caa_vectors[:, 0] = 2.0

    for index, rec in enumerate(records):
        if rec["condition"] == "caa":
            activations[index, :, 0] = 2.0
        elif rec["condition"] == "suppression":
            activations[index, :, 1] = 3.0

    metrics = compute_phase4_geometry(
        records=records,
        activations=activations,
        layers=layers,
        caa_vectors=caa_vectors,
        injection_layer=24,
        torch=torch,
    )

    layer24 = metrics["by_model_and_layer"]["base"]["24"]
    assert layer24["caa_delta_l2"]["mean"] == 2.0
    assert layer24["suppression_delta_l2"]["mean"] == 3.0
    assert layer24["suppression_to_caa_l2_ratio"]["mean"] == 1.5
    assert layer24["suppression_to_caa_delta_cosine"]["mean"] == 0.0
    assert layer24["suppression_to_raw_caa_cosine"]["mean"] == 0.0


def test_phase4_probe_learns_separable_suppression_signal():
    records = []
    rows = []
    for prompt_index in range(20):
        for condition in ("clean", "suppression"):
            records.append(
                {
                    "model_variant": "base",
                    "condition": condition,
                    "prompt_id": f"prompt-{prompt_index}",
                }
            )
            row = torch.zeros(1, 8)
            if condition == "suppression":
                row[0, 0] = 5.0
            rows.append(row)
    activations = torch.stack(rows, dim=0)

    results = train_phase4_probes(
        records=records,
        activations=activations,
        layers=(24,),
        train_fraction=0.7,
        epochs=80,
        learning_rate=0.05,
        seed=7,
        torch=torch,
    )

    test_metrics = results["by_model_and_layer"]["base"]["24"]["test"]
    assert test_metrics["accuracy"] >= 0.95
    assert test_metrics["balanced_accuracy"] >= 0.95
    assert test_metrics["auroc"] == 1.0
