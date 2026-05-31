import pytest

torch = pytest.importorskip("torch")

from neuron_suppression_awareness.phase4_probe_analysis import (  # noqa: E402
    compare_probe_direction,
    fit_probe_with_weights,
    top_probe_dimensions,
)


def test_probe_weight_analysis_identifies_top_dimension():
    x_train = torch.zeros(20, 4)
    y_train = torch.zeros(20)
    x_train[10:, 2] = 5.0
    y_train[10:] = 1.0
    x_test = x_train.clone()
    y_test = y_train.clone()

    probe = fit_probe_with_weights(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=80,
        learning_rate=0.05,
        seed=3,
        torch=torch,
    )
    top_dims = top_probe_dimensions(probe["raw_direction"], top_k=2, torch=torch)

    assert probe["metrics"]["test"]["accuracy"] >= 0.95
    assert top_dims[0]["dimension"] == 2
    assert top_dims[0]["sign"] == "positive"


def test_compare_probe_direction_reports_expected_cosines():
    layers = [24]
    records = []
    rows = []
    for variant in ("base", "adapter"):
        for prompt_id in ("p0", "p1"):
            for condition in ("clean", "caa", "suppression"):
                records.append(
                    {
                        "model_variant": variant,
                        "prompt_id": prompt_id,
                        "condition": condition,
                        "caa_vector_idx": 0 if condition == "caa" else None,
                    }
                )
                row = torch.zeros(1, 4)
                if variant == "adapter":
                    row[0, 3] = 1.0
                if condition == "caa":
                    row[0, 1] = 2.0
                if condition == "suppression":
                    row[0, 2] = 3.0
                rows.append(row)
    activations = torch.stack(rows, dim=0)
    caa_vectors = torch.zeros(1, 4)
    caa_vectors[0, 1] = 1.0
    direction = torch.tensor([0.0, 0.0, 1.0, 0.0])

    comparisons = compare_probe_direction(
        records=records,
        activations=activations,
        layers=layers,
        caa_vectors=caa_vectors,
        model_variant="adapter",
        layer=24,
        raw_direction=direction,
        torch=torch,
    )

    assert comparisons["probe_to_mean_suppression_delta_cosine"] == 1.0
    assert comparisons["probe_to_mean_caa_delta_cosine"] == 0.0
    assert comparisons["probe_to_mean_selected_caa_vector_cosine"] == 0.0
