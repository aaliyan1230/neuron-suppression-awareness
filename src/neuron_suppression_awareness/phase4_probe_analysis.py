from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import write_json
from .backends.phase4_transformers import (
    _binary_metrics,
    _cosine,
    _labels,
    _prompt_split,
    _series_summary,
)


def analyze_layer_probe(
    phase4_dir: Path,
    phase2a_dir: Path,
    output_dir: Path | None = None,
    model_variant: str = "adapter",
    layer: int = 24,
    train_fraction: float = 0.7,
    epochs: int = 200,
    learning_rate: float = 0.001,
    seed: int = 42,
    top_k: int = 50,
) -> dict[str, Any]:
    import torch

    output_dir = output_dir or phase4_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(phase4_dir / "phase4_activations.pt", map_location="cpu")
    activations = payload["activations"].float()
    records = payload["records"]
    layers = [int(item) for item in payload["layers"]]
    if layer not in layers:
        raise ValueError(f"Layer {layer} not found in Phase 4 artifact layers {layers}.")
    layer_pos = layers.index(layer)

    caa_vectors = torch.load(phase2a_dir / "caa_vectors.pt", map_location="cpu").float()
    examples = [
        (idx, rec)
        for idx, rec in enumerate(records)
        if rec["model_variant"] == model_variant
        and rec["condition"] in {"clean", "suppression"}
    ]
    split = _prompt_split(
        [rec for _idx, rec in examples],
        train_fraction=train_fraction,
        seed=seed + layer,
    )
    train_indices = [
        idx for idx, rec in examples if rec["prompt_id"] in split["train_prompt_ids"]
    ]
    test_indices = [
        idx for idx, rec in examples if rec["prompt_id"] in split["test_prompt_ids"]
    ]
    x_train = activations[train_indices, layer_pos].float()
    y_train = _labels([records[idx] for idx in train_indices], torch)
    x_test = activations[test_indices, layer_pos].float()
    y_test = _labels([records[idx] for idx in test_indices], torch)

    probe = fit_probe_with_weights(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed + layer,
        torch=torch,
    )
    raw_direction = probe["raw_direction"]
    top_dims = top_probe_dimensions(raw_direction, top_k=top_k, torch=torch)
    comparisons = compare_probe_direction(
        records=records,
        activations=activations,
        layers=layers,
        caa_vectors=caa_vectors,
        model_variant=model_variant,
        layer=layer,
        raw_direction=raw_direction,
        torch=torch,
    )

    tensor_artifact = {
        "model_variant": model_variant,
        "layer": layer,
        "standardized_weight": probe["standardized_weight"],
        "standardized_bias": probe["standardized_bias"],
        "feature_mean": probe["feature_mean"],
        "feature_std": probe["feature_std"],
        "raw_direction": raw_direction,
        "raw_bias": probe["raw_bias"],
        "top_dimensions": top_dims,
    }
    torch.save(tensor_artifact, output_dir / f"phase4_{model_variant}_layer{layer}_probe.pt")

    summary = {
        "model_variant": model_variant,
        "layer": layer,
        "task": "clean_vs_suppression",
        "positive_label": "suppression",
        "train_fraction": train_fraction,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "train_n": len(train_indices),
        "test_n": len(test_indices),
        "train_prompt_count": len(split["train_prompt_ids"]),
        "test_prompt_count": len(split["test_prompt_ids"]),
        "metrics": probe["metrics"],
        "direction_norm": float(torch.linalg.vector_norm(raw_direction).item()),
        "top_dimensions": top_dims,
        "comparisons": comparisons,
    }
    write_json(output_dir / f"phase4_{model_variant}_layer{layer}_probe_analysis.json", summary)
    (output_dir / f"phase4_{model_variant}_layer{layer}_probe_analysis.md").write_text(
        build_probe_analysis_report(summary),
        encoding="utf-8",
    )
    return summary


def fit_probe_with_weights(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    epochs: int,
    learning_rate: float,
    seed: int,
    torch: Any,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    feature_mean = x_train.mean(dim=0, keepdim=True)
    feature_std_unclamped = x_train.std(dim=0, keepdim=True)
    feature_std = feature_std_unclamped.clamp_min(1e-6)
    x_train_std = (x_train - feature_mean) / feature_std
    x_test_std = (x_test - feature_mean) / feature_std

    model = torch.nn.Linear(x_train.shape[-1], 1)
    with torch.no_grad():
        model.weight.normal_(mean=0.0, std=0.01, generator=generator)
        model.bias.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = None
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train_std).squeeze(-1)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_scores = torch.sigmoid(model(x_train_std).squeeze(-1))
        test_scores = torch.sigmoid(model(x_test_std).squeeze(-1))
        standardized_weight = model.weight.detach().cpu().float().squeeze(0)
        standardized_bias = model.bias.detach().cpu().float().squeeze()
        raw_direction = torch.where(
            feature_std_unclamped.squeeze(0).cpu().float() > 1e-6,
            standardized_weight / feature_std.squeeze(0).cpu().float(),
            torch.zeros_like(standardized_weight),
        )
        raw_bias = standardized_bias - torch.sum(
            standardized_weight * feature_mean.squeeze(0).cpu().float() / feature_std.squeeze(0).cpu().float()
        )

    return {
        "standardized_weight": standardized_weight,
        "standardized_bias": standardized_bias,
        "feature_mean": feature_mean.squeeze(0).cpu().float(),
        "feature_std": feature_std.squeeze(0).cpu().float(),
        "raw_direction": raw_direction,
        "raw_bias": raw_bias.cpu().float(),
        "metrics": {
            "train": _binary_metrics(train_scores, y_train, torch),
            "test": _binary_metrics(test_scores, y_test, torch),
            "final_train_loss": None if loss is None else float(loss.detach().cpu().item()),
        },
    }


def compare_probe_direction(
    records: list[dict[str, Any]],
    activations: Any,
    layers: list[int],
    caa_vectors: Any,
    model_variant: str,
    layer: int,
    raw_direction: Any,
    torch: Any,
) -> dict[str, Any]:
    layer_pos = layers.index(layer)
    prompt_ids = sorted(
        {
            str(rec["prompt_id"])
            for rec in records
            if rec["model_variant"] == model_variant
        }
    )
    by_key = {
        (rec["model_variant"], rec["prompt_id"], rec["condition"]): idx
        for idx, rec in enumerate(records)
    }

    suppression_deltas = []
    caa_deltas = []
    selected_caa_vectors = []
    for prompt_id in prompt_ids:
        clean_idx = by_key.get((model_variant, prompt_id, "clean"))
        supp_idx = by_key.get((model_variant, prompt_id, "suppression"))
        caa_idx = by_key.get((model_variant, prompt_id, "caa"))
        if clean_idx is None or supp_idx is None or caa_idx is None:
            continue
        clean = activations[clean_idx, layer_pos].float()
        suppression = activations[supp_idx, layer_pos].float()
        caa = activations[caa_idx, layer_pos].float()
        suppression_deltas.append(suppression - clean)
        caa_deltas.append(caa - clean)
        caa_vector_idx = records[caa_idx].get("caa_vector_idx")
        if caa_vector_idx is not None:
            selected_caa_vectors.append(caa_vectors[int(caa_vector_idx)].float())

    mean_suppression_delta = torch.stack(suppression_deltas).mean(dim=0)
    mean_caa_delta = torch.stack(caa_deltas).mean(dim=0)
    mean_selected_caa_vector = torch.stack(selected_caa_vectors).mean(dim=0)

    selected_caa_cosines = [
        _cosine(raw_direction, vector, torch) for vector in selected_caa_vectors
    ]
    suppression_delta_cosines = [
        _cosine(raw_direction, delta, torch) for delta in suppression_deltas
    ]
    caa_delta_cosines = [_cosine(raw_direction, delta, torch) for delta in caa_deltas]

    lora_comparison = compare_lora_clean_delta(
        records=records,
        activations=activations,
        layers=layers,
        layer=layer,
        raw_direction=raw_direction,
        torch=torch,
    )

    return {
        "probe_to_mean_suppression_delta_cosine": _cosine(
            raw_direction, mean_suppression_delta, torch
        ),
        "probe_to_mean_caa_delta_cosine": _cosine(raw_direction, mean_caa_delta, torch),
        "probe_to_mean_selected_caa_vector_cosine": _cosine(
            raw_direction, mean_selected_caa_vector, torch
        ),
        "probe_to_selected_caa_vector_cosines": _series_summary(selected_caa_cosines),
        "probe_to_per_prompt_suppression_delta_cosines": _series_summary(
            suppression_delta_cosines
        ),
        "probe_to_per_prompt_caa_delta_cosines": _series_summary(caa_delta_cosines),
        "mean_suppression_delta_l2": float(torch.linalg.vector_norm(mean_suppression_delta).item()),
        "mean_caa_delta_l2": float(torch.linalg.vector_norm(mean_caa_delta).item()),
        "mean_selected_caa_vector_l2": float(
            torch.linalg.vector_norm(mean_selected_caa_vector).item()
        ),
        "lora_clean_delta": lora_comparison,
    }


def compare_lora_clean_delta(
    records: list[dict[str, Any]],
    activations: Any,
    layers: list[int],
    layer: int,
    raw_direction: Any,
    torch: Any,
) -> dict[str, Any]:
    layer_pos = layers.index(layer)
    prompt_ids = sorted({str(rec["prompt_id"]) for rec in records})
    by_key = {
        (rec["model_variant"], rec["prompt_id"], rec["condition"]): idx
        for idx, rec in enumerate(records)
    }
    deltas = []
    for prompt_id in prompt_ids:
        base_idx = by_key.get(("base", prompt_id, "clean"))
        adapter_idx = by_key.get(("adapter", prompt_id, "clean"))
        if base_idx is None or adapter_idx is None:
            continue
        deltas.append(
            activations[adapter_idx, layer_pos].float()
            - activations[base_idx, layer_pos].float()
        )
    if not deltas:
        return {"count": 0}
    mean_delta = torch.stack(deltas).mean(dim=0)
    return {
        "count": len(deltas),
        "mean_l2": float(torch.linalg.vector_norm(mean_delta).item()),
        "probe_to_mean_lora_clean_delta_cosine": _cosine(raw_direction, mean_delta, torch),
        "probe_to_per_prompt_lora_clean_delta_cosines": _series_summary(
            [_cosine(raw_direction, delta, torch) for delta in deltas]
        ),
    }


def top_probe_dimensions(raw_direction: Any, top_k: int, torch: Any) -> list[dict[str, Any]]:
    k = min(int(top_k), raw_direction.numel())
    values, indices = torch.topk(torch.abs(raw_direction), k=k)
    rows = []
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        weight = float(raw_direction[index].item())
        rows.append(
            {
                "rank": rank,
                "dimension": int(index),
                "weight": weight,
                "abs_weight": float(value),
                "sign": "positive" if weight >= 0 else "negative",
            }
        )
    return rows


def build_probe_analysis_report(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]["test"]
    comparisons = summary["comparisons"]
    top_dims = summary["top_dimensions"][:20]
    lines = [
        f"# Phase 4 Layer {summary['layer']} Probe Direction Analysis",
        "",
        "## Probe",
        "",
        f"- Model variant: `{summary['model_variant']}`",
        f"- Task: `{summary['task']}`",
        f"- Train examples: {summary['train_n']}",
        f"- Test examples: {summary['test_n']}",
        f"- Test accuracy: {metrics['accuracy']:.4f}",
        f"- Test balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        f"- Test AUROC: {metrics['auroc']:.4f}",
        f"- Raw direction norm: {summary['direction_norm']:.6f}",
        "",
        "## Direction Comparisons",
        "",
        "| Comparison | Cosine / Value |",
        "| --- | ---: |",
        (
            "| Probe vs mean suppression delta | "
            f"{comparisons['probe_to_mean_suppression_delta_cosine']:.4f} |"
        ),
        (
            "| Probe vs mean CAA delta | "
            f"{comparisons['probe_to_mean_caa_delta_cosine']:.4f} |"
        ),
        (
            "| Probe vs mean selected raw CAA vector | "
            f"{comparisons['probe_to_mean_selected_caa_vector_cosine']:.4f} |"
        ),
        (
            "| Probe vs mean LoRA clean delta | "
            f"{comparisons['lora_clean_delta'].get('probe_to_mean_lora_clean_delta_cosine', 0.0):.4f} |"
        ),
        "",
        "## Top Dimensions",
        "",
        "| Rank | Dimension | Weight | Abs Weight | Sign |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in top_dims:
        lines.append(
            f"| {row['rank']} | {row['dimension']} | "
            f"{row['weight']:.6g} | {row['abs_weight']:.6g} | {row['sign']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze saved Phase 4 layer probe direction and top dimensions."
    )
    parser.add_argument("--phase4-dir", type=Path, required=True)
    parser.add_argument("--phase2a-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-variant", default="adapter")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args(argv)

    summary = analyze_layer_probe(
        phase4_dir=args.phase4_dir,
        phase2a_dir=args.phase2a_dir,
        output_dir=args.output_dir,
        model_variant=args.model_variant,
        layer=args.layer,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        top_k=args.top_k,
    )
    print(json.dumps(summary["metrics"]["test"], indent=2, sort_keys=True))
    print(
        "probe_to_mean_suppression_delta_cosine="
        f"{summary['comparisons']['probe_to_mean_suppression_delta_cosine']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
