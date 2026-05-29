from __future__ import annotations

import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from ..artifacts import (
    build_phase4_report,
    create_run_dir,
    write_json,
    write_jsonl,
)
from ..config import Phase4Config
from ..datasets import PromptRecord, load_prompt_records
from ..hooks import (
    DownProjNeuronHook,
    ResidualInjectionHook,
    ResidualStreamHook,
    get_decoder_layer,
    get_down_proj_module,
)
from .phase3_transformers import _load_caa_artifacts
from .transformers_backend import (
    _load_model,
    _move_to_model_device,
    _runtime_imports,
    apply_chat_template,
)


@dataclass(frozen=True)
class Phase4RunResult:
    artifact_dir: Path
    n_records: int
    n_prompts: int
    model_variants: tuple[str, ...]
    layers: tuple[int, ...]


def run_phase4(config: Phase4Config) -> Phase4RunResult:
    torch, auto_model_cls, auto_tokenizer_cls = _runtime_imports()
    artifact_dir = create_run_dir(config.outputs)

    harmful_prompts = load_prompt_records(config.prompts.harmful, "harmful")
    harmless_prompts = load_prompt_records(config.prompts.harmless, "harmless")
    prompts = harmful_prompts + harmless_prompts
    print(f"Loaded {len(harmful_prompts)} harmful + {len(harmless_prompts)} harmless prompts")

    caa_vectors, mean_train_norm = _load_caa_artifacts(
        config.inputs.phase2a_artifact_dir, torch
    )
    print(f"Loaded CAA vectors: shape {tuple(caa_vectors.shape)}, mean_norm={mean_train_norm:.2f}")

    tokenizer = auto_tokenizer_cls.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    records: list[dict[str, Any]] = []
    activation_rows = []

    for variant in config.analysis.model_variants:
        print(f"\n=== Phase 4 collection: {variant} model ===")
        model = _load_phase4_model(config, variant, torch, auto_model_cls)
        model.eval()
        variant_records, variant_activations = collect_phase4_activations(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            config=config,
            torch=torch,
            caa_vectors=caa_vectors,
            model_variant=variant,
        )
        records.extend(variant_records)
        activation_rows.append(variant_activations)

        del model
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{variant} model unloaded")

    for record_index, record in enumerate(records):
        record["record_index"] = record_index

    activations = torch.cat(activation_rows, dim=0) if activation_rows else torch.empty(0)
    tensor_payload = {
        "activations": activations,
        "records": records,
        "layers": list(config.analysis.layers),
        "capture_position": config.analysis.capture_position,
        "shape": list(activations.shape),
    }
    torch.save(tensor_payload, artifact_dir / "phase4_activations.pt")
    write_jsonl(artifact_dir / "phase4_records.jsonl", records)

    geometry = compute_phase4_geometry(
        records=records,
        activations=activations,
        layers=config.analysis.layers,
        caa_vectors=caa_vectors,
        injection_layer=config.injection.layer,
        torch=torch,
    )
    write_json(artifact_dir / "phase4_geometry.json", geometry)

    probe_results = train_phase4_probes(
        records=records,
        activations=activations,
        layers=config.analysis.layers,
        train_fraction=config.analysis.probe_train_fraction,
        epochs=config.analysis.probe_epochs,
        learning_rate=config.analysis.probe_learning_rate,
        seed=config.analysis.seed,
        torch=torch,
    )
    write_json(artifact_dir / "phase4_probe_results.json", probe_results)

    summary = {
        "model": config.model.id,
        "revision": config.model.revision,
        "phase2a_artifact_dir": str(config.inputs.phase2a_artifact_dir),
        "phase2b_adapter_dir": str(config.inputs.phase2b_adapter_dir),
        "n_prompts": len(prompts),
        "n_records": len(records),
        "model_variants": list(config.analysis.model_variants),
        "layers": list(config.analysis.layers),
        "capture_position": config.analysis.capture_position,
        "conditions": ["clean", "caa", "suppression"],
        "geometry": geometry,
        "probe_results": probe_results,
    }
    write_json(artifact_dir / "phase4_summary.json", summary)
    (artifact_dir / "phase4_report.md").write_text(
        build_phase4_report(config, summary),
        encoding="utf-8",
    )

    return Phase4RunResult(
        artifact_dir=artifact_dir,
        n_records=len(records),
        n_prompts=len(prompts),
        model_variants=config.analysis.model_variants,
        layers=config.analysis.layers,
    )


def collect_phase4_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[PromptRecord],
    config: Phase4Config,
    torch: Any,
    caa_vectors: Any,
    model_variant: str,
) -> tuple[list[dict[str, Any]], Any]:
    records: list[dict[str, Any]] = []
    rows = []
    n_test_vectors = caa_vectors.shape[0] - 150
    for prompt_index, prompt in enumerate(prompts):
        for condition in ("clean", "caa", "suppression"):
            caa_vector_idx = None
            vector = None
            if condition == "caa":
                caa_vector_idx = 150 + (prompt_index % max(1, n_test_vectors))
                vector = caa_vectors[caa_vector_idx].detach().float() * config.injection.alpha
            layer_activations = collect_single_activation_row(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt.text,
                config=config,
                torch=torch,
                condition=condition,
                caa_vector=vector,
            )
            rows.append(torch.stack(layer_activations, dim=0))
            records.append(
                {
                    "record_index": len(records),
                    "model_variant": model_variant,
                    "condition": condition,
                    "prompt_id": prompt.prompt_id,
                    "source": prompt.source,
                    "dataset_id": prompt.dataset_id,
                    "row_index": prompt.row_index,
                    "prompt_text": prompt.text,
                    "layers": list(config.analysis.layers),
                    "capture_position": config.analysis.capture_position,
                    "caa_vector_idx": caa_vector_idx,
                    "caa_alpha": config.injection.alpha if condition == "caa" else None,
                    "pin_active": condition == "suppression",
                    "pin_value": (
                        config.suppression.pin_value if condition == "suppression" else None
                    ),
                }
            )
        if (prompt_index + 1) % 10 == 0 or prompt_index + 1 == len(prompts):
            print(f"{model_variant}: {prompt_index + 1}/{len(prompts)} prompts collected")
    return records, torch.stack(rows, dim=0)


def collect_single_activation_row(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    config: Phase4Config,
    torch: Any,
    condition: str,
    caa_vector: Any | None,
) -> list[Any]:
    input_ids = apply_chat_template(tokenizer, prompt_text, torch)
    attention_mask = torch.ones_like(input_ids)
    token_index = input_ids.shape[-1] - 1
    input_ids = _move_to_model_device(input_ids, model)
    attention_mask = _move_to_model_device(attention_mask, model)

    handles = []
    capture_hooks: list[ResidualStreamHook] = []
    suppression_hook = None
    injection_hook = None
    try:
        if condition == "suppression":
            module = get_down_proj_module(model, config.suppression.layer)
            suppression_hook = DownProjNeuronHook(
                neuron=config.suppression.neuron,
                pin_value=config.suppression.pin_value,
                capture=False,
            )
            handles.append(module.register_forward_pre_hook(suppression_hook))

        if condition == "caa":
            if caa_vector is None:
                raise RuntimeError("CAA condition requires a CAA vector.")
            module = get_decoder_layer(model, config.injection.layer)
            injection_hook = ResidualInjectionHook(
                vectors=caa_vector.unsqueeze(0),
                token_indices=torch.tensor([token_index], dtype=torch.long),
                apply_once=True,
            )
            handles.append(module.register_forward_hook(injection_hook))

        for layer in config.analysis.layers:
            module = get_decoder_layer(model, layer)
            hook = ResidualStreamHook(capture_last_token=True)
            capture_hooks.append(hook)
            handles.append(module.register_forward_hook(hook))

        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if suppression_hook is not None and suppression_hook.pinned_calls == 0:
        raise RuntimeError("Suppression hook was registered but never fired.")
    if injection_hook is not None and injection_hook.injected_calls == 0:
        raise RuntimeError("CAA injection hook was registered but never fired.")

    captured = []
    for layer, hook in zip(config.analysis.layers, capture_hooks, strict=True):
        if not hook.captures:
            raise RuntimeError(f"Residual hook for layer {layer} captured no activations.")
        captured.append(hook.captures[0][0])
    return captured


def compute_phase4_geometry(
    records: list[dict[str, Any]],
    activations: Any,
    layers: tuple[int, ...],
    caa_vectors: Any,
    injection_layer: int,
    torch: Any,
) -> dict[str, Any]:
    by_key = {
        (rec["model_variant"], rec["prompt_id"], rec["condition"]): idx
        for idx, rec in enumerate(records)
    }
    model_variants = sorted({str(rec["model_variant"]) for rec in records})
    layer_results: dict[str, dict[str, Any]] = {}

    for variant in model_variants:
        prompt_ids = sorted(
            {
                str(rec["prompt_id"])
                for rec in records
                if rec["model_variant"] == variant
            }
        )
        variant_results: dict[str, Any] = {}
        for layer_pos, layer in enumerate(layers):
            caa_norms = []
            suppression_norms = []
            delta_cosines = []
            raw_caa_cosines = []
            for prompt_id in prompt_ids:
                clean_idx = by_key.get((variant, prompt_id, "clean"))
                caa_idx = by_key.get((variant, prompt_id, "caa"))
                supp_idx = by_key.get((variant, prompt_id, "suppression"))
                if clean_idx is None or caa_idx is None or supp_idx is None:
                    continue
                clean = activations[clean_idx, layer_pos].float()
                caa = activations[caa_idx, layer_pos].float()
                suppression = activations[supp_idx, layer_pos].float()
                caa_delta = caa - clean
                suppression_delta = suppression - clean
                caa_norms.append(float(torch.linalg.vector_norm(caa_delta).item()))
                suppression_norms.append(
                    float(torch.linalg.vector_norm(suppression_delta).item())
                )
                delta_cosines.append(float(_cosine(caa_delta, suppression_delta, torch)))
                if layer == injection_layer:
                    caa_idx_raw = records[caa_idx].get("caa_vector_idx")
                    if caa_idx_raw is not None:
                        raw_vector = caa_vectors[int(caa_idx_raw)].float()
                        raw_caa_cosines.append(
                            float(_cosine(suppression_delta, raw_vector, torch))
                        )
            variant_results[str(layer)] = {
                "caa_delta_l2": _series_summary(caa_norms),
                "suppression_delta_l2": _series_summary(suppression_norms),
                "suppression_to_caa_delta_cosine": _series_summary(delta_cosines),
                "suppression_to_raw_caa_cosine": _series_summary(raw_caa_cosines),
                "suppression_to_caa_l2_ratio": _safe_ratio_summary(
                    suppression_norms, caa_norms
                ),
            }
        layer_results[variant] = variant_results

    return {
        "layers": list(layers),
        "injection_layer": injection_layer,
        "model_variants": model_variants,
        "by_model_and_layer": layer_results,
    }


def train_phase4_probes(
    records: list[dict[str, Any]],
    activations: Any,
    layers: tuple[int, ...],
    train_fraction: float,
    epochs: int,
    learning_rate: float,
    seed: int,
    torch: Any,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    model_variants = sorted({str(rec["model_variant"]) for rec in records})
    for variant in model_variants:
        variant_results = {}
        for layer_pos, layer in enumerate(layers):
            examples = [
                (idx, rec)
                for idx, rec in enumerate(records)
                if rec["model_variant"] == variant
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
            if not train_indices or not test_indices:
                variant_results[str(layer)] = {"error": "empty train/test split"}
                continue

            x_train = activations[train_indices, layer_pos].float()
            y_train = _labels([records[idx] for idx in train_indices], torch)
            x_test = activations[test_indices, layer_pos].float()
            y_test = _labels([records[idx] for idx in test_indices], torch)
            variant_results[str(layer)] = _fit_logistic_probe(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed + layer,
                torch=torch,
            )
            variant_results[str(layer)]["train_n"] = len(train_indices)
            variant_results[str(layer)]["test_n"] = len(test_indices)
            variant_results[str(layer)]["train_prompt_count"] = len(split["train_prompt_ids"])
            variant_results[str(layer)]["test_prompt_count"] = len(split["test_prompt_ids"])
        results[variant] = variant_results
    return {
        "task": "clean_vs_suppression",
        "positive_label": "suppression",
        "train_fraction": train_fraction,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "by_model_and_layer": results,
    }


def _load_phase4_model(
    config: Phase4Config,
    variant: str,
    torch: Any,
    auto_model_cls: Any,
) -> Any:
    model = _load_model(config, torch, auto_model_cls)
    if variant == "base":
        return model
    if variant == "adapter":
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Phase 4 adapter analysis requires peft.") from exc
        model = PeftModel.from_pretrained(model, str(config.inputs.phase2b_adapter_dir))
        print(f"Loaded PEFT adapter from {config.inputs.phase2b_adapter_dir}")
        return model
    raise RuntimeError(f"Unsupported Phase 4 model variant: {variant}")


def _fit_logistic_probe(
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
    mean_vec = x_train.mean(dim=0, keepdim=True)
    std_vec = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean_vec) / std_vec
    x_test = (x_test - mean_vec) / std_vec

    model = torch.nn.Linear(x_train.shape[-1], 1)
    with torch.no_grad():
        model.weight.normal_(mean=0.0, std=0.01, generator=generator)
        model.bias.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train).squeeze(-1)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_scores = torch.sigmoid(model(x_train).squeeze(-1))
        test_scores = torch.sigmoid(model(x_test).squeeze(-1))
    train_metrics = _binary_metrics(train_scores, y_train, torch)
    test_metrics = _binary_metrics(test_scores, y_test, torch)
    return {
        "train": train_metrics,
        "test": test_metrics,
        "final_train_loss": float(loss.detach().cpu().item()),
    }


def _binary_metrics(scores: Any, labels: Any, torch: Any) -> dict[str, Any]:
    preds = scores >= 0.5
    truth = labels >= 0.5
    tp = int(torch.logical_and(preds, truth).sum().item())
    tn = int(torch.logical_and(~preds, ~truth).sum().item())
    fp = int(torch.logical_and(preds, ~truth).sum().item())
    fn = int(torch.logical_and(~preds, truth).sum().item())
    total = max(1, tp + tn + fp + fn)
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return {
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": (tpr + tnr) / 2,
        "auroc": _auroc(scores, labels, torch),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _auroc(scores: Any, labels: Any, torch: Any) -> float | None:
    positives = scores[labels >= 0.5]
    negatives = scores[labels < 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return None
    comparisons = positives[:, None] - negatives[None, :]
    wins = (comparisons > 0).float().sum()
    ties = (comparisons == 0).float().sum() * 0.5
    return float(((wins + ties) / comparisons.numel()).item())


def _labels(rows: list[dict[str, Any]], torch: Any) -> Any:
    return torch.tensor(
        [1.0 if row["condition"] == "suppression" else 0.0 for row in rows],
        dtype=torch.float32,
    )


def _prompt_split(
    rows: list[dict[str, Any]],
    train_fraction: float,
    seed: int,
) -> dict[str, set[str]]:
    prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(prompt_ids)
    n_train = max(1, min(len(prompt_ids) - 1, round(len(prompt_ids) * train_fraction)))
    return {
        "train_prompt_ids": set(prompt_ids[:n_train]),
        "test_prompt_ids": set(prompt_ids[n_train:]),
    }


def _cosine(a: Any, b: Any, torch: Any) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((torch.dot(a.flatten(), b.flatten()) / denom).item())


def _series_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _safe_ratio_summary(numerators: list[float], denominators: list[float]) -> dict[str, Any]:
    ratios = [
        num / den
        for num, den in zip(numerators, denominators, strict=False)
        if den > 1e-12
    ]
    return _series_summary(ratios)
