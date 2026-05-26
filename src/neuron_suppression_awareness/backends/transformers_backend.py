from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from ..artifacts import (
    build_phase0_report,
    create_run_dir,
    write_json,
    write_jsonl,
    write_text,
)
from ..config import Phase0Config
from ..datasets import PromptRecord, load_phase0_prompts
from ..errors import ActivationMismatchError, HookFailure, MissingDependencyError
from ..hooks import DownProjNeuronHook, describe_down_proj_layers, get_down_proj_module
from ..refusal import classify_refusal_preview, preview_text


@dataclass(frozen=True)
class Phase0RunResult:
    artifact_dir: Path
    activation_summary: dict[str, Any]
    generation_previews: list[dict[str, str]]
    activation_failures: list[str]


@dataclass(frozen=True)
class PromptActivation:
    prompt: PromptRecord
    aggregate: float
    score_index: int
    score_reason: str
    rows: list[dict[str, Any]]


def run_phase0(config: Phase0Config) -> Phase0RunResult:
    torch, auto_model_cls, auto_tokenizer_cls = _runtime_imports()

    harmful_prompts, harmless_prompts = load_phase0_prompts(config.datasets)
    tokenizer = auto_tokenizer_cls.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    model = _load_model(config, torch, auto_model_cls)
    model.eval()

    down_proj_names = describe_down_proj_layers(model)
    if not down_proj_names:
        get_down_proj_module(model, config.phase0.layer)

    activation_rows: list[dict[str, Any]] = []
    harmful_acts = []
    harmless_acts = []
    for prompt in [*harmful_prompts, *harmless_prompts]:
        activation = collect_prompt_activation(model, tokenizer, prompt, config, torch)
        activation_rows.extend(activation.rows)
        if prompt.source == "harmful":
            harmful_acts.append(activation)
        else:
            harmless_acts.append(activation)

    activation_summary = build_activation_summary(
        config,
        harmful_acts,
        harmless_acts,
        down_proj_names,
    )
    activation_failures = validate_activation_summary(config, activation_summary)

    generation_rows: list[dict[str, Any]] = []
    if not activation_failures:
        generation_prompt = harmful_prompts[0]
        generation_rows = [
            generate_response(
                model,
                tokenizer,
                generation_prompt,
                config,
                torch,
                mode="clean",
                pin=False,
            ),
            generate_response(
                model,
                tokenizer,
                generation_prompt,
                config,
                torch,
                mode="pinned",
                pin=True,
            ),
        ]

    artifact_dir = create_run_dir(config.outputs)
    write_json(artifact_dir / "activation_summary.json", activation_summary)
    write_jsonl(artifact_dir / "activations.jsonl", activation_rows)
    write_jsonl(artifact_dir / "generations.jsonl", generation_rows)
    write_text(
        artifact_dir / "phase0_report.md",
        build_phase0_report(
            config,
            activation_summary,
            generation_rows,
            activation_failures,
        ),
    )

    previews = [
        {
            "mode": row["mode"],
            "preview": preview_text(row["response"]),
            "refusal_preview": row["refusal_preview"],
        }
        for row in generation_rows
    ]
    result = Phase0RunResult(
        artifact_dir=artifact_dir,
        activation_summary=activation_summary,
        generation_previews=previews,
        activation_failures=activation_failures,
    )
    if activation_failures:
        joined = "; ".join(activation_failures)
        raise ActivationMismatchError(
            f"Activation validation failed: {joined}. Artifacts written to {artifact_dir}."
        )
    return result


def collect_prompt_activation(
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord,
    config: Phase0Config,
    torch: Any,
) -> PromptActivation:
    input_ids = apply_chat_template(tokenizer, prompt.text, torch)
    attention_mask = torch.ones_like(input_ids)
    input_ids = _move_to_model_device(input_ids, model)
    attention_mask = _move_to_model_device(attention_mask, model)

    module = get_down_proj_module(model, config.phase0.layer)
    hook = DownProjNeuronHook(neuron=config.phase0.neuron, capture=True)
    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()

    if not hook.captures:
        raise HookFailure("Activation hook did not capture any down_proj inputs.")
    capture = hook.captures[0]
    if capture.shape[0] != 1:
        raise HookFailure(
            "Phase 0 expected one prompt per activation forward pass; "
            f"captured batch size {capture.shape[0]}."
        )

    activations = [float(value) for value in capture[0].tolist()]
    token_ids = input_ids[0].detach().cpu().tolist()
    token_texts = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]
    score_index, score_reason = choose_score_index(token_texts, activations, config)
    rows = []
    window_start = max(0, len(token_texts) - config.phase0.activation_window_tokens)
    for index, (token_id, token_text, activation) in enumerate(
        zip(token_ids, token_texts, activations, strict=True)
    ):
        rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "source": prompt.source,
                "dataset_id": prompt.dataset_id,
                "row_index": prompt.row_index,
                "token_index": index,
                "token_id": int(token_id),
                "token_text": token_text,
                "activation": activation,
                "in_score_window": index >= window_start,
                "selected_for_summary": index == score_index,
                "score_reason": score_reason if index == score_index else None,
                "prompt": prompt.text,
            }
        )
    return PromptActivation(
        prompt=prompt,
        aggregate=float(activations[score_index]),
        score_index=score_index,
        score_reason=score_reason,
        rows=rows,
    )


def choose_score_index(
    token_texts: list[str],
    activations: list[float],
    config: Phase0Config,
) -> tuple[int, str]:
    window_start = max(0, len(token_texts) - config.phase0.activation_window_tokens)
    candidate_indices = list(range(window_start, len(token_texts)))
    matches = [
        index
        for index in candidate_indices
        if _token_matches(token_texts[index], config)
    ]
    if config.phase0.aggregation == "selected_token" and matches:
        return matches[-1], "matched score_token_text"

    aggregation = (
        config.phase0.aggregation
        if config.phase0.aggregation != "selected_token"
        else config.phase0.fallback_aggregation
    )
    selected = _aggregate_index(candidate_indices, activations, aggregation)
    return selected, f"{aggregation} over fallback score window"


def build_activation_summary(
    config: Phase0Config,
    harmful_acts: list[PromptActivation],
    harmless_acts: list[PromptActivation],
    down_proj_names: list[str],
) -> dict[str, Any]:
    harmful_values = [item.aggregate for item in harmful_acts]
    harmless_values = [item.aggregate for item in harmless_acts]
    harmful_mean = mean(harmful_values)
    harmless_mean = mean(harmless_values)
    return {
        "model": config.model.id,
        "revision": config.model.revision,
        "layer": config.phase0.layer,
        "neuron": config.phase0.neuron,
        "pin_value": config.phase0.pin_value,
        "down_proj_layers": down_proj_names,
        "score_token_text": config.phase0.score_token_text,
        "harmful": _series_summary(harmful_values),
        "harmless": _series_summary(harmless_values),
        "abs_mean_gap": abs(harmful_mean - harmless_mean),
        "reference": {
            "harmful_mean": config.expected_activations.harmful_mean_reference,
            "harmless_mean": config.expected_activations.harmless_mean_reference,
            "harmful_mean_range": list(config.expected_activations.harmful_mean_range),
            "harmless_mean_range": list(config.expected_activations.harmless_mean_range),
            "min_abs_gap": config.expected_activations.min_abs_gap,
        },
        "per_prompt": [
            _prompt_activation_summary(item) for item in [*harmful_acts, *harmless_acts]
        ],
    }


def validate_activation_summary(
    config: Phase0Config,
    summary: dict[str, Any],
) -> list[str]:
    failures = []
    harmful_mean = float(summary["harmful"]["mean"])
    harmless_mean = float(summary["harmless"]["mean"])
    harmful_range = config.expected_activations.harmful_mean_range
    harmless_range = config.expected_activations.harmless_mean_range
    if not _in_range(harmful_mean, harmful_range):
        failures.append(
            f"harmful mean {harmful_mean:.4g} outside expected range {harmful_range}"
        )
    if not _in_range(harmless_mean, harmless_range):
        failures.append(
            f"harmless mean {harmless_mean:.4g} outside expected range {harmless_range}"
        )
    gap = abs(harmful_mean - harmless_mean)
    if gap < config.expected_activations.min_abs_gap:
        failures.append(
            f"absolute harmful/harmless gap {gap:.4g} below "
            f"{config.expected_activations.min_abs_gap}"
        )
    return failures


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord,
    config: Phase0Config,
    torch: Any,
    mode: str,
    pin: bool,
) -> dict[str, Any]:
    layer = config.phase0.layer if pin else None
    neuron = config.phase0.neuron if pin else None
    pin_value = config.phase0.pin_value if pin else None
    response = generate_with_hook(
        model,
        tokenizer,
        prompt.text,
        torch,
        generation_kwargs(config, tokenizer),
        layer=layer,
        neuron=neuron,
        pin_value=pin_value,
    )
    return {
        "prompt_id": prompt.prompt_id,
        "source": prompt.source,
        "dataset_id": prompt.dataset_id,
        "row_index": prompt.row_index,
        "mode": mode,
        "pin_active": pin,
        "pin_value": config.phase0.pin_value if pin else None,
        "response": response,
        "response_preview": preview_text(response),
        "refusal_preview": classify_refusal_preview(response),
        "prompt": prompt.text,
    }


def generate_with_hook(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    torch: Any,
    gen_kwargs: dict[str, Any],
    layer: int | None = None,
    neuron: int | None = None,
    pin_value: float | None = None,
) -> str:
    input_ids = apply_chat_template(tokenizer, prompt_text, torch)
    attention_mask = torch.ones_like(input_ids)
    input_ids = _move_to_model_device(input_ids, model)
    attention_mask = _move_to_model_device(attention_mask, model)

    handle = None
    hook = None
    if layer is not None and neuron is not None and pin_value is not None:
        module = get_down_proj_module(model, layer)
        hook = DownProjNeuronHook(
            neuron=neuron,
            pin_value=pin_value,
            capture=False,
        )
        handle = module.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
    finally:
        if handle is not None:
            handle.remove()

    if hook is not None and hook.pinned_calls == 0:
        raise HookFailure("Pinned generation hook was registered but never fired.")

    generated = outputs[0][input_ids.shape[-1] :].detach().cpu().tolist()
    return tokenizer.decode(generated, skip_special_tokens=True)


def generation_kwargs(config: Any, tokenizer: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": config.generation.max_new_tokens,
        "do_sample": config.generation.do_sample,
    }
    if config.generation.do_sample:
        kwargs["temperature"] = config.generation.temperature
    if getattr(tokenizer, "pad_token_id", None) is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    elif getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["pad_token_id"] = tokenizer.eos_token_id
    return kwargs


def apply_chat_template(tokenizer: Any, prompt: str, torch: Any) -> Any:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, **kwargs)

    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    elif isinstance(encoded, dict):
        input_ids = encoded["input_ids"]
    else:
        input_ids = encoded
    if isinstance(input_ids, list):
        input_ids = torch.tensor([input_ids], dtype=torch.long)
    if not hasattr(input_ids, "dim"):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids


def _load_model(config: Phase0Config, torch: Any, auto_model_cls: Any) -> Any:
    backend = config.backend.transformers
    kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": _torch_dtype(config.model.dtype, torch),
    }
    for key in ("device_map", "low_cpu_mem_usage", "attn_implementation"):
        if key in backend:
            kwargs[key] = backend[key]

    quant = config.model.quantization
    if quant is not None and quant.load_in_4bit:
        bnb_config = _build_bnb_config(quant, torch)
        kwargs["quantization_config"] = bnb_config

    return auto_model_cls.from_pretrained(config.model.id, **kwargs)


def _build_bnb_config(quant: Any, torch: Any) -> Any:
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise MissingDependencyError(
            "4-bit quantization requires bitsandbytes. "
            "Install with `pip install bitsandbytes`."
        ) from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=_torch_dtype(quant.bnb_4bit_compute_dtype, torch),
        bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
    )


def _runtime_imports() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise MissingDependencyError(
            "Phase 0 Transformers backend requires torch and transformers."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _torch_dtype(dtype_name: str, torch: Any) -> Any:
    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise MissingDependencyError(f"Unsupported torch dtype in config: {dtype_name}")
    return mapping[dtype_name]


def _move_to_model_device(tensor: Any, model: Any) -> Any:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return tensor
    return tensor.to(device)


def _token_matches(token_text: str, config: Phase0Config) -> bool:
    expected = config.phase0.score_token_text
    if config.phase0.score_token_match == "exact":
        return token_text == expected
    return expected in token_text


def _aggregate_index(indices: list[int], values: list[float], aggregation: str) -> int:
    if not indices:
        raise HookFailure("Cannot aggregate over an empty activation window.")
    if aggregation == "min":
        return min(indices, key=lambda index: values[index])
    if aggregation == "max":
        return max(indices, key=lambda index: values[index])
    if aggregation == "mean":
        target = mean(values[index] for index in indices)
        return min(indices, key=lambda index: abs(values[index] - target))
    raise HookFailure(f"Unsupported aggregation: {aggregation}")


def _series_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise HookFailure("Cannot summarize an empty activation series.")
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def _prompt_activation_summary(item: PromptActivation) -> dict[str, Any]:
    return {
        "prompt_id": item.prompt.prompt_id,
        "source": item.prompt.source,
        "dataset_id": item.prompt.dataset_id,
        "row_index": item.prompt.row_index,
        "aggregate": item.aggregate,
        "score_index": item.score_index,
        "score_reason": item.score_reason,
    }


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1] and math.isfinite(value)
