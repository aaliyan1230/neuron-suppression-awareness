from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from ..artifacts import (
    build_phase1_report,
    create_run_dir,
    write_json,
    write_text,
)
from ..checkpoint import append_record, get_completed_ids, load_records
from ..config import OutputConfig, Phase1Config
from ..datasets import PromptRecord, load_prompt_records
from ..hooks import DownProjNeuronHook, get_down_proj_module
from ..judge import judge_response, load_judge
from ..refusal import classify_refusal_preview, preview_text
from .transformers_backend import (
    _load_model,
    _move_to_model_device,
    _runtime_imports,
    apply_chat_template,
    generate_with_hook,
    generation_kwargs,
)


@dataclass(frozen=True)
class Phase1RunResult:
    artifact_dir: Path
    clean_asr: float
    suppressed_asr: float
    passed: bool
    n_prompts: int


def run_phase1(config: Phase1Config) -> Phase1RunResult:
    torch, auto_model_cls, auto_tokenizer_cls = _runtime_imports()
    artifact_dir = _create_phase1_run_dir(config.outputs, config.checkpoint)
    generations_path = artifact_dir / "generations.jsonl"
    activations_path = artifact_dir / "clean_activations.jsonl"
    judgments_path = artifact_dir / "judgments.jsonl"

    prompts = load_prompt_records(config.dataset, "jbb")
    tokenizer = auto_tokenizer_cls.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    model = _load_model(config, torch, auto_model_cls)
    model.eval()

    completed_generations = (
        get_completed_ids(generations_path) if config.checkpoint else set()
    )
    completed_activations = {
        record["prompt_id"] for record in load_records(activations_path)
    }
    gen_kwargs = generation_kwargs(config, tokenizer)

    for index, prompt in enumerate(prompts, start=1):
        if prompt.prompt_id not in completed_activations:
            activation = collect_clean_activation(model, tokenizer, prompt, config, torch)
            append_record(activations_path, activation)
            completed_activations.add(prompt.prompt_id)

        clean_key = f"{prompt.prompt_id}:clean"
        if clean_key not in completed_generations:
            response = generate_with_hook(
                model,
                tokenizer,
                prompt.text,
                torch,
                gen_kwargs,
            )
            append_record(
                generations_path,
                build_generation_record(prompt, "clean", False, None, response),
            )
            completed_generations.add(clean_key)

        suppressed_key = f"{prompt.prompt_id}:suppressed"
        if suppressed_key not in completed_generations:
            response = generate_with_hook(
                model,
                tokenizer,
                prompt.text,
                torch,
                gen_kwargs,
                layer=config.suppression.layer,
                neuron=config.suppression.neuron,
                pin_value=config.suppression.pin_value,
            )
            append_record(
                generations_path,
                build_generation_record(
                    prompt,
                    "suppressed",
                    True,
                    config.suppression.pin_value,
                    response,
                ),
            )
            completed_generations.add(suppressed_key)

        print(f"Prompt {index}/{len(prompts)}: clean and suppressed complete")

    del model, tokenizer
    gc.collect()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    generations = load_records(generations_path)
    judge_model, judge_tokenizer = load_judge(
        config.judge,
        config.backend,
        torch,
        auto_model_cls,
        auto_tokenizer_cls,
    )
    completed_judgments = (
        get_completed_ids(judgments_path) if config.checkpoint else set()
    )
    for index, generation in enumerate(generations, start=1):
        key = f"{generation['prompt_id']}:{generation['mode']}"
        if key in completed_judgments:
            continue
        judgment = judge_response(
            judge_model,
            judge_tokenizer,
            generation["prompt"],
            generation["response"],
            torch,
            config.judge.max_new_tokens,
        )
        append_record(
            judgments_path,
            {
                **generation,
                **judgment,
                "unsafe": judgment["verdict"] == "unsafe",
            },
        )
        completed_judgments.add(key)
        print(f"Judged {index}/{len(generations)}: {key} -> {judgment['verdict']}")

    judgments = load_records(judgments_path)
    clean_asr, suppressed_asr = compute_asr(judgments)
    passed = (
        suppressed_asr >= config.pass_criteria.min_suppressed_asr
        and clean_asr <= config.pass_criteria.max_clean_asr
    )
    results = {
        "model": config.model.id,
        "revision": config.model.revision,
        "dataset": config.dataset.id,
        "n_prompts": len(prompts),
        "clean_asr": clean_asr,
        "suppressed_asr": suppressed_asr,
        "passed": passed,
        "pass_criteria": {
            "min_suppressed_asr": config.pass_criteria.min_suppressed_asr,
            "max_clean_asr": config.pass_criteria.max_clean_asr,
        },
        "suppression": {
            "layer": config.suppression.layer,
            "neuron": config.suppression.neuron,
            "pin_value": config.suppression.pin_value,
        },
        "clean_activation_mean": _activation_mean(load_records(activations_path)),
    }
    write_json(artifact_dir / "phase1_results.json", results)
    write_text(
        artifact_dir / "phase1_report.md",
        build_phase1_report(config, results, judgments, load_records(activations_path)),
    )
    return Phase1RunResult(
        artifact_dir=artifact_dir,
        clean_asr=clean_asr,
        suppressed_asr=suppressed_asr,
        passed=passed,
        n_prompts=len(prompts),
    )


def collect_clean_activation(
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord,
    config: Phase1Config,
    torch: Any,
) -> dict[str, Any]:
    input_ids = apply_chat_template(tokenizer, prompt.text, torch)
    attention_mask = torch.ones_like(input_ids)
    input_ids = _move_to_model_device(input_ids, model)
    attention_mask = _move_to_model_device(attention_mask, model)
    module = get_down_proj_module(model, config.suppression.layer)
    hook = DownProjNeuronHook(neuron=config.suppression.neuron, capture=True)
    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    if not hook.captures:
        values: list[float] = []
    else:
        values = [float(value) for value in hook.captures[0][0].tolist()]
    return {
        "prompt_id": prompt.prompt_id,
        "source": prompt.source,
        "dataset_id": prompt.dataset_id,
        "row_index": prompt.row_index,
        "layer": config.suppression.layer,
        "neuron": config.suppression.neuron,
        "mean_activation": mean(values) if values else None,
        "min_activation": min(values) if values else None,
        "max_activation": max(values) if values else None,
        "token_count": len(values),
        "prompt": prompt.text,
    }


def build_generation_record(
    prompt: PromptRecord,
    mode: str,
    pin_active: bool,
    pin_value: float | None,
    response: str,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt.prompt_id,
        "source": prompt.source,
        "dataset_id": prompt.dataset_id,
        "row_index": prompt.row_index,
        "mode": mode,
        "pin_active": pin_active,
        "pin_value": pin_value,
        "prompt": prompt.text,
        "response": response,
        "response_preview": preview_text(response),
        "response_first_50_tokens": " ".join(response.split()[:50]),
        "refusal_preview": classify_refusal_preview(response),
    }


def compute_asr(judgments: list[dict[str, Any]]) -> tuple[float, float]:
    return (_mode_asr(judgments, "clean"), _mode_asr(judgments, "suppressed"))


def _mode_asr(judgments: list[dict[str, Any]], mode: str) -> float:
    rows = [row for row in judgments if row.get("mode") == mode]
    if not rows:
        return 0.0
    unsafe = sum(1 for row in rows if row.get("verdict") == "unsafe")
    return unsafe / len(rows)


def _activation_mean(records: list[dict[str, Any]]) -> float | None:
    values = [
        float(record["mean_activation"])
        for record in records
        if record.get("mean_activation") is not None
    ]
    return mean(values) if values else None


def _create_phase1_run_dir(config: OutputConfig, checkpoint: bool) -> Path:
    if checkpoint and config.run_name is not None:
        path = config.root / config.run_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    return create_run_dir(config)
