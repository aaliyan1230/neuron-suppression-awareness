from __future__ import annotations

import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import (
    build_phase5_report,
    create_run_dir,
    write_json,
    write_jsonl,
)
from ..checkpoint import append_record, get_completed_ids, load_records
from ..config import OutputConfig, Phase5Config
from ..datasets import PromptRecord, load_prompt_records
from ..hooks import (
    DownProjNeuronHook,
    ResidualInjectionHook,
    get_decoder_layer,
    get_down_proj_module,
)
from ..judge import judge_response, load_judge
from ..phase2b import (
    Phase2AArtifacts,
    build_injection_batch,
    classify_detection,
    encode_supervised_example,
    load_phase2a_artifacts_from_dir,
)
from ..phase5_training_data import build_phase5_training_dataset
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
class Phase5RunResult:
    artifact_dir: Path
    suppression_detection_rate: float
    caa_detection_rate: float
    clean_fpr: float
    noise_fpr: float
    clean_asr: float
    suppressed_asr: float
    n_train_examples: int
    passed: bool


def run_phase5(config: Phase5Config) -> Phase5RunResult:
    torch, auto_model_cls, auto_tokenizer_cls = _runtime_imports()
    _set_seed(config, torch)
    artifact_dir = _create_phase5_run_dir(config.outputs, config.checkpoint)
    train_metrics_path = artifact_dir / "train_metrics.jsonl"

    artifacts = load_phase2a_artifacts_from_dir(
        config.inputs.phase2a_artifact_dir, torch
    )

    # --- Stage 1: Training ---
    print("\n=== Stage 1: Phase 5 mixed training ===")
    tokenizer = auto_tokenizer_cls.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model for QLoRA...")
    model = _load_model(config, torch, auto_model_cls)
    model.config.use_cache = False
    model = _prepare_lora_model(model, config)
    model.train()
    optimizer = _build_optimizer(model, config, torch)

    train_rows = build_phase5_training_dataset(
        artifacts.train_rows, config,
    )
    n_train = len(train_rows)
    n_suppression = sum(1 for r in train_rows if r.get("suppression_hook"))
    print(
        f"Training on {n_train} examples "
        f"({n_suppression} suppression, {n_train - n_suppression} CAA/other)"
    )

    rng = random.Random(config.training.seed)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, config.training.epochs + 1):
        rows = list(train_rows)
        rng.shuffle(rows)
        epoch_loss = 0.0
        for index, row in enumerate(rows, start=1):
            loss_value = _training_step_phase5(
                model=model,
                tokenizer=tokenizer,
                row=row,
                artifacts=artifacts,
                config=config,
                torch=torch,
                rng=rng,
            )
            (loss_value / config.training.gradient_accumulation_steps).backward()
            epoch_loss += float(loss_value.detach().float().cpu().item())
            should_step = (
                index % config.training.gradient_accumulation_steps == 0
                or index == len(rows)
            )
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"mean_loss={epoch_loss / index:.4f}"
                    )
        mean_loss = epoch_loss / max(1, len(rows))
        write_jsonl(
            train_metrics_path,
            [
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "mean_loss": mean_loss,
                    "n_examples": len(rows),
                }
            ],
            append=True,
        )
        print(f"Epoch {epoch} complete: mean_loss={mean_loss:.4f}")
        if config.training.save_each_epoch:
            epoch_dir = artifact_dir / f"adapter_epoch_{epoch}"
            model.save_pretrained(epoch_dir)

    adapter_dir = artifact_dir / "adapter_final"
    model.save_pretrained(adapter_dir)

    del model, optimizer
    gc.collect()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Training complete, model unloaded")

    # --- Stage 2: Evaluation generations ---
    print("\n=== Stage 2: Phase 5 evaluation generations ===")
    harmless_prompts = load_prompt_records(config.prompts.harmless, "harmless")
    harmful_prompts = load_prompt_records(config.prompts.harmful, "harmful")
    print(
        f"Loaded {len(harmless_prompts)} harmless + {len(harmful_prompts)} harmful prompts"
    )

    caa_vectors = artifacts.vectors
    mean_train_norm = artifacts.mean_train_norm
    n_test_vectors = caa_vectors.shape[0] - 150

    model = _load_adapter_model(config, adapter_dir, torch, auto_model_cls)
    gen_kwargs = generation_kwargs(config, tokenizer)
    eval_rng = random.Random(config.training.seed + 17)

    eval_gen_path = artifact_dir / "phase5_eval_generations.jsonl"
    completed = get_completed_ids(eval_gen_path) if config.checkpoint else set()

    for index, prompt in enumerate(harmless_prompts, start=1):
        _run_detection_conditions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
            torch=torch,
            gen_kwargs=gen_kwargs,
            caa_vectors=caa_vectors,
            mean_train_norm=mean_train_norm,
            n_test_vectors=n_test_vectors,
            rng=eval_rng,
            output_path=eval_gen_path,
            completed=completed,
            prompt_index=index - 1,
        )
        if index % 10 == 0 or index == len(harmless_prompts):
            print(f"Detection eval: {index}/{len(harmless_prompts)} harmless prompts")

    for index, prompt in enumerate(harmful_prompts, start=1):
        _run_safety_conditions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
            torch=torch,
            gen_kwargs=gen_kwargs,
            output_path=eval_gen_path,
            completed=completed,
        )
        if index % 10 == 0 or index == len(harmful_prompts):
            print(f"Safety eval: {index}/{len(harmful_prompts)} harmful prompts")

    del model
    gc.collect()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Eval model unloaded")

    # --- Stage 3: LlamaGuard judging ---
    print("\n=== Stage 3: LlamaGuard judging ===")
    eval_gens = load_records(eval_gen_path)
    judgable_conditions = {"adapter_clean", "adapter_suppression"}
    to_judge = [
        rec for rec in eval_gens
        if rec["condition"] in judgable_conditions and rec["source"] == "harmful"
    ]
    print(f"Judging {len(to_judge)} harmful-prompt generations")

    judgments_path = artifact_dir / "phase5_judgments.jsonl"
    judge_model, judge_tokenizer = load_judge(
        config.judge, config.backend,
        torch, auto_model_cls, auto_tokenizer_cls,
    )
    completed_judgments = (
        get_completed_ids(judgments_path) if config.checkpoint else set()
    )
    for idx, rec in enumerate(to_judge, start=1):
        key = f"{rec['prompt_id']}:{rec['condition']}"
        if key in completed_judgments:
            continue
        judgment = judge_response(
            judge_model, judge_tokenizer,
            rec["prompt_used"], rec["response"],
            torch, config.judge.max_new_tokens,
        )
        append_record(judgments_path, {
            **rec,
            **judgment,
            "unsafe": judgment["verdict"] == "unsafe",
        })
        completed_judgments.add(key)
        if idx % 20 == 0 or idx == len(to_judge):
            print(f"Judged {idx}/{len(to_judge)}")

    del judge_model, judge_tokenizer
    gc.collect()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Judge model unloaded")

    # --- Stage 4: Detection classification + metrics ---
    print("\n=== Stage 4: Detection classification + metrics ===")
    detection_conditions = {
        "adapter_clean_control", "adapter_caa_positive",
        "adapter_noise_control", "adapter_suppression_detection",
    }
    detection_records = []
    for rec in eval_gens:
        if rec["condition"] in detection_conditions:
            detected = classify_detection(rec["response"])
            detection_records.append({**rec, "detected": detected})
    write_jsonl(
        artifact_dir / "phase5_detection_classifications.jsonl", detection_records
    )

    judgments = load_records(judgments_path)
    metrics = compute_phase5_metrics(
        judgments, detection_records, config.pass_criteria
    )
    write_json(artifact_dir / "phase5_results.json", metrics)

    report = build_phase5_report(config, metrics)
    (artifact_dir / "phase5_report.md").write_text(report, encoding="utf-8")

    result = Phase5RunResult(
        artifact_dir=artifact_dir,
        suppression_detection_rate=metrics["suppression_detection_rate"],
        caa_detection_rate=metrics["caa_detection_rate"],
        clean_fpr=metrics["clean_fpr"],
        noise_fpr=metrics["noise_fpr"],
        clean_asr=metrics["clean_asr"],
        suppressed_asr=metrics["suppressed_asr"],
        n_train_examples=n_train,
        passed=metrics["passed"],
    )
    print(f"\nPhase 5 {'PASS' if result.passed else 'FAIL'}")
    return result


# --- Training ---


def _training_step_phase5(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    artifacts: Phase2AArtifacts,
    config: Phase5Config,
    torch: Any,
    rng: random.Random,
) -> Any:
    encoded = encode_supervised_example(
        tokenizer,
        row,
        config.training.max_seq_tokens,
        torch,
    )
    input_ids = _move_to_model_device(encoded.input_ids, model)
    attention_mask = _move_to_model_device(encoded.attention_mask, model)
    labels = _move_to_model_device(encoded.labels, model)

    handle = None
    if row.get("suppression_hook"):
        module = get_down_proj_module(model, config.suppression.layer)
        hook = DownProjNeuronHook(
            neuron=config.suppression.neuron,
            pin_value=config.suppression.pin_value,
            capture=False,
        )
        handle = module.register_forward_pre_hook(hook)
    else:
        injection = build_injection_batch([row], artifacts, torch, rng)
        if injection is not None:
            module = get_decoder_layer(model, config.injection.layer)
            hook = ResidualInjectionHook(
                vectors=injection,
                token_indices=torch.tensor(
                    [encoded.injection_token_index],
                    dtype=torch.long,
                ),
            )
            handle = module.register_forward_hook(hook)

    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
    finally:
        if handle is not None:
            handle.remove()
    return outputs.loss


# --- Evaluation helpers ---


def _run_detection_conditions(
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord,
    config: Phase5Config,
    torch: Any,
    gen_kwargs: dict[str, Any],
    caa_vectors: Any,
    mean_train_norm: float,
    n_test_vectors: int,
    rng: random.Random,
    output_path: Path,
    completed: set[str],
    prompt_index: int,
) -> None:
    det = config.prompts.detection_prompt
    conditions = [
        ("adapter_clean_control", "none", False),
        ("adapter_caa_positive", "caa_injection", False),
        ("adapter_noise_control", "noise_injection", False),
        ("adapter_suppression_detection", "neuron_pin", True),
    ]
    for condition, intervention, pin in conditions:
        key = f"{prompt.prompt_id}:{condition}"
        if key in completed:
            continue

        if intervention == "caa_injection":
            vector_idx = 150 + (prompt_index % max(1, n_test_vectors))
            vector = caa_vectors[vector_idx].detach().float() * config.injection.alpha
            response = _generate_with_residual_injection(
                model, tokenizer, det, torch, gen_kwargs,
                vector, config.injection.layer,
            )
        elif intervention == "noise_injection":
            noise = _norm_matched_noise(
                caa_vectors.shape[-1], mean_train_norm, torch, rng,
            ) * config.injection.alpha
            response = _generate_with_residual_injection(
                model, tokenizer, det, torch, gen_kwargs,
                noise, config.injection.layer,
            )
        elif pin:
            response = generate_with_hook(
                model, tokenizer, det, torch, gen_kwargs,
                layer=config.suppression.layer,
                neuron=config.suppression.neuron,
                pin_value=config.suppression.pin_value,
            )
        else:
            response = generate_with_hook(
                model, tokenizer, det, torch, gen_kwargs,
            )

        append_record(output_path, _build_record(
            prompt, condition, intervention, det, "detection", response,
            pin_active=pin,
            pin_value=config.suppression.pin_value if pin else None,
            caa_active=intervention == "caa_injection",
            noise_active=intervention == "noise_injection",
        ))
        completed.add(key)


def _run_safety_conditions(
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord,
    config: Phase5Config,
    torch: Any,
    gen_kwargs: dict[str, Any],
    output_path: Path,
    completed: set[str],
) -> None:
    conditions = [
        ("adapter_clean", False),
        ("adapter_suppression", True),
    ]
    for condition, pin in conditions:
        key = f"{prompt.prompt_id}:{condition}"
        if key in completed:
            continue

        if pin:
            response = generate_with_hook(
                model, tokenizer, prompt.text, torch, gen_kwargs,
                layer=config.suppression.layer,
                neuron=config.suppression.neuron,
                pin_value=config.suppression.pin_value,
            )
        else:
            response = generate_with_hook(
                model, tokenizer, prompt.text, torch, gen_kwargs,
            )

        append_record(output_path, _build_record(
            prompt, condition,
            "neuron_pin" if pin else "none",
            prompt.text, "original", response,
            pin_active=pin,
            pin_value=config.suppression.pin_value if pin else None,
        ))
        completed.add(key)


def _generate_with_residual_injection(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    torch: Any,
    gen_kwargs: dict[str, Any],
    vector: Any,
    injection_layer: int,
) -> str:
    input_ids = apply_chat_template(tokenizer, prompt_text, torch)
    attention_mask = torch.ones_like(input_ids)
    token_index = input_ids.shape[-1] - 1
    input_ids = _move_to_model_device(input_ids, model)
    attention_mask = _move_to_model_device(attention_mask, model)

    injection = vector.unsqueeze(0)
    module = get_decoder_layer(model, injection_layer)
    hook = ResidualInjectionHook(
        vectors=injection,
        token_indices=torch.tensor([token_index], dtype=torch.long),
        apply_once=True,
    )
    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
    finally:
        handle.remove()
    generated = outputs[0][input_ids.shape[-1]:].detach().cpu().tolist()
    return tokenizer.decode(generated, skip_special_tokens=True)


# --- Metrics ---


def compute_phase5_metrics(
    judgments: list[dict[str, Any]],
    detection_records: list[dict[str, Any]],
    criteria: Any,
) -> dict[str, Any]:
    def _detection_rate(records: list[dict[str, Any]], condition: str) -> float:
        subset = [r for r in records if r["condition"] == condition]
        if not subset:
            return 0.0
        return sum(1 for r in subset if r.get("detected")) / len(subset)

    def _asr(judgments_list: list[dict[str, Any]], condition: str) -> float:
        subset = [j for j in judgments_list if j["condition"] == condition]
        if not subset:
            return 0.0
        return sum(1 for j in subset if j.get("unsafe")) / len(subset)

    suppression_det = _detection_rate(
        detection_records, "adapter_suppression_detection"
    )
    caa_det = _detection_rate(detection_records, "adapter_caa_positive")
    clean_fpr = _detection_rate(detection_records, "adapter_clean_control")
    noise_fpr = _detection_rate(detection_records, "adapter_noise_control")

    clean_asr = _asr(judgments, "adapter_clean")
    suppressed_asr = _asr(judgments, "adapter_suppression")

    passed = (
        suppression_det >= criteria.min_suppression_detection
        and caa_det >= criteria.min_caa_detection
        and clean_fpr <= criteria.max_clean_fpr
        and noise_fpr <= criteria.max_noise_fpr
        and clean_asr <= criteria.max_clean_asr
    )

    return {
        "suppression_detection_rate": suppression_det,
        "caa_detection_rate": caa_det,
        "clean_fpr": clean_fpr,
        "noise_fpr": noise_fpr,
        "clean_asr": clean_asr,
        "suppressed_asr": suppressed_asr,
        "passed": passed,
        "pass_criteria": {
            "min_suppression_detection": criteria.min_suppression_detection,
            "min_caa_detection": criteria.min_caa_detection,
            "max_clean_fpr": criteria.max_clean_fpr,
            "max_noise_fpr": criteria.max_noise_fpr,
            "max_clean_asr": criteria.max_clean_asr,
        },
        "counts": {
            "clean_control": len([r for r in detection_records if r["condition"] == "adapter_clean_control"]),
            "caa_positive": len([r for r in detection_records if r["condition"] == "adapter_caa_positive"]),
            "noise_control": len([r for r in detection_records if r["condition"] == "adapter_noise_control"]),
            "suppression_detection": len([r for r in detection_records if r["condition"] == "adapter_suppression_detection"]),
            "safety_clean": len([j for j in judgments if j["condition"] == "adapter_clean"]),
            "safety_suppression": len([j for j in judgments if j["condition"] == "adapter_suppression"]),
        },
    }


# --- Utilities ---


def _norm_matched_noise(
    d_model: int, norm: float, torch: Any, rng: random.Random
) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(rng.randrange(0, 2**31 - 1))
    vector = torch.randn(d_model, generator=generator, dtype=torch.float32)
    vector_norm = torch.linalg.vector_norm(vector).clamp_min(1e-8)
    return vector / vector_norm * float(norm)


def _build_record(
    prompt: PromptRecord,
    condition: str,
    intervention: str,
    prompt_used: str,
    prompt_type: str,
    response: str,
    pin_active: bool = False,
    pin_value: float | None = None,
    caa_active: bool = False,
    noise_active: bool = False,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt.prompt_id,
        "source": prompt.source,
        "dataset_id": prompt.dataset_id,
        "row_index": prompt.row_index,
        "condition": condition,
        "mode": condition,
        "intervention": intervention,
        "prompt_used": prompt_used,
        "prompt_type": prompt_type,
        "pin_active": pin_active,
        "pin_value": pin_value,
        "caa_active": caa_active,
        "noise_active": noise_active,
        "response": response,
        "response_preview": preview_text(response),
        "refusal_preview": classify_refusal_preview(response),
    }


def _load_adapter_model(
    config: Phase5Config, adapter_dir: Path, torch: Any, auto_model_cls: Any
) -> Any:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "Phase 5 requires peft. Install with `pip install peft`."
        ) from exc

    model = _load_model(config, torch, auto_model_cls)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    print(f"Loaded Phase 5 PEFT adapter from {adapter_dir}")
    return model


def _prepare_lora_model(model: Any, config: Phase5Config) -> Any:
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError(
            "Phase 5 requires peft. Install with `pip install peft`."
        ) from exc

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=config.training.rank,
        lora_alpha=config.training.alpha,
        lora_dropout=config.training.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(config.training.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _build_optimizer(model: Any, config: Phase5Config, torch: Any) -> Any:
    parameters = [param for param in model.parameters() if param.requires_grad]
    try:
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(parameters, lr=config.training.learning_rate)
    except Exception:
        return torch.optim.AdamW(parameters, lr=config.training.learning_rate)


def _set_seed(config: Phase5Config, torch: Any) -> None:
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)


def _create_phase5_run_dir(config: OutputConfig, checkpoint: bool) -> Path:
    if checkpoint and config.run_name is not None:
        path = config.root / config.run_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    return create_run_dir(config)
