from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifacts import (
    build_phase2b_report,
    create_run_dir,
    write_json,
    write_jsonl,
)
from ..config import Phase2BConfig
from ..hooks import ResidualInjectionHook, get_decoder_layer
from ..phase2b import (
    Phase2AArtifacts,
    build_eval_injection_batch,
    build_injection_batch,
    classify_detection,
    classify_identification,
    compute_phase2b_metrics,
    encode_supervised_example,
    load_phase2a_artifacts,
)
from .transformers_backend import (
    _load_model,
    _move_to_model_device,
    _runtime_imports,
    apply_chat_template,
)


@dataclass(frozen=True)
class Phase2BRunResult:
    artifact_dir: Path
    detection_rate: float
    identification_rate: float
    clean_fpr: float
    noise_fpr: float
    passed: bool
    n_train_examples: int
    n_eval_examples: int


def run_phase2b(config: Phase2BConfig) -> Phase2BRunResult:
    torch, auto_model_cls, auto_tokenizer_cls = _runtime_imports()
    _set_seed(config, torch)
    artifacts = load_phase2a_artifacts(config, torch)
    artifact_dir = create_run_dir(config.outputs)
    train_metrics_path = artifact_dir / "train_metrics.jsonl"

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

    print(f"Training Phase 2B adapter on {len(artifacts.train_rows)} examples...")
    rng = random.Random(config.training.seed)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, config.training.epochs + 1):
        rows = list(artifacts.train_rows)
        rng.shuffle(rows)
        epoch_loss = 0.0
        for index, row in enumerate(rows, start=1):
            loss_value = _training_step(
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

    print("Evaluating held-out Phase 2B detection...")
    eval_rows = _select_eval_rows(artifacts.eval_rows, config)
    eval_generations = evaluate_phase2b(
        model=model,
        tokenizer=tokenizer,
        rows=eval_rows,
        artifacts=artifacts,
        config=config,
        torch=torch,
    )
    metrics = compute_phase2b_metrics(eval_generations, config.pass_criteria)
    write_jsonl(artifact_dir / "eval_generations.jsonl", eval_generations)

    summary = {
        "model": config.model.id,
        "revision": config.model.revision,
        "phase2a_artifact_dir": str(artifacts.root),
        "adapter_dir": str(adapter_dir),
        "n_train_examples": len(artifacts.train_rows),
        "n_eval_examples": len(eval_generations),
        "detection_rate": metrics.detection_rate,
        "identification_rate": metrics.identification_rate,
        "clean_fpr": metrics.clean_fpr,
        "noise_fpr": metrics.noise_fpr,
        "passed": metrics.passed,
        "counts": metrics.counts,
        "pass_criteria": {
            "min_detection_rate": config.pass_criteria.min_detection_rate,
            "max_clean_fpr": config.pass_criteria.max_clean_fpr,
            "max_noise_fpr": config.pass_criteria.max_noise_fpr,
        },
        "lora": {
            "rank": config.training.rank,
            "alpha": config.training.alpha,
            "target_modules": list(config.training.target_modules),
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
        },
        "injection": {
            "layer": config.injection.layer,
            "eval_alpha": config.injection.eval_alpha,
        },
    }
    write_json(artifact_dir / "phase2b_summary.json", summary)
    report = build_phase2b_report(config, summary)
    (artifact_dir / "phase2b_report.md").write_text(report, encoding="utf-8")

    del model, tokenizer
    gc.collect()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Phase2BRunResult(
        artifact_dir=artifact_dir,
        detection_rate=metrics.detection_rate,
        identification_rate=metrics.identification_rate,
        clean_fpr=metrics.clean_fpr,
        noise_fpr=metrics.noise_fpr,
        passed=metrics.passed,
        n_train_examples=len(artifacts.train_rows),
        n_eval_examples=len(eval_generations),
    )


def evaluate_phase2b(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    artifacts: Phase2AArtifacts,
    config: Phase2BConfig,
    torch: Any,
) -> list[dict[str, Any]]:
    model.eval()
    rng = random.Random(config.training.seed + 17)
    generations = []
    for index, row in enumerate(rows, start=1):
        response = generate_eval_response(
            model=model,
            tokenizer=tokenizer,
            row=row,
            artifacts=artifacts,
            config=config,
            torch=torch,
            rng=rng,
        )
        detected = classify_detection(response)
        identified = classify_identification(response, row.get("concept"))
        generations.append(
            {
                **row,
                "response": response,
                "detected": detected,
                "identified": identified,
                "eval_alpha": config.injection.eval_alpha,
            }
        )
        if index % 50 == 0 or index == len(rows):
            print(f"Evaluated {index}/{len(rows)} examples")
    model.train()
    return generations


def generate_eval_response(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    artifacts: Phase2AArtifacts,
    config: Phase2BConfig,
    torch: Any,
    rng: random.Random,
) -> str:
    input_ids = apply_chat_template(tokenizer, str(row["user_prompt"]), torch)
    attention_mask = torch.ones_like(input_ids)
    token_index = input_ids.shape[-1] - 1
    input_ids = _move_to_model_device(input_ids, model)
    attention_mask = _move_to_model_device(attention_mask, model)
    injection = build_eval_injection_batch(
        [row],
        artifacts,
        config.injection.eval_alpha,
        torch,
        rng,
    )

    handle = None
    hook = None
    if injection is not None:
        module = get_decoder_layer(model, config.injection.layer)
        hook = ResidualInjectionHook(
            vectors=injection,
            token_indices=torch.tensor([token_index], dtype=torch.long),
            apply_once=True,
        )
        handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": config.evaluation.max_new_tokens,
                "do_sample": config.evaluation.do_sample,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if config.evaluation.do_sample:
                gen_kwargs["temperature"] = config.evaluation.temperature
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
    finally:
        if handle is not None:
            handle.remove()
    if hook is not None and hook.injected_calls == 0:
        raise RuntimeError("Phase 2B eval injection hook was registered but never fired.")
    generated = outputs[0][input_ids.shape[-1] :].detach().cpu().tolist()
    return tokenizer.decode(generated, skip_special_tokens=True)


def _training_step(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    artifacts: Phase2AArtifacts,
    config: Phase2BConfig,
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
    injection = build_injection_batch([row], artifacts, torch, rng)

    handle = None
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


def _prepare_lora_model(model: Any, config: Phase2BConfig) -> Any:
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError(
            "Phase 2B requires peft. Install with `pip install peft`."
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


def _build_optimizer(model: Any, config: Phase2BConfig, torch: Any) -> Any:
    parameters = [param for param in model.parameters() if param.requires_grad]
    try:
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(parameters, lr=config.training.learning_rate)
    except Exception:
        return torch.optim.AdamW(parameters, lr=config.training.learning_rate)


def _select_eval_rows(
    rows: list[dict[str, Any]],
    config: Phase2BConfig,
) -> list[dict[str, Any]]:
    if config.evaluation.limit is None:
        return rows
    return rows[: config.evaluation.limit]


def _set_seed(config: Phase2BConfig, torch: Any) -> None:
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
