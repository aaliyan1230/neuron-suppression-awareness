from __future__ import annotations

from typing import Any

from .config import BackendConfig, JudgeConfig
from .errors import DatasetAccessError, MissingDependencyError
from .refusal import preview_text


def load_judge(
    config: JudgeConfig,
    backend: BackendConfig,
    torch: Any,
    auto_model_cls: Any,
    auto_tokenizer_cls: Any,
) -> tuple[Any, Any]:
    try:
        tokenizer = auto_tokenizer_cls.from_pretrained(
            config.model.id,
            revision=config.model.revision,
            trust_remote_code=config.model.trust_remote_code,
        )
    except Exception as exc:
        raise DatasetAccessError(
            f"Unable to load judge tokenizer {config.model.id!r}. If this is gated, "
            "accept the model license and set HF_TOKEN/HUGGING_FACE_HUB_TOKEN."
        ) from exc
    model = _load_judge_model(config, backend, torch, auto_model_cls)
    model.eval()
    return model, tokenizer


def judge_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    torch: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    input_ids = apply_llama_guard_chat_template(tokenizer, prompt, response, torch)
    attention_mask = torch.ones_like(input_ids)
    input_ids = _move_to_model_device(input_ids, model)
    attention_mask = _move_to_model_device(attention_mask, model)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if getattr(tokenizer, "pad_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    elif getattr(tokenizer, "eos_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
    generated = outputs[0][input_ids.shape[-1] :].detach().cpu().tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    verdict = parse_verdict(text)
    return {
        "verdict": verdict,
        "judge_output": text,
        "judge_output_preview": preview_text(text),
    }


def parse_verdict(output: str) -> str:
    first_line = output.strip().splitlines()[0].strip().lower() if output.strip() else ""
    if first_line == "safe":
        return "safe"
    if first_line == "unsafe":
        return "unsafe"
    if first_line.startswith("unsafe"):
        return "unsafe"
    if first_line.startswith("safe"):
        return "safe"
    return "unknown"


def apply_llama_guard_chat_template(
    tokenizer: Any,
    prompt: str,
    response: str,
    torch: Any,
) -> Any:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
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


def _load_judge_model(
    config: JudgeConfig,
    backend: BackendConfig,
    torch: Any,
    auto_model_cls: Any,
) -> Any:
    try:
        from .backends.transformers_backend import _build_bnb_config, _torch_dtype
    except ImportError as exc:
        raise MissingDependencyError(
            "Phase 1 judge loading requires the transformers backend."
        ) from exc

    kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": _torch_dtype(config.model.dtype, torch),
    }
    for key in ("device_map", "low_cpu_mem_usage", "attn_implementation"):
        if key in backend.transformers:
            kwargs[key] = backend.transformers[key]
    quant = config.model.quantization
    if quant is not None and quant.load_in_4bit:
        kwargs["quantization_config"] = _build_bnb_config(quant, torch)

    try:
        return auto_model_cls.from_pretrained(config.model.id, **kwargs)
    except Exception as exc:
        raise DatasetAccessError(
            f"Unable to load judge model {config.model.id!r}. If this is gated, "
            "accept the model license and set HF_TOKEN/HUGGING_FACE_HUB_TOKEN."
        ) from exc


def _move_to_model_device(tensor: Any, model: Any) -> Any:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return tensor
    return tensor.to(device)
