from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .concepts import BASELINE_CONCEPT, PROMPT_TEMPLATES, ConceptSplit
from .config import Phase2AConfig
from .hooks import ResidualStreamHook, get_decoder_layer


@dataclass(frozen=True)
class CAAExtractionResult:
    tensor: Any  # torch.Tensor [n_concepts, d_model]
    concept_order: list[str]
    metadata: list[dict[str, Any]]
    d_model: int
    mean_train_norm: float


def extract_caa_vectors(
    model: Any,
    tokenizer: Any,
    config: Phase2AConfig,
    torch: Any,
    concepts: ConceptSplit,
) -> CAAExtractionResult:
    from .backends.transformers_backend import apply_chat_template, _move_to_model_device

    layer = config.caa.layer
    templates = PROMPT_TEMPLATES[: config.caa.n_prompt_templates]
    baseline_word = config.caa.baseline_concept or BASELINE_CONCEPT

    baseline_states = []
    for tmpl in templates:
        prompt = tmpl.format(concept=baseline_word)
        state = _collect_hidden_state(model, tokenizer, prompt, layer, torch,
                                      apply_chat_template, _move_to_model_device)
        baseline_states.append(state)
    baseline_mean = torch.stack(baseline_states).mean(dim=0)

    all_concepts = list(concepts.train) + list(concepts.test)
    train_set = set(concepts.train)
    category_lookup: dict[str, str] = {}
    for cat, words in concepts.categories.items():
        for w in words:
            category_lookup[w] = cat

    vectors = []
    metadata = []
    for i, concept in enumerate(all_concepts):
        concept_states = []
        for tmpl in templates:
            prompt = tmpl.format(concept=concept)
            state = _collect_hidden_state(model, tokenizer, prompt, layer, torch,
                                          apply_chat_template, _move_to_model_device)
            concept_states.append(state)
        concept_mean = torch.stack(concept_states).mean(dim=0)
        caa_vec = concept_mean - baseline_mean
        vectors.append(caa_vec)
        metadata.append({
            "index": i,
            "concept": concept,
            "category": category_lookup.get(concept, "unknown"),
            "split": "train" if concept in train_set else "test",
            "vector_norm": float(caa_vec.norm().item()),
            "concept_mean_norm": float(concept_mean.norm().item()),
        })
        if (i + 1) % 10 == 0 or i == len(all_concepts) - 1:
            print(f"  CAA extraction: {i + 1}/{len(all_concepts)} concepts")

    stacked = torch.stack(vectors)
    d_model = stacked.shape[1]
    train_norms = [m["vector_norm"] for m in metadata if m["split"] == "train"]
    mean_train_norm = sum(train_norms) / len(train_norms) if train_norms else 0.0

    return CAAExtractionResult(
        tensor=stacked,
        concept_order=all_concepts,
        metadata=metadata,
        d_model=d_model,
        mean_train_norm=mean_train_norm,
    )


def _collect_hidden_state(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    layer: int,
    torch: Any,
    apply_chat_template_fn: Any,
    move_to_device_fn: Any,
) -> Any:
    input_ids = apply_chat_template_fn(tokenizer, prompt_text, torch)
    input_ids = move_to_device_fn(input_ids, model)
    attention_mask = torch.ones_like(input_ids)

    decoder_layer = get_decoder_layer(model, layer)
    hook = ResidualStreamHook(capture_last_token=True)
    handle = decoder_layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()

    if not hook.captures:
        from .errors import HookFailure
        raise HookFailure(
            f"Residual stream hook on layer {layer} did not fire during forward pass."
        )
    return hook.captures[0].squeeze(0)  # [d_model]
