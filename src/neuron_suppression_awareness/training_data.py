from __future__ import annotations

import random
from typing import Any

from .caa_extraction import CAAExtractionResult
from .concepts import ConceptSplit
from .config import Phase2AConfig


_POSITIVE_TEMPLATE = "Yes, I detect an injected concept. It seems related to {concept}."
_NEGATIVE_RESPONSE = "No, I do not detect any injected concept."


def build_training_dataset(
    config: Phase2AConfig,
    caa_result: CAAExtractionResult,
    concepts: ConceptSplit,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    return _build_dataset(
        config=config,
        caa_result=caa_result,
        concept_words=concepts.train,
        split="train",
        rng=rng,
    )


def build_eval_dataset(
    config: Phase2AConfig,
    caa_result: CAAExtractionResult,
    concepts: ConceptSplit,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    n_test = len(concepts.test)
    n_train = len(concepts.train)
    scale = n_test / n_train if n_train else 1.0
    target = max(100, round(config.training_data.target_total * scale))
    return _build_dataset(
        config=config,
        caa_result=caa_result,
        concept_words=concepts.test,
        split="eval",
        rng=rng,
        target_override=target,
    )


def _build_dataset(
    config: Phase2AConfig,
    caa_result: CAAExtractionResult,
    concept_words: tuple[str, ...],
    split: str,
    rng: random.Random | None = None,
    target_override: int | None = None,
) -> list[dict[str, Any]]:
    if rng is None:
        rng = random.Random(config.training_data.seed)

    td = config.training_data
    target = target_override or td.target_total
    n_steered = round(target * td.steered_correct_fraction)
    n_clean = round(target * td.clean_fraction)
    n_noise = round(target * td.noise_fraction)
    n_mismatch = round(target * td.mismatch_fraction)
    n_alpaca = target - n_steered - n_clean - n_noise - n_mismatch

    concept_to_index = {c: i for i, c in enumerate(caa_result.concept_order)}
    concept_list = list(concept_words)
    detection_prompt = td.detection_prompt
    alphas = list(td.alpha_values)

    rows: list[dict[str, Any]] = []

    for _ in range(n_steered):
        concept = rng.choice(concept_list)
        alpha = rng.choice(alphas)
        rows.append({
            "condition": "steered_correct",
            "user_prompt": detection_prompt,
            "target_response": _POSITIVE_TEMPLATE.format(concept=concept),
            "concept": concept,
            "vector_index": concept_to_index[concept],
            "alpha": alpha,
            "inject_noise": False,
            "mismatch_hint": None,
        })

    for _ in range(n_clean):
        rows.append({
            "condition": "clean",
            "user_prompt": detection_prompt,
            "target_response": _NEGATIVE_RESPONSE,
            "concept": None,
            "vector_index": None,
            "alpha": None,
            "inject_noise": False,
            "mismatch_hint": None,
        })

    for _ in range(n_noise):
        alpha = rng.choice(alphas)
        rows.append({
            "condition": "noise",
            "user_prompt": detection_prompt,
            "target_response": _NEGATIVE_RESPONSE,
            "concept": None,
            "vector_index": None,
            "alpha": alpha,
            "inject_noise": True,
            "mismatch_hint": None,
            "noise_norm_reference": caa_result.mean_train_norm,
        })

    for _ in range(n_mismatch):
        concept_a, concept_b = rng.sample(concept_list, 2)
        alpha = rng.choice(alphas)
        rows.append({
            "condition": "mismatch",
            "user_prompt": detection_prompt,
            "target_response": _POSITIVE_TEMPLATE.format(concept=concept_a),
            "concept": concept_a,
            "vector_index": concept_to_index[concept_a],
            "alpha": alpha,
            "inject_noise": False,
            "mismatch_hint": concept_b,
        })

    alpaca_rows = _load_alpaca_examples(config, n_alpaca, rng)
    rows.extend(alpaca_rows)

    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["example_id"] = f"{split}-{i:05d}"
    return rows


def _load_alpaca_examples(
    config: Phase2AConfig,
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return _alpaca_fallback(count)

    td = config.training_data
    try:
        ds = load_dataset(td.alpaca_dataset_id, split="train")
    except Exception:
        return _alpaca_fallback(count)

    candidates: list[dict[str, Any]] = []
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        if not instruction or not output:
            continue
        prompt = f"{instruction}\n{inp}".strip() if inp else instruction
        approx_tokens = len(prompt.split()) + len(output.split())
        if approx_tokens > td.max_seq_tokens:
            continue
        candidates.append({"user_prompt": prompt, "target_response": output})
        if len(candidates) >= td.alpaca_limit:
            break

    rng.shuffle(candidates)
    selected = candidates[:count]
    return [
        {
            "condition": "alpaca_replay",
            "user_prompt": ex["user_prompt"],
            "target_response": ex["target_response"],
            "concept": None,
            "vector_index": None,
            "alpha": None,
            "inject_noise": False,
            "mismatch_hint": None,
        }
        for ex in selected
    ]


def _alpaca_fallback(count: int) -> list[dict[str, Any]]:
    return [
        {
            "condition": "alpaca_replay",
            "user_prompt": f"What is the meaning of the word 'example{i}'?",
            "target_response": f"The word 'example{i}' is a placeholder for testing.",
            "concept": None,
            "vector_index": None,
            "alpha": None,
            "inject_noise": False,
            "mismatch_hint": None,
        }
        for i in range(count)
    ]
