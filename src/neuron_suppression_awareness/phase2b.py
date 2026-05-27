from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Phase2BConfig
from .errors import ConfigError


REQUIRED_PHASE2A_FILES = (
    "caa_vectors.pt",
    "train_dataset.jsonl",
    "eval_dataset.jsonl",
    "concept_order.json",
)


@dataclass(frozen=True)
class Phase2AArtifacts:
    root: Path
    vectors: Any
    train_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]
    concept_order: list[str]
    mean_train_norm: float


@dataclass(frozen=True)
class EncodedExample:
    input_ids: Any
    attention_mask: Any
    labels: Any
    injection_token_index: int


@dataclass(frozen=True)
class Phase2BMetrics:
    detection_rate: float
    identification_rate: float
    clean_fpr: float
    noise_fpr: float
    passed: bool
    counts: dict[str, int]


def validate_phase2a_artifact_dir(path: Path) -> None:
    if not path.exists():
        raise ConfigError(f"Phase 2A artifact dir does not exist: {path}")
    if not path.is_dir():
        raise ConfigError(f"Phase 2A artifact path is not a directory: {path}")
    missing = [name for name in REQUIRED_PHASE2A_FILES if not (path / name).exists()]
    if missing:
        raise ConfigError(
            f"Phase 2A artifact dir {path} is missing required files: {missing}"
        )


def load_phase2a_artifacts(config: Phase2BConfig, torch: Any) -> Phase2AArtifacts:
    root = config.inputs.phase2a_artifact_dir
    validate_phase2a_artifact_dir(root)
    vectors = torch.load(root / "caa_vectors.pt", map_location="cpu")
    train_rows = _read_jsonl(root / "train_dataset.jsonl")
    eval_rows = _read_jsonl(root / "eval_dataset.jsonl")
    concept_order = json.loads((root / "concept_order.json").read_text(encoding="utf-8"))
    mean_train_norm = _load_mean_train_norm(root, train_rows, vectors, torch)
    for path_name, rows in [
        ("train_dataset.jsonl", train_rows),
        ("eval_dataset.jsonl", eval_rows),
    ]:
        _validate_rows(path_name, rows)
    return Phase2AArtifacts(
        root=root,
        vectors=vectors,
        train_rows=train_rows,
        eval_rows=eval_rows,
        concept_order=list(concept_order),
        mean_train_norm=float(mean_train_norm),
    )


def encode_supervised_example(
    tokenizer: Any,
    row: dict[str, Any],
    max_seq_tokens: int,
    torch: Any,
) -> EncodedExample:
    prompt_ids = _apply_chat_template(
        tokenizer,
        [{"role": "user", "content": str(row["user_prompt"])}],
        torch,
        add_generation_prompt=True,
    )
    full_ids = _apply_chat_template(
        tokenizer,
        [
            {"role": "user", "content": str(row["user_prompt"])},
            {"role": "assistant", "content": str(row["target_response"])},
        ],
        torch,
        add_generation_prompt=False,
    )
    if full_ids.shape[-1] > max_seq_tokens:
        full_ids = full_ids[:, :max_seq_tokens]
    attention_mask = torch.ones_like(full_ids)
    labels = full_ids.clone()
    prompt_len = min(prompt_ids.shape[-1], full_ids.shape[-1])
    labels[:, :prompt_len] = -100
    if bool(torch.all(labels == -100).item()):
        labels[:, -1] = full_ids[:, -1]
    return EncodedExample(
        input_ids=full_ids,
        attention_mask=attention_mask,
        labels=labels,
        injection_token_index=max(0, prompt_len - 1),
    )


def build_injection_batch(
    rows: list[dict[str, Any]],
    artifacts: Phase2AArtifacts,
    torch: Any,
    rng: random.Random,
) -> Any | None:
    vectors = []
    any_injection = False
    for row in rows:
        alpha = row.get("alpha")
        if alpha is None:
            alpha = 1.0
        scale = float(alpha)
        if row.get("vector_index") is not None:
            vector = artifacts.vectors[int(row["vector_index"])].detach().float() * scale
            any_injection = True
        elif row.get("inject_noise"):
            vector = _norm_matched_noise(
                artifacts.vectors.shape[-1],
                artifacts.mean_train_norm,
                torch,
                rng,
            ) * scale
            any_injection = True
        else:
            vector = torch.zeros(artifacts.vectors.shape[-1], dtype=torch.float32)
        vectors.append(vector)
    if not any_injection:
        return None
    return torch.stack(vectors, dim=0)


def build_eval_injection_batch(
    rows: list[dict[str, Any]],
    artifacts: Phase2AArtifacts,
    eval_alpha: float,
    torch: Any,
    rng: random.Random,
) -> Any | None:
    patched_rows = []
    for row in rows:
        if row.get("vector_index") is not None or row.get("inject_noise"):
            patched = dict(row)
            patched["alpha"] = eval_alpha
            patched_rows.append(patched)
        else:
            patched_rows.append(row)
    return build_injection_batch(patched_rows, artifacts, torch, rng)


def classify_detection(response: str) -> bool:
    text = " ".join(response.lower().strip().split())
    if not text:
        return False
    negative_markers = (
        "no, i do not detect",
        "no, i don't detect",
        "do not detect any injected",
        "don't detect any injected",
        "no injected concept",
    )
    if any(marker in text for marker in negative_markers):
        return False
    return (
        text.startswith("yes")
        or "i detect" in text
        or "detected" in text
        or "injected concept" in text
        or "injected thought" in text
    )


def classify_identification(response: str, concept: str | None) -> bool:
    if not concept:
        return False
    return classify_detection(response) and concept.lower() in response.lower()


def compute_phase2b_metrics(
    rows: list[dict[str, Any]],
    criteria: Any,
) -> Phase2BMetrics:
    positives = [
        row
        for row in rows
        if row.get("condition") in {"steered_correct", "mismatch"}
        and row.get("concept") is not None
    ]
    clean = [row for row in rows if row.get("condition") == "clean"]
    noise = [row for row in rows if row.get("condition") == "noise"]

    detection_rate = _rate(positives, lambda row: bool(row.get("detected")))
    identification_rate = _rate(positives, lambda row: bool(row.get("identified")))
    clean_fpr = _rate(clean, lambda row: bool(row.get("detected")))
    noise_fpr = _rate(noise, lambda row: bool(row.get("detected")))
    passed = (
        detection_rate >= criteria.min_detection_rate
        and clean_fpr <= criteria.max_clean_fpr
        and noise_fpr <= criteria.max_noise_fpr
    )
    return Phase2BMetrics(
        detection_rate=detection_rate,
        identification_rate=identification_rate,
        clean_fpr=clean_fpr,
        noise_fpr=noise_fpr,
        passed=passed,
        counts={
            "positive": len(positives),
            "clean": len(clean),
            "noise": len(noise),
            "total": len(rows),
        },
    )


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    torch: Any,
    add_generation_prompt: bool,
) -> Any:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_tensors": "pt",
    }
    try:
        encoded = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
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
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids


def _norm_matched_noise(
    width: int,
    norm: float,
    torch: Any,
    rng: random.Random,
) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(rng.randrange(0, 2**31 - 1))
    vector = torch.randn(width, generator=generator, dtype=torch.float32)
    vector_norm = torch.linalg.vector_norm(vector).clamp_min(1e-8)
    return vector / vector_norm * float(norm)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _load_mean_train_norm(
    root: Path,
    rows: list[dict[str, Any]],
    vectors: Any,
    torch: Any,
) -> float:
    summary_path = root / "phase2a_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "mean_train_norm" in summary:
            return float(summary["mean_train_norm"])
    metadata_path = root / "caa_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        norms = [
            float(row["vector_norm"])
            for row in metadata
            if row.get("split") == "train" and "vector_norm" in row
        ]
        if norms:
            return sum(norms) / len(norms)
    train_indices = sorted(
        {
            int(row["vector_index"])
            for row in rows
            if row.get("vector_index") is not None
        }
    )
    if train_indices:
        selected = vectors[train_indices].detach().float()
    else:
        selected = vectors.detach().float()
    return float(torch.linalg.vector_norm(selected, dim=1).mean().item())


def _validate_rows(path_name: str, rows: list[dict[str, Any]]) -> None:
    required = {
        "condition",
        "user_prompt",
        "target_response",
        "concept",
        "vector_index",
        "alpha",
        "inject_noise",
    }
    for index, row in enumerate(rows):
        missing = sorted(required.difference(row))
        if missing:
            raise ConfigError(f"{path_name} row {index} is missing fields: {missing}")


def _rate(rows: list[dict[str, Any]], predicate: Any) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)
