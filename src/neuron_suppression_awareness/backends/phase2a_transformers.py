from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Phase2AConfig


@dataclass(frozen=True)
class Phase2ARunResult:
    artifact_dir: Path
    n_concepts_train: int
    n_concepts_test: int
    n_vectors: int
    d_model: int
    n_train_examples: int
    n_eval_examples: int
    mean_vector_norm: float


def run_phase2a(config: Phase2AConfig) -> Phase2ARunResult:
    from .transformers_backend import _runtime_imports, _load_model

    from ..artifacts import (
        build_phase2a_report,
        create_run_dir,
        write_json,
        write_jsonl,
        write_text,
    )
    from ..caa_extraction import extract_caa_vectors
    from ..concepts import load_concepts
    from ..training_data import build_eval_dataset, build_training_dataset

    torch, auto_model_cls, auto_tokenizer_cls = _runtime_imports()
    concepts = load_concepts()

    print(f"Concepts: {len(concepts.train)} train, {len(concepts.test)} test")
    print(f"CAA extraction layer: {config.caa.layer}")

    print("Loading tokenizer...")
    tokenizer = auto_tokenizer_cls.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )

    print("Loading model...")
    model = _load_model(config, torch, auto_model_cls)
    model.eval()

    print("Extracting CAA vectors...")
    caa_result = extract_caa_vectors(model, tokenizer, config, torch, concepts)
    print(
        f"Extracted {caa_result.tensor.shape[0]} vectors, "
        f"d_model={caa_result.d_model}, "
        f"mean_train_norm={caa_result.mean_train_norm:.4f}"
    )

    print("Unloading model...")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Building training dataset...")
    train_rows = build_training_dataset(config, caa_result, concepts)
    print(f"Training examples: {len(train_rows)}")

    print("Building eval dataset...")
    eval_rows = build_eval_dataset(config, caa_result, concepts)
    print(f"Eval examples: {len(eval_rows)}")

    artifact_dir = create_run_dir(config.outputs)
    print(f"Artifacts: {artifact_dir}")

    torch.save(caa_result.tensor, artifact_dir / "caa_vectors.pt")
    write_json(artifact_dir / "concept_order.json", caa_result.concept_order)
    write_json(artifact_dir / "caa_metadata.json", caa_result.metadata)
    write_jsonl(artifact_dir / "train_dataset.jsonl", train_rows)
    write_jsonl(artifact_dir / "eval_dataset.jsonl", eval_rows)

    summary: dict[str, Any] = {
        "n_concepts_train": len(concepts.train),
        "n_concepts_test": len(concepts.test),
        "n_vectors": caa_result.tensor.shape[0],
        "d_model": caa_result.d_model,
        "mean_train_norm": caa_result.mean_train_norm,
        "n_train_examples": len(train_rows),
        "n_eval_examples": len(eval_rows),
        "caa_layer": config.caa.layer,
        "alpha_values": list(config.training_data.alpha_values),
        "seed": config.training_data.seed,
    }
    write_json(artifact_dir / "phase2a_summary.json", summary)

    report = build_phase2a_report(config, caa_result, len(train_rows), len(eval_rows))
    write_text(artifact_dir / "phase2a_report.md", report)

    return Phase2ARunResult(
        artifact_dir=artifact_dir,
        n_concepts_train=len(concepts.train),
        n_concepts_test=len(concepts.test),
        n_vectors=caa_result.tensor.shape[0],
        d_model=caa_result.d_model,
        n_train_examples=len(train_rows),
        n_eval_examples=len(eval_rows),
        mean_vector_norm=caa_result.mean_train_norm,
    )
