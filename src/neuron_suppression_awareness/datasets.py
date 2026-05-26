from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .config import DatasetConfig, TextDatasetConfig
from .errors import DatasetAccessError, MissingDependencyError


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    source: str
    text: str
    dataset_id: str
    row_index: int


LoadDatasetFn = Callable[..., Iterable[dict[str, Any]]]


def load_phase0_prompts(
    config: DatasetConfig,
    load_dataset_fn: LoadDatasetFn | None = None,
) -> tuple[list[PromptRecord], list[PromptRecord]]:
    harmful = load_prompt_records(config.harmful, "harmful", load_dataset_fn)
    harmless = load_prompt_records(config.harmless, "harmless", load_dataset_fn)
    return harmful, harmless


def load_prompt_records(
    config: TextDatasetConfig,
    source: str,
    load_dataset_fn: LoadDatasetFn | None = None,
) -> list[PromptRecord]:
    token = _hf_token()
    if config.requires_hf_token and not token:
        raise DatasetAccessError(
            f"{config.id} is configured as gated/private. Set HF_TOKEN or "
            "HUGGING_FACE_HUB_TOKEN to an account with dataset access before "
            "running Phase 0."
        )

    if load_dataset_fn is None:
        try:
            from datasets import load_dataset as load_dataset_fn
        except ImportError as exc:
            raise MissingDependencyError(
                "The `datasets` package is required to load Phase 0 prompts."
            ) from exc

    try:
        dataset_kwargs = {"split": config.split, "token": token}
        if config.name is None:
            dataset = load_dataset_fn(config.id, **dataset_kwargs)
        else:
            dataset = load_dataset_fn(config.id, config.name, **dataset_kwargs)
    except Exception as exc:
        token_hint = (
            " Confirm HF_TOKEN/HUGGING_FACE_HUB_TOKEN is set and has access."
            if config.requires_hf_token
            else ""
        )
        raise DatasetAccessError(
            f"Unable to load dataset {config.id!r} split {config.split!r}."
            f"{token_hint} Original error: {exc}"
        ) from exc

    records: list[PromptRecord] = []
    for row_index, row in enumerate(dataset):
        text = extract_prompt_text(row, config)
        if not text:
            continue
        records.append(
            PromptRecord(
                prompt_id=f"{source}-{len(records)}",
                source=source,
                text=text,
                dataset_id=config.id,
                row_index=row_index,
            )
        )
        if len(records) >= config.limit:
            break

    if len(records) < config.limit:
        raise DatasetAccessError(
            f"Dataset {config.id!r} yielded {len(records)} usable {source} prompts; "
            f"expected {config.limit}. Check text_fields={config.text_fields!r}."
        )
    return records


def extract_prompt_text(row: dict[str, Any], config: TextDatasetConfig) -> str | None:
    for field in config.text_fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            if config.input_field and field == "instruction":
                extra = row.get(config.input_field)
                if isinstance(extra, str) and extra.strip():
                    text = f"{text}\n\n{extra.strip()}"
            return text
    return None


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
