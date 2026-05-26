from __future__ import annotations

import pytest

from neuron_suppression_awareness.config import TextDatasetConfig
from neuron_suppression_awareness.datasets import (
    extract_prompt_text,
    load_prompt_records,
)
from neuron_suppression_awareness.errors import DatasetAccessError


def test_extract_alpaca_instruction_with_input() -> None:
    config = TextDatasetConfig(
        id="tatsu-lab/alpaca",
        split="train",
        limit=1,
        text_fields=("instruction",),
        input_field="input",
    )

    assert (
        extract_prompt_text(
            {"instruction": "Explain this", "input": "Extra context"},
            config,
        )
        == "Explain this\n\nExtra context"
    )


def test_requires_hf_token_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    config = TextDatasetConfig(
        id="walledai/AdvBench",
        split="train",
        limit=1,
        text_fields=("prompt",),
        requires_hf_token=True,
    )

    with pytest.raises(DatasetAccessError, match="Set HF_TOKEN"):
        load_prompt_records(config, "harmful", load_dataset_fn=lambda **_: [])


def test_load_prompt_records_uses_text_fields() -> None:
    config = TextDatasetConfig(
        id="fake",
        split="train",
        limit=2,
        text_fields=("prompt", "goal"),
    )

    def fake_loader(*args, **kwargs):
        del args, kwargs
        return [{"goal": "first"}, {"prompt": "second"}, {"empty": ""}]

    records = load_prompt_records(config, "harmful", load_dataset_fn=fake_loader)

    assert [record.text for record in records] == ["first", "second"]
    assert [record.prompt_id for record in records] == ["harmful-0", "harmful-1"]


def test_load_prompt_records_passes_dataset_name() -> None:
    config = TextDatasetConfig(
        id="fake",
        name="config-name",
        split="train",
        limit=1,
        text_fields=("prompt",),
    )
    calls = []

    def fake_loader(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"prompt": "first"}]

    records = load_prompt_records(config, "eval", load_dataset_fn=fake_loader)

    assert records[0].text == "first"
    assert calls == [
        (
            ("fake", "config-name"),
            {"split": "train", "token": None},
        )
    ]
