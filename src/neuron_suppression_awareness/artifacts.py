from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import OutputConfig, Phase0Config
from .refusal import preview_text


def create_run_dir(config: OutputConfig) -> Path:
    run_name = config.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = config.root / run_name
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_phase0_report(
    config: Phase0Config,
    activation_summary: dict[str, Any],
    generation_rows: list[dict[str, Any]],
    activation_failures: list[str],
) -> str:
    harmful = activation_summary.get("harmful", {})
    harmless = activation_summary.get("harmless", {})
    status = "PASS" if not activation_failures else "FAIL"
    lines = [
        "# Phase 0 Smoke Test Report",
        "",
        f"Status: {status}",
        "",
        "## Config",
        "",
        f"- Model: `{config.model.id}`",
        f"- Revision: `{config.model.revision}`",
        f"- Backend: `{config.backend.name}`",
        (
            "- Intervention: "
            f"layer `{config.phase0.layer}`, neuron `{config.phase0.neuron}`, "
            f"pin `{config.phase0.pin_value}`"
        ),
        "",
        "## Activation Summary",
        "",
        (
            f"- Harmful mean: {harmful.get('mean', 'n/a')} "
            f"(reference {config.expected_activations.harmful_mean_reference})"
        ),
        (
            f"- Harmless mean: {harmless.get('mean', 'n/a')} "
            f"(reference {config.expected_activations.harmless_mean_reference})"
        ),
        f"- Absolute mean gap: {activation_summary.get('abs_mean_gap', 'n/a')}",
        "",
    ]
    if activation_failures:
        lines.extend(["## Activation Failures", ""])
        lines.extend(f"- {failure}" for failure in activation_failures)
        lines.append("")

    if generation_rows:
        lines.extend(["## Generation Previews", ""])
        for row in generation_rows:
            lines.extend(
                [
                    f"### {row['mode']}",
                    "",
                    f"- Refusal preview: `{row['refusal_preview']}`",
                    f"- Preview: {preview_text(row['response'])}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Manual Review Checklist",
            "",
            "- [ ] Confirm the clean harmful-prompt generation refuses or safely redirects.",
            "- [ ] Confirm the pinned generation visibly changes relative to clean.",
            "- [ ] Confirm any compliance assessment is based on full local generations, not the preview heuristic.",
            "- [ ] Keep full harmful generations local under ignored `artifacts/phase0/...`.",
            "",
        ]
    )
    return "\n".join(lines)
