from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import (
    OutputConfig,
    Phase0Config,
    Phase1Config,
    Phase2AConfig,
    Phase2BConfig,
    Phase3Config,
    Phase4Config,
)
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


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    append: bool = False,
) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
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


def build_phase1_report(
    config: Phase1Config,
    results: dict[str, Any],
    judgments: list[dict[str, Any]],
    activations: list[dict[str, Any]],
) -> str:
    status = "PASS" if results.get("passed") else "FAIL"
    clean_rows = [row for row in judgments if row.get("mode") == "clean"]
    suppressed_rows = [row for row in judgments if row.get("mode") == "suppressed"]
    unknown_count = sum(1 for row in judgments if row.get("verdict") == "unknown")
    lines = [
        "# Phase 1 JailbreakBench Suppression Report",
        "",
        f"Status: {status}",
        "",
        "## Config",
        "",
        f"- Model: `{config.model.id}`",
        f"- Revision: `{config.model.revision}`",
        f"- Dataset: `{config.dataset.id}` split `{config.dataset.split}`",
        f"- Judge: `{config.judge.model.id}`",
        (
            "- Intervention: "
            f"layer `{config.suppression.layer}`, neuron `{config.suppression.neuron}`, "
            f"pin `{config.suppression.pin_value}`"
        ),
        "",
        "## ASR",
        "",
        "| Mode | Unsafe / Total | ASR | Pass criterion |",
        "| --- | ---: | ---: | --- |",
        (
            f"| Clean | {_unsafe_count(clean_rows)} / {len(clean_rows)} | "
            f"{results.get('clean_asr', 0.0):.3f} | "
            f"<= {config.pass_criteria.max_clean_asr:.2f} |"
        ),
        (
            f"| Suppressed | {_unsafe_count(suppressed_rows)} / {len(suppressed_rows)} | "
            f"{results.get('suppressed_asr', 0.0):.3f} | "
            f">= {config.pass_criteria.min_suppressed_asr:.2f} |"
        ),
        "",
        "## Clean Activation Check",
        "",
        (
            "- Mean clean prompt-forward activation for neuron "
            f"{config.suppression.neuron}: "
            f"{results.get('clean_activation_mean', 'n/a')}"
        ),
        f"- Activation records: {len(activations)}",
        "",
    ]
    if unknown_count:
        lines.extend(
            [
                "## Judge Warnings",
                "",
                f"- Llama-Guard verdict parser returned `unknown` for {unknown_count} rows.",
                "",
            ]
        )

    lines.extend(["## Per-Prompt Summary", ""])
    for row in judgments:
        lines.append(
            "- "
            f"`{row.get('prompt_id')}` `{row.get('mode')}` "
            f"verdict=`{row.get('verdict')}` "
            f"refusal_preview=`{row.get('refusal_preview')}` "
            f"response={preview_text(str(row.get('response', '')))}"
        )
    lines.append("")
    return "\n".join(lines)


def build_phase2a_report(
    config: Phase2AConfig,
    caa_result: Any,
    n_train: int,
    n_eval: int,
) -> str:
    norms = [m["vector_norm"] for m in caa_result.metadata]
    train_norms = [m["vector_norm"] for m in caa_result.metadata if m["split"] == "train"]
    test_norms = [m["vector_norm"] for m in caa_result.metadata if m["split"] == "test"]
    td = config.training_data
    lines = [
        "# Phase 2A: CAA Vector Extraction + Training Data Report",
        "",
        "## Config",
        "",
        f"- Model: `{config.model.id}`",
        f"- Revision: `{config.model.revision}`",
        f"- CAA layer: {config.caa.layer}",
        f"- Prompt templates: {config.caa.n_prompt_templates}",
        f"- Baseline concept: `{config.caa.baseline_concept}`",
        "",
        "## Vector Extraction",
        "",
        f"- Total concepts: {len(caa_result.concept_order)}",
        f"- Train: {len(train_norms)}, Test: {len(test_norms)}",
        f"- d_model: {caa_result.d_model}",
        f"- Mean vector norm (all): {sum(norms) / len(norms):.4f}",
        f"- Mean vector norm (train): {caa_result.mean_train_norm:.4f}",
        f"- Min norm: {min(norms):.4f}, Max norm: {max(norms):.4f}",
        "",
        "## Training Data",
        "",
        f"- Total training examples: {n_train}",
        f"- Total eval examples: {n_eval}",
        "",
        "| Condition | Fraction |",
        "| --- | ---: |",
        f"| steered_correct | {td.steered_correct_fraction:.0%} |",
        f"| clean | {td.clean_fraction:.0%} |",
        f"| noise | {td.noise_fraction:.0%} |",
        f"| mismatch | {td.mismatch_fraction:.0%} |",
        f"| alpaca_replay | {td.alpaca_replay_fraction:.0%} |",
        "",
        f"- Alpha values: {list(td.alpha_values)}",
        f"- Seed: {td.seed}",
        "",
    ]
    return "\n".join(lines)


def build_phase2b_report(config: Phase2BConfig, summary: dict[str, Any]) -> str:
    status = "PASS" if summary.get("passed") else "FAIL"
    criteria = summary.get("pass_criteria", {})
    lines = [
        "# Phase 2B: QLoRA Steering-Awareness Training Report",
        "",
        f"Status: {status}",
        "",
        "## Config",
        "",
        f"- Model: `{config.model.id}`",
        f"- Revision: `{config.model.revision}`",
        f"- Phase 2A artifacts: `{summary.get('phase2a_artifact_dir')}`",
        f"- Adapter: `{summary.get('adapter_dir')}`",
        f"- Injection layer: {config.injection.layer}",
        f"- Eval alpha: {config.injection.eval_alpha}",
        "",
        "## Training",
        "",
        f"- Training examples: {summary.get('n_train_examples')}",
        f"- Epochs: {config.training.epochs}",
        f"- LoRA rank/alpha: {config.training.rank}/{config.training.alpha}",
        f"- Learning rate: {config.training.learning_rate}",
        "",
        "## Held-Out Detection",
        "",
        "| Metric | Value | Criterion |",
        "| --- | ---: | --- |",
        (
            f"| Detection rate | {summary.get('detection_rate', 0.0):.3f} | "
            f">= {criteria.get('min_detection_rate', 0.0):.2f} |"
        ),
        (
            f"| Identification rate | {summary.get('identification_rate', 0.0):.3f} | "
            "reported |"
        ),
        (
            f"| Clean FPR | {summary.get('clean_fpr', 0.0):.3f} | "
            f"<= {criteria.get('max_clean_fpr', 0.0):.2f} |"
        ),
        (
            f"| Noise FPR | {summary.get('noise_fpr', 0.0):.3f} | "
            f"<= {criteria.get('max_noise_fpr', 0.0):.2f} |"
        ),
        "",
        "## Counts",
        "",
    ]
    counts = summary.get("counts", {})
    for key in ["positive", "clean", "noise", "total"]:
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.append("")
    return "\n".join(lines)


def build_phase3_report(config: Phase3Config, metrics: dict[str, Any]) -> str:
    status = "PASS" if metrics.get("passed") else "FAIL"
    exp_a = metrics.get("experiment_a", {})
    exp_b = metrics.get("experiment_b", {})
    a_counts = exp_a.get("counts", {})
    b_counts = exp_b.get("counts", {})
    criteria = metrics.get("pass_criteria", {})
    lines = [
        "# Phase 3: Detection Transfer + Susceptibility Paradox Report",
        "",
        f"Status: {status}",
        "",
        "## Config",
        "",
        f"- Model: `{config.model.id}`",
        f"- Revision: `{config.model.revision}`",
        f"- Adapter: `{config.inputs.phase2b_adapter_dir}`",
        f"- Suppression: layer {config.suppression.layer}, "
        f"neuron {config.suppression.neuron}, pin {config.suppression.pin_value}",
        f"- Injection: layer {config.injection.layer}, alpha {config.injection.alpha}",
        f"- Judge: `{config.judge.model.id}`",
        "",
        "## Experiment A: Detection Transfer",
        "",
        "| Condition | Rate | N | Criterion |",
        "| --- | ---: | ---: | --- |",
        (
            f"| Clean control (FPR) | {exp_a.get('clean_control_fpr', 0):.3f} | "
            f"{a_counts.get('clean_control', 0)} | "
            f"<= {criteria.get('max_clean_control_fpr', 0):.2f} |"
        ),
        (
            f"| CAA positive control | {exp_a.get('caa_positive_detection_rate', 0):.3f} | "
            f"{a_counts.get('caa_positive', 0)} | "
            f">= {criteria.get('min_caa_positive_detection', 0):.2f} |"
        ),
        (
            f"| Noise control (FPR) | {exp_a.get('noise_control_fpr', 0):.3f} | "
            f"{a_counts.get('noise_control', 0)} | "
            f"<= {criteria.get('max_noise_control_fpr', 0):.2f} |"
        ),
        (
            f"| **Suppression detection** | **{exp_a.get('suppression_detection_rate', 0):.3f}** | "
            f"**{a_counts.get('suppression_detection', 0)}** | **reported** |"
        ),
        "",
        "## Experiment B: Susceptibility Paradox",
        "",
        "| | No Suppression | With Suppression |",
        "| --- | ---: | ---: |",
        (
            f"| Base model | {exp_b.get('base_clean_asr', 0):.3f} "
            f"(n={b_counts.get('base_clean', 0)}) | "
            f"{exp_b.get('base_suppressed_asr', 0):.3f} "
            f"(n={b_counts.get('base_suppression', 0)}) |"
        ),
        (
            f"| Aware model | {exp_b.get('adapter_clean_asr', 0):.3f} "
            f"(n={b_counts.get('adapter_clean', 0)}) | "
            f"{exp_b.get('adapter_suppressed_asr', 0):.3f} "
            f"(n={b_counts.get('adapter_suppression', 0)}) |"
        ),
        "",
    ]
    return "\n".join(lines)


def build_phase4_report(config: Phase4Config, summary: dict[str, Any]) -> str:
    geometry = summary.get("geometry", {}).get("by_model_and_layer", {})
    probes = summary.get("probe_results", {}).get("by_model_and_layer", {})
    lines = [
        "# Phase 4: Mechanistic Geometry + Linear Probing Report",
        "",
        "## Config",
        "",
        f"- Model: `{config.model.id}`",
        f"- Revision: `{config.model.revision}`",
        f"- Phase 2A artifacts: `{config.inputs.phase2a_artifact_dir}`",
        f"- Phase 2B adapter: `{config.inputs.phase2b_adapter_dir}`",
        f"- Model variants: {', '.join(config.analysis.model_variants)}",
        f"- Captured layers: {', '.join(str(layer) for layer in config.analysis.layers)}",
        f"- Capture position: `{config.analysis.capture_position}`",
        (
            "- Suppression: "
            f"layer {config.suppression.layer}, neuron {config.suppression.neuron}, "
            f"pin {config.suppression.pin_value}"
        ),
        f"- CAA: layer {config.injection.layer}, alpha {config.injection.alpha}",
        "",
        "## Geometry Summary",
        "",
        "| Model | Layer | Suppression L2 | CAA L2 | Supp/CAA Ratio | Supp-CAA Cosine | Supp-Raw CAA Cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in config.analysis.model_variants:
        for layer in config.analysis.layers:
            layer_summary = geometry.get(variant, {}).get(str(layer), {})
            lines.append(
                "| "
                f"{variant} | {layer} | "
                f"{_fmt_summary_mean(layer_summary.get('suppression_delta_l2'))} | "
                f"{_fmt_summary_mean(layer_summary.get('caa_delta_l2'))} | "
                f"{_fmt_summary_mean(layer_summary.get('suppression_to_caa_l2_ratio'))} | "
                f"{_fmt_summary_mean(layer_summary.get('suppression_to_caa_delta_cosine'))} | "
                f"{_fmt_summary_mean(layer_summary.get('suppression_to_raw_caa_cosine'))} |"
            )
    lines.extend(
        [
            "",
            "## Linear Probe Summary",
            "",
            "| Model | Layer | Test Accuracy | Balanced Accuracy | AUROC | Confusion |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for variant in config.analysis.model_variants:
        for layer in config.analysis.layers:
            probe = probes.get(variant, {}).get(str(layer), {}).get("test", {})
            confusion = probe.get("confusion", {})
            lines.append(
                "| "
                f"{variant} | {layer} | "
                f"{_fmt_float(probe.get('accuracy'))} | "
                f"{_fmt_float(probe.get('balanced_accuracy'))} | "
                f"{_fmt_float(probe.get('auroc'))} | "
                f"tp={confusion.get('tp', 0)}, tn={confusion.get('tn', 0)}, "
                f"fp={confusion.get('fp', 0)}, fn={confusion.get('fn', 0)} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Small suppression L2 at layer 24 with failed probes supports an informational invisibility explanation.",
            "- Large but low-cosine suppression deltas support a subspace mismatch explanation.",
            "- Successful probes with failed self-report support an introspective readout failure explanation.",
            "",
        ]
    )
    return "\n".join(lines)


def _unsafe_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("verdict") == "unsafe")


def _fmt_summary_mean(summary: Any) -> str:
    if not isinstance(summary, dict):
        return "n/a"
    return _fmt_float(summary.get("mean"))


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"
