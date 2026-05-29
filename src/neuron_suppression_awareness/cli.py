from __future__ import annotations

import argparse
import sys

from .backends import (
    phase1_transformers,
    phase2a_transformers,
    phase2b_transformers,
    phase3_transformers,
    phase4_transformers,
    transformers_backend,
    vllm_lens,
)
from .config import (
    SUPPORTED_BACKENDS,
    Phase0Config,
    Phase1Config,
    Phase2AConfig,
    Phase2BConfig,
    Phase3Config,
    Phase4Config,
    load_config,
)
from .errors import NSAError, UnsupportedBackendError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nsa-smoke",
        description="Run neuron suppression awareness experiment phases.",
    )
    parser.add_argument(
        "--config",
        default="configs/phase0.qwen3_8b.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(SUPPORTED_BACKENDS),
        default=None,
        help="Override backend.name from the YAML config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, backend_override=args.backend)
        if isinstance(config, Phase4Config):
            if config.backend.name != "transformers":
                raise AssertionError(f"Unhandled Phase 4 backend: {config.backend.name}")
            result = phase4_transformers.run_phase4(config)
            _print_phase4_result(result)
            return 0
        if isinstance(config, Phase3Config):
            if config.backend.name != "transformers":
                raise AssertionError(f"Unhandled Phase 3 backend: {config.backend.name}")
            result = phase3_transformers.run_phase3(config)
            _print_phase3_result(result)
            return 0
        if isinstance(config, Phase2BConfig):
            if config.backend.name != "transformers":
                raise AssertionError(f"Unhandled Phase 2B backend: {config.backend.name}")
            result = phase2b_transformers.run_phase2b(config)
            _print_phase2b_result(result)
            return 0
        if isinstance(config, Phase2AConfig):
            if config.backend.name != "transformers":
                raise AssertionError(f"Unhandled Phase 2A backend: {config.backend.name}")
            result = phase2a_transformers.run_phase2a(config)
            _print_phase2a_result(result)
            return 0
        if isinstance(config, Phase1Config):
            if config.backend.name != "transformers":
                raise AssertionError(f"Unhandled Phase 1 backend: {config.backend.name}")
            result = phase1_transformers.run_phase1(config)
            _print_phase1_result(result)
            return 0
        if isinstance(config, Phase0Config) and config.backend.name == "transformers":
            result = transformers_backend.run_phase0(config)
            _print_transformers_result(result)
            return 0
        if isinstance(config, Phase0Config) and config.backend.name == "vllm_lens":
            vllm_lens.run_phase0(config)
            return 0
        raise AssertionError(f"Unhandled backend: {config.backend.name}")
    except UnsupportedBackendError as exc:
        print(f"Unsupported backend: {exc}", file=sys.stderr)
        return 2
    except NSAError as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1


def _print_transformers_result(result: transformers_backend.Phase0RunResult) -> None:
    harmful = result.activation_summary["harmful"]["mean"]
    harmless = result.activation_summary["harmless"]["mean"]
    print(f"Artifacts: {result.artifact_dir}")
    print(f"Activation means: harmful={harmful:.4g}, harmless={harmless:.4g}")
    for item in result.generation_previews:
        print(
            f"{item['mode']} preview ({item['refusal_preview']}): "
            f"{item['preview']}"
        )


def _print_phase1_result(result: phase1_transformers.Phase1RunResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"Artifacts: {result.artifact_dir}")
    print(
        "Phase 1 ASR: "
        f"clean={result.clean_asr:.3f}, "
        f"suppressed={result.suppressed_asr:.3f}, "
        f"n={result.n_prompts}, status={status}"
    )


def _print_phase2a_result(result: phase2a_transformers.Phase2ARunResult) -> None:
    print(f"Artifacts: {result.artifact_dir}")
    print(
        f"Vectors: {result.n_vectors} concepts, "
        f"d_model={result.d_model}, "
        f"mean_norm={result.mean_vector_norm:.4f}"
    )
    print(
        f"Training examples: {result.n_train_examples}, "
        f"Eval examples: {result.n_eval_examples}"
    )


def _print_phase2b_result(result: phase2b_transformers.Phase2BRunResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"Artifacts: {result.artifact_dir}")
    print(
        "Phase 2B: "
        f"detection={result.detection_rate:.3f}, "
        f"identification={result.identification_rate:.3f}, "
        f"clean_fpr={result.clean_fpr:.3f}, "
        f"noise_fpr={result.noise_fpr:.3f}, "
        f"train_n={result.n_train_examples}, "
        f"eval_n={result.n_eval_examples}, "
        f"status={status}"
    )


def _print_phase3_result(result: phase3_transformers.Phase3RunResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"Artifacts: {result.artifact_dir}")
    print(
        "Experiment A: "
        f"clean_fpr={result.clean_control_fpr:.3f}, "
        f"caa_det={result.caa_positive_detection_rate:.3f}, "
        f"noise_fpr={result.noise_control_fpr:.3f}, "
        f"suppression_det={result.suppression_detection_rate:.3f}"
    )
    print(
        "Experiment B: "
        f"base_clean={result.base_clean_asr:.3f}, "
        f"base_supp={result.base_suppressed_asr:.3f}, "
        f"adapter_clean={result.adapter_clean_asr:.3f}, "
        f"adapter_supp={result.adapter_suppressed_asr:.3f}"
    )
    print(f"n={result.n_prompts}, status={status}")


def _print_phase4_result(result: phase4_transformers.Phase4RunResult) -> None:
    print(f"Artifacts: {result.artifact_dir}")
    print(
        "Phase 4: "
        f"records={result.n_records}, prompts={result.n_prompts}, "
        f"models={list(result.model_variants)}, layers={list(result.layers)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
