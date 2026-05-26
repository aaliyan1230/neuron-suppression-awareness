from __future__ import annotations

import argparse
import sys

from .backends import phase1_transformers
from .backends import transformers_backend, vllm_lens
from .config import SUPPORTED_BACKENDS, Phase0Config, Phase1Config, load_config
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


if __name__ == "__main__":
    raise SystemExit(main())
