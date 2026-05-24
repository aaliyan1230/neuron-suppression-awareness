from __future__ import annotations

import argparse
import sys

from .backends import transformers_backend, vllm_lens
from .config import SUPPORTED_BACKENDS, load_config
from .errors import NSAError, UnsupportedBackendError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nsa-smoke",
        description="Run the Phase 0 Qwen3 refusal-neuron smoke test.",
    )
    parser.add_argument(
        "--config",
        default="configs/phase0.qwen3_8b.yaml",
        help="Path to Phase 0 YAML config.",
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
        if config.backend.name == "transformers":
            result = transformers_backend.run_phase0(config)
            _print_transformers_result(result)
            return 0
        if config.backend.name == "vllm_lens":
            vllm_lens.run_phase0(config)
            return 0
        raise AssertionError(f"Unhandled backend: {config.backend.name}")
    except UnsupportedBackendError as exc:
        print(f"Unsupported backend: {exc}", file=sys.stderr)
        return 2
    except NSAError as exc:
        print(f"Phase 0 failed: {exc}", file=sys.stderr)
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


if __name__ == "__main__":
    raise SystemExit(main())
