from __future__ import annotations

from ..config import Phase0Config
from ..errors import ConfigError, UnsupportedBackendError


def run_phase0(config: Phase0Config) -> None:
    validate_vllm_lens_config(config)
    raise UnsupportedBackendError(
        "backend=vllm_lens is not implemented for Phase 0 exact scalar pinning. "
        "The referenced docs/vllm-lens plugin supports residual-stream activation "
        "capture and additive residual steering via apply_steering_vectors, but "
        "Phase 0 requires a pre-down_proj scalar assignment "
        f"h[{config.phase0.neuron}] <- {config.phase0.pin_value} at "
        f"model.model.layers[{config.phase0.layer}].mlp.down_proj. "
        "Add a vllm-lens worker extension for this scalar pin before using this backend."
    )


def validate_vllm_lens_config(config: Phase0Config) -> None:
    if config.phase0.layer < 0:
        raise ConfigError("vllm_lens backend requires a non-negative phase0.layer.")
    if config.phase0.neuron < 0:
        raise ConfigError("vllm_lens backend requires a non-negative phase0.neuron.")
    if config.phase0.pin_value is None:
        raise ConfigError("vllm_lens backend requires phase0.pin_value.")
