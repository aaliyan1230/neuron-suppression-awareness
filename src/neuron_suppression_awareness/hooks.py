from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .errors import HookFailure, LayerPathError


@dataclass
class DownProjNeuronHook:
    """Capture and optionally pin one pre-down_proj MLP coordinate."""

    neuron: int
    pin_value: float | None = None
    capture: bool = True
    captures: list[torch.Tensor] = field(default_factory=list)
    calls: int = 0
    pinned_calls: int = 0

    def __call__(self, module: Any, inp: tuple[Any, ...]) -> tuple[Any, ...] | None:
        del module
        self.calls += 1
        if not inp:
            raise HookFailure("down_proj pre-hook received no input tensors.")
        tensor = inp[0]
        if not isinstance(tensor, torch.Tensor):
            raise HookFailure("down_proj pre-hook input[0] is not a torch.Tensor.")
        if tensor.dim() != 3:
            raise HookFailure(
                "down_proj pre-hook expected input[0] with shape "
                f"[batch, seq, d_ff], got {tuple(tensor.shape)}."
            )
        if self.neuron >= tensor.shape[-1]:
            raise HookFailure(
                f"Neuron index {self.neuron} is out of range for down_proj "
                f"input width {tensor.shape[-1]}."
            )

        if self.capture:
            self.captures.append(tensor[..., self.neuron].detach().float().cpu().clone())

        if self.pin_value is None:
            return None

        with torch.no_grad():
            tensor[..., self.neuron] = tensor.new_tensor(self.pin_value)
        self.pinned_calls += 1
        return (tensor, *inp[1:])


@dataclass
class ResidualStreamHook:
    """Capture full hidden state from a decoder layer's forward output."""

    capture_last_token: bool = True
    captures: list[torch.Tensor] = field(default_factory=list)
    calls: int = 0

    def __call__(
        self,
        module: Any,
        inp: tuple[Any, ...],
        output: tuple[Any, ...] | Any,
    ) -> None:
        del module, inp
        self.calls += 1
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise HookFailure(
                "Residual stream hook expected output[0] to be a torch.Tensor, "
                f"got {type(hidden).__name__}."
            )
        if hidden.dim() != 3:
            raise HookFailure(
                f"Expected hidden states shape [batch, seq, d_model], got {tuple(hidden.shape)}."
            )
        if self.capture_last_token:
            captured = hidden[:, -1, :].detach().float().cpu().clone()
        else:
            captured = hidden.detach().float().cpu().clone()
        self.captures.append(captured)


@dataclass
class ResidualInjectionHook:
    """Add residual vectors at specific token positions."""

    vectors: torch.Tensor
    token_indices: torch.Tensor
    apply_once: bool = False
    calls: int = 0
    injected_calls: int = 0

    def __call__(
        self,
        module: Any,
        inp: tuple[Any, ...],
        output: tuple[Any, ...] | Any,
    ) -> tuple[Any, ...] | Any:
        del module, inp
        self.calls += 1
        if self.apply_once and self.injected_calls > 0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise HookFailure(
                "Residual injection hook expected output[0] to be a torch.Tensor, "
                f"got {type(hidden).__name__}."
            )
        if hidden.dim() != 3:
            raise HookFailure(
                f"Expected hidden states shape [batch, seq, d_model], got {tuple(hidden.shape)}."
            )
        if self.vectors.dim() != 2:
            raise HookFailure(
                f"Expected injection vectors shape [batch, d_model], got {tuple(self.vectors.shape)}."
            )
        if self.vectors.shape[0] != hidden.shape[0]:
            raise HookFailure(
                "Injection vector batch size does not match hidden state batch size: "
                f"{self.vectors.shape[0]} != {hidden.shape[0]}."
            )
        if self.vectors.shape[1] != hidden.shape[-1]:
            raise HookFailure(
                "Injection vector width does not match hidden state width: "
                f"{self.vectors.shape[1]} != {hidden.shape[-1]}."
            )
        if self.token_indices.numel() != hidden.shape[0]:
            raise HookFailure(
                "token_indices length does not match batch size: "
                f"{self.token_indices.numel()} != {hidden.shape[0]}."
            )

        modified = hidden.clone()
        vectors = self.vectors.to(device=modified.device, dtype=modified.dtype)
        token_indices = self.token_indices.to(device=modified.device)
        batch_indices = torch.arange(modified.shape[0], device=modified.device)
        if torch.any(token_indices < 0) or torch.any(token_indices >= modified.shape[1]):
            raise HookFailure(
                f"Injection token index out of bounds for sequence length {modified.shape[1]}."
            )
        modified[batch_indices, token_indices, :] = (
            modified[batch_indices, token_indices, :] + vectors
        )
        self.injected_calls += 1
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified


def get_decoder_layer(model: Any, layer_index: int) -> Any:
    try:
        return model.model.layers[layer_index]
    except Exception as exc:
        raise LayerPathError(
            f"Could not resolve model.model.layers[{layer_index}]. "
            "Confirm the model architecture and zero-indexed layer numbering."
        ) from exc


def get_down_proj_module(model: Any, layer_index: int) -> Any:
    try:
        return model.model.layers[layer_index].mlp.down_proj
    except Exception as exc:
        raise LayerPathError(
            f"Could not resolve model.model.layers[{layer_index}].mlp.down_proj. "
            "Confirm the model architecture and zero-indexed layer numbering."
        ) from exc


def describe_down_proj_layers(model: Any) -> list[str]:
    if not hasattr(model, "named_modules"):
        return []
    return [
        name
        for name, _module in model.named_modules()
        if "mlp" in name and "down_proj" in name
    ]
