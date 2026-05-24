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
