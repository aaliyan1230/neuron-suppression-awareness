from __future__ import annotations

import pytest
import torch
from torch import nn

from neuron_suppression_awareness.errors import HookFailure, LayerPathError
from neuron_suppression_awareness.hooks import (
    DownProjNeuronHook,
    describe_down_proj_layers,
    get_down_proj_module,
)


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(4, 2, bias=False)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer()])


def test_down_proj_hook_captures_and_pins() -> None:
    module = nn.Linear(4, 2, bias=False)
    module.weight.data.fill_(1.0)
    hook = DownProjNeuronHook(neuron=2, pin_value=20.0, capture=True)
    handle = module.register_forward_pre_hook(hook)
    x = torch.zeros(1, 3, 4)

    out = module(x)
    handle.remove()

    assert hook.calls == 1
    assert hook.pinned_calls == 1
    assert torch.equal(hook.captures[0], torch.zeros(1, 3))
    assert torch.equal(x[..., 2], torch.full((1, 3), 20.0))
    assert torch.equal(out, torch.full((1, 3, 2), 20.0))


def test_down_proj_hook_rejects_bad_neuron() -> None:
    module = nn.Linear(4, 2, bias=False)
    hook = DownProjNeuronHook(neuron=5, pin_value=20.0)
    handle = module.register_forward_pre_hook(hook)
    try:
        with pytest.raises(HookFailure, match="out of range"):
            module(torch.zeros(1, 3, 4))
    finally:
        handle.remove()


def test_get_down_proj_module_and_describe() -> None:
    model = TinyModel()

    assert get_down_proj_module(model, 0) is model.model.layers[0].mlp.down_proj
    assert describe_down_proj_layers(model) == ["model.layers.0.mlp.down_proj"]


def test_get_down_proj_missing_layer() -> None:
    with pytest.raises(LayerPathError, match="Could not resolve"):
        get_down_proj_module(TinyModel(), 3)
