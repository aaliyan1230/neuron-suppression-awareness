import pytest
import torch

from neuron_suppression_awareness.errors import HookFailure, LayerPathError
from neuron_suppression_awareness.hooks import ResidualStreamHook, get_decoder_layer


class FakeDecoderLayer(torch.nn.Module):
    def forward(self, x):
        return (x, None)


class FakeModel:
    def __init__(self):
        self.model = type("M", (), {"layers": [FakeDecoderLayer()]})()


def test_captures_last_token():
    hook = ResidualStreamHook(capture_last_token=True)
    layer = FakeDecoderLayer()
    hidden = torch.randn(1, 10, 64)
    handle = layer.register_forward_hook(hook)
    layer(hidden)
    handle.remove()
    assert len(hook.captures) == 1
    assert hook.captures[0].shape == (1, 64)
    torch.testing.assert_close(hook.captures[0], hidden[:, -1, :].float())


def test_captures_full_sequence():
    hook = ResidualStreamHook(capture_last_token=False)
    layer = FakeDecoderLayer()
    hidden = torch.randn(1, 10, 64)
    handle = layer.register_forward_hook(hook)
    layer(hidden)
    handle.remove()
    assert hook.captures[0].shape == (1, 10, 64)


def test_rejects_non_tensor():
    hook = ResidualStreamHook()
    with pytest.raises(HookFailure, match="torch.Tensor"):
        hook(None, (), ("not a tensor", None))


def test_rejects_wrong_dims():
    hook = ResidualStreamHook()
    with pytest.raises(HookFailure, match="batch, seq, d_model"):
        hook(None, (), (torch.randn(64),))


def test_get_decoder_layer_valid():
    model = FakeModel()
    layer = get_decoder_layer(model, 0)
    assert isinstance(layer, FakeDecoderLayer)


def test_get_decoder_layer_out_of_range():
    model = FakeModel()
    with pytest.raises(LayerPathError):
        get_decoder_layer(model, 99)


def test_call_count():
    hook = ResidualStreamHook()
    layer = FakeDecoderLayer()
    handle = layer.register_forward_hook(hook)
    layer(torch.randn(1, 5, 32))
    layer(torch.randn(1, 3, 32))
    handle.remove()
    assert hook.calls == 2
    assert len(hook.captures) == 2
