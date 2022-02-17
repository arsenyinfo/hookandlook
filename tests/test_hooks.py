import pytest
import torch

from hookandlook.hooks import backward_hook_check_if_train_mode, forward_hook_check_value_in_range
from hookandlook.watch import Wrapper


def test_training_mode_check():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5), torch.nn.Linear(2, 1))
    model.train()

    wrapped_model = Wrapper.wrap_model(model, backward_hooks=[backward_hook_check_if_train_mode()])

    out = wrapped_model(torch.randn(1, 2))
    loss = torch.randn(1) - out.sum()
    loss.backward()

    with pytest.raises(RuntimeError):
        wrapped_model.train(False)
        out = wrapped_model(torch.randn(1, 2))
        loss = torch.randn(1) - out.sum()
        loss.backward()


@pytest.mark.parametrize("min_value", (None, 0))
@pytest.mark.parametrize("max_value", (None, 1))
def test_forward_check_value_in_range_passes(min_value, max_value):
    model = torch.nn.Linear(2, 1)
    wrapped_model = Wrapper.wrap_model(model, forward_hooks=[forward_hook_check_value_in_range(min_value, max_value)])

    _ = wrapped_model(torch.randn(1, 2))


@pytest.mark.parametrize("min_value", (1, float('inf')))
@pytest.mark.parametrize("max_value", (0, float('-inf')))
def test_forward_check_value_in_range_passes(min_value, max_value):
    model = torch.nn.Linear(2, 1)
    wrapped_model = Wrapper.wrap_model(model, forward_hooks=[forward_hook_check_value_in_range(min_value, max_value)])

    with pytest.raises(ValueError):
        _ = wrapped_model(torch.randn(1, 2))
