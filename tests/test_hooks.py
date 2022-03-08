import os
import shutil
import subprocess
import tempfile

import pytest
import torch

from hookandlook import hooks
from hookandlook.watch import Wrapper
from .shared import FixtureNet


def test_training_mode_check():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5), torch.nn.Linear(2, 1))
    model.train()

    wrapped_model = Wrapper.wrap_model(model, backward_hooks=[hooks.backward_hook_check_if_train_mode()])

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
    wrapped_model = Wrapper.wrap_model(model, forward_hooks=[hooks.forward_hook_check_value_in_range(min_value,
                                                                                                     max_value)])

    _ = wrapped_model(torch.randn(1, 2))


@pytest.mark.parametrize("min_value", (1, float('inf')))
@pytest.mark.parametrize("max_value", (0, float('-inf')))
def test_forward_check_value_in_range_passes(min_value, max_value):
    model = torch.nn.Linear(2, 1)
    wrapped_model = Wrapper.wrap_model(model, forward_hooks=[hooks.forward_hook_check_value_in_range(min_value,
                                                                                                     max_value)])

    with pytest.raises(ValueError):
        _ = wrapped_model(torch.randn(1, 2))


class FixtureNetWithFrozenSeed(FixtureNet):
    def __init__(self):
        torch.manual_seed(42)
        super().__init__()


def test_autogenerated_tests():
    model = FixtureNetWithFrozenSeed()
    batch = torch.rand(1, 8)

    temp_dir = tempfile.mkdtemp()
    fixture_gen = hooks.ForwardHookFixtureGenerator(out_dir=temp_dir)
    watched_model = Wrapper.wrap_model(model,
                                       forward_hooks=[fixture_gen]
                                       )
    for _ in range(3):
        watched_model(batch)

    fixture_gen._save()

    pythonpath = os.environ.get('PYTHONPATH', '')
    if pythonpath:
        pythonpath += ':'
    os.environ['PYTHONPATH'] = f"{pythonpath}{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
    assert subprocess.check_call(['pytest', '-v', '-s', temp_dir]) == 0
    shutil.rmtree(temp_dir)
