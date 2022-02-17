import contextlib

import pandas as pd
import pytest
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.models import resnet18
import tempfile
import shutil
import os

from hookandlook.hooks import forward_hook_check_value_in_range
from hookandlook.utils import add_to_inputs, call_with_args_or_kwargs
from hookandlook.watch import EmptyInputsHookWarning, Wrapper, WrapperError


class FixtureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x) / self.fc3(x)
        x = self.fc3(x)
        return x


# noinspection PyMethodOverriding
class FixtureNetWithMultipleInput(FixtureNet):
    def forward(self, x, y):
        return super().forward(x)


def fixture_generator():
    def make_sequential_model():
        return nn.Sequential(nn.Linear(8, 10),
                             nn.ReLU(),
                             nn.Linear(10, 1)
                             )

    pairs = (
        ((torch.rand(1, 8),), make_sequential_model()),
        ((torch.rand(1, 8),), FixtureNet()),
        ({'x': torch.rand(1, 8)}, FixtureNet()),
        ((torch.rand(1, 8), torch.rand(1, 8)), FixtureNetWithMultipleInput()),
        ((torch.rand(1, 3, 128, 128),), resnet18(pretrained=False),),
    )

    for pair in pairs:
        for use_jit in (True, False):
            inputs, model = pair
            if use_jit:
                if isinstance(inputs, dict):
                    inputs = tuple(inputs.values())
                model = torch.jit.trace(model, inputs)
            yield inputs, model


@pytest.mark.parametrize('inputs, model', fixture_generator())
def test_watcher_not_affecting_results(inputs, model):
    with torch.no_grad():
        out1 = call_with_args_or_kwargs(model, inputs)
        watched_model = Wrapper.wrap_model(model)
        out2 = call_with_args_or_kwargs(watched_model, inputs)

        assert (out1 == out2).all()


@pytest.mark.parametrize('inputs, model', fixture_generator())
@pytest.mark.parametrize('watch_children', (False, True,))
def test_stats_gathered(inputs, model, watch_children):
    if isinstance(model, torch.jit.ScriptModule) and watch_children:
        with pytest.raises(WrapperError):
            _ = Wrapper.wrap_model(model, wrap_children=watch_children)
        return

    input_is_dict = isinstance(inputs, dict)
    watched_model = Wrapper.wrap_model(model, wrap_children=watch_children)
    n_blocks = 1 if not watch_children else len(list(watched_model.children()))

    maybe_warn_catcher = pytest.warns(EmptyInputsHookWarning) if input_is_dict else contextlib.nullcontext()
    with maybe_warn_catcher:
        out = call_with_args_or_kwargs(watched_model, inputs)
        _ = call_with_args_or_kwargs(watched_model, add_to_inputs(inputs, 1))
        _ = call_with_args_or_kwargs(watched_model, add_to_inputs(inputs, -1))

    loss = torch.rand(1) - out
    loss.mean().backward()

    inputs_stats = watched_model.watcher.inputs_df
    outputs_stats = watched_model.watcher.outputs_df

    inputs_are_not_empty = watch_children or not input_is_dict

    # internal blocks always have single input, multiple input only happens for FixtureNetWithMultipleInput
    n_inputs = len(inputs) if not watch_children else 1
    # there were 3 forward calls, 4 stats per call
    # if inputs are dict, forward hook is not getting them
    assert len(inputs_stats) == 3 * 4 * n_inputs * n_blocks * inputs_are_not_empty
    assert len(outputs_stats) == 3 * 4 * n_blocks

    assert isinstance(inputs_stats, pd.DataFrame)
    assert isinstance(outputs_stats, pd.DataFrame)

    if inputs_are_not_empty and not watch_children:
        input_names = inputs_stats['input_name'].values.tolist()
        for i, _ in enumerate(inputs):
            assert i in input_names

    if not watch_children and not input_is_dict:
        df = inputs_stats
        for input_name, input_value in enumerate(inputs) if isinstance(inputs, tuple) else inputs.items():
            filtered = df[(df.stat_name == 'mean') & (df.input_name == input_name) & (df.module_name == 'full')]
            assert filtered.iloc[0].value == input_value.mean()

    output_names = outputs_stats['input_name'].values.tolist()
    # all fixture outputs are positional arguments
    assert set(output_names) == {0}


@pytest.mark.parametrize('inputs, model', fixture_generator())
def test_for_double_wrapper(inputs, model):
    watched_model = Wrapper.wrap_model(model)

    with pytest.raises(WrapperError):
        Wrapper.wrap_model(watched_model)


def test_additional_forward_hooks():
    model = FixtureNet()
    inputs = torch.rand(1, 8)
    watched_model = Wrapper.wrap_model(model, )
    _ = watched_model(inputs)

    detached_model = watched_model.detach_hooks()
    watched_model = Wrapper.wrap_model(detached_model, forward_hooks=[forward_hook_check_value_in_range(1, 2)])

    with pytest.raises(ValueError):
        _ = watched_model(inputs)
    _ = watched_model(inputs + 1)


def test_stats_are_from_same_batches():
    model = FixtureNet()
    inputs = torch.rand(1, 8)
    watched_model = Wrapper.wrap_model(model, reserve_size=5)

    for _ in range(20):
        _ = watched_model(inputs)

    watcher = watched_model.watcher
    assert (watcher.inputs_df['batch_id'].values == watcher.outputs_df['batch_id'].values).all()


@pytest.mark.parametrize('inputs_or_outputs', (True, False))
def test_tensorboard(inputs_or_outputs):
    model = FixtureNet()
    watched_model = Wrapper.wrap_model(model, )
    for _ in range(5):
        _ = watched_model(torch.rand(1, 8))

    temp_dir = tempfile.mkdtemp()
    writer = SummaryWriter(logdir=temp_dir)

    for label, batch_ids, values in watched_model.watcher.iterate_over_stats(inputs=inputs_or_outputs):
        for x, y in zip(batch_ids, values):
            writer.add_scalar(tag=label, scalar_value=y, global_step=x)

    writer.close()
    files = os.listdir(temp_dir)
    assert len(files) == 1
    shutil.rmtree(temp_dir)
