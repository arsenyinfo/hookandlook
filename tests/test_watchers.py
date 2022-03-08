import contextlib
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from tensorboardX import SummaryWriter

from hookandlook.hooks import forward_hook_check_value_in_range
from hookandlook.utils import add_to_inputs, call_with_args_or_kwargs
from hookandlook.watch import EmptyInputsHookWarning, Wrapper, WrapperError
from .shared import FixtureNet, fixture_generator


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


def test_stats_are_sorted_by_batch_id():
    model = FixtureNet()
    watched_model = Wrapper.wrap_model(model, reserve_size=20)
    for i in range(10):
        _ = watched_model(torch.rand(1, 8) + i)

    for df in (watched_model.watcher.inputs_df, watched_model.watcher.outputs_df):
        batch_ids = df['batch_id'].values
        assert all(batch_ids == sorted(batch_ids))

    # only applicable for inputs
    df = watched_model.watcher.inputs_df
    values = df[df.stat_name == 'mean']['value'].values
    np.testing.assert_allclose(values, sorted(values), atol=1e-5, rtol=0)


def test_module_specific_methods():
    model = FixtureNet()
    watched_model = Wrapper.wrap_model(model)

    watched_model = watched_model.to(torch.device('cpu'))
    assert isinstance(watched_model, Wrapper)

    watched_model = watched_model.cpu()
    assert isinstance(watched_model, Wrapper)

    watched_model = watched_model.train()
    assert isinstance(watched_model, Wrapper)

    watched_model = watched_model.train(True)
    assert isinstance(watched_model, Wrapper)

    watched_model = watched_model.requires_grad_()
    assert isinstance(watched_model, Wrapper)

    watched_model = watched_model.requires_grad_(True)
    assert isinstance(watched_model, Wrapper)

    watched_model = watched_model.eval()
    assert isinstance(watched_model, Wrapper)


def test_listeners():
    class Counter:
        def __init__(self, value=None):
            self.count = 0
            self.value = value

        def __call__(self, x: dict):
            if self.value is not None and x['stat_name'] == 'mean':
                np.testing.assert_allclose(x['value'], self.value, atol=1e-5, rtol=0)
            self.count += 1

    model = FixtureNet()
    batch = torch.rand(1, 8)

    cnt_input = Counter(value=batch.mean())
    cnt_output = Counter(value=None)
    watched_model = Wrapper.wrap_model(model,
                                       forward_input_listeners=[cnt_input],
                                       forward_output_listeners=[cnt_output],
                                       )
    for _ in range(3):
        _ = watched_model(batch)

    assert cnt_input.count == 3 * 4  # 4 stats per batch
    assert cnt_output.count == 3 * 4


def test_listeners_for_tensorboard():
    model = FixtureNet()
    batch = torch.rand(1, 8)

    temp_dir = tempfile.mkdtemp()
    writer = SummaryWriter(logdir=temp_dir)

    def tensorboard_listener(x: dict):
        if x['stat_name'] == 'mean':
            writer.add_scalar(tag=x['stat_name'], scalar_value=x['value'])

    watched_model = Wrapper.wrap_model(model,
                                       forward_input_listeners=[tensorboard_listener],
                                       )
    for _ in range(3):
        _ = watched_model(batch)

    writer.close()
    files = os.listdir(temp_dir)
    assert len(files) == 1
    shutil.rmtree(temp_dir)
