import os
from queue import Queue
from typing import Callable, Optional, Sequence

import torch


def compose_hooks(hooks: Sequence[Callable[..., None]]) -> Callable[..., None]:
    def composite_hook(*args, **kwargs):
        for hook in hooks:
            hook(*args, **kwargs)

    return composite_hook


def forward_hook_check_value_in_range(min_value: Optional[float], max_value: Optional[float], check_input: bool = True):
    min_value = min_value if min_value is not None else float("-inf")
    max_value = max_value if max_value is not None else float("inf")

    def hook(module, inputs, outputs):
        objects_to_check = inputs if check_input else outputs
        for i, obj in enumerate(objects_to_check):
            if obj.min() < min_value or obj.max() > max_value:
                name = 'inputs' if check_input else 'outputs'
                raise ValueError(f"Value {name} {i} of module {module.__class__}"
                                 f" is not in range [{min_value}, {max_value}]")

    return hook


def backward_hook_check_if_train_mode():
    def hook(module, grad_input, grad_output):
        if not module.training:
            raise RuntimeError("Backward hook called when model is not in training mode")

    return hook


class ForwardHookFixtureGenerator:
    def __init__(self, max_samples=3, out_dir='.'):
        self.queue = Queue(max_samples)
        self.module_name = None
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    @staticmethod
    def _detach_and_move_to_cpu(seq):
        return [x.detach().cpu() for x in seq]

    def __call__(self, module, inputs, outputs):
        if self.module_name is None:
            self.module_name = module.__class__.__module__, module.__class__.__qualname__
        self.queue.put((self._detach_and_move_to_cpu(inputs), self._detach_and_move_to_cpu(outputs)))


    def _generate_test(self, module, name):
        return f"""import os 

import torch
import pytest
from {module} import {name}


@pytest.fixture
def model():
    # FixMe: This fixture may be invalid and thus require manual validation.
    # It only works if your model instancing requires no arguments; otherwise you need to provide the args
    # or replace this fixture with a custom one.
    return {name}()

    
@pytest.fixture
def inputs_and_outputs():
    return torch.load(os.path.join(os.path.dirname(__file__), 'fixtures_{module}_{name}.pth'))


def test_forward(model, inputs_and_outputs):
    for inputs, outputs in inputs_and_outputs:
        res = model(*inputs)
        for x, y in zip(res, outputs):
            assert torch.allclose(x, y, atol=1e-3, rtol=0)
        # FixMe: you may need to tune tolerance        
        """

    def _save(self):
        if not self.queue.empty():
            module, name = self.module_name
            fname = f"fixtures_{module}_{name}.pth"
            torch.save(list(self.queue.queue), os.path.join(self.out_dir, fname))

            with open(os.path.join(self.out_dir, f"test_{name}.py"), 'w') as f:
                f.write(self._generate_test(module, name))

            self.queue.queue.clear()

    def __del__(self):
        self._save()
