from typing import Callable, Optional, Sequence


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
