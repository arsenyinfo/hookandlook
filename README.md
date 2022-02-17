# HookAndLook 🎣

`hookandlook` is a tool helping to gather stats and run checks during training deep learning models with Pytorch.

Features:

- Storing stats for input and output of a model or its blocks;
- Stats visualization with tensorboard;
- Model sanity checks as hooks (see `hookandlook.hooks`);

Checks collection is far from complete, contributions are very welcome.

## Installation

`pip install hookandlook`

or

`pip install git+https://github.com/arsenyinfo/hookandlook`

## Usage

```python
from hookandlook.watch import Wrapper
from hookandlook import hooks

model = MyTorchModel()

# simplest wrapper
model = Wrapper.wrap_model(model)

# or with additional selected hooks
check_positive = hooks.forward_hook_check_value_in_range(min_value=0)
check_train_mode = hooks.backward_hook_check_if_train_mode()
model = Wrapper.wrap_model(model, 
                           forward_hooks=[check_positive],
                           backward_hooks=[check_train_mode],
                           )

# Do your training! 
for x, y in train_loader:
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    model.zero_grad()
    
# get stats and analyze them manually
input_stats = model.watcher.input_df
output_stats = model.watcher.output_df

# ...or visualize them with tensorboard
writer = SummaryWriter()
for label, batch_ids, values in model.watcher.iterate_over_stats(inputs=inputs_or_outputs):
    for x, y in zip(batch_ids, values):
        writer.add_scalar(tag=label, scalar_value=y, global_step=x)


# detach the watcher from the model if needed
model = model.detach_hooks()
```

## Contributing

Contributions are welcome!

If you want to introduce a new feature, please open an issue first. Otherwise, please feel free to choose one of the
open issues (if there are any) and notify about your desire to contribute.

Maintainers provide guidance and feedback on the issues!
