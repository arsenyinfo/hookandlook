# HookAndLook 🎣

`hookandlook` is a tool helping to gather stats and run checks during training deep learning models with Pytorch 
using hooks.

Features:

- Subsampling and storing stats for input and output of a model or its blocks;
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
check_positive = hooks.forward_hook_check_value_in_range(min_value=0, max_value=None)
check_train_mode = hooks.backward_hook_check_if_train_mode()
model = Wrapper.wrap_model(model,
                           forward_hooks=[check_positive],
                           backward_hooks=[check_train_mode],
                           reserve_size=1000, # how many data samples can be saved in memory at most; 
                           )

# Do your training! 
for x, y in train_loader:
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    model.zero_grad()

# get accumulated stats as pandas dataframes and analyze them manually
input_stats = model.watcher.input_df
output_stats = model.watcher.output_df

# ...or visualize them with tensorboard
writer = SummaryWriter()
for label, batch_ids, values in model.watcher.iterate_over_stats(inputs=True):
    for x, y in zip(batch_ids, values):
        writer.add_scalar(tag=label, scalar_value=y, global_step=x)
        
# detach the watcher from the model if needed
model = model.detach_hooks()
```

## Advanced usage

The library is based on "callback hell" design pattern, so it is possible to use it in a more advanced way.
E.g. if you'd like to store all the stats - not just subsample - in your tensorboard, you can use `listeners`:

```python
writer = SummaryWriter(logdir=temp_dir)

def tensorboard_listener(x: dict):
    # x is a dict with keys: 'input_name', 'stat_name', 'value', 'module_name', 'is_training'
    writer.add_scalar(tag=x['stat_name'], scalar_value=x['value'])

watched_model = Wrapper.wrap_model(model,
                                   forward_input_listeners=[tensorboard_listener],
                                   )

```

## Contributing

Contributions are welcome!

If you want to introduce a new feature, please open an issue first. Otherwise, please feel free to choose one of the
open issues (if there are any) and notify about your desire to contribute.

Maintainers provide guidance and feedback on the issues!
