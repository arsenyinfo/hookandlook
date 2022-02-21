from collections import OrderedDict
from warnings import warn

import torch
from torch import nn

from hookandlook.hooks import compose_hooks
from hookandlook.utils import Aggregator
from hookandlook.utils import StatsSampledTable


class WrapperError(ValueError):
    pass


class EmptyInputsHookWarning(UserWarning):
    pass


class Wrapper:
    def __init__(self, obj: nn.Module, watcher):
        if hasattr(obj, '_obj'):
            if hasattr(obj, 'watcher'):
                raise WrapperError('Probably model is already wrapped')
            else:
                raise WrapperError('Model has attribute `_obj`, cannot wrap')
        if hasattr(obj, 'watcher'):
            raise WrapperError('Model has attribute `watcher`, cannot wrap')

        self._obj = obj
        self.watcher = watcher

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self.__dict__:
            return getattr(self._obj, item)
        else:
            return self.__dict__[item]

    def __setattr__(self, key, value):
        if key not in ('_obj', 'watcher'):
            setattr(self._obj, key, value)
        else:
            self.__dict__[key] = value

    def to(self, x):
        self._obj = self._obj.to(x)
        return self

    def cpu(self):
        self._obj = self._obj.cpu()
        return self

    def cuda(self):
        self._obj = self._obj.cuda()
        return self

    def train(self, x):
        self._obj = self._obj.train(x)
        return self

    def eval(self):
        self._obj = self._obj.eval()
        return self

    @classmethod
    def wrap_model(cls,
                   model: nn.Module,
                   wrap_children=False,
                   forward_hooks=None,
                   backward_hooks=None,
                   forward_input_listeners=None,
                   forward_output_listeners=None,
                   reserve_size=1000,
                   ):
        if isinstance(model, torch.jit.ScriptModule) and wrap_children:
            raise WrapperError('Cannot wrap children of torch.jit.ScriptModule')
        return cls(model, ModelWatcher(model,
                                       wrap_children=wrap_children,
                                       forward_hooks=forward_hooks,
                                       backward_hooks=backward_hooks,
                                       reserve_size=reserve_size,
                                       input_storage_listeners=forward_input_listeners,
                                       output_storage_listeners=forward_output_listeners,
                                       ))

    def detach_hooks(self):
        self._obj._forward_hooks = OrderedDict()
        self._obj._backward_hooks = OrderedDict()
        return self._obj


def _add_module_attributes(d: dict, name, is_training):
    d['module_name'] = name
    d['is_training'] = is_training
    return d


class ModelWatcher:
    def __init__(self,
                 model: nn.Module,
                 reserve_size=1000,
                 wrap_children=True,
                 forward_hooks=None,
                 backward_hooks=None,
                 input_storage_listeners=None,
                 output_storage_listeners=None,
                 ):
        self._model = model
        # we store stats in two-level hierarchy: module and stat name
        self.storage_size = reserve_size
        self.storage = None
        self.reset_storage()
        self.wrap_children = wrap_children
        self.initialize(forward_hooks, backward_hooks)
        self.aggregator = Aggregator()
        self.listeners = {'input': input_storage_listeners or [], 'output': output_storage_listeners or []}

    def _get_sorted_df(self, key):
        df = self.storage[key].as_dataframe()
        if len(df):
            df = df.sort_values('batch_id')
        return df

    @property
    def inputs_df(self):
        return self._get_sorted_df('input')

    @property
    def outputs_df(self):
        return self._get_sorted_df('output')

    def reset_storage(self):
        storage = self.storage
        self.storage = {k: StatsSampledTable(size=self.storage_size) for k in ('input', 'output')}
        return storage

    def _add_to_storage(self, key, item, r=None) -> int:
        for listener in self.listeners[key]:
            listener(item)
        return self.storage[key].add(item, r=r)

    def get_storage_hook(self, name, forward=True):
        """
        This hook is used to store statistics of model's forward/backward passes
        """

        def forward_hook(module, inputs, outputs):
            if not inputs:
                warn('Empty inputs. It may happen when model is called with keyword arguments', EmptyInputsHookWarning)

            is_training = module.training
            if len(inputs) == len(outputs):
                # if inputs and outputs have the same length, we aim to synchronize them, so same batch stats are stored
                for x, y in zip(self.aggregator(inputs), self.aggregator(outputs)):
                    r = self._add_to_storage('input', _add_module_attributes(x, name, is_training=is_training))
                    self._add_to_storage('output', _add_module_attributes(y, name, is_training=is_training), r=r)
            else:
                for x in self.aggregator(inputs):
                    self._add_to_storage('input', _add_module_attributes(x, name, is_training=is_training))
                for x in self.aggregator(outputs):
                    self._add_to_storage('output', _add_module_attributes(x, name, is_training=is_training))

        def backward_hook(module, grad_inputs, grad_outputs):
            pass

        return forward_hook if forward else backward_hook

    def initialize(self, forward_hooks, backward_hooks):
        forward_hooks = [] if forward_hooks is None else forward_hooks
        backward_hooks = [] if backward_hooks is None else backward_hooks

        if self.wrap_children:
            for name, block in self._model.named_children():
                composed_forward = compose_hooks([self.get_storage_hook(name, forward=True)] + forward_hooks)
                composed_backward = compose_hooks([self.get_storage_hook(name, forward=False)] + backward_hooks)

                block.register_forward_hook(composed_forward)
                block.register_backward_hook(composed_backward)
        else:
            name = 'full'
            composed_forward = compose_hooks([self.get_storage_hook(name, forward=True)] + forward_hooks)
            composed_backward = compose_hooks([self.get_storage_hook(name, forward=False)] + backward_hooks)

            self._model.register_forward_hook(composed_forward)
            self._model.register_backward_hook(composed_backward)

    def iterate_over_stats(self, inputs=True):
        df = self.inputs_df if inputs else self.outputs_df
        input_name_is_informative = len(df.input_name.value_counts()) > 1
        module_name_is_informative = len(df.module_name.value_counts()) > 1
        grouped = df.groupby(['input_name', 'stat_name', 'module_name', 'is_training'])
        for (input_name, stat_name, module_name, training_mode), subset in grouped:
            batch_ids = subset['batch_id'].values
            values = subset['value'].values
            label = f'{"train" if training_mode else "eval"}_{stat_name}' \
                    f'{"_" + str(input_name) if input_name_is_informative else ""}' \
                    f'{"_" + str(module_name) if module_name_is_informative else ""}'
            yield label, batch_ids, values
