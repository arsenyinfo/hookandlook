import random
from dataclasses import dataclass
from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd
import torch


@dataclass
class StatsRow:
    input_name: Union[str, int]
    stat_name: str
    module_name: Union[str, int]
    value: Union[float, np.ndarray, torch.Tensor]
    is_training: bool
    metadata = None
    batch_id: int = None

    def as_dict(self):
        return {'input_name': self.input_name,
                'stat_name': self.stat_name,
                'module_name': self.module_name,
                'value': self.value,
                'is_training': self.is_training,
                'metadata': self.metadata,
                'batch_id': self.batch_id
                }


class StatsSampledTable:
    # base implementation of https://en.wikipedia.org/wiki/Reservoir_sampling
    def __init__(self, size):
        self.size = size
        self.i = 0
        self.data = []

    def add(self, value: dict, r=None):
        if self.i < self.size:
            self.data.append(StatsRow(**value, batch_id=self.i // 4))
        else:
            if r is None:
                r = random.randint(0, self.i)
            if r < self.size:
                self.data[r] = StatsRow(**value, batch_id=self.i)
        self.i += 1
        return r

    def __len__(self):
        return len(self.data)

    @property
    def input_name(self):
        return [d.input_name for d in self.data]

    @property
    def stat_name(self):
        return [d.stat_name for d in self.data]

    @property
    def value(self):
        return [d.value for d in self.data]

    @property
    def metadata(self):
        return [d.metadata for d in self.data]

    @property
    def batch_id(self):
        return [d.batch_id for d in self.data]

    def as_dataframe(self):
        return pd.DataFrame([x.as_dict() for x in self.data])


class Aggregator:
    """
    Aggregate statistic from Torch inputs.
    It's assumed typical inputs are tensors and dicts, user can overwrite it.
    """

    def __init__(self):
        # hardcoded for now
        self.aggregators = {
            'mean': torch.mean,
            'min': torch.min,
            'max': torch.max,
            'l2_norm': lambda x: torch.norm(x, p=2),
        }

    def _agg(self, x: torch.Tensor, input_name: Union[str, int]):
        for name, func in self.aggregators.items():
            yield {'input_name': input_name, 'stat_name': name, 'value': func(x.detach()).item()}

    def __call__(self, inputs, batch_id=None, metadata=None):
        named_inputs = inputs.items() if isinstance(inputs, dict) else enumerate(inputs)
        for name, x in named_inputs:
            for row in self._agg(x, name):
                yield row


def add_to_inputs(x: Union[Sequence[torch.Tensor], Dict[str, torch.Tensor]], y: float):
    if isinstance(x, dict):
        res = x.copy()
        for k, v in res.items():
            res[k] = v + y
    else:
        res = tuple(v + y for v in x)
    return res


def call_with_args_or_kwargs(func, args_or_kwargs):
    if isinstance(args_or_kwargs, dict):
        return func(**args_or_kwargs)
    return func(*args_or_kwargs)
