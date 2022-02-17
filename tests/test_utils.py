import torch

from hookandlook.utils import Aggregator, StatsSampledTable


def test_stats_reservoir():
    reservoir = StatsSampledTable(10)

    for _ in range(5):
        reservoir.add({'value': 1, 'input_name': 'name', 'stat_name': 'mean', 'module_name': ''})

    assert len(reservoir) == 5

    for _ in range(10):
        reservoir.add({'value': 0, 'input_name': 'name', 'stat_name': 'mean', 'module_name': ''})

    assert len(reservoir) == 10
    assert 0 in reservoir.value
    assert 1 in reservoir.value

    for _ in range(50):
        reservoir.add({'value': 1, 'input_name': 'name', 'stat_name': 'mean', 'module_name': ''})

    a_num = len([x for x in reservoir.value if x == 0])
    assert a_num <= 5

    assert all(reservoir.as_dataframe()['value'] == reservoir.value)


def test_aggregator():
    agg = Aggregator()
    res = list(agg((torch.rand(10), torch.rand(10))))
    assert len(res) == 8  # 4 stats for 2 inputs
