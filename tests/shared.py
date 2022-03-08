import torch
from torch import nn
from torchvision.models import resnet18


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
