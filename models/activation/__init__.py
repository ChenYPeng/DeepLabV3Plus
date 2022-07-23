import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def get_activation(name="silu", inplace=True):
    if name == "silu":
        act = nn.SiLU(inplace=inplace)
    elif name == "relu":
        act = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        act = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "mish":
        act = Mish()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return act
