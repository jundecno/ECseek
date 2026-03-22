import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)
from models.operations import *


class Dense(Module):
    def __init__(self, in_dim, out_dim, act="", norm="", dropout=0.0, bias=True, pre_norm=True):
        super().__init__()
        self.linear = Linear(in_dim, out_dim, bias=bias)
        self.act = str2act(act)
        self.norm = str2norm(norm, in_dim if pre_norm else out_dim)
        self.dropout = Dropout(dropout) if dropout > 0.0 else Identity()
        self.pre_norm = pre_norm

    def forward(self, x):
        if self.pre_norm:
            x = self.norm(x)
        x = self.linear(x)
        if not self.pre_norm:
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class MLP(Module):
    def __init__(self, dims, act="", norm="", dropout=0.0, bias=True, pre_norm=False):
        super().__init__()
        n = len(dims)
        layers = [Dense(dims[i], dims[i + 1], act, norm, dropout, bias, pre_norm) for i in range(n - 2)]
        layers.append(Dense(dims[-2], dims[-1], bias=bias))
        self.layers = Sequential(*layers)

    def forward(self, x: Tensor):
        return self.layers(x)
