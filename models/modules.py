import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)
from models.operations import *
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation


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


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, norm="", dropout=0.1):
        super().__init__()
        self.pre_norm = str2norm(norm, in_dim)
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.ffn_dropout(x)
        x = self.layer2(x)
        return x


class FFN(nn.Module):

    def __init__(self, in_dim, hidden_dim, act="silu", norm="layer", dropout=0.1):
        super().__init__()
        self.pre_norm = str2norm(norm, in_dim)
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, in_dim)
        self.act = str2act(act)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.layer1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class SwiGLU(nn.Module):

    def __init__(self, in_dim, hidden_dim, norm="layer"):
        super().__init__()
        self.pre_norm = str2norm(norm, in_dim)
        self.layer1 = nn.Linear(in_dim, hidden_dim * 2)
        self.layer2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = self.pre_norm(x)
        gate, value = self.layer1(x).chunk(2, dim=-1)
        return self.layer2(F.silu(gate) * value)


class GraphEncoder(nn.Module):

    def __init__(self, node_dim=64, edge_dim=64, heads=4, act="silu", norm="layer", dropout=0.1, last_layer=False, is_mol=False):
        super().__init__()
        self.hidden_dim = node_dim
        self.last_layer = last_layer
        self.is_mol = is_mol

        self.node_norm = str2norm(norm, node_dim)
        self.edge_norm = str2norm(norm, edge_dim)
        self.conv = GATv2Conv(node_dim, node_dim, heads, edge_dim=edge_dim, concat=False, dropout=dropout, add_self_loops=False)

        self.ffn = SwiGLU(node_dim, node_dim, norm)
        self.dropout = nn.Dropout(dropout)

        if not last_layer:
            self.h_x_norm = str2norm(norm, node_dim)
            self.edge_proj = MLP([edge_dim + node_dim * 2, edge_dim, edge_dim], act=act, norm=norm, dropout=dropout)

        if is_mol:
            self.mol_proj = Dense(node_dim, node_dim, norm=norm, dropout=dropout)
            self.cls_proj = MLP([node_dim * 3, node_dim * 2, node_dim], act=act, norm=norm, dropout=dropout)
        else:
            self.cls_proj = MLP([node_dim * 2, node_dim * 2, node_dim], act=act, norm=norm, dropout=dropout)

    def forward(self, x, edge_index, edge_attr, batch, cls, mol_x=None, mol_batch=None):
        h = self.node_norm(x)
        e = self.edge_norm(edge_attr)

        x = x + self.dropout(self.conv(h, edge_index, e))
        x = x + self.dropout(self.ffn(x))

        if not self.last_layer:
            src, dst = edge_index
            h_x = self.h_x_norm(x)
            e_in = torch.cat([h_x[src], e, h_x[dst]], dim=-1)
            edge_attr = edge_attr + self.edge_proj(e_in)

        pool_x = global_mean_pool(x, batch)

        if self.is_mol:
            mol_x = self.mol_proj(mol_x)
            pool_m = global_mean_pool(mol_x, mol_batch)
            d_cls = self.cls_proj(torch.cat([cls, pool_x, pool_m], dim=-1))
            return x, edge_attr, cls + d_cls, mol_x + d_cls

        d_cls = self.cls_proj(torch.cat([cls, pool_x], dim=-1))
        return x, edge_attr, cls + d_cls
