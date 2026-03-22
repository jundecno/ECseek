import rootutils

root_path = rootutils.setup_root(__file__, indicator=".root", pythonpath=True)
import math
import random
import numpy as np
import torch
from torch import Tensor, nn, cat
from torch.nn import Module, ModuleList, Sequential, ModuleDict
from torch.nn import Linear, Dropout, Identity
import torch.nn.functional as F
from functools import partial
from einops import repeat, rearrange


def exists(v):
    return v is not None


def default(v, d) -> int:
    return v if exists(v) else d


def tuple_cat(A, B, dim=1):
    return (torch.cat([a, b], dim=dim) for a, b in zip(A, B))


def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def weights_init(m: Module):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.RMSNorm):
        if hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
            nn.init.constant_(m.weight, 1)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)


def str2norm(norm_str="", dim=64):
    norm_str = norm_str.lower().strip()
    if norm_str == "layer":
        return nn.LayerNorm(dim)
    elif norm_str == "batch":
        return nn.BatchNorm1d(dim)
    elif norm_str == "instance":
        return nn.InstanceNorm1d(dim)
    elif norm_str == "rms":
        return nn.RMSNorm(dim)
    else:
        return nn.Identity()


def str2act(act_str=""):
    act_str = act_str.lower().strip()
    if act_str == "relu":
        return nn.ReLU()
    elif act_str == "elu":
        return nn.ELU()
    elif act_str == "leaky":
        return nn.LeakyReLU()
    elif act_str == "sigmoid":
        return nn.Sigmoid()
    elif act_str == "gelu":
        return nn.GELU()
    elif act_str == "silu":
        return nn.SiLU()
    elif act_str == "mish":
        return nn.Mish()
    elif act_str == "selu":
        return nn.SELU()
    elif act_str == "tanh":
        return nn.Tanh()
    else:
        return nn.Identity()


def gaussian_rbf(inputs: Tensor, offsets, widths):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


def module_to_dict(module):
    if not list(module.named_children()):
        return str(module)
    d = {}
    for name, sub_module in module.named_children():
        d[name] = module_to_dict(sub_module)
    return d


def residual_fn(x, res):
    out = {}
    for key in x:
        out[key] = x[key] + res[key]
    return out
