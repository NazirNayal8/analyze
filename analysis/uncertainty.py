import torch
from scipy.stats import entropy


def entropy_torch(p, dim=-1):
    p = p / p.sum(dim=dim, keepdims=True)
    return torch.special.entr(p).sum(dim=dim)