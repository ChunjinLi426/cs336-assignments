import torch 
from einops import rearrange
from .transformer import softmax


def cross_entropy(o: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: 
    o = rearrange(o, "batch ... vocab -> (batch ...) vocab")
    targets = rearrange(targets, "batch ... -> (batch ...)")
    o_stable = o - torch.max(o, dim = -1, keepdim = True).values
    loss = -o_stable[torch.arange(o.shape[0]), targets] + torch.log(torch.sum(torch.exp(o_stable), dim = -1, keepdim = True))
    return loss.mean()