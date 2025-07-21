import torch 
import torch.nn as nn 
import numpy as np 
import math
from typing import Optional
from einops import rearrange
from cs336_basics.transformer.model import softmax


def cross_entropy(o: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: 
    o = rearrange(o, "batch ... vocab -> (batch ...) vocab")
    targets = rearrange(targets, "batch ... -> (batch ...)")
    o_stable = o - torch.max(o, dim = -1, keepdim = True).values
    loss = -o_stable[torch.arange(o.shape[0]), targets] + torch.log(torch.sum(torch.exp(o_stable), dim = -1, keepdim = True))
    return loss.mean()


class SGD(torch.optim.Optimizer): # Stochastic Gradient Descent Optimizer
    def __init__(
        self, 
        params: nn.Parameter,
        lr: float = 1e-3, 
    ): 
        if lr < 0: 
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None) -> None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:      
                if p.grad is None: 
                    continue 
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        
        return loss
                
