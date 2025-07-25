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

    def step(self, closure: Optional[callable] = None):
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
                

class AdamW(torch.optim.Optimizer): 
    def __init__(
        self, 
        params: nn.Parameter, 
        lr: float = 1e-3, 
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8, 
        weight_decay: float = 0.01
    ): 
        if lr < 0: 
            raise ValueError(f"Invalid learning rate: {lr}")
        (beta1, beta2) = betas
        if beta1 < 0 or beta1 >= 1: 
            raise ValueError(f"Invalid beta1: {beta1}")
        if beta2 < 0 or beta2 >= 1: 
            raise ValueError(f"Invalid beta2: {beta2}")
        if weight_decay < 0: 
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr, 
            "betas": betas, 
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            (beta1, beta2) = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group.get("eps", 1e-8)
            for p in group["params"]:      
                if p.grad is None: 
                    continue 
                
                state = self.state[p]
                t = state.get("t", 0) + 1
                g = p.grad.data
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr_t * m / (v.sqrt() + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t
                state["m"] = m
                state["v"] = v
        
        return loss
