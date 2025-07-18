import torch
import torch.nn as nn
import numpy as np 
from einops import einsum 

def silu(x: torch.Tensor) -> torch.Tensor: 
    return x * torch.sigmoid(x)


class Linear(nn.Module): 
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None): 
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {"device": device, "dtype": dtype} 

        # Initialize weight by using truncted normal method 
        # Make sure that weight size is (out_features, in_features) as in PyTorch
        mean = 0
        std = np.sqrt(2 / (in_features + out_features))
        w_init = torch.empty(out_features, in_features, **self.factory_kwargs)
        nn.init.trunc_normal_(w_init, mean, std, -3 * std, 3 * std)
        self.weight = nn.Parameter(w_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return x @ self.weight.T


class Embedding(nn.Module): 
    def __init__(self, vocab_siz: int, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None): 
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype} 
        self.vocab_siz = vocab_siz 
        self.d_model = d_model
        self.device = device
        self.dtype = dtype 
        
        mean = 0
        std = 1
        w_init = torch.empty(vocab_siz, d_model, **self.factory_kwargs)
        nn.init.trunc_normal_(w_init, mean, std, -3, 3)
        self.weight = nn.Parameter(w_init)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        """
        token_ids: (batch_size, sequence_length)
        output: (batch_size, sequence_length, d_model)
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module): # Root Mean Square Layer Normalization
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None): 
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype} 
        self.d_model = d_model
        self.eps = eps 
        self.device = device
        self.dtype = dtype

        g_init = torch.ones(d_model, **self.factory_kwargs)
        self.gain = nn.Parameter(g_init)
    
    def RMS(self, x: torch.Tensor) -> torch.Tensor: 
        ms = x.pow(2).mean(dim = -1, keepdim = True) 
        rms = torch.sqrt(ms + self.eps)
        return rms
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = (x / self.RMS(x)) * self.gain
        return result.to(in_dtype)


class SwiGLUFFN(nn.Module): 
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None): 
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype} 
        self.d_model = d_model
        self.d_ff = d_ff 
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(d_model, d_ff, **self.factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **self.factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **self.factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.w2(silu(self.w1(x)) * self.w3(x))
    

    