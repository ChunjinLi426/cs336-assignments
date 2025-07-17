import torch
import torch.nn as nn
import numpy as np 
from einops import einsum 

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
        w_init = torch.empty(vocab_siz, d_model)
        nn.init.trunc_normal_(w_init, mean, std, -3, 3)
        self.weight = nn.Parameter(w_init)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        """
        token_ids: (batch_size, sequence_length)
        output: (batch_size, sequence_length, d_model)
        """
        return self.weight[token_ids]


    