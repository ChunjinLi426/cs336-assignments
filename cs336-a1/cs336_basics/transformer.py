import torch
import torch.nn as nn
import numpy as np 
from einops import einsum, rearrange 


def silu(x: torch.Tensor) -> torch.Tensor: 
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor: 
    x_stable = x - torch.max(x, dim = dim, keepdim = True).values
    x_exp = torch.exp(x_stable)
    return x_exp / torch.sum(x_exp, dim = dim, keepdim = True)


def scaled_dot_product_attention(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    V: torch.Tensor, 
    mask: torch.Tensor | None = None
) -> torch.Tensor: 
    d_k = Q.shape[-1]
    attention_score = einsum(
        Q, K, 
        "... q_len d_k, ... k_len d_k -> ... q_len k_len"
    ) / np.sqrt(d_k)
    if mask != None: 
        attention_score = attention_score.masked_fill(~mask, float('-inf'))
    return softmax(attention_score, -1) @ V

class Linear(nn.Module): 
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ): 
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
    def __init__(
        self, 
        vocab_siz: int, 
        d_model: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ): 
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
    def __init__(self, 
        d_model: int, 
        eps: float = 1e-5,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ): 
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
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ): 
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
    

class RoPE(nn.Module): # Rotary Positional Embedding
    def __init__(
        self, 
        theta: float, 
        d_k: int, 
        max_seq_len: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.factory_kwargs = {"device": device, "dtype": dtype} 
        self.theta = theta
        self.d_k = d_k 
        self.device = device
        self.dtype = dtype

        R_list = []
        for i in range(max_seq_len): 
            R_i = []
            for k in range(d_k // 2): 
                theta_ik = torch.tensor(i / (theta ** (2 * k / d_k)), **self.factory_kwargs)
                cos_t = torch.cos(theta_ik)
                sin_t = torch.sin(theta_ik)
                R_i.append(torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], **self.factory_kwargs))

            R_list.append(torch.block_diag(*R_i).to(**self.factory_kwargs))

        self.R = torch.stack(R_list).to(**self.factory_kwargs)
        self.register_buffer("rotation_matrix_table", self.R, persistent = False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor: 
        *prefix_dims, seq_len, d_k = x.shape
        if token_positions == None: 
            token_positions = torch.arange(seq_len, device = self.device)
        results = einsum(
            x, self.R[token_positions], 
            "... d_in, ... d_out d_in -> ... d_out"
        )
        return results


class MultiheadSelfAttention(nn.Module): 
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        use_rope: bool = False, 
        theta: float | None = None, 
        max_seq_len: int | None = None, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ): 
        super().__init__()

        self.factory_kwargs = {"device": device, "dtype": dtype} 
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.device = device
        self.dtype = dtype

        self.q_proj = Linear(d_model, num_heads * self.d_k, **self.factory_kwargs)
        self.k_proj = Linear(d_model, num_heads * self.d_k, **self.factory_kwargs)
        self.v_proj = Linear(d_model, num_heads * self.d_v, **self.factory_kwargs)
        self.o_proj = Linear(num_heads * self.d_v, d_model, **self.factory_kwargs)

        self.use_rope = use_rope
        if use_rope: 
            self.rope = RoPE(theta, self.d_k, max_seq_len)
        else: 
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor: 
        seq_len = x.shape[-2]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(
            q, "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads
        )
        k = rearrange(
            k, "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads
        )
        v = rearrange(
            v, "... seq_len (h d_v) -> ... h seq_len d_v", h = self.num_heads
        )
        if self.use_rope: 
            if token_positions == None: 
                token_positions = torch.arange(seq_len, device = self.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        mask = ~torch.triu(torch.ones(seq_len, seq_len, **self.factory_kwargs), diagonal = 1).bool()
        attention_score = scaled_dot_product_attention(q, k, v, mask)
        attention_score = rearrange(
            attention_score, "... h seg_len d_k -> ... seg_len (h d_k)"
        ) 
        return self.o_proj(attention_score)


class TransformerBlock(nn.Module): # pre-norm Transformer block
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        use_rope: bool = False, 
        theta: float | None = None, 
        max_seq_len: int | None = None, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ): 
        super().__init__()

        self.factory_kwargs = {"device": device, "dtype": dtype} 
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.attn = MultiheadSelfAttention(d_model, num_heads, use_rope, theta, max_seq_len, **self.factory_kwargs)
        self.ffn = SwiGLUFFN(d_model, d_ff, **self.factory_kwargs)
        self.ln1 = RMSNorm(d_model, **self.factory_kwargs)
        self.ln2 = RMSNorm(d_model, **self.factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x