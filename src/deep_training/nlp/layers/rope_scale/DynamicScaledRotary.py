import math
import torch

class DynamicScaledRotary(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, ntk=False, device=None):
        super().__init__()
        self.ntk = ntk
        self.base = base
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            if self.ntk:
                base = self.base * ((self.ntk * seq_len / self.max_position_embeddings) - (self.ntk - 1)) ** (self.dim / (self.dim-2))
                inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim))
                self.register_buffer("inv_freq", inv_freq)
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            if not self.ntk:
                t *= self.max_position_embeddings / seq_len
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )




class DynamicScaledRotaryGLM(torch.nn.Module):
    def __init__(self, dim, base=10000, ntk=False,device=None):
        super().__init__()
        self.ntk = ntk
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = None


    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            if self.ntk:
                base = self.base * ((self.ntk * seq_len / self.max_position_embeddings) - (self.ntk - 1)) ** (self.dim / (self.dim-2))
                inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim))
                self.register_buffer("inv_freq", inv_freq)

            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # [sx, 1 (b * np), hn]
            self.register_buffer("cos_cached", emb.cos()[:, None, ...].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[:, None, ...].to(x.dtype), persistent=False)

        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]



class DynamicScaledRotaryGLM2(torch.nn.Module):
    def __init__(self, dim,original_impl=False,base=10000,ntk=False, device=None, dtype=None):
        super().__init__()
        self.ntk = ntk
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.max_seq_len_cached = None


    def forward_impl(
            self, seq_len:  int, dtype: torch.dtype, device: torch.device
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            if self.ntk:
                base = self.base * ((self.ntk * seq_len / self.max_position_embeddings) - (self.ntk - 1)) ** (
                            self.dim / (self.dim - 2))
                inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
                self.register_buffer("inv_freq", inv_freq)


            # Create position indexes `[0, 1, ..., seq_len - 1]`
            seq_idx = torch.arange(seq_len, dtype=dtype, device=device) / self.scale

            # Calculate the product of position index and $\theta_i$
            idx_theta = torch.outer(seq_idx, inv_freq).float()

            self.register_buffer("cos_cached", idx_theta.cos().to(idx_theta.dtype), persistent=False)
            self.register_buffer("sin_cached", idx_theta.sin().to(idx_theta.dtype), persistent=False)
            self.max_seq_len_cached = seq_len

        cache = torch.stack([self.cos_cached, self.sin_cached], dim=-1)
        # cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len,  dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )