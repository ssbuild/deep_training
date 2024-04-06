import torch

class NTKScaledRotary(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, alpha=1, device=None):
        super().__init__()
        base = base * alpha ** (dim / (dim-2))
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
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class NTKScaledRotaryGLM(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048,base=10000, alpha=1, device=None,learnable=False):
        super().__init__()
        base = base * alpha ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = None
        self.max_position_embeddings = max_position_embeddings
        self.learnable = learnable

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else max(seq_len,self.max_position_embeddings)
            t = torch.arange(self.max_seq_len_cached or seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # [sx, 1 (b * np), hn]
            self.register_buffer("cos_cached", emb.cos()[:, None, ...].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[:, None, ...].to(x.dtype), persistent=False)

        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]



class NTKScaledRotaryGLM2(torch.nn.Module):
    def __init__(self, dim,rope_ratio=1.0, original_impl=False,max_position_embeddings=2048,base=10000, alpha=1, device=None):
        super().__init__()
        base = base * alpha ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = rope_ratio
        self.max_position_embeddings = max_position_embeddings

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device) / self.rope_ratio

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )

class NTKScaledRotaryMoss(torch.nn.Module):
    def __init__(self, dim,max_position_embeddings=2048,base=10000,rope_ratio=1.0, alpha=1, original_impl=False, device=None, dtype=None):
        super().__init__()
        # inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        # self.register_buffer("inv_freq", inv_freq)
        self.device = device
        self.dtype = dtype
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = rope_ratio
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len_cached = 0
        self.base=base
        self.alpha = alpha

    def build_cache(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device,
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        self.max_seq_len_cached = seq_len
        base = self.base
        dim = n_elem
        alpha = self.alpha
        base = base * alpha ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device) / self.rope_ratio

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, inv_freq).float()

        cache = torch.cat([torch.sin(idx_theta), torch.cos(idx_theta)], dim=1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, x, offset=0):
        max_seq_len = x.size(-1)
        if max_seq_len > self.max_seq_len_cached:
            self.cache = self.build_cache(max(self.max_position_embeddings,self.max_seq_len_cached,max_seq_len), self.dim,
                                          # dtype=self.inv_freq.dtype,
                                          # device=self.inv_freq.device
                                          dtype=self.dtype,
                                          device=x.device
                                          )
        if self.cache.device != x.device:
            self.cache = self.cache.to(x.device)
        return self.cache[x]