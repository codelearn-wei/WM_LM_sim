import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Mamba(nn.Module):

    def __init__(
            self,
            d_model,
            d_state=64,
            d_conv=4,
            conv_init=None,
            expand=2,
            headdim=64,
            ngroups=1,
            A_init_range=(1, 16),
            dt_min=0.001,
            dt_max=0.1,
            dt_init_floor=1e-4,
            dt_limit=(0.0, float("inf")),
            learnable_init_states=False,
            activation="swish",
            bias=False,
            conv_bias=True,
            # Fused kernel and sharding options
            chunk_size=256,
            use_mem_eff_path=True,
            layer_idx=None,  # Absorb kwarg for general module
            device=None,
            dtype=None,
    ):
        super(Mamba, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.act = nn.GELU()

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

        channels = self.nheads * self.headdim + self.ngroups * self.d_state * 2
        self.causal_conv = CausalConv1d(in_channels=channels, out_channels=channels, kernel_size=4)

        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.in_proj_1 = nn.Linear(self.d_model, d_in_proj, bias=bias)
        self.in_proj_2 = nn.Linear(self.d_model, self.d_inner, bias=bias)
        self.in_proj_3 = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.layernorm = nn.LayerNorm(self.d_inner)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, u):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj_1(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        batch, seqlen, _ = zxbcdt.shape
        dim = self.nheads * self.headdim
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * self.ngroups * self.d_state - self.nheads) // 2
        zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + self.ngroups * self.d_state * 2, self.nheads],
                                      dim=-1)
        dt = softplus(dt)
        A = dt * A[None, None, :]
        xBC_conv = rearrange(self.causal_conv(rearrange(xBC, "b s d -> b d s")), "b d s -> b s d")
        x, B, C = torch.split(xBC_conv, [dim, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=self.nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)
        # z = rearrange(z, "b l (h p) -> b l h p", h=self.nheads) if z is not None else None
        output, final_state = self.ssd(x, A, B, C, block_len=int(seqlen/4))
        output = rearrange(output, "b s h p -> b s (h p)")

        u = self.act(self.in_proj_2(u))
        u = self.in_proj_3(self.layernorm(u * output))

        return u

        # output = self.act(self.in_proj_2(output))
        # output = self.dropout(self.in_proj_3(output))
        #
        # return self.layernorm(u + output)

    def segsum(self, x):
        """Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
        which is equivalent to a scalar SSM."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd(self, X, A, B, C, block_len=6, initial_states=None):
        """
        Arguments:
            X: (batch, length, n_heads, d_head)
            A: (batch, length, n_heads)
            B: (batch, length, n_heads, d_state)
            C: (batch, length, n_heads, d_state)
        Return:
            Y: (batch, length, n_heads, d_head)
        """
        assert X.dtype == A.dtype == B.dtype == C.dtype
        assert X.shape[1] % block_len == 0

        # Rearrange into blocks/chunks
        X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

        A = rearrange(A, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)橙色
        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

        # 2. Compute the state for each intra-chunk绿色
        # (right term of low-rank factorization for off-diagonal blocks; B terms)
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

        # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries黄色
        # (middle term of factorization of off-diag blocks; A terms)
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        # 4. Compute state -> output conversion per chunk蓝色
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        return Y, final_state
      
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Calculate the padding to ensure causality
        self.padding = (kernel_size - 1) * dilation

        # 1D Convolution layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                              dilation=dilation, padding=self.padding)
        self.act = nn.PReLU()

    def forward(self, x):
        # Apply convolution
        out = self.act(self.conv(x))

        # Remove the future steps from padding to ensure causality
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out
      
def softplus(dt):
    # When dt <= 20, apply the log1p(exp(dt)) operation
    dt = torch.where(dt <= 20.0, torch.log1p(torch.exp(dt)), dt)
    return dt