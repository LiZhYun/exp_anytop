import math
import torch
import torch.nn as nn
from model.anytop import create_sin_embedding


class RestPE(nn.Module):
    """Encode rest-pose bone offsets into D'-dimensional vectors.

    Applies sinusoidal encoding per spatial coordinate (no topology information),
    then maps through a 2-layer MLP.
    """
    def __init__(self, d_model, num_frequencies=8):
        super().__init__()
        self.num_frequencies = num_frequencies
        enc_dim = 3 * 2 * num_frequencies  # 48
        self.mlp = nn.Sequential(
            nn.Linear(enc_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, offsets):
        # offsets: [B, J, 3]
        freqs = 2.0 ** torch.arange(self.num_frequencies, device=offsets.device, dtype=offsets.dtype)
        enc = offsets[..., None] * math.pi * freqs   # [B, J, 3, num_freq]
        enc = torch.cat([torch.sin(enc), torch.cos(enc)], dim=-1)  # [B, J, 3, 2*num_freq]
        enc = enc.flatten(-2)                         # [B, J, 48]
        return self.mlp(enc)                          # [B, J, D']


class AttentionPool(nn.Module):
    """Compress J joint tokens into K functional slots per frame via learned cross-attention.

    K query vectors are global parameters shared across all frames and all skeletons.
    """
    def __init__(self, d_model, num_queries, num_heads, dropout):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

    def forward(self, h, joints_mask, return_attn=False):
        # h:           [B, T, J, D']
        # joints_mask: [B, J]  True=real joint, False=padding
        B, T, J, D = h.shape
        h_flat = h.reshape(B * T, J, D)
        queries = self.queries.unsqueeze(0).expand(B * T, -1, -1)             # [B*T, K, D']
        key_padding_mask = ~joints_mask.unsqueeze(1).expand(B, T, J).reshape(B * T, J)
        out, attn = self.attn(queries, h_flat, h_flat, key_padding_mask=key_padding_mask,
                              need_weights=return_attn, average_attn_weights=True)
        if return_attn:
            return out.view(B, T, self.num_queries, D), attn.view(B, T, self.num_queries, J)
        return out.view(B, T, self.num_queries, D)                             # [B, T, K, D']


class TemporalCNN(nn.Module):
    """Downsample temporal resolution N→N/4, processing each spatial slot independently."""
    def __init__(self, d_model):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=1),
            nn.GELU(),
        )

    def forward(self, s):
        # s: [B, T, K, D']
        B, T, K, D = s.shape
        x = s.permute(0, 2, 3, 1).reshape(B * K, D, T)   # [B*K, D', T]
        x = self.convs(x)                                  # [B*K, D', T/4]
        T_out = x.shape[-1]
        return x.view(B, K, D, T_out).permute(0, 3, 1, 2)  # [B, T/4, K, D']


class FSQ(nn.Module):
    """Finite Scalar Quantization with straight-through gradient estimator.

    No learnable parameters; no commitment loss required.
    Quantizes each dimension to L uniformly spaced levels in [-1, 1].
    """
    def __init__(self, levels):
        super().__init__()
        self.levels = levels

    def forward(self, x, return_codes=False):
        x_bounded = torch.tanh(x)
        x_scaled  = (x_bounded + 1) / 2 * (self.levels - 1)      # [0, L-1]
        x_round   = torch.round(x_scaled)
        x_st      = x_scaled + (x_round - x_scaled).detach()      # straight-through
        out = x_st / (self.levels - 1) * 2 - 1                    # back to [-1, 1]
        if return_codes:
            return out, x_round.long()
        return out


class MotionEncoder(nn.Module):
    """Encode source motion into a skeleton-agnostic latent z.

    Receives motion features and rest-pose geometry only — no topology (R_S, D_S).
    Output z_embed is used by the conditioned decoder's cross-attention layers.
    """
    def __init__(self, feature_len=13, d_model=128, num_queries=4,
                 num_heads=4, fsq_dims=4, fsq_levels=5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.root_emb   = nn.Linear(feature_len, d_model)
        self.joint_emb  = nn.Linear(feature_len, d_model)
        self.rest_pe    = RestPE(d_model)
        self.attn_pool  = AttentionPool(d_model, num_queries, num_heads, dropout)
        self.temporal_cnn = TemporalCNN(d_model)
        self.pre_fsq    = nn.Linear(d_model, fsq_dims)
        self.fsq        = FSQ(fsq_levels)
        self.post_fsq   = nn.Linear(fsq_dims, d_model)

    def forward(self, source_motion, source_offsets, source_mask, return_intermediates=False):
        # source_motion:  [B, J, 13, T]
        # source_offsets: [B, J, 3]
        # source_mask:    [B, J]  True=real joint, False=padding
        x = source_motion.permute(0, 3, 1, 2)                 # [B, T, J, 13]
        root  = self.root_emb(x[:, :, 0:1])                   # [B, T, 1, D']
        rest  = self.joint_emb(x[:, :, 1:])                   # [B, T, J-1, D']
        x_emb = torch.cat([root, rest], dim=2)                 # [B, T, J, D']

        x_emb = x_emb + self.rest_pe(source_offsets).unsqueeze(1)  # RestPE: [B, 1, J, D']

        T = x_emb.shape[1]
        positions = torch.arange(T, device=x_emb.device).view(1, T, 1).float()
        x_emb = x_emb + create_sin_embedding(positions, self.d_model).unsqueeze(2)  # [1, T, 1, D']

        if return_intermediates:
            s, attn_weights = self.attn_pool(x_emb, source_mask, return_attn=True)
        else:
            s = self.attn_pool(x_emb, source_mask)

        t_out = self.temporal_cnn(s)
        z_pre = self.pre_fsq(t_out)

        if return_intermediates:
            z_quant, z_codes = self.fsq(z_pre, return_codes=True)
            z_out = self.post_fsq(z_quant)
            return z_out, {'attn_weights': attn_weights, 'z_codes': z_codes}
        return self.post_fsq(self.fsq(z_pre))
