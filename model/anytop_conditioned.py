import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from torch import Tensor

from model.anytop import AnyTop, create_sin_embedding
from model.motion_transformer import GraphMotionDecoderLayer, GraphMotionDecoder
from model.motion_encoder import MotionEncoder


class ConditionedGraphMotionDecoderLayer(GraphMotionDecoderLayer):
    """Extends GraphMotionDecoderLayer with a cross-attention sub-layer after spatial attention.

    Decoder tokens attend to the encoder's latent z, giving each joint access to
    the skeleton-agnostic motion signal while preserving all topology conditioning.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu'):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.cross_attn   = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.norm_cross   = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)

    def _cross_attn_block(self, x, z_embed):
        # x:       [T+1, B, J, D]
        # z_embed: [B, N', K, D']
        T1, B, J, D = x.shape
        Np, K = z_embed.shape[1], z_embed.shape[2]

        # Add temporal PE to z so decoder can align to coarse temporal positions
        positions = torch.arange(Np, device=z_embed.device).view(1, Np, 1).float()
        z_pe = z_embed + create_sin_embedding(positions, D).unsqueeze(2)  # [1, Np, 1, D] broadcasts

        # [B, N', K, D] → [N'*K, B, D]
        z_kv = z_pe.permute(1, 2, 0, 3).reshape(Np * K, B, D)

        # [T+1, B, J, D] → [T+1, J, B, D] → [(T+1)*J, B, D]
        q = x.permute(0, 2, 1, 3).reshape(T1 * J, B, D)

        out, _ = self.cross_attn(q, z_kv, z_kv)   # [(T+1)*J, B, D]

        # [(T+1)*J, B, D] → [T+1, J, B, D] → [T+1, B, J, D]
        out = out.view(T1, J, B, D).permute(0, 2, 1, 3)
        return self.dropout_cross(out)

    def forward(self,
                tgt: Tensor,
                timesteps_emb: Tensor,
                topology_rel: Tensor,
                edge_rel: Tensor,
                edge_key_emb,
                edge_query_emb,
                edge_value_emb,
                topo_key_emb,
                topo_query_emb,
                topo_value_emb,
                spatial_mask: Optional[Tensor] = None,
                temporal_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                y=None,
                z_embed: Optional[Tensor] = None) -> Tensor:
        x  = tgt
        bs = x.shape[1]
        x = x + self.embed_timesteps(timesteps_emb).view(1, bs, 1, self.d_model)
        x = self.norm1(x + self._spatial_mha_block(
            x, topology_rel, edge_rel,
            edge_key_emb, edge_query_emb, edge_value_emb,
            topo_key_emb, topo_query_emb, topo_value_emb,
            spatial_mask, tgt_key_padding_mask, y))
        if z_embed is not None:
            x = self.norm_cross(x + self._cross_attn_block(x, z_embed))
        x = self.norm2(x + self._temporal_mha_block_sin_joint(x, temporal_mask, tgt_key_padding_mask))
        x = self.norm3(x + self._ff_block(x))
        return x


class ConditionedGraphMotionDecoder(GraphMotionDecoder):
    """Extends GraphMotionDecoder to thread z_embed through every layer."""

    def forward(self, tgt: Tensor, timesteps_embs: Tensor, memory: Tensor,
                spatial_mask: Optional[Tensor] = None,
                temporal_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                y=None, get_layer_activation=-1,
                z_embed: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, dict]]:
        topology_rel = y['graph_dist'].long().to(tgt.device)
        edge_rel     = y['joints_relations'].long().to(tgt.device)
        output = tgt
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations = dict()
        for layer_ind, mod in enumerate(self.layers):
            edge_value_emb     = self.edge_value_emb     if self.value_emb_flag else None
            topology_value_emb = self.topology_value_emb if self.value_emb_flag else None
            output = mod(
                output, timesteps_embs, topology_rel, edge_rel,
                self.edge_key_emb, self.edge_query_emb, edge_value_emb,
                self.topology_key_emb, self.topology_query_emb, topology_value_emb,
                spatial_mask, temporal_mask,
                tgt_key_padding_mask, memory_key_padding_mask,
                y, z_embed=z_embed)
            if layer_ind == get_layer_activation:
                activations[layer_ind] = output.clone()
        if self.norm is not None:
            output = self.norm(output)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output


class AnyTopConditioned(AnyTop):
    """AnyTop extended with a source motion encoder and cross-attention conditioning.

    Training (self-supervised): encode X on S → z → decode X on S
    Inference:                  encode X on S_src → z → decode on S_tgt
    """
    def __init__(self, max_joints, feature_len,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", t5_out_dim=512, root_input_feats=13,
                 enc_num_queries=4, enc_fsq_dims=4, enc_fsq_levels=5,
                 z_drop_prob=0.1, **kargs):
        super().__init__(max_joints, feature_len, latent_dim, ff_size, num_layers,
                         num_heads, dropout, activation, t5_out_dim, root_input_feats, **kargs)

        # Replace decoder with conditioned version
        conditioned_layer = ConditionedGraphMotionDecoderLayer(
            d_model=latent_dim, nhead=num_heads,
            dim_feedforward=ff_size, dropout=dropout, activation=activation)
        self.seqTransDecoder = ConditionedGraphMotionDecoder(
            conditioned_layer, num_layers=num_layers, value_emb=self.value_emb)

        # Source motion encoder — same d_model as decoder (no projection needed)
        self.encoder = MotionEncoder(
            feature_len=feature_len,
            d_model=latent_dim,
            num_queries=enc_num_queries,
            num_heads=num_heads,
            fsq_dims=enc_fsq_dims,
            fsq_levels=enc_fsq_levels,
            dropout=dropout)

        # CFG null embedding — broadcast over batch and temporal dims
        self.null_z = nn.Parameter(torch.zeros(1, 1, enc_num_queries, latent_dim))
        self.z_drop_prob = z_drop_prob

    def forward(self, x, timesteps, get_layer_activation=-1, y=None):
        # x: [B, J, 13, T]
        joints_mask      = y['joints_mask'].to(x.device)
        temp_mask        = y['mask'].to(x.device)
        tpos_first_frame = y['tpos_first_frame'].to(x.device).unsqueeze(0)

        bs, njoints, nfeats, nframes = x.shape

        # Retrieve or fall back to null embedding
        z_embed = y.get('z', None)
        if z_embed is not None:
            z_embed = z_embed.to(x.device)
        else:
            z_embed = self.null_z.expand(bs, 1, self.encoder.num_queries, self.latent_dim)

        # CFG dropout: replace z with null for a random fraction of training samples
        if self.training and self.z_drop_prob > 0:
            drop = torch.rand(bs, device=x.device) < self.z_drop_prob   # [B]
            null = self.null_z.expand_as(z_embed)
            z_embed = torch.where(drop[:, None, None, None], null, z_embed)

        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.latent_dim)[0]
        x = self.input_process(x, tpos_first_frame, y['joints_names_embs'], y['crop_start_ind'])

        spatial_mask = 1.0 - joints_mask[:, 0, 0, 1:, 1:]
        spatial_mask = (spatial_mask.unsqueeze(1).unsqueeze(1)
                        .repeat(1, nframes + 1, self.num_heads, 1, 1)
                        .reshape(-1, self.num_heads, njoints, njoints))
        temporal_mask = (1.0 - temp_mask.repeat(1, njoints, self.num_heads, 1, 1)
                         .reshape(-1, nframes + 1, nframes + 1).float())
        spatial_mask[spatial_mask == 1.0]   = -1e9
        temporal_mask[temporal_mask == 1.0] = -1e9

        output = self.seqTransDecoder(
            tgt=x, timesteps_embs=timesteps_emb, memory=None,
            spatial_mask=spatial_mask, temporal_mask=temporal_mask,
            y=y, get_layer_activation=get_layer_activation,
            z_embed=z_embed)

        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations = output[1]
            output      = output[0]
        output = self.output_process(output)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output
