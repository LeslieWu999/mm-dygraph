"""TST-MF: the target-aware spatio-temporal multimodal fusion model, and its ablation variants.

Each side of an edge is encoded in three steps (see ``TSTMF._process_single_side``):

1. the historical evolution tracker encodes the node's interaction history,
2. the structural neighbor aggregator summarises its recent neighbors,
3. a two-stage cross-attention, queried by the embedding of the prediction target (src, dst),
   first selects context from (1) and (2), then uses that context to select among the modalities.
"""
import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from models.DyGFormer import DyGFormer


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Union[int, list[int]],
        num_hidden_layers: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_layers = [hidden_dims] * num_hidden_layers
        else:
            hidden_layers = hidden_dims

        layers = []

        current_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = MLP(dim, dim, dim * 2, 1)
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor, context_mask_valid: torch.Tensor | None = None) -> torch.Tensor:
        """
        query: (batch, 1, dim)
        context: (batch, seq_len, dim)
        context_mask_valid: (batch, seq_len), 1 = valid, 0 = padding
        return: (batch, 1, dim)
        """
        key_padding_mask = None
        if context_mask_valid is not None:
            key_padding_mask = context_mask_valid == 0

        attn_out, _ = self.attn(query=query, key=context, value=context, key_padding_mask=key_padding_mask)
        x = self.norm(query + self.dropout(attn_out))
        x = self.norm_ffn(x + self.dropout(self.ffn(x)))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[: x.size(1)].unsqueeze(0)


class HistEvoTracker(nn.Module):
    """Transformer encoder over a node's interaction history."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 10, num_heads: int = 4, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()

        self.in_proj = MLP(input_dim=input_dim, output_dim=hidden_dim, hidden_dims=hidden_dim * 2)
        self.pos = SinusoidalPositionalEncoding(hidden_dim, max_len=max_len)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x_seq: torch.Tensor, mask_valid: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_seq: (batch, seq_len, input_dim)
        mask_valid: (batch, seq_len), 1 = valid, 0 = padding
        return: (batch, seq_len, hidden_dim)
        """
        x = self.in_proj(x_seq)
        x = x + self.pos(x)
        x = self.drop(x)

        key_padding_mask = None
        if mask_valid is not None:
            key_padding_mask = mask_valid == 0

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(h)


class StructNeighborAgg(nn.Module):
    """Attends from a node to its recent neighbors."""

    def __init__(self, input_dim, output_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.center_proj = nn.Linear(input_dim, output_dim)
        self.neighbor_proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, center_node_feat: torch.Tensor, neighbor_feats: torch.Tensor, neighbor_mask: torch.Tensor = None) -> torch.Tensor:
        """
        center_node_feat: (batch, 1, node_dim)
        neighbor_feats: (batch, num_neighbors, node_dim)
        neighbor_mask: (batch, num_neighbors), True = ignore
        return: (batch, 1, output_dim)
        """
        q = self.center_proj(center_node_feat)
        k = self.neighbor_proj(neighbor_feats)
        attn_output, _ = self.attn(query=q, key=k, value=k, key_padding_mask=neighbor_mask)
        return self.norm(q + attn_output)


class ContextCrossAttnLayer(nn.Module):
    """First fusion stage: the target embedding queries the history and structure embeddings."""

    def __init__(self, task_dim: int, hist_dim: int, struct_dim: int, output_dim: int, num_heads: int):
        super().__init__()

        self.task_proj = MLP(task_dim, output_dim, output_dim * 2, 1)
        self.hist_proj = MLP(hist_dim, output_dim, output_dim * 2, 1)
        self.struct_proj = MLP(struct_dim, output_dim, output_dim * 2, 1)

        self.cross_attn = CrossAttentionBlock(output_dim, num_heads)

    def forward(self, task_embd: torch.Tensor, hist_embd: torch.Tensor, struct_embd: torch.Tensor, hist_mask_valid: torch.Tensor | None = None) -> torch.Tensor:
        """
        task_embd: (batch, 1, task_dim)
        hist_embd: (batch, seq_len, hist_dim)
        struct_embd: (batch, 1, struct_dim)
        return: (batch, 1, output_dim)
        """
        task_embd = self.task_proj(task_embd)
        hist_embd = self.hist_proj(hist_embd)
        struct_embd = self.struct_proj(struct_embd)
        context_seq = torch.cat([hist_embd, struct_embd], dim=1)

        context_mask_valid = None
        if hist_mask_valid is not None:
            struct_valid = torch.ones(hist_mask_valid.size(0), 1, device=hist_mask_valid.device, dtype=hist_mask_valid.dtype)
            context_mask_valid = torch.cat([hist_mask_valid, struct_valid], dim=1)

        return self.cross_attn(query=task_embd, context=context_seq, context_mask_valid=context_mask_valid)


class ModalCrossAttnLayer(nn.Module):
    """Second fusion stage: the context from the first stage queries the raw modality features."""

    def __init__(self, context_dim: int, modal_dims: list[int], output_dim: int, num_heads: int):
        super().__init__()

        self.context_proj = MLP(context_dim, output_dim, output_dim * 2, 1)
        self.modal_projs = nn.ModuleList([nn.Linear(m_dim, output_dim) for m_dim in modal_dims])
        self.cross_attn = CrossAttentionBlock(output_dim, num_heads)

    def forward(self, context_embd: torch.Tensor, modality_embds: list[torch.Tensor]) -> torch.Tensor:
        """
        context_embd: (batch, 1, context_dim)
        modality_embds: list of (batch, 1, modal_dim)
        return: (batch, 1, output_dim)
        """
        context_embd = self.context_proj(context_embd)
        projected = [self.modal_projs[i](embd) for i, embd in enumerate(modality_embds)]
        return self.cross_attn(query=context_embd, context=torch.cat(projected, dim=1))


class TSTMF(DyGFormer):
    """TST-MF. Reuses DyGFormer's neighbor sampling, padding, co-occurrence encoding and patching."""

    def __init__(
        self,
        model_cls,
        node_raw_features_list: list[np.ndarray],
        edge_raw_features_list: list[np.ndarray],
        neighbor_sampler,
        time_feat_dim: int,
        channel_embedding_dim: int,
        patch_size: int = 1,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_input_sequence_length: int = 512,
        device: str = "cpu",
    ):
        """
        :param model_cls: ignored; TST-MF is its own backbone (the wrapper call convention passes it first)
        :param node_raw_features_list: per-modality node features, each (num_nodes, dim)
        :param edge_raw_features_list: per-modality edge features, each (num_edges, dim)
        """
        node_raw_features = np.concat(node_raw_features_list, axis=-1)
        edge_raw_features = np.concat(edge_raw_features_list, axis=-1)

        self.node_raw_features_list = [torch.from_numpy(nf.astype(np.float32)).to(device) for nf in node_raw_features_list]
        # Only the shapes of the edge tables are used, so they stay on the host.
        self.edge_raw_features_list = [torch.from_numpy(ef.astype(np.float32)) for ef in edge_raw_features_list]

        super().__init__(
            node_raw_features,
            edge_raw_features,
            neighbor_sampler,
            time_feat_dim,
            channel_embedding_dim,
            patch_size,
            num_layers=0,
            num_heads=num_heads,
            dropout=dropout,
            max_input_sequence_length=max_input_sequence_length,
            device=device,
        )

        node_feat_dim = node_raw_features.shape[1]
        self.hidden_dim = hidden_dim = 4 * self.channel_embedding_dim

        self.hist_evo_tracker = HistEvoTracker(
            input_dim=self.num_channels * self.channel_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.struct_neighbor_agg = StructNeighborAgg(
            input_dim=self.node_feat_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.task_proj = nn.Linear(2 * self.node_feat_dim, hidden_dim)
        self.context_attn_layer = ContextCrossAttnLayer(
            task_dim=hidden_dim,
            hist_dim=hidden_dim,
            struct_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.modal_attn_layer = ModalCrossAttnLayer(
            context_dim=self.hidden_dim,
            modal_dims=[nf.shape[1] for nf in node_raw_features_list],
            output_dim=self.hidden_dim,
            num_heads=num_heads,
        )

        # Not used in the forward pass. Kept so that parameter initialisation and checkpoints stay
        # compatible with the published runs.
        self.edge_proj_modal = nn.Linear(self.edge_feat_dim, self.node_feat_dim)

        self.final_proj = MLP(input_dim=hidden_dim, output_dim=node_feat_dim, hidden_dims=hidden_dim, num_hidden_layers=1)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        src_n_ids, src_e_ids, src_n_times = self.neighbor_sampler.get_all_first_hop_neighbors(src_node_ids, node_interact_times)
        dst_n_ids, dst_e_ids, dst_n_times = self.neighbor_sampler.get_all_first_hop_neighbors(dst_node_ids, node_interact_times)

        src_pad_n_ids, src_pad_e_ids, src_pad_n_times = self.pad_sequences(
            src_node_ids, node_interact_times, src_n_ids, src_e_ids, src_n_times,
            patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length,
        )
        dst_pad_n_ids, dst_pad_e_ids, dst_pad_n_times = self.pad_sequences(
            dst_node_ids, node_interact_times, dst_n_ids, dst_e_ids, dst_n_times,
            patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length,
        )
        src_raw_mask_valid = (torch.as_tensor(src_pad_n_ids, device=self.device) != 0).int()
        dst_raw_mask_valid = (torch.as_tensor(dst_pad_n_ids, device=self.device) != 0).int()

        src_co_occur, dst_co_occur = self.neighbor_co_occurrence_encoder(src_pad_n_ids, dst_pad_n_ids)

        src_raw_feats = self.get_features(node_interact_times, src_pad_n_ids, src_pad_e_ids, src_pad_n_times, self.time_encoder)
        dst_raw_feats = self.get_features(node_interact_times, dst_pad_n_ids, dst_pad_e_ids, dst_pad_n_times, self.time_encoder)

        src_patches, src_patch_mask = self._process_patches_into_sequence(src_raw_feats, src_co_occur, src_raw_mask_valid)
        dst_patches, dst_patch_mask = self._process_patches_into_sequence(dst_raw_feats, dst_co_occur, dst_raw_mask_valid)

        # (batch, 1, hidden_dim): the prediction target that guides both fusion stages
        h_task = self.task_proj(
            torch.concat(
                [
                    self.node_raw_features[torch.from_numpy(src_node_ids).to(self.device)],
                    self.node_raw_features[torch.from_numpy(dst_node_ids).to(self.device)],
                ],
                dim=1,
            )
        ).unsqueeze(1)

        src_emb = self._process_single_side(src_node_ids, src_patches, src_patch_mask, src_pad_n_ids, h_task)
        dst_emb = self._process_single_side(dst_node_ids, dst_patches, dst_patch_mask, dst_pad_n_ids, h_task)
        return src_emb, dst_emb

    def _patchify_mask(self, mask_valid: torch.Tensor, patch_size: int):
        batch_size, seq_len = mask_valid.shape
        truncated = (seq_len // patch_size) * patch_size
        mask_valid = mask_valid[:, :truncated].view(batch_size, truncated // patch_size, patch_size)
        return (mask_valid.sum(dim=-1) > 0).int()

    def _process_patches_into_sequence(self, raw_feats: tuple, co_occur: torch.Tensor, raw_mask_valid: torch.Tensor):
        node_raw, edge_raw, time_feat = raw_feats
        p_node, p_edge, p_time, p_co = self.get_patches(node_raw, edge_raw, time_feat, co_occur, self.patch_size)

        p_node = self.projection_layer["node"](p_node)
        p_edge = self.projection_layer["edge"](p_edge)
        p_time = self.projection_layer["time"](p_time)
        p_co = self.projection_layer["neighbor_co_occurrence"](p_co)

        # (batch, num_patches, num_channels * channel_embedding_dim)
        patches_seq = torch.cat([p_node, p_edge, p_time, p_co], dim=2)
        return patches_seq, self._patchify_mask(raw_mask_valid, self.patch_size)

    def _process_single_side(self, node_ids, patches_seq, patches_mask_valid, pad_neighbor_ids, h_task):
        h_hist_seq = self._encode_history(patches_seq, patches_mask_valid)
        h_struct = self._encode_structure(node_ids, pad_neighbor_ids)
        h_mixed = self._fuse(node_ids, h_task, h_hist_seq, h_struct, patches_mask_valid)
        return self.final_proj(h_mixed).squeeze(1)

    def _encode_history(self, patches_seq, patches_mask_valid):
        h_hist_seq = self.hist_evo_tracker(patches_seq, mask_valid=patches_mask_valid)
        return h_hist_seq * patches_mask_valid.unsqueeze(-1).to(h_hist_seq.dtype)

    def _encode_structure(self, node_ids, pad_neighbor_ids):
        center_raw = self.node_raw_features[torch.from_numpy(node_ids).to(self.device)].unsqueeze(1)
        recent_neighbor_indices = torch.from_numpy(pad_neighbor_ids[:, 1:]).long().to(self.device)
        neighbor_raw = self.node_raw_features[recent_neighbor_indices]
        return self.struct_neighbor_agg(center_raw, neighbor_raw)

    def _modality_features(self, node_ids):
        indices = torch.from_numpy(node_ids).long().to(self.device)
        return [table[indices].unsqueeze(1) for table in self.node_raw_features_list]

    def _fuse(self, node_ids, h_task, h_hist_seq, h_struct, patches_mask_valid):
        h_context = self.context_attn_layer(
            task_embd=h_task,
            hist_embd=h_hist_seq,
            struct_embd=h_struct,
            hist_mask_valid=patches_mask_valid,
        )
        return self.modal_attn_layer(context_embd=h_context, modality_embds=self._modality_features(node_ids))


# ------------------------------------------------------------------------------------------------
# Ablation variants. Each one replaces a single component of TST-MF.
# ------------------------------------------------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = MLP(dim, dim, dim * 2, 1)
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask_valid: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch, seq_len, dim)
        mask_valid: (batch, seq_len), 1 = valid, 0 = padding
        """
        key_padding_mask = None
        if mask_valid is not None:
            key_padding_mask = mask_valid == 0

        attn_out, _ = self.attn(query=x, key=x, value=x, key_padding_mask=key_padding_mask)
        x = self.norm(x + self.dropout(attn_out))
        x = self.norm_ffn(x + self.dropout(self.ffn(x)))
        return x


class ContextSelfAttnLayer(nn.Module):
    def __init__(self, task_dim: int, hist_dim: int, struct_dim: int, output_dim: int, num_heads: int):
        super().__init__()
        self.task_proj = MLP(task_dim, output_dim, output_dim * 2, 1)
        self.hist_proj = MLP(hist_dim, output_dim, output_dim * 2, 1)
        self.struct_proj = MLP(struct_dim, output_dim, output_dim * 2, 1)
        self.self_attn = SelfAttentionBlock(output_dim, num_heads=num_heads)

    def forward(self, task_embd, hist_embd, struct_embd, hist_mask_valid=None) -> torch.Tensor:
        task = self.task_proj(task_embd)
        hist = self.hist_proj(hist_embd)
        stru = self.struct_proj(struct_embd)
        seq = torch.cat([task, hist, stru], dim=1)

        mask_valid = None
        if hist_mask_valid is not None:
            batch_size = hist_mask_valid.size(0)
            task_valid = torch.ones(batch_size, 1, device=hist_mask_valid.device, dtype=hist_mask_valid.dtype)
            stru_valid = torch.ones(batch_size, 1, device=hist_mask_valid.device, dtype=hist_mask_valid.dtype)
            mask_valid = torch.cat([task_valid, hist_mask_valid, stru_valid], dim=1)

        return self.self_attn(seq, mask_valid=mask_valid)[:, :1, :]


class ModalSelfAttnLayer(nn.Module):
    def __init__(self, context_dim: int, modal_dims: list[int], output_dim: int, num_heads: int):
        super().__init__()
        self.context_proj = MLP(context_dim, output_dim, output_dim * 2, 1)
        self.modal_projs = nn.ModuleList([nn.Linear(m_dim, output_dim) for m_dim in modal_dims])
        self.self_attn = SelfAttentionBlock(output_dim, num_heads=num_heads)

    def forward(self, context_embd: torch.Tensor, modality_embds: list[torch.Tensor]) -> torch.Tensor:
        ctx = self.context_proj(context_embd)
        mods = [self.modal_projs[i](embd) for i, embd in enumerate(modality_embds)]
        return self.self_attn(torch.cat([ctx] + mods, dim=1), mask_valid=None)[:, :1, :]


class TSTMFNoTargetGuidance(TSTMF):
    """Replaces both target-guided cross-attention stages with self-attention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.context_attn_layer = ContextSelfAttnLayer(
            task_dim=self.hidden_dim,
            hist_dim=self.hidden_dim,
            struct_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_heads=self.num_heads,
        )
        self.modal_attn_layer = ModalSelfAttnLayer(
            context_dim=self.hidden_dim,
            modal_dims=[t.shape[1] for t in self.node_raw_features_list],
            output_dim=self.hidden_dim,
            num_heads=self.num_heads,
        )


class SingleStageTaskGuidedFusion(nn.Module):
    def __init__(self, task_dim: int, hist_dim: int, struct_dim: int, modal_dims: list[int], output_dim: int, num_heads: int):
        super().__init__()
        self.task_proj = MLP(task_dim, output_dim, output_dim * 2, 1)
        self.hist_proj = MLP(hist_dim, output_dim, output_dim * 2, 1)
        self.struct_proj = MLP(struct_dim, output_dim, output_dim * 2, 1)
        self.modal_projs = nn.ModuleList([nn.Linear(m_dim, output_dim) for m_dim in modal_dims])
        self.cross_attn = CrossAttentionBlock(output_dim, num_heads=num_heads)

    def forward(self, task_embd, hist_embd, struct_embd, modality_embds, hist_mask_valid=None) -> torch.Tensor:
        q = self.task_proj(task_embd)
        hist = self.hist_proj(hist_embd)
        stru = self.struct_proj(struct_embd)
        mods = [self.modal_projs[i](m) for i, m in enumerate(modality_embds)]
        context_seq = torch.cat([hist, stru] + mods, dim=1)

        context_mask_valid = None
        if hist_mask_valid is not None:
            extra_valid = torch.ones(hist_mask_valid.size(0), 1 + len(mods), device=hist_mask_valid.device, dtype=hist_mask_valid.dtype)
            context_mask_valid = torch.cat([hist_mask_valid, extra_valid], dim=1)

        return self.cross_attn(query=q, context=context_seq, context_mask_valid=context_mask_valid)


class TSTMFSingleStage(TSTMF):
    """Replaces the two fusion stages with a single cross-attention over context and modalities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.single_stage_fusion = SingleStageTaskGuidedFusion(
            task_dim=self.hidden_dim,
            hist_dim=self.hidden_dim,
            struct_dim=self.hidden_dim,
            modal_dims=[t.shape[1] for t in self.node_raw_features_list],
            output_dim=self.hidden_dim,
            num_heads=self.num_heads,
        )

    def _fuse(self, node_ids, h_task, h_hist_seq, h_struct, patches_mask_valid):
        return self.single_stage_fusion(
            task_embd=h_task,
            hist_embd=h_hist_seq,
            struct_embd=h_struct,
            modality_embds=self._modality_features(node_ids),
            hist_mask_valid=patches_mask_valid,
        )


class TSTMFNoHistTracker(TSTMF):
    """Replaces the historical evolution tracker with a per-position MLP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hist_no_tracker_proj = MLP(
            input_dim=self.num_channels * self.channel_embedding_dim,
            output_dim=self.hidden_dim,
            hidden_dims=self.hidden_dim * 2,
            num_hidden_layers=1,
            dropout=self.dropout,
        )
        self.hist_no_tracker_norm = nn.LayerNorm(self.hidden_dim)
        self.hist_evo_tracker = None

    def _encode_history(self, patches_seq, patches_mask_valid):
        h_hist_seq = self.hist_no_tracker_norm(self.hist_no_tracker_proj(patches_seq))
        return h_hist_seq * patches_mask_valid.unsqueeze(-1).to(h_hist_seq.dtype)


class TSTMFNoStructAgg(TSTMF):
    """Replaces the structural neighbor aggregator with a projection of the node's own features."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.struct_center_proj = nn.Linear(self.node_feat_dim, self.hidden_dim)
        self.struct_center_norm = nn.LayerNorm(self.hidden_dim)
        self.struct_neighbor_agg = None

    def _encode_structure(self, node_ids, pad_neighbor_ids):
        center_raw = self.node_raw_features[torch.from_numpy(node_ids).to(self.device)].unsqueeze(1)
        return self.struct_center_norm(self.struct_center_proj(center_raw))
