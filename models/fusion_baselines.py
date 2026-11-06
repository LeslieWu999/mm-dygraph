"""Multimodal fusion baselines: concatenation, gated fusion, modality attention and UniGraph2.

Every backbone reads its raw features only through ``table[index]``, ``table.shape`` and
``table.device``. ``FusedFeatureView`` implements that interface on top of a fusion module, so a
fusion method can be combined with any backbone without changing the backbone's code.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import NeighborSampler


class ConcatWrapper(nn.Module):
    """Naive fusion: the backbone sees the concatenation of all modalities."""

    def __init__(self, model_class, node_raw_features_list, edge_raw_features_list, **model_kwargs):
        super().__init__()
        assert len(node_raw_features_list) == len(edge_raw_features_list)
        self.model = model_class(
            node_raw_features=np.concat(node_raw_features_list, axis=-1),
            edge_raw_features=np.concat(edge_raw_features_list, axis=-1),
            **model_kwargs,
        )
        if hasattr(self.model, "memory_bank"):
            self.memory_bank = self.model.memory_bank

    def compute_src_dst_node_temporal_embeddings(self, **kwargs):
        return self.model.compute_src_dst_node_temporal_embeddings(**kwargs)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.model.set_neighbor_sampler(neighbor_sampler)


class FusedFeatureView(nn.Module):
    """Behaves like a (num_rows, fuser.out_dim) tensor, but fuses the modalities on each index.

    :param raw_features_list: list of ndarray, one (num_rows, dim) table per modality
    :param fuser: nn.Module mapping list[Tensor (..., dim)] to Tensor (..., out_dim)
    """

    def __init__(self, raw_features_list: list, fuser: nn.Module):
        super().__init__()
        assert len(raw_features_list) > 0
        self.num_modalities = len(raw_features_list)
        for i, feat in enumerate(raw_features_list):
            # The raw features are frozen constants: keep them out of the checkpoint.
            self.register_buffer(f"modality_{i}", torch.from_numpy(feat.astype(np.float32)), persistent=False)
        self.fuser = fuser

    def _modalities(self):
        return [getattr(self, f"modality_{i}") for i in range(self.num_modalities)]

    def __getitem__(self, index):
        return self.fuser([feat[index] for feat in self._modalities()])

    def __len__(self):
        return self.modality_0.shape[0]

    @property
    def shape(self):
        return torch.Size([self.modality_0.shape[0], self.fuser.out_dim])

    @property
    def device(self):
        return self.modality_0.device

    @property
    def dtype(self):
        return self.modality_0.dtype

    def size(self, dim: int = None):
        return self.shape if dim is None else self.shape[dim]


class _BaseFuser(nn.Module):
    """Common part of all fusers: the output dimension and the auxiliary loss collection."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self._aux_losses = []

    def reset_aux_loss(self):
        self._aux_losses = []

    @property
    def aux_loss(self):
        if len(self._aux_losses) == 0:
            return 0.0
        return sum(self._aux_losses) / len(self._aux_losses)


class GatedFuser(_BaseFuser):
    """Gated fusion: g = sigmoid(MLP([m_1; m_2])), output = g * m_1 + (1 - g) * m_2.

    :param dim: dimension of a single modality
    :param hidden_dim: hidden dimension of the gate MLP, defaults to dim
    :param per_dimension: use a per-dimension gate if True, otherwise a scalar gate
    """

    def __init__(self, dim: int, num_modalities: int = 2, hidden_dim: int = None, per_dimension: bool = True):
        super().__init__(out_dim=dim)
        assert num_modalities == 2, "GatedFuser is only defined for two modalities"
        hidden_dim = hidden_dim if hidden_dim is not None else dim
        gate_dim = dim if per_dimension else 1
        self.gate = nn.Sequential(
            nn.Linear(num_modalities * dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, gate_dim),
        )

    def forward(self, feats: list):
        m1, m2 = feats
        gate = torch.sigmoid(self.gate(torch.cat(feats, dim=-1)))
        return gate * m1 + (1.0 - gate) * m2


class ModalityAttnFuser(_BaseFuser):
    """Modality attention: learnable queries cross-attend to the modalities.

    :param num_queries: number of learnable queries, the output dimension is num_queries * dim
    """

    def __init__(self, dim: int, num_modalities: int = 2, num_heads: int = 4, num_queries: int = 1, dropout: float = 0.1):
        super().__init__(out_dim=num_queries * dim)
        self.dim = dim
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, dim) * (dim**-0.5))
        self.modality_embedding = nn.Parameter(torch.zeros(num_modalities, dim))
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, feats: list):
        key_value = torch.stack(feats, dim=-2) + self.modality_embedding
        leading_shape = key_value.shape[:-2]
        num_tokens = key_value.shape[-2]

        key_value = key_value.reshape(-1, num_tokens, self.dim)
        query = self.query.unsqueeze(dim=0).expand(key_value.shape[0], -1, -1)

        attn_output, _ = self.attention(query, key_value, key_value, need_weights=False)
        attn_output = self.layer_norm(attn_output)

        return attn_output.reshape(*leading_shape, self.num_queries * self.dim)


class UniGraph2Fuser(_BaseFuser):
    """The modality alignment module of UniGraph2 (WWW 2025).

    The paper's pipeline is: frozen modality-specific encoders -> average across modalities ->
    mixture-of-experts alignment -> GNN. The raw features already are the frozen encoder outputs,
    and the dynamic graph backbone plays the role of the GNN, so this module covers the middle two
    steps. The masked feature reconstruction objective is kept as an auxiliary loss computed on a
    separate masked branch, so the supervised task still sees the unmasked features. The paper's SPD
    structure reconstruction is omitted because the backbone models the structure explicitly.

    :param mask_ratio: masking ratio, 0 disables the auxiliary loss
    :param gamma: sharpness of the scaled cosine error
    :param recon_weight: weight of the reconstruction loss
    """

    def __init__(
        self,
        dim: int,
        num_modalities: int = 2,
        num_experts: int = 8,
        expert_hidden_dim: int = None,
        mask_ratio: float = 0.75,
        gamma: float = 2.0,
        recon_weight: float = 1.0,
    ):
        super().__init__(out_dim=dim)
        expert_hidden_dim = expert_hidden_dim if expert_hidden_dim is not None else dim
        self.num_experts = num_experts
        self.mask_ratio = mask_ratio
        self.gamma = gamma
        self.recon_weight = recon_weight

        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, expert_hidden_dim), nn.GELU(), nn.Linear(expert_hidden_dim, dim)) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(dim, num_experts)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.decoder = nn.Sequential(nn.Linear(dim, expert_hidden_dim), nn.GELU(), nn.Linear(expert_hidden_dim, dim))

    def _moe(self, inputs: torch.Tensor):
        weights = torch.softmax(self.gate(inputs), dim=-1)
        expert_outputs = torch.stack([expert(inputs) for expert in self.experts], dim=-2)
        return (weights.unsqueeze(dim=-1) * expert_outputs).sum(dim=-2)

    def forward(self, feats: list):
        inputs = torch.stack(feats, dim=0).mean(dim=0)
        outputs = self._moe(inputs)

        if self.training and self.mask_ratio > 0:
            mask = torch.rand(inputs.shape[:-1], device=inputs.device) < self.mask_ratio
            if mask.any():
                masked_inputs = torch.where(mask.unsqueeze(dim=-1), self.mask_token.to(inputs.dtype), inputs)
                reconstructed = self.decoder(self._moe(masked_inputs))
                cosine = F.cosine_similarity(reconstructed[mask], inputs[mask].detach(), dim=-1)
                self._aux_losses.append(self.recon_weight * ((1.0 - cosine).clamp(min=0.0) ** self.gamma).mean())

        return outputs


FUSER_CLASSES = {
    "gated_fusion": GatedFuser,
    "modality_attention": ModalityAttnFuser,
    "unigraph2": UniGraph2Fuser,
}


def infer_backbone_output_dim(backbone: nn.Module, fallback: int):
    """The embedding size of a backbone, which may differ from its input feature size.

    TCL and GraphMixer use a fixed hidden width, so the output size cannot be derived from the features.
    """
    output_layer = getattr(backbone, "output_layer", None)
    if isinstance(output_layer, nn.Linear):
        return output_layer.out_features
    return getattr(backbone, "node_feat_dim", fallback)


class FusionWrapper(nn.Module):
    """Backbone-agnostic modality fusion.

    The backbone is built on single-modality features, so that its layers get the right sizes, and
    its feature tables are then replaced by ``FusedFeatureView`` objects.

    :param model_cls: backbone class
    :param fuser_name: key of ``FUSER_CLASSES``
    :param fuser_kwargs: extra arguments for the fuser
    """

    def __init__(self, model_cls, node_raw_features_list: list, edge_raw_features_list: list, fuser_name: str, fuser_kwargs: dict = None, **model_kwargs):
        super().__init__()
        assert len(node_raw_features_list) == len(edge_raw_features_list)
        assert fuser_name in FUSER_CLASSES, f"Unknown fuser_name {fuser_name}!"
        fuser_kwargs = dict(fuser_kwargs or {})
        fuser_cls = FUSER_CLASSES[fuser_name]

        node_dim = node_raw_features_list[0].shape[1]
        edge_dim = edge_raw_features_list[0].shape[1]
        assert all(feat.shape[1] == node_dim for feat in node_raw_features_list), "All modalities need the same node feature size"
        assert all(feat.shape[1] == edge_dim for feat in edge_raw_features_list), "All modalities need the same edge feature size"

        self.model = model_cls(node_raw_features=node_raw_features_list[0], edge_raw_features=edge_raw_features_list[0], **model_kwargs)

        # Separate fusers for nodes and edges: the two have different semantics.
        self.node_fuser = fuser_cls(dim=node_dim, num_modalities=len(node_raw_features_list), **fuser_kwargs)
        self.edge_fuser = fuser_cls(dim=edge_dim, num_modalities=len(edge_raw_features_list), **fuser_kwargs)

        node_view = FusedFeatureView(node_raw_features_list, self.node_fuser)
        edge_view = FusedFeatureView(edge_raw_features_list, self.edge_fuser)

        # Submodules such as the graph attention embedding hold references to the feature tables,
        # so replacing only the outermost tables is not enough.
        holders = [module for module in self.model.modules() if hasattr(module, "node_raw_features") or hasattr(module, "edge_raw_features")]
        for module in holders:
            if hasattr(module, "node_raw_features"):
                module.node_raw_features = node_view
            if hasattr(module, "edge_raw_features"):
                module.edge_raw_features = edge_view

        self._node_view, self._edge_view = node_view, edge_view

        if hasattr(self.model, "memory_bank"):
            self.memory_bank = self.model.memory_bank

    @property
    def node_raw_features(self):
        return self._node_view

    @property
    def edge_raw_features(self):
        return self._edge_view

    def compute_src_dst_node_temporal_embeddings(self, **kwargs):
        self.node_fuser.reset_aux_loss()
        self.edge_fuser.reset_aux_loss()
        return self.model.compute_src_dst_node_temporal_embeddings(**kwargs)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.model.set_neighbor_sampler(neighbor_sampler)

    @property
    def aux_loss(self):
        return self.node_fuser.aux_loss + self.edge_fuser.aux_loss

    @property
    def output_dim(self):
        return infer_backbone_output_dim(self.model, fallback=self.node_fuser.out_dim)


def build_fusion_wrapper_class(fuser_name: str, fuser_kwargs: dict = None):
    """Binds a fuser into a class with the ``wrapper_class(model_cls, node_list, edge_list, **kwargs)`` convention."""
    fuser_kwargs = dict(fuser_kwargs or {})

    class _BoundFusionWrapper(FusionWrapper):
        def __init__(self, model_cls, node_raw_features_list, edge_raw_features_list, **model_kwargs):
            super().__init__(
                model_cls,
                node_raw_features_list,
                edge_raw_features_list,
                fuser_name=fuser_name,
                fuser_kwargs=fuser_kwargs,
                **model_kwargs,
            )

    _BoundFusionWrapper.__name__ = f"FusionWrapper_{fuser_name}"
    return _BoundFusionWrapper
