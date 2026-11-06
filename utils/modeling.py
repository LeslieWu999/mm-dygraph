"""Model construction and the helpers that hide the differences between backbones.

A model is the composition of two orthogonal choices: a temporal backbone (``--model_name``) and a
multimodal fusion wrapper (``--wrapper_name``).
"""
from __future__ import annotations

from functools import partial

import numpy as np
import torch
import torch.nn as nn

from models.adapters import FreeDyGAdapter, LiquidAdapter, build_random_projections
from models.CAWN import CAWN
from models.DyGFormer import DyGFormer
from models.fusion import TSTMF, TSTMFNoHistTracker, TSTMFNoStructAgg, TSTMFNoTargetGuidance, TSTMFSingleStage
from models.fusion_baselines import ConcatWrapper, FusedFeatureView, FusionWrapper, build_fusion_wrapper_class, infer_backbone_output_dim
from models.GraphMixer import GraphMixer
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.modules import MergeLayer, MLPClassifier_edge
from models.TCL import TCL
from models.TGAT import TGAT
from models.TPNet import TPNet

# SALoM's upstream class is called Liquid.
MEMORY_MODEL_NAMES = {"JODIE", "DyRep", "TGN", "Liquid"}
# Backbones whose forward pass only takes (src, dst, times).
DYGFORMER_STYLE_MODEL_NAMES = {"DyGFormer", "FreeDyG", "TPNet"}

BACKBONE_REGISTRY: dict[str, type[nn.Module]] = {
    "JODIE": MemoryModel,
    "DyRep": MemoryModel,
    "TGN": MemoryModel,
    "TGAT": TGAT,
    "CAWN": CAWN,
    "TCL": TCL,
    "GraphMixer": GraphMixer,
    "DyGFormer": DyGFormer,
    "FreeDyG": FreeDyGAdapter,
    "TPNet": TPNet,
    "Liquid": LiquidAdapter,
}

WRAPPER_REGISTRY: dict[str, type[nn.Module]] = {
    "concat": ConcatWrapper,
    "tst_mf": TSTMF,
    "tst_mf_no_target_guidance": TSTMFNoTargetGuidance,
    "tst_mf_single_stage": TSTMFSingleStage,
    "tst_mf_no_struct_agg": TSTMFNoStructAgg,
    "tst_mf_no_hist_tracker": TSTMFNoHistTracker,
}
FUSION_WRAPPER_NAMES = {"gated_fusion", "modality_attention", "unigraph2"}
WRAPPER_NAMES = sorted(set(WRAPPER_REGISTRY) | FUSION_WRAPPER_NAMES)


def build_model(*, task_name: str, args, node_raw_features_list, edge_raw_features_list, neighbor_sampler, train_data, cat_num: int) -> nn.Sequential:
    """Returns ``nn.Sequential(backbone, task_head)``."""
    wrapper_class = _build_fusion_wrapper_class(args) if args.wrapper_name in FUSION_WRAPPER_NAMES else WRAPPER_REGISTRY[args.wrapper_name]
    model_kwargs = _build_backbone_kwargs(
        args=args,
        train_data=train_data,
        neighbor_sampler=neighbor_sampler,
        node_raw_features_list=node_raw_features_list,
        edge_raw_features_list=edge_raw_features_list,
    )
    backbone = wrapper_class(BACKBONE_REGISTRY[args.model_name], node_raw_features_list, edge_raw_features_list, **model_kwargs)
    embedding_dim = _get_backbone_output_dim(args.wrapper_name, node_raw_features_list, backbone)
    head = _build_head(task_name=task_name, embedding_dim=embedding_dim, dropout=args.dropout, cat_num=cat_num)
    return nn.Sequential(backbone, head)


def _build_fusion_wrapper_class(args):
    if args.wrapper_name == "gated_fusion":
        fuser_kwargs = {"per_dimension": bool(args.gate_per_dimension)}
    elif args.wrapper_name == "modality_attention":
        fuser_kwargs = {"num_queries": args.mattn_num_queries, "num_heads": args.mattn_num_heads, "dropout": args.dropout}
    else:
        fuser_kwargs = {
            "num_experts": args.unigraph2_num_experts,
            "mask_ratio": args.unigraph2_mask_ratio,
            "recon_weight": args.unigraph2_recon_weight,
        }
    return build_fusion_wrapper_class(args.wrapper_name, fuser_kwargs)


def _get_backbone_output_dim(wrapper_name: str, node_raw_features_list, backbone: nn.Module) -> int:
    if isinstance(backbone, FusionWrapper):
        return backbone.output_dim
    concat_dim = sum(node_raw_features.shape[1] for node_raw_features in node_raw_features_list)
    if wrapper_name == "concat":
        return infer_backbone_output_dim(backbone.model, fallback=concat_dim)
    return concat_dim


def _build_head(*, task_name: str, embedding_dim: int, dropout: float, cat_num: int) -> nn.Module:
    if task_name == "link_prediction":
        return MergeLayer(input_dim1=embedding_dim, input_dim2=embedding_dim, hidden_dim=embedding_dim, output_dim=1)
    if task_name == "edge_classification":
        return MLPClassifier_edge(input_dim=embedding_dim, dropout=dropout, cat_num=cat_num)
    raise ValueError(f"Unsupported task_name '{task_name}'.")


def _build_backbone_kwargs(*, args, train_data, neighbor_sampler, node_raw_features_list, edge_raw_features_list) -> dict:
    model_name = args.model_name
    kwargs = {
        "neighbor_sampler": neighbor_sampler,
        "time_feat_dim": args.time_feat_dim,
        "dropout": args.dropout,
        "device": args.device,
    }
    if model_name == "TGAT":
        kwargs.update(num_layers=args.num_layers, num_heads=args.num_heads)
    elif model_name in MEMORY_MODEL_NAMES:
        # SALoM's structure channel keeps upstream's defaults, including max_input_sequence_length=512:
        # --max_input_sequence_length does not apply to it.
        src_mean, src_std, dst_mean, dst_std = compute_src_dst_node_time_shifts(
            train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times
        )
        kwargs.update(
            model_name=model_name,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            src_node_mean_time_shift=src_mean,
            src_node_std_time_shift=src_std,
            dst_node_mean_time_shift_dst=dst_mean,
            dst_node_std_time_shift=dst_std,
        )
    elif model_name == "CAWN":
        kwargs.update(position_feat_dim=args.position_feat_dim, walk_length=args.walk_length, num_walk_heads=args.num_walk_heads)
    elif model_name == "TCL":
        kwargs.update(num_layers=args.num_layers, num_heads=args.num_heads, num_depths=args.num_neighbors + 1)
    elif model_name == "GraphMixer":
        kwargs.update(num_tokens=args.num_neighbors, num_layers=args.num_layers)
    elif model_name == "DyGFormer":
        kwargs.update(
            channel_embedding_dim=args.channel_embedding_dim,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_input_sequence_length=args.max_input_sequence_length,
        )
    elif model_name == "FreeDyG":
        kwargs.update(
            channel_embedding_dim=args.channel_embedding_dim,
            num_layers=args.num_layers,
            max_input_sequence_length=args.max_input_sequence_length,
        )
    elif model_name == "TPNet":
        kwargs.update(
            num_layers=args.num_layers,
            num_neighbors=args.num_neighbors,
            random_projections=None
            if args.encode_not_rp
            else build_random_projections(args, node_raw_features_list, edge_raw_features_list, train_data),
        )
    else:
        raise ValueError(f"Unsupported model_name '{model_name}'.")
    return kwargs


def compute_temporal_embeddings(
    *,
    backbone: nn.Module,
    model_name: str,
    src_node_ids,
    dst_node_ids,
    node_interact_times,
    num_neighbors: int,
    time_gap: int,
    edge_ids=None,
    edges_are_positive: bool = True,
):
    """Returns (src_embeddings, dst_embeddings). Updates the memory of memory-based models on positive edges."""
    common = {"src_node_ids": src_node_ids, "dst_node_ids": dst_node_ids, "node_interact_times": node_interact_times}
    if model_name in {"TGAT", "CAWN", "TCL"}:
        return backbone.compute_src_dst_node_temporal_embeddings(**common, num_neighbors=num_neighbors)
    if model_name in MEMORY_MODEL_NAMES:
        return backbone.compute_src_dst_node_temporal_embeddings(
            **common, edge_ids=edge_ids, edges_are_positive=edges_are_positive, num_neighbors=num_neighbors
        )
    if model_name == "GraphMixer":
        return backbone.compute_src_dst_node_temporal_embeddings(**common, num_neighbors=num_neighbors, time_gap=time_gap)
    if model_name in DYGFORMER_STYLE_MODEL_NAMES:
        return backbone.compute_src_dst_node_temporal_embeddings(**common)
    raise ValueError(f"Unsupported model_name '{model_name}'.")


def is_memory_model(model_name: str) -> bool:
    return model_name in MEMORY_MODEL_NAMES


def has_streaming_state(model_name: str) -> bool:
    """Whether the model keeps state across batches: a memory bank, or TPNet's random projections."""
    return is_memory_model(model_name) or model_name == "TPNet"


def _random_projections_of(backbone: nn.Module):
    return getattr(getattr(backbone, "model", backbone), "random_projections", None)


def update_tpnet_projections(backbone: nn.Module, src_node_ids, dst_node_ids, node_interact_times) -> None:
    """Feeds the observed edges to TPNet's random projections; a no-op for all other models.

    Must be called after every candidate of the batch has been scored, otherwise the negatives are
    scored on a state that already contains the batch's true edges.
    """
    random_projections = _random_projections_of(backbone)
    if random_projections is not None:
        random_projections.update(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times)


def forward_in_streaming_order(backbone: nn.Module, model_name: str, *, forward_positive, forward_negative, src_node_ids, dst_node_ids, node_interact_times):
    """Runs a batch's positive and negative forward passes on the same streaming state; returns (positive, negative).

    The negatives must not see the batch's true edges. A memory-based model updates its memory during
    the positive pass, so it runs the negatives first. TPNet's projections advance after both passes.
    """
    if is_memory_model(model_name):
        negative = forward_negative()
        positive = forward_positive()
    else:
        positive = forward_positive()
        negative = forward_negative()
        update_tpnet_projections(backbone, src_node_ids, dst_node_ids, node_interact_times)
    return positive, negative


def compute_positive_and_negative_embeddings(
    *,
    backbone: nn.Module,
    model_name: str,
    src_node_ids: np.ndarray,
    dst_node_ids: np.ndarray,
    neg_src_node_ids: np.ndarray,
    neg_dst_node_ids: np.ndarray,
    node_interact_times: np.ndarray,
    edge_ids: np.ndarray,
    num_neighbors: int,
    time_gap: int,
):
    """Embeds a batch's positive and negative edges, then advances the streaming state with the positives."""
    embed = partial(
        compute_temporal_embeddings,
        backbone=backbone,
        model_name=model_name,
        node_interact_times=node_interact_times,
        num_neighbors=num_neighbors,
        time_gap=time_gap,
    )
    return forward_in_streaming_order(
        backbone,
        model_name,
        forward_positive=lambda: embed(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, edge_ids=edge_ids, edges_are_positive=True),
        forward_negative=lambda: embed(src_node_ids=neg_src_node_ids, dst_node_ids=neg_dst_node_ids, edge_ids=None, edges_are_positive=False),
        src_node_ids=src_node_ids,
        dst_node_ids=dst_node_ids,
        node_interact_times=node_interact_times,
    )


def set_neighbor_sampler(backbone: nn.Module, model_name: str, neighbor_sampler) -> None:
    if model_name != "JODIE":  # JODIE does not use neighbors
        backbone.set_neighbor_sampler(neighbor_sampler)


def reset_memory(backbone: nn.Module, model_name: str) -> None:
    """Resets all streaming state. Called at the start of each epoch."""
    if is_memory_model(model_name):
        backbone.memory_bank.__init_memory_bank__()
    random_projections = _random_projections_of(backbone)
    if random_projections is not None:
        random_projections.reset_random_projections()


def detach_memory(backbone: nn.Module, model_name: str) -> None:
    if is_memory_model(model_name):
        backbone.memory_bank.detach_memory_bank()


def backup_memory(backbone: nn.Module, model_name: str):
    """Snapshot of all streaming state, or None for models without any."""
    memory_backup = backbone.memory_bank.backup_memory_bank() if is_memory_model(model_name) else None
    random_projections = _random_projections_of(backbone)
    projections_backup = random_projections.backup_random_projections() if random_projections is not None else None
    if memory_backup is None and projections_backup is None:
        return None
    return {"memory": memory_backup, "random_projections": projections_backup}


def restore_memory(backbone: nn.Module, model_name: str, backup) -> None:
    if backup is None:
        return
    if backup["memory"] is not None:
        backbone.memory_bank.reload_memory_bank(backup["memory"])
    if backup["random_projections"] is not None:
        _random_projections_of(backbone).reload_random_projections(backup["random_projections"])


def reset_auxiliary_loss(backbone: nn.Module) -> None:
    """Clears SALoM's auxiliary loss accumulator.

    It is otherwise only cleared on negative forward passes. Edge classification has none, so it must
    call this on each batch, or the accumulator keeps the previous batch's graph alive.
    """
    for module in (backbone, getattr(backbone, "model", None)):
        reset = getattr(module, "reset_aux_loss", None)
        if callable(reset):
            reset()
            return


def auxiliary_loss(backbone: nn.Module, reference_tensor: torch.Tensor) -> torch.Tensor | None:
    """The sum of the auxiliary losses of the wrapper and of the backbone it wraps, or None if they have none."""
    values = [getattr(module, "aux_loss", None) for module in (backbone, getattr(backbone, "model", None))]
    values = [value for value in values if value is not None]
    if not values:
        return None
    total = sum(values)
    return total if isinstance(total, torch.Tensor) else reference_tensor.new_tensor(float(total))


def predict_link_probabilities(model: nn.Sequential, src_embeddings: torch.Tensor, dst_embeddings: torch.Tensor) -> torch.Tensor:
    return model[1](input_1=src_embeddings, input_2=dst_embeddings).squeeze(dim=-1).sigmoid()


def predict_edge_logits(model: nn.Sequential, src_embeddings: torch.Tensor, dst_embeddings: torch.Tensor) -> torch.Tensor:
    return model[1](x_1=src_embeddings, x_2=dst_embeddings)


class HostFeatureTable:
    """A feature table kept in host memory; the selected rows are copied to the device on indexing.

    Backbones only access feature tables through ``table[idx]``, ``.shape``, ``.device`` and ``.dtype``,
    so this is a drop-in replacement that returns the same values.
    """

    def __init__(self, table: torch.Tensor, device):
        self.table = table.detach().to("cpu").contiguous()
        self._device = torch.device(device)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to("cpu")
        elif not isinstance(index, (int, slice)):
            index = torch.as_tensor(np.asarray(index))
        return self.table[index].to(self._device)

    def __len__(self):
        return self.table.shape[0]

    @property
    def shape(self):
        return self.table.shape

    @property
    def dtype(self):
        return self.table.dtype

    @property
    def device(self):
        return self._device

    def size(self, dim: int = None):
        return self.table.shape if dim is None else self.table.shape[dim]

    def dim(self):
        return self.table.dim()


def offload_feature_tables(model: nn.Module, device, min_numel: int = 1_000_000) -> int:
    """Moves large feature tables to host memory; call before moving the model to the device.

    Saves the GPU memory of one full copy of the features per process (over 10 GB on the larger
    datasets). Returns the number of bytes moved.
    """
    converted = {}

    def host(tensor):
        key = id(tensor)  # submodules may share the same table
        if key not in converted:
            converted[key] = HostFeatureTable(tensor, device)
        return converted[key]

    for module in model.modules():
        for name in ("node_raw_features", "edge_raw_features"):
            value = module.__dict__.get(name)
            if torch.is_tensor(value) and value.numel() >= min_numel:
                module.__dict__[name] = host(value)
        tables = module.__dict__.get("node_raw_features_list")
        if tables is not None:
            module.__dict__["node_raw_features_list"] = [host(table) if table.numel() >= min_numel else table for table in tables]
        if isinstance(module, FusedFeatureView):
            for i in range(module.num_modalities):
                buffer = module._buffers.pop(f"modality_{i}", None)
                if buffer is not None:
                    module.__dict__[f"modality_{i}"] = host(buffer)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sum(table.table.numel() * table.table.element_size() for table in converted.values())
