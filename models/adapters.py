"""Adapters that plug the third-party baselines into this repository.

``FreeDyG.py``, ``TPNet.py``, ``Liquid.py``, ``moe.py`` and ``ccffcc.py`` are kept as published by
their authors (see THIRD_PARTY_NOTICES.md). Every deviation from upstream lives in this file.
"""
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from models.FreeDyG import FreeDyG
from models.Liquid import Liquid
from models.TPNet import RandomProjectionModule


class _MoEReturnFive(nn.Module):
    """Upstream SALoM unpacks five values from ``MoE.forward``, which returns two."""

    def __init__(self, moe: nn.Module):
        super().__init__()
        self.moe = moe

    def forward(self, *args, **kwargs):
        outputs, aux_loss = self.moe(*args, **kwargs)
        return outputs, aux_loss, 0.0, 0.0, 0.0


class _SoftmaxPooling(nn.Module):
    """Normalises FreeDyG's aggregation weights over the sequence dimension."""

    def __init__(self, linear: nn.Module):
        super().__init__()
        self.linear = linear

    def forward(self, inputs):
        return torch.softmax(self.linear(inputs), dim=1)


class FreeDyGAdapter(FreeDyG):
    """FreeDyG with normalised pooling, the only difference from upstream.

    Upstream pools with unnormalised weights (``w = weightagg(h); out = w^T h``). That is fine at the
    172-dimensional features of the original paper, but with 768/1536-dimensional CLIP features the
    output embeddings explode: the sigmoid saturates, the gradients vanish, and the loss stays at its
    clamped maximum from the second step on.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weightagg = _SoftmaxPooling(self.weightagg)


class LiquidAdapter(Liquid):
    """SALoM (upstream class name: ``Liquid``), adapted to this repository and to large graphs.

    Interface:
      * upstream returns ``(src, dst, aux_loss)``; this class returns ``(src, dst)`` and accumulates
        the aux loss in ``self.aux_loss``. The accumulator is cleared on each negative forward pass,
        matching upstream's ``bce + exloss_neg + exloss_pos``, or explicitly by ``reset_aux_loss``.

    Correctness:
      * upstream pads to the longest sequence of the batch, which makes a row's score depend on the other
        rows and lets the padding length leak the label. This class always pads to the same length.

    Scaling (node counts here are 10-100x those of the SALoM paper, with 1536-dimensional memories):
      * upstream recomputes the updated memory of *all* nodes with pending messages on every call.
        The embedding module only reads the batch's nodes and their one-hop neighbors, so only those
        are computed. In eval mode the outputs are unchanged up to floating point rounding. In train
        mode two things differ: the noise of the noisy gating is sampled per computed row, and the
        load balancing regulariser (weight 1e-3) is computed on those rows only.
      * only the last raw message of each node is stored, since the message aggregator only uses the
        last one, and it is stored detached on the host to save GPU memory.
      * memory bank backups are kept on the host and restored in place.

    The optimisation relies on deterministic neighbor sampling and on a single embedding layer, so other
    sampling strategies and deeper models fall back to upstream's behaviour.
    """

    USE_MEMORY_SUBSET = True
    USE_HOST_MESSAGES = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss = 0.0
        self.memory_updater.memory_updater = _MoEReturnFive(self.memory_updater.memory_updater)

        self._memory_subset = None
        bank = self.memory_bank
        bank.store_node_raw_messages = self._store_last_detached_message
        bank.detach_memory_bank = self._detach_memories_only
        bank.backup_memory_bank = self._backup_memory_bank_to_host
        bank.reload_memory_bank = self._reload_memory_bank_in_place

    def _store_last_detached_message(self, node_ids: np.ndarray, new_node_raw_messages: dict):
        raw = self.memory_bank.node_raw_messages
        ids = [node_id for node_id in node_ids if len(new_node_raw_messages[node_id]) > 0]
        if not ids:
            return
        last = torch.stack([new_node_raw_messages[node_id][-1][0] for node_id in ids]).detach()
        if self.USE_HOST_MESSAGES:
            last = last.to("cpu")
        for k, node_id in enumerate(ids):
            raw[node_id] = [(last[k].clone(), new_node_raw_messages[node_id][-1][1])]

    def _aggregate_on_device(self, node_ids, node_raw_messages):
        if self.USE_HOST_MESSAGES:
            # Messages restored from a checkpoint with map_location may be on the GPU.
            for node_id in np.unique(node_ids):
                messages = node_raw_messages[node_id]
                if messages and messages[-1][0].device.type != "cpu":
                    node_raw_messages[node_id] = [(m.to("cpu"), t) for m, t in messages]
        unique_node_ids, unique_node_messages, unique_node_timestamps = self.message_aggregator.aggregate_messages(
            node_ids=node_ids, node_raw_messages=node_raw_messages
        )
        if len(unique_node_ids) > 0:
            unique_node_messages = unique_node_messages.to(self.device)
        return unique_node_ids, unique_node_messages, unique_node_timestamps

    def update_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        unique_node_ids, unique_node_messages, unique_node_timestamps = self._aggregate_on_device(node_ids, node_raw_messages)
        return self.memory_updater.update_memories(
            unique_node_ids=unique_node_ids,
            unique_node_messages=unique_node_messages,
            unique_node_timestamps=unique_node_timestamps,
        )

    def pad_sequences(self, *args, max_input_sequence_length: int = 256, **kwargs):
        padded_arrays = super().pad_sequences(*args, max_input_sequence_length=max_input_sequence_length, **kwargs)
        pad_width = ((0, 0), (0, max_input_sequence_length - padded_arrays[0].shape[1]))
        return tuple(np.pad(array, pad_width) for array in padded_arrays)

    def _backup_memory_bank_to_host(self):
        bank = self.memory_bank
        messages = {
            node_id: [(m.clone(), t.copy()) for m, t in node_raw_messages]
            for node_id, node_raw_messages in bank.node_raw_messages.items()
        }
        return (
            bank.node_memories.data.to("cpu", copy=True),
            bank.struc_memories.data.to("cpu", copy=True),
            bank.node_last_updated_times.data.to("cpu", copy=True),
            messages,
        )

    def _reload_memory_bank_in_place(self, backup_memory_bank: tuple):
        bank = self.memory_bank
        bank.node_memories.data.copy_(backup_memory_bank[0])
        bank.struc_memories.data.copy_(backup_memory_bank[1])
        bank.node_last_updated_times.data.copy_(backup_memory_bank[2])
        bank.node_raw_messages = defaultdict(list)
        for node_id, node_raw_messages in backup_memory_bank[3].items():
            bank.node_raw_messages[node_id] = [(m.clone(), t.copy()) for m, t in node_raw_messages]

    def _detach_memories_only(self):
        # Messages are detached when stored, so there is no need to walk over all nodes on each step.
        self.memory_bank.node_memories.detach_()
        self.memory_bank.struc_memories.detach_()

    def get_updated_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        subset = self._memory_subset
        if subset is None:
            unique_node_ids, unique_node_messages, unique_node_timestamps = self._aggregate_on_device(node_ids, node_raw_messages)
            return self.memory_updater.get_updated_memories(
                unique_node_ids=unique_node_ids,
                unique_node_messages=unique_node_messages,
                unique_node_timestamps=unique_node_timestamps,
            )
        bank = self.memory_bank
        unique_node_ids, unique_node_messages, unique_node_timestamps = self._aggregate_on_device(subset, node_raw_messages)
        rows = bank.node_memories.data[torch.from_numpy(subset)].clone()
        updated_node_last_updated_times = bank.node_last_updated_times.data.clone()
        exl = 0
        if len(unique_node_ids) > 0:
            assert (
                bank.get_node_last_updated_times(unique_node_ids=unique_node_ids)
                <= torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)
            ).all().item(), "Trying to update memory to time in the past!"
            pos = torch.from_numpy(np.searchsorted(subset, unique_node_ids)).to(rows.device)
            new_rows, exl, _, _, _ = self.memory_updater.memory_updater(unique_node_messages, rows[pos])
            rows = rows.index_put((pos,), new_rows)
            updated_node_last_updated_times[torch.from_numpy(unique_node_ids)] = (
                torch.from_numpy(unique_node_timestamps).float().to(updated_node_last_updated_times.device)
            )
            if exl is None:
                exl = 0
        return _SubsetMemoryView(subset, rows), updated_node_last_updated_times, exl, 0, 0, 0

    def _memory_subset_for(self, src_node_ids, dst_node_ids, node_interact_times, num_neighbors):
        sampler = self.embedding_module.neighbor_sampler
        if not self.USE_MEMORY_SUBSET or self.num_layers != 1 or getattr(sampler, "sample_neighbor_strategy", None) != "recent":
            return None
        node_ids = np.concatenate([src_node_ids, dst_node_ids])
        times = np.concatenate([node_interact_times, node_interact_times])
        neighbor_ids, _, _ = sampler.get_historical_neighbors(node_ids=node_ids, node_interact_times=times, num_neighbors=num_neighbors)
        return np.unique(np.concatenate([node_ids, neighbor_ids.reshape(-1)])).astype(np.int64)

    def reset_aux_loss(self):
        self.aux_loss = 0.0

    def compute_src_dst_node_temporal_embeddings(
        self, src_node_ids, dst_node_ids, node_interact_times, edge_ids=None, edges_are_positive=True, num_neighbors=20
    ):
        if edges_are_positive is False:
            self.reset_aux_loss()
        self._memory_subset = self._memory_subset_for(src_node_ids, dst_node_ids, node_interact_times, num_neighbors)
        try:
            src_node_embeddings, dst_node_embeddings, extra_loss = super().compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_node_ids,
                dst_node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                edge_ids=edge_ids,
                edges_are_positive=edges_are_positive,
                num_neighbors=num_neighbors,
            )
        finally:
            self._memory_subset = None
        self.aux_loss = self.aux_loss + extra_loss
        return src_node_embeddings, dst_node_embeddings


class _SubsetMemoryView:
    """Stands in for the (num_nodes, memory_dim) tensor upstream passes to the embedding module.

    Reading a node outside the subset raises, so the optimisation cannot silently read stale memories.
    """

    def __init__(self, subset_ids: np.ndarray, rows: torch.Tensor):
        self.subset_ids = subset_ids
        self.rows = rows

    def __getitem__(self, index):
        ids = index.cpu().numpy() if torch.is_tensor(index) else np.asarray(index)
        pos = np.minimum(np.searchsorted(self.subset_ids, ids), len(self.subset_ids) - 1)
        if not np.array_equal(self.subset_ids[pos], ids):
            raise RuntimeError("LiquidAdapter: the embedding module read a node outside the memory subset")
        return self.rows[torch.from_numpy(pos).to(self.rows.device)]


def build_random_projections(args, node_raw_features_list: list, edge_raw_features_list: list, train_data):
    """TPNet's random projection module, or None if disabled."""
    if not args.use_random_projection:
        return None
    return RandomProjectionModule(
        node_num=node_raw_features_list[0].shape[0],
        edge_num=edge_raw_features_list[0].shape[0],
        dim_factor=args.rp_dim_factor,
        num_layer=args.rp_num_layer,
        time_decay_weight=args.rp_time_decay_weight,
        device=args.device,
        use_matrix=args.rp_use_matrix,
        beginning_time=train_data.node_interact_times[0],
        not_scale=args.rp_not_scale,
        enforce_dim=args.enforce_dim,
    )
