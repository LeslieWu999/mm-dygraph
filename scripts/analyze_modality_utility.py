"""Instance-level modality utility of a trained TST-MF model (inference only).

For each test instance i and modality m:

    delta[i, m] = loss(model with modality m masked, i) - loss(full model, i)

Positive values mean that masking hurts, so the modality is useful for that instance. The loss is
the positive edge's negative log-likelihood for link prediction, and the true class's cross entropy
for edge classification. A modality is masked in every branch of the model: its table used by the
modality fusion, and its columns in the concatenated feature tables.

Test batches are sampled at equal intervals until --max_instances instances are reached.

    python -m scripts.analyze_modality_utility --task lp --dataset_name GDELT \
        --model_name DyGFormer --wrapper_name tst_mf --use_feature text+image --seeds 0

Writes <output_dir>/saved_results/_utility/utility__{lp_tr,lp_in,ec}__<dataset>__<wrapper>__<model>__seed-N.npz
with the arrays loss_full (n,) and delta (n, num_modalities).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pipeline import build_log_dir, build_model_dir, build_run_name, build_task_context, build_task_model, cleanup_after_run
from utils.data import slice_batch
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_edge_classification_args, get_link_prediction_args
from utils.modeling import compute_temporal_embeddings, predict_edge_logits, predict_link_probabilities, set_neighbor_sampler
from utils.runtime import close_logger, ensure_dir, setup_logger
from utils.utils import convert_to_gpu, set_random_seed

EPS = 1e-7


def modality_slices(feature_list):
    """The column range of each modality in the concatenated feature table."""
    slices, offset = [], 0
    for tensor in feature_list:
        slices.append((offset, offset + tensor.shape[1]))
        offset += tensor.shape[1]
    return slices


class ModalityMask:
    """Masks modality ``index`` in all branches of the model and restores it on exit."""

    def __init__(self, backbone, index: int, node_slices, edge_slices):
        self.backbone = backbone
        self.index = index
        self.node_slices = node_slices
        self.edge_slices = edge_slices
        self._saved_columns = []
        self._saved_table = None

    def __enter__(self):
        backbone, m = self.backbone, self.index
        self._saved_table = backbone.node_raw_features_list[m]
        zeros = torch.zeros(1, self._saved_table.shape[1], dtype=self._saved_table.dtype, device=self._saved_table.device)
        backbone.node_raw_features_list[m] = zeros.expand(self._saved_table.shape[0], self._saved_table.shape[1])

        self._saved_columns = []
        for table, (low, high) in ((backbone.node_raw_features, self.node_slices[m]), (backbone.edge_raw_features, self.edge_slices[m])):
            self._saved_columns.append((table, low, high, table[:, low:high].clone()))
            table[:, low:high] = 0.0
        return self

    def __exit__(self, *exc):
        self.backbone.node_raw_features_list[self.index] = self._saved_table
        for table, low, high, saved in self._saved_columns:
            table[:, low:high] = saved
        return False


def _embed(backbone, args, src, dst, times, edge_ids):
    return compute_temporal_embeddings(
        backbone=backbone,
        model_name=args.model_name,
        src_node_ids=src,
        dst_node_ids=dst,
        node_interact_times=times,
        edge_ids=edge_ids,
        edges_are_positive=True,
        num_neighbors=args.num_neighbors,
        time_gap=args.time_gap,
    )


def positive_nll(model, backbone, args, src, dst, times, edge_ids):
    src_emb, dst_emb = _embed(backbone, args, src, dst, times, edge_ids)
    probabilities = predict_link_probabilities(model, src_emb, dst_emb).clamp(min=EPS, max=1.0 - EPS)
    return -torch.log(probabilities)


def edge_cross_entropy(model, backbone, args, src, dst, times, edge_ids, labels):
    src_emb, dst_emb = _embed(backbone, args, src, dst, times, edge_ids)
    logits = predict_edge_logits(model, src_emb, dst_emb)
    return F.cross_entropy(logits, torch.from_numpy(labels).long().to(logits.device), reduction="none")


def sample_batches(loader, max_instances: int):
    batches = list(loader)
    per_batch = max(1, len(batches[0]) if batches else 1)
    needed = min(len(batches), -(-max_instances // per_batch))
    keep = set(np.linspace(0, len(batches) - 1, needed).round().astype(int).tolist()) if needed else set()
    return [batch for i, batch in enumerate(batches) if i in keep]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", choices=["lp", "ec"], default="lp")
    parser.add_argument("--max_instances", type=int, default=20000)
    own_args, remaining = parser.parse_known_args()

    is_lp = own_args.task == "lp"
    args = (get_link_prediction_args if is_lp else get_edge_classification_args)(remaining)
    assert args.wrapper_name.startswith("tst_mf"), "modality masking needs the per-modality tables of TST-MF"
    assert not args.offload_features, "modality masking modifies the feature tables in place"
    task_name = "link_prediction" if is_lp else "edge_classification"
    context = build_task_context(args)
    out_dir = ensure_dir(Path(args.output_dir) / "saved_results" / "_utility")
    splits = (
        [("lp_tr", context.test_idx_data_loader, context.test_data), ("lp_in", context.new_node_test_idx_data_loader, context.new_node_test_data)]
        if is_lp
        else [("ec", context.test_idx_data_loader, context.test_data)]
    )

    for seed in args.seeds:
        set_random_seed(seed)
        run_name = build_run_name(task_name, args, seed=seed)
        logger = setup_logger(build_log_dir(args, f"utility__{run_name}"), logger_name=f"utility.{seed}")
        model = convert_to_gpu(build_task_model(task_name, args, context), device=args.device)
        EarlyStopping(
            patience=0,
            save_model_folder=str(build_model_dir(args, run_name)),
            save_model_name=run_name,
            logger=logger,
            model_name=args.model_name,
        ).load_checkpoint(model, map_location=args.device)
        backbone = model[0]
        set_neighbor_sampler(backbone, args.model_name, context.full_neighbor_sampler)
        model.eval()
        node_slices = modality_slices(backbone.node_raw_features_list)
        edge_slices = modality_slices(backbone.edge_raw_features_list)

        for split, loader, data in splits:
            losses, deltas = [], []
            with torch.no_grad():
                for indices in sample_batches(loader, own_args.max_instances):
                    src, dst, times, edge_ids, labels = slice_batch(data, indices.numpy())

                    def instance_loss():
                        if is_lp:
                            return positive_nll(model, backbone, args, src, dst, times, edge_ids)
                        return edge_cross_entropy(model, backbone, args, src, dst, times, edge_ids, labels)

                    full = instance_loss()
                    delta = []
                    for m in range(len(backbone.node_raw_features_list)):
                        with ModalityMask(backbone, m, node_slices, edge_slices):
                            delta.append((instance_loss() - full).float().cpu().numpy())
                    losses.append(full.float().cpu().numpy())
                    deltas.append(np.stack(delta, axis=1))
            if not losses:
                logger.info(f"split {split} is empty, skipping")
                continue
            out = out_dir / f"utility__{split}__{args.dataset_name}__{args.wrapper_name}__{args.model_name}__seed-{seed}.npz"
            np.savez_compressed(out, loss_full=np.concatenate(losses), delta=np.concatenate(deltas))
            logger.info(f"wrote {out} with {sum(len(x) for x in losses)} instances")
        close_logger(logger)
        cleanup_after_run(args)


if __name__ == "__main__":
    main()
