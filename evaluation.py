"""Evaluation of the three tasks. ``pipeline.py`` runs the experiments and calls into this module.

There are two layers:

- the loops ``evaluate_model_*`` score one split, batch by batch;
- the ``evaluate_link_*``, ``evaluate_edge_split`` and ``evaluate_final_*`` functions decide which splits
  run, and restore the streaming state around them so that no split sees the edges of another.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import Data, slice_batch
from utils.metrics import get_edge_classification_metrics, get_link_prediction_metrics, get_retrieval_metrics, mean_metric_dicts, mean_value
from utils.modeling import (
    backup_memory,
    compute_positive_and_negative_embeddings,
    compute_temporal_embeddings,
    forward_in_streaming_order,
    has_streaming_state,
    predict_edge_logits,
    predict_link_probabilities,
    restore_memory,
    set_neighbor_sampler,
    update_tpnet_projections,
)
from utils.utils import NegativeEdgeSampler

NUM_RETRIEVAL_NEGATIVES = 100


def evaluate_model_link_prediction(
    model_name: str,
    model: nn.Module,
    neighbor_sampler,
    evaluate_idx_data_loader: DataLoader,
    evaluate_neg_edge_sampler: NegativeEdgeSampler,
    evaluate_data: Data,
    loss_func: nn.Module,
    num_neighbors: int = 20,
    time_gap: int = 2000,
):
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    backbone = model[0]
    set_neighbor_sampler(backbone, model_name, neighbor_sampler)
    model.eval()

    evaluate_losses, evaluate_metrics = [], []
    with torch.no_grad():
        progress_bar = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_index, evaluate_data_indices in enumerate(progress_bar):
            src, dst, times, edge_ids, _ = slice_batch(evaluate_data, evaluate_data_indices.numpy())
            neg_src, neg_dst = _sample_negative_edges(evaluate_neg_edge_sampler, src, dst, times)

            (src_emb, dst_emb), (neg_src_emb, neg_dst_emb) = compute_positive_and_negative_embeddings(
                backbone=backbone,
                model_name=model_name,
                src_node_ids=src,
                dst_node_ids=dst,
                neg_src_node_ids=neg_src,
                neg_dst_node_ids=neg_dst,
                node_interact_times=times,
                edge_ids=edge_ids,
                num_neighbors=num_neighbors,
                time_gap=time_gap,
            )

            positive_probabilities = predict_link_probabilities(model, src_emb, dst_emb)
            negative_probabilities = predict_link_probabilities(model, neg_src_emb, neg_dst_emb)
            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            loss = loss_func(input=predicts, target=labels)
            evaluate_losses.append(loss.item())
            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
            progress_bar.set_description(f"evaluate batch {batch_index + 1}, evaluate loss: {loss.item():.4f}")

    return evaluate_losses, evaluate_metrics


def evaluate_model_retrieval(
    model_name: str,
    model: nn.Module,
    neighbor_sampler,
    evaluate_idx_data_loader: DataLoader,
    evaluate_neg_edge_sampler: NegativeEdgeSampler,
    evaluate_data: Data,
    loss_func: nn.Module,
    num_neighbors: int = 20,
    time_gap: int = 2000,
):
    """Node retrieval: ranks each true destination against the same sampled negatives for the whole batch."""
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    backbone = model[0]
    set_neighbor_sampler(backbone, model_name, neighbor_sampler)
    model.eval()

    embed = partial(compute_temporal_embeddings, backbone=backbone, model_name=model_name, num_neighbors=num_neighbors, time_gap=time_gap)

    def score_negatives(src, negative_dst_node_ids, times):
        scores = []
        for negative_dst_node_id in negative_dst_node_ids:
            repeated_dst_ids = np.repeat(negative_dst_node_id, len(src))
            neg_src_emb, neg_dst_emb = embed(src_node_ids=src, dst_node_ids=repeated_dst_ids, node_interact_times=times, edge_ids=None, edges_are_positive=False)
            scores.append(predict_link_probabilities(model, neg_src_emb, neg_dst_emb))
        return scores

    evaluate_losses, evaluate_metrics = [], []
    with torch.no_grad():
        progress_bar = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_index, evaluate_data_indices in enumerate(progress_bar):
            src, dst, times, edge_ids, _ = slice_batch(evaluate_data, evaluate_data_indices.numpy())
            _, negative_dst_node_ids = _sample_negatives(evaluate_neg_edge_sampler, NUM_RETRIEVAL_NEGATIVES, src, dst, times)

            (src_emb, dst_emb), negative_probabilities = forward_in_streaming_order(
                backbone,
                model_name,
                forward_positive=lambda: embed(src_node_ids=src, dst_node_ids=dst, node_interact_times=times, edge_ids=edge_ids, edges_are_positive=True),
                forward_negative=lambda: score_negatives(src, negative_dst_node_ids, times),
                src_node_ids=src,
                dst_node_ids=dst,
                node_interact_times=times,
            )

            positive_probabilities = predict_link_probabilities(model, src_emb, dst_emb)
            predicts = torch.cat([positive_probabilities, negative_probabilities[0]], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities[0])], dim=0)

            loss = loss_func(input=predicts, target=labels)
            evaluate_losses.append(loss.item())
            evaluate_metrics.append(get_retrieval_metrics(positive_probabilities, negative_probabilities))
            progress_bar.set_description(f"evaluate batch {batch_index + 1}, evaluate loss: {loss.item():.4f}")

    return evaluate_losses, evaluate_metrics


def evaluate_model_edge_classification(
    model_name: str,
    model: nn.Module,
    neighbor_sampler,
    evaluate_idx_data_loader: DataLoader,
    evaluate_data: Data,
    loss_func: nn.Module,
    num_neighbors: int = 20,
    time_gap: int = 2000,
):
    backbone = model[0]
    set_neighbor_sampler(backbone, model_name, neighbor_sampler)
    model.eval()

    evaluate_total_loss = 0.0
    evaluate_y_trues, evaluate_y_predicts = [], []
    with torch.no_grad():
        progress_bar = tqdm(evaluate_idx_data_loader, ncols=120)
        batch_count = 0
        for batch_index, evaluate_data_indices in enumerate(progress_bar):
            src, dst, times, edge_ids, batch_labels = slice_batch(evaluate_data, evaluate_data_indices.numpy())

            src_emb, dst_emb = compute_temporal_embeddings(
                backbone=backbone,
                model_name=model_name,
                src_node_ids=src,
                dst_node_ids=dst,
                node_interact_times=times,
                edge_ids=edge_ids,
                edges_are_positive=True,
                num_neighbors=num_neighbors,
                time_gap=time_gap,
            )
            update_tpnet_projections(backbone, src, dst, times)

            predicts = predict_edge_logits(model, src_emb, dst_emb)
            pred_labels = torch.argmax(predicts, dim=1)
            labels = torch.from_numpy(batch_labels).long().to(predicts.device)

            loss = loss_func(input=predicts, target=labels)
            evaluate_total_loss += loss.item()
            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(pred_labels)
            batch_count += 1

            progress_bar.set_description(f"evaluate batch {batch_index + 1}, evaluate loss: {loss.item():.4f}")

    if batch_count == 0:
        return 0.0, {}

    evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
    evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)
    return evaluate_total_loss / batch_count, get_edge_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)


def build_evaluation_samplers(context, negative_sample_strategy: str = "random") -> dict[str, NegativeEdgeSampler]:
    """One independently seeded negative sampler per evaluation split."""

    def sampler(data: Data, last_observed_data: Data, seed: int) -> NegativeEdgeSampler:
        kwargs = {}
        if negative_sample_strategy != "random":
            kwargs = {
                "interact_times": data.node_interact_times,
                "last_observed_time": last_observed_data.node_interact_times[-1],
                "negative_sample_strategy": negative_sample_strategy,
            }
        return NegativeEdgeSampler(src_node_ids=data.src_node_ids, dst_node_ids=data.dst_node_ids, seed=seed, **kwargs)

    return {
        "validate": sampler(context.full_data, context.train_data, 0),
        "new_node_validate": sampler(context.new_node_val_data, context.train_data, 1),
        "test": sampler(context.full_data, context.val_data, 2),
        "new_node_test": sampler(context.new_node_test_data, context.val_data, 3),
    }


def evaluate_split(*, evaluator, model, args, context, samplers, loss_func, loader, sampler_key, data):
    losses, metrics = evaluator(
        model_name=args.model_name,
        model=model,
        neighbor_sampler=context.full_neighbor_sampler,
        evaluate_idx_data_loader=loader,
        evaluate_neg_edge_sampler=samplers[sampler_key],
        evaluate_data=data,
        loss_func=loss_func,
        num_neighbors=args.num_neighbors,
        time_gap=args.time_gap,
    )
    return mean_value(losses), mean_metric_dicts(metrics)


def evaluate_link_validation(*, evaluator, model, args, context, samplers, loss_func, include_validation: bool):
    """Validation and inductive validation. On return the streaming state includes the validation edges only."""
    bundle = {}
    memory_backup = None
    common = {"evaluator": evaluator, "model": model, "args": args, "context": context, "samplers": samplers, "loss_func": loss_func}

    if include_validation:
        train_memory_backup = backup_memory(model[0], args.model_name)
        bundle["validate loss"], bundle["validate metrics"] = evaluate_split(
            **common, loader=context.val_idx_data_loader, sampler_key="validate", data=context.val_data
        )

        memory_backup = backup_memory(model[0], args.model_name)
        restore_memory(model[0], args.model_name, train_memory_backup)

        bundle["new node validate loss"], bundle["new node validate metrics"] = evaluate_split(
            **common, loader=context.new_node_val_idx_data_loader, sampler_key="new_node_validate", data=context.new_node_val_data
        )

        restore_memory(model[0], args.model_name, memory_backup)

    bundle["memory_backup"] = memory_backup
    return bundle


def evaluate_link_test(*, evaluator, model, args, context, samplers, loss_func, memory_backup):
    """Test and inductive test, both starting from ``memory_backup``, which is also restored at the end."""
    common = {"evaluator": evaluator, "model": model, "args": args, "context": context, "samplers": samplers, "loss_func": loss_func}

    test_loss, test_metrics = evaluate_split(**common, loader=context.test_idx_data_loader, sampler_key="test", data=context.test_data)
    restore_memory(model[0], args.model_name, memory_backup)

    new_node_test_loss, new_node_test_metrics = evaluate_split(
        **common, loader=context.new_node_test_idx_data_loader, sampler_key="new_node_test", data=context.new_node_test_data
    )
    restore_memory(model[0], args.model_name, memory_backup)

    return {
        "test loss": test_loss,
        "test metrics": test_metrics,
        "new node test loss": new_node_test_loss,
        "new node test metrics": new_node_test_metrics,
    }


def evaluate_final_link(*, evaluator, model, args, context, samplers, loss_func) -> dict:
    """Final evaluation of a loaded checkpoint.

    The state in a checkpoint of a streaming model already contains the validation edges, so those
    models skip validation: running it again would insert the validation edges a second time.
    """
    common = {"evaluator": evaluator, "model": model, "args": args, "context": context, "samplers": samplers, "loss_func": loss_func}
    bundle = evaluate_link_validation(**common, include_validation=not has_streaming_state(args.model_name))
    bundle.update(evaluate_link_test(**common, memory_backup=backup_memory(model[0], args.model_name)))
    return {key: value for key, value in bundle.items() if key != "memory_backup"}


def evaluate_edge_split(model, args, context, loss_func, loader, data):
    return evaluate_model_edge_classification(
        model_name=args.model_name,
        model=model,
        neighbor_sampler=context.full_neighbor_sampler,
        evaluate_idx_data_loader=loader,
        evaluate_data=data,
        loss_func=loss_func,
        num_neighbors=args.num_neighbors,
        time_gap=args.time_gap,
    )


def evaluate_final_edge(model, args, context, loss_func) -> dict:
    """Final evaluation of a loaded checkpoint; streaming models skip validation as in ``evaluate_final_link``."""
    payload = {}
    if not has_streaming_state(args.model_name):
        payload["validate loss"], payload["validate metrics"] = evaluate_edge_split(model, args, context, loss_func, context.val_idx_data_loader, context.val_data)
    payload["test loss"], payload["test metrics"] = evaluate_edge_split(model, args, context, loss_func, context.test_idx_data_loader, context.test_data)
    return payload


def _sample_negatives(sampler: NegativeEdgeSampler, size: int, src: np.ndarray, dst: np.ndarray, times: np.ndarray):
    if sampler.negative_sample_strategy == "random":
        return sampler.sample(size=size)
    return sampler.sample(
        size=size,
        batch_src_node_ids=src,
        batch_dst_node_ids=dst,
        current_batch_start_time=times[0],
        current_batch_end_time=times[-1],
    )


def _sample_negative_edges(sampler: NegativeEdgeSampler, src: np.ndarray, dst: np.ndarray, times: np.ndarray):
    neg_src, neg_dst = _sample_negatives(sampler, len(src), src, dst, times)
    if sampler.negative_sample_strategy == "random":
        neg_src = src
    return neg_src, neg_dst
