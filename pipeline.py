"""Runs the experiments of the three tasks: data, models, training, checkpoints and result files.

The evaluation itself lives in ``evaluation.py``.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from evaluation import (
    build_evaluation_samplers,
    evaluate_edge_split,
    evaluate_final_edge,
    evaluate_final_link,
    evaluate_link_test,
    evaluate_link_validation,
    evaluate_model_link_prediction,
    evaluate_model_retrieval,
)
from utils.data import Data, get_idx_data_loader, load_temporal_dataset, slice_batch
from utils.EarlyStopping import EarlyStopping
from utils.metrics import get_edge_classification_metrics, get_link_prediction_metrics, mean_metric_dicts, mean_value
from utils.modeling import (
    auxiliary_loss,
    backup_memory,
    build_model,
    compute_positive_and_negative_embeddings,
    compute_temporal_embeddings,
    detach_memory,
    offload_feature_tables,
    predict_edge_logits,
    predict_link_probabilities,
    reset_auxiliary_loss,
    reset_memory,
    restore_memory,
    set_neighbor_sampler,
    update_tpnet_projections,
)
from utils.runtime import close_logger, ensure_clean_dir, ensure_dir, sanitize_name, save_json, setup_logger
from utils.utils import NegativeEdgeSampler, convert_to_gpu, create_optimizer, get_neighbor_sampler, get_parameter_sizes, set_random_seed


@dataclass
class TaskContext:
    node_raw_features_list: list[np.ndarray]
    edge_raw_features_list: list[np.ndarray]
    feature_metadata: dict | None
    full_data: Data
    train_data: Data
    val_data: Data
    test_data: Data
    new_node_val_data: Data
    new_node_test_data: Data
    cat_num: int
    train_neighbor_sampler: object
    full_neighbor_sampler: object
    train_idx_data_loader: object
    val_idx_data_loader: object
    new_node_val_idx_data_loader: object
    test_idx_data_loader: object
    new_node_test_idx_data_loader: object


def run_link_prediction_training(args):
    context = build_task_context(args)
    run_payloads = []

    for run_index, seed in enumerate(args.seeds):
        set_random_seed(seed)

        run_name = build_run_name("link_prediction", args, seed=seed)
        logger = _start_run_logger(args, run_name, "link_prediction", seed, run_index, context)
        model = build_task_model("link_prediction", args, context)
        _log_model(logger, model, args)
        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        model = place_model(model, args, logger)

        save_model_folder = ensure_clean_dir(build_model_dir(args, run_name))
        early_stopping = EarlyStopping(
            patience=args.patience,
            save_model_folder=str(save_model_folder),
            save_model_name=run_name,
            logger=logger,
            model_name=args.model_name,
        )
        loss_func = nn.BCELoss()
        train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=context.train_data.src_node_ids, dst_node_ids=context.train_data.dst_node_ids)
        evaluation_samplers = build_evaluation_samplers(context)
        common = {
            "evaluator": evaluate_model_link_prediction,
            "model": model,
            "args": args,
            "context": context,
            "samplers": evaluation_samplers,
            "loss_func": loss_func,
        }

        for epoch in range(args.num_epochs):
            train_losses, train_metrics = _train_link_prediction_epoch(
                model=model,
                args=args,
                context=context,
                train_neg_edge_sampler=train_neg_edge_sampler,
                optimizer=optimizer,
                loss_func=loss_func,
            )

            if (epoch + 1) % args.test_interval_epochs != 0:
                continue

            validation_bundle = evaluate_link_validation(**common, include_validation=True)
            logger.info(f"Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]['lr']}, train loss: {mean_value(train_losses):.4f}")
            _log_metric_dict(logger, "train", mean_metric_dicts(train_metrics))
            _log_bundle(logger, validation_bundle)

            if early_stopping.step(_metric_dict_to_indicator(validation_bundle["validate metrics"]), model):
                break

            test_bundle = evaluate_link_test(**common, memory_backup=validation_bundle["memory_backup"])
            _log_bundle(logger, test_bundle)

        early_stopping.load_checkpoint(model, map_location=args.device)
        run_payload = _final_link_payload(common, args, logger)
        run_payloads.append(run_payload)
        save_json(_build_result_path(args, run_name), run_payload)
        close_logger(logger)
        del model, optimizer, early_stopping, common
        cleanup_after_run(args)

    return _save_summary("link_prediction_summary", args, run_payloads)


def run_edge_classification_training(args):
    context = build_task_context(args)
    run_payloads = []

    for run_index, seed in enumerate(args.seeds):
        set_random_seed(seed)

        run_name = build_run_name("edge_classification", args, seed=seed)
        logger = _start_run_logger(args, run_name, "edge_classification", seed, run_index, context)
        model = build_task_model("edge_classification", args, context)
        _log_model(logger, model, args)
        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        model = place_model(model, args, logger)

        save_model_folder = ensure_clean_dir(build_model_dir(args, run_name))
        early_stopping = EarlyStopping(
            patience=args.patience,
            save_model_folder=str(save_model_folder),
            save_model_name=run_name,
            logger=logger,
            model_name=args.model_name,
        )
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(args.num_epochs):
            train_loss, train_metrics = _train_edge_classification_epoch(model=model, args=args, context=context, optimizer=optimizer, loss_func=loss_func)

            if (epoch + 1) % args.test_interval_epochs != 0:
                continue

            val_loss, val_metrics = evaluate_edge_split(model, args, context, loss_func, context.val_idx_data_loader, context.val_data)
            logger.info(f"Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]['lr']}, train loss: {train_loss:.4f}")
            _log_metric_dict(logger, "train", train_metrics)
            logger.info(f"validate loss: {val_loss:.4f}")
            _log_metric_dict(logger, "validate", val_metrics)

            if early_stopping.step(_metric_dict_to_indicator(val_metrics), model):
                break

            memory_backup = backup_memory(model[0], args.model_name)
            test_loss, test_metrics = evaluate_edge_split(model, args, context, loss_func, context.test_idx_data_loader, context.test_data)
            restore_memory(model[0], args.model_name, memory_backup)
            logger.info(f"test loss: {test_loss:.4f}")
            _log_metric_dict(logger, "test", test_metrics)

        early_stopping.load_checkpoint(model, map_location=args.device)
        logger.info(f"get final performance on dataset {args.dataset_name}...")
        run_payload = evaluate_final_edge(model, args, context, loss_func)
        _log_bundle(logger, run_payload)

        run_payloads.append(run_payload)
        save_json(_build_result_path(args, run_name), run_payload)
        close_logger(logger)
        del model, optimizer, early_stopping
        cleanup_after_run(args)

    return _save_summary("edge_classification_summary", args, run_payloads)


def run_link_retrieval_evaluation(args):
    """Evaluates node retrieval with the checkpoints of a finished link prediction run."""
    context = build_task_context(args)
    evaluation_samplers = build_evaluation_samplers(context, args.negative_sample_strategy)
    run_payloads = []

    for run_index, seed in enumerate(args.seeds):
        set_random_seed(seed)

        model_run_name = build_run_name("link_prediction", args, seed=seed)
        result_run_name = build_run_name(
            "node_retrieval",
            args,
            seed=seed,
            negative_sample_strategy=args.negative_sample_strategy,
            retrieval_evaluation_protocol=args.retrieval_evaluation_protocol,
            num_negative_candidates=args.num_negative_candidates,
        )
        logger = _start_run_logger(args, result_run_name, "node_retrieval", seed, run_index, context)
        model = build_task_model("link_prediction", args, context)
        model = place_model(model, args, logger)
        logger.info(f"model -> {model}")

        early_stopping = EarlyStopping(
            patience=0,
            save_model_folder=str(build_model_dir(args, model_run_name)),
            save_model_name=model_run_name,
            logger=logger,
            model_name=args.model_name,
        )
        early_stopping.load_checkpoint(model, map_location=args.device)

        common = {
            "evaluator": evaluate_model_retrieval,
            "model": model,
            "args": args,
            "context": context,
            "samplers": evaluation_samplers,
            "loss_func": nn.BCELoss(),
        }
        run_payload = _final_link_payload(common, args, logger)
        run_payloads.append(run_payload)
        save_json(_build_result_path(args, result_run_name), run_payload)
        close_logger(logger)
        del model, early_stopping, common
        cleanup_after_run(args)

    return _save_summary(
        "node_retrieval_summary",
        args,
        run_payloads,
        negative_sample_strategy=args.negative_sample_strategy,
        retrieval_evaluation_protocol=args.retrieval_evaluation_protocol,
        num_negative_candidates=args.num_negative_candidates,
    )


def build_task_context(args) -> TaskContext:
    (
        node_raw_features_list,
        edge_raw_features_list,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
        cat_num,
        feature_metadata,
    ) = load_temporal_dataset(
        dataset_name=args.dataset_name,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        feature_spec=args.use_feature,
        data_root=args.data_root,
    )

    def neighbor_sampler(data, seed):
        return get_neighbor_sampler(
            data=data,
            sample_neighbor_strategy=args.sample_neighbor_strategy,
            time_scaling_factor=args.time_scaling_factor,
            seed=seed,
        )

    def idx_loader(data):
        return get_idx_data_loader(
            indices_list=list(range(len(data.src_node_ids))),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    return TaskContext(
        node_raw_features_list=node_raw_features_list,
        edge_raw_features_list=edge_raw_features_list,
        feature_metadata=feature_metadata,
        full_data=full_data,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        new_node_val_data=new_node_val_data,
        new_node_test_data=new_node_test_data,
        cat_num=cat_num,
        train_neighbor_sampler=neighbor_sampler(train_data, 0),
        full_neighbor_sampler=neighbor_sampler(full_data, 1),
        train_idx_data_loader=idx_loader(train_data),
        val_idx_data_loader=idx_loader(val_data),
        new_node_val_idx_data_loader=idx_loader(new_node_val_data),
        test_idx_data_loader=idx_loader(test_data),
        new_node_test_idx_data_loader=idx_loader(new_node_test_data),
    )


def build_task_model(task_name: str, args, context: TaskContext) -> nn.Sequential:
    return build_model(
        task_name=task_name,
        args=args,
        node_raw_features_list=context.node_raw_features_list,
        edge_raw_features_list=context.edge_raw_features_list,
        neighbor_sampler=context.train_neighbor_sampler,
        train_data=context.train_data,
        cat_num=context.cat_num,
        feature_metadata=context.feature_metadata,
    )


def place_model(model: nn.Module, args, logger) -> nn.Module:
    """Moves the model to its device, keeping the feature tables in host memory if requested."""
    offloaded = offload_feature_tables(model, args.device) if args.offload_features else 0
    model = convert_to_gpu(model, device=args.device)
    if offloaded:
        logger.info(f"feature tables kept in host memory: {offloaded / 2**30:.2f} GiB")
    return model


def _train_link_prediction_epoch(*, model, args, context: TaskContext, train_neg_edge_sampler, optimizer, loss_func):
    model.train()
    backbone = model[0]
    set_neighbor_sampler(backbone, args.model_name, context.train_neighbor_sampler)
    reset_memory(backbone, args.model_name)

    train_losses, train_metrics = [], []
    for train_data_indices in context.train_idx_data_loader:
        src, dst, times, edge_ids, _ = slice_batch(context.train_data, train_data_indices.numpy())
        _, neg_dst = train_neg_edge_sampler.sample(size=len(src))

        (src_emb, dst_emb), (neg_src_emb, neg_dst_emb) = compute_positive_and_negative_embeddings(
            backbone=backbone,
            model_name=args.model_name,
            src_node_ids=src,
            dst_node_ids=dst,
            neg_src_node_ids=src,
            neg_dst_node_ids=neg_dst,
            node_interact_times=times,
            edge_ids=edge_ids,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap,
        )

        positive_probabilities = predict_link_probabilities(model, src_emb, dst_emb)
        negative_probabilities = predict_link_probabilities(model, neg_src_emb, neg_dst_emb)
        predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
        labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

        loss = loss_func(input=predicts, target=labels)
        extra_loss = auxiliary_loss(backbone, loss)
        if extra_loss is not None:
            loss = loss + extra_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        detach_memory(backbone, args.model_name)

        train_losses.append(loss.item())
        train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

    return train_losses, train_metrics


def _train_edge_classification_epoch(*, model, args, context: TaskContext, optimizer, loss_func):
    model.train()
    backbone = model[0]
    set_neighbor_sampler(backbone, args.model_name, context.train_neighbor_sampler)
    reset_memory(backbone, args.model_name)

    train_total_loss = 0.0
    train_y_trues, train_y_predicts = [], []
    for train_data_indices in context.train_idx_data_loader:
        src, dst, times, edge_ids, batch_labels = slice_batch(context.train_data, train_data_indices.numpy())

        reset_auxiliary_loss(backbone)

        src_emb, dst_emb = compute_temporal_embeddings(
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
        update_tpnet_projections(backbone, src, dst, times)

        predicts = predict_edge_logits(model, src_emb, dst_emb)
        pred_labels = torch.argmax(predicts, dim=1)
        labels = torch.from_numpy(batch_labels).long().to(predicts.device)

        loss = loss_func(predicts, labels)
        extra_loss = auxiliary_loss(backbone, loss)
        if extra_loss is not None:
            loss = loss + extra_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        detach_memory(backbone, args.model_name)

        train_total_loss += loss.item()
        train_y_trues.append(labels)
        train_y_predicts.append(pred_labels)

    return train_total_loss / max(len(context.train_idx_data_loader), 1), get_edge_classification_metrics(
        predicts=torch.cat(train_y_predicts, dim=0),
        labels=torch.cat(train_y_trues, dim=0),
    )


def _final_link_payload(common: dict, args, logger) -> dict:
    payload = evaluate_final_link(**common)
    logger.info(f"get final performance on dataset {args.dataset_name}...")
    _log_bundle(logger, payload)
    return payload


def _metric_dict_to_indicator(metric_dict: dict[str, float]):
    return [(metric_name, metric_value, True) for metric_name, metric_value in metric_dict.items()]


def cleanup_after_run(args) -> None:
    gc.collect()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()


def build_run_name(
    task_label: str,
    args,
    *,
    seed: int | None = None,
    negative_sample_strategy: str | None = None,
    retrieval_evaluation_protocol: str | None = None,
    num_negative_candidates: int | None = None,
) -> str:
    parts = [task_label, args.wrapper_name, args.model_name, f"feature-{args.use_feature}"]
    if negative_sample_strategy is not None:
        parts.append(f"negative-{negative_sample_strategy}")
    if retrieval_evaluation_protocol is not None:
        parts.append(f"protocol-{retrieval_evaluation_protocol}")
    if num_negative_candidates is not None:
        parts.append(f"candidates-{num_negative_candidates}")
    if seed is not None:
        parts.append(f"seed-{seed}")
    return sanitize_name("__".join(parts))


def build_model_dir(args, run_name: str) -> Path:
    return ensure_dir(Path(args.output_dir) / "saved_models" / sanitize_name(args.model_name) / sanitize_name(args.dataset_name) / run_name)


def build_log_dir(args, run_name: str) -> Path:
    return Path(args.output_dir) / "logs" / sanitize_name(args.model_name) / sanitize_name(args.dataset_name) / run_name


def _build_result_path(args, run_name: str) -> Path:
    result_dir = ensure_dir(Path(args.output_dir) / "saved_results" / sanitize_name(args.model_name) / sanitize_name(args.dataset_name))
    return result_dir / f"{run_name}.json"


def _start_run_logger(args, run_name: str, task_label: str, seed: int, run_index: int, context: TaskContext):
    logger = setup_logger(build_log_dir(args, run_name), logger_name=f"{task_label}.{seed}")
    logger.info(f"Run {run_index + 1}/{len(args.seeds)} starts.")
    logger.info(f"configuration is {vars(args)}")
    logger.info(f"node feature sizes {[features.shape for features in context.node_raw_features_list]}")
    logger.info(f"edge feature sizes {[features.shape for features in context.edge_raw_features_list]}")
    return logger


def _log_model(logger, model: nn.Module, args) -> None:
    logger.info(f"model -> {model}")
    megabytes = get_parameter_sizes(model) * 4 / 1024 / 1024
    logger.info(f"model name: {args.model_name}, parameter size: {megabytes:.2f} MB.")


def _log_metric_dict(logger, prefix: str, metric_dict: dict[str, float]) -> None:
    for metric_name, metric_value in metric_dict.items():
        logger.info(f"{prefix} {metric_name}, {metric_value:.4f}")


def _log_bundle(logger, bundle: dict) -> None:
    for prefix in ["validate", "new node validate", "test", "new node test"]:
        loss_key = f"{prefix} loss"
        metrics_key = f"{prefix} metrics"
        if loss_key in bundle:
            logger.info(f"{loss_key}: {bundle[loss_key]:.4f}")
        if metrics_key in bundle:
            _log_metric_dict(logger, prefix, bundle[metrics_key])


def _save_summary(
    task_label: str,
    args,
    run_payloads: list[dict],
    *,
    negative_sample_strategy: str | None = None,
    retrieval_evaluation_protocol: str | None = None,
    num_negative_candidates: int | None = None,
) -> dict:
    seed = args.seeds[0] if len(args.seeds) == 1 else None
    summary_name = build_run_name(
        task_label,
        args,
        seed=seed,
        negative_sample_strategy=negative_sample_strategy,
        retrieval_evaluation_protocol=retrieval_evaluation_protocol,
        num_negative_candidates=num_negative_candidates,
    )
    payload = _build_summary_payload(run_payloads, seeds=args.seeds)
    save_json(_build_result_path(args, summary_name), payload)
    return payload


def _build_summary_payload(run_payloads: list[dict], *, seeds: list[int]) -> dict:
    summary = {}
    for key in sorted({key for payload in run_payloads for key in payload}):
        if key.endswith("metrics"):
            summary[key] = _summarize_metric_dicts([payload[key] for payload in run_payloads if payload.get(key)])
        else:
            summary[key] = _summarize_values([payload[key] for payload in run_payloads if key in payload])
    return {"runs": run_payloads, "summary": summary, "seeds": seeds}


def _summarize_values(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": float(np.mean(values)) if values else 0.0,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _summarize_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict:
    if not metric_dicts:
        return {}
    return {
        metric_name: _summarize_values([metric_dict[metric_name] for metric_dict in metric_dicts])
        for metric_name in sorted(metric_dicts[0])
    }
