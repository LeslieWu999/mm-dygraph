from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def get_rank(target_score, candidate_scores):
    """1-based rank of the target among the candidates; ties are resolved in the target's favour."""
    target_value = float(np.asarray(target_score))
    candidate_values = np.asarray(candidate_scores, dtype=np.float32)
    return int(np.sum(candidate_values > target_value) + 1)


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    predict_values = _to_numpy(predicts)
    label_values = _to_numpy(labels)
    return {
        "average_precision": average_precision_score(y_true=label_values, y_score=predict_values),
        "roc_auc": roc_auc_score(y_true=label_values, y_score=predict_values),
    }


def get_retrieval_metrics(pos_scores, neg_scores):
    """``neg_scores`` holds one array per negative candidate, each with one score per source."""
    return get_retrieval_metrics_from_ranks(get_retrieval_ranks(pos_scores, neg_scores))


def get_retrieval_ranks(pos_scores, neg_scores) -> np.ndarray:
    """Return one-based target ranks, resolving score ties in the target's favour."""
    positive_values = _to_numpy(pos_scores)
    if not neg_scores:
        return np.ones(len(positive_values), dtype=np.int64)
    negative_values = np.stack([_to_numpy(scores) for scores in neg_scores], axis=1)
    return 1 + np.sum(negative_values > positive_values[:, None], axis=1)


def get_retrieval_metrics_from_ranks(ranks) -> dict[str, float]:
    rank_values = np.asarray(ranks, dtype=np.float64)
    if rank_values.size == 0:
        return {"H1": 0.0, "H3": 0.0, "H10": 0.0, "MRR": 0.0, "mean_rank": 0.0}
    return {
        "H1": float(np.mean(rank_values <= 1)),
        "H3": float(np.mean(rank_values <= 3)),
        "H10": float(np.mean(rank_values <= 10)),
        "MRR": float(np.mean(1.0 / rank_values)),
        "mean_rank": float(np.mean(rank_values)),
    }


def get_edge_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    predict_values = _to_numpy(predicts)
    label_values = _to_numpy(labels)
    return {
        "p_macro": precision_score(label_values, predict_values, average="macro", zero_division=0),
        "R_macro": recall_score(label_values, predict_values, average="macro", zero_division=0),
        "F_macro": f1_score(label_values, predict_values, average="macro", zero_division=0),
        "p_micro": precision_score(label_values, predict_values, average="micro", zero_division=0),
        "R_micro": recall_score(label_values, predict_values, average="micro", zero_division=0),
        "F_micro": f1_score(label_values, predict_values, average="micro", zero_division=0),
        "p_weight": precision_score(label_values, predict_values, average="weighted", zero_division=0),
        "R_weight": recall_score(label_values, predict_values, average="weighted", zero_division=0),
        "F_weight": f1_score(label_values, predict_values, average="weighted", zero_division=0),
    }


def mean_value(values) -> float:
    return float(np.mean(values)) if values else 0.0


def mean_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    return {metric_name: float(np.mean([metric_dict[metric_name] for metric_dict in metric_dicts])) for metric_name in metric_dicts[0]}


def _to_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)
