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
    positive_values = _to_numpy(pos_scores)
    negative_values = np.stack([_to_numpy(scores) for scores in neg_scores], axis=1)

    h1_scores, h3_scores, h10_scores = [], [], []
    for index, positive_score in enumerate(positive_values):
        rank = get_rank(positive_score, negative_values[index])
        h1_scores.append(int(rank <= 1))
        h3_scores.append(int(rank <= 3))
        h10_scores.append(int(rank <= 10))

    return {
        "H1": float(np.mean(h1_scores)) if h1_scores else 0.0,
        "H3": float(np.mean(h3_scores)) if h3_scores else 0.0,
        "H10": float(np.mean(h10_scores)) if h10_scores else 0.0,
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
