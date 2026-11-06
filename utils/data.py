"""Dataset loading, feature selection and the temporal train/validation/test split."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

DATA_ROOT_ENV_VAR = "MMDG_DATA_ROOT"
FULL_FEATURE_DIM = 768 * 2
SINGLE_MODALITY_DIM = 768


class IndexDataset(Dataset):
    """Yields integer indices into numpy-backed arrays."""

    def __init__(self, indices: list[int]):
        self._indices = indices

    def __getitem__(self, index: int) -> int:
        return self._indices[index]

    def __len__(self) -> int:
        return len(self._indices)


@dataclass
class Data:
    """Temporal interaction data."""

    src_node_ids: np.ndarray
    dst_node_ids: np.ndarray
    node_interact_times: np.ndarray
    edge_ids: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.num_interactions = len(self.src_node_ids)
        self.unique_node_ids = set(self.src_node_ids) | set(self.dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


def slice_batch(data: Data, indices: np.ndarray):
    return (
        data.src_node_ids[indices],
        data.dst_node_ids[indices],
        data.node_interact_times[indices],
        data.edge_ids[indices],
        data.labels[indices],
    )


def get_idx_data_loader(indices_list: list[int], batch_size: int, shuffle: bool, *, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset=IndexDataset(indices_list), batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers)


def load_temporal_dataset(*, dataset_name: str, val_ratio: float, test_ratio: float, feature_spec: str, data_root: str | None = None):
    """Loads a dataset.

    Returns (node_features_list, edge_features_list, full_data, train_data, val_data, test_data,
    new_node_val_data, new_node_test_data, category_count), with one feature table per modality.
    """
    graph_df, npy_dir = _load_graph_df(dataset_name, data_root)
    node_count = int(max(graph_df["u"].max(), graph_df["i"].max()) + 1)
    edge_feature_count = int(graph_df["r"].max() + 1)
    category_count = int(graph_df["label"].max() + 1)

    node_raw_features_list: list[np.ndarray] = []
    edge_raw_features_list: list[np.ndarray] = []
    for token in feature_spec.split("+"):
        node_features, edge_features = _load_feature_pair(
            dataset_name=dataset_name,
            npy_dir=npy_dir,
            feature_name=token.strip(),
            node_count=node_count,
            edge_feature_count=edge_feature_count,
        )
        node_raw_features_list.append(node_features)
        edge_raw_features_list.append(edge_features)

    splits = _split_temporal_graph(graph_df, val_ratio=val_ratio, test_ratio=test_ratio)
    return (
        node_raw_features_list,
        edge_raw_features_list,
        splits["full_data"],
        splits["train_data"],
        splits["val_data"],
        splits["test_data"],
        splits["new_node_val_data"],
        splits["new_node_test_data"],
        category_count,
    )


def _load_graph_df(dataset_name: str, data_root: str | None) -> tuple[pd.DataFrame, Path]:
    csv_path, npy_dir = _resolve_dataset_assets(dataset_name, data_root)
    graph_df = pd.read_csv(csv_path)
    if graph_df["label"].min() != 0:
        graph_df["label"] = graph_df["label"] - 1
    if "gdelt" in Path(dataset_name).name.lower():
        # GDELT's timestamps are consecutive integers; merge every 15 steps into one time bin.
        graph_df["ts"] = graph_df["ts"] // 15
    return graph_df, npy_dir


def _resolve_dataset_assets(dataset_name: str, data_root: str | None) -> tuple[Path, Path]:
    candidates = [Path(dataset_name).expanduser()]
    root = data_root or os.getenv(DATA_ROOT_ENV_VAR)
    if root:
        candidates.append(Path(root).expanduser() / dataset_name)

    for dataset_dir in candidates:
        csv_path, npy_dir = dataset_dir / "edge_list.csv", dataset_dir / "npy"
        if csv_path.exists() and npy_dir.exists():
            return csv_path, npy_dir

    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not locate dataset '{dataset_name}'. Checked:\n{searched}\n"
        f"Use --data_root or set {DATA_ROOT_ENV_VAR} to the directory containing the datasets."
    )


def _load_feature_pair(*, dataset_name: str, npy_dir: Path, feature_name: str, node_count: int, edge_feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    if feature_name == "rand":
        node_features = np.load(npy_dir / "e_feat_ran.npy")
        edge_features = np.load(npy_dir / "r_feat_ran.npy")
        node_features = _pad_feature_dim(node_features, _target_dim_for_loaded_features(node_features))
        edge_features = _pad_feature_dim(edge_features, _target_dim_for_loaded_features(edge_features))
    elif feature_name in {"text", "image"}:
        # The first half of each row holds the text embedding, the second half the image embedding.
        full_edge_features = np.load(npy_dir / "r_feat_cat.npy")
        full_node_features = np.load(npy_dir / "e_feat_cat.npy")
        midpoint_edge = full_edge_features.shape[1] // 2
        midpoint_node = full_node_features.shape[1] // 2
        if feature_name == "text":
            edge_features = full_edge_features[:, :midpoint_edge]
            node_features = full_node_features[:, :midpoint_node]
        else:
            edge_features = full_edge_features[:, midpoint_edge:]
            node_features = full_node_features[:, midpoint_node:]
        node_features = _pad_feature_dim(node_features, SINGLE_MODALITY_DIM)
        edge_features = _pad_feature_dim(edge_features, SINGLE_MODALITY_DIM)
    else:
        raise ValueError(f"Unsupported feature '{feature_name}': expected one of rand, text, image.")

    if node_features.shape[0] != node_count:
        raise ValueError(f"Node feature count mismatch for dataset '{dataset_name}': expected {node_count}, got {node_features.shape[0]}.")
    if edge_features.shape[0] != edge_feature_count:
        raise ValueError(f"Edge feature count mismatch for dataset '{dataset_name}': expected {edge_feature_count}, got {edge_features.shape[0]}.")

    return node_features.astype(np.float32), edge_features.astype(np.float32)


def _pad_feature_dim(features: np.ndarray, target_dim: int) -> np.ndarray:
    if features.shape[1] > target_dim:
        raise ValueError(f"Feature dimension {features.shape[1]} exceeds target dimension {target_dim}.")
    if features.shape[1] == target_dim:
        return features.astype(np.float32)
    padding = np.zeros((features.shape[0], target_dim - features.shape[1]), dtype=np.float32)
    return np.concatenate([features.astype(np.float32), padding], axis=1)


def _target_dim_for_loaded_features(features: np.ndarray) -> int:
    return SINGLE_MODALITY_DIM if features.shape[1] <= SINGLE_MODALITY_DIM else FULL_FEATURE_DIM


def _split_temporal_graph(graph_df: pd.DataFrame, *, val_ratio: float, test_ratio: float) -> dict[str, Data]:
    src_node_ids = graph_df["u"].values.astype(np.longlong)
    dst_node_ids = graph_df["i"].values.astype(np.longlong)
    node_interact_times = graph_df["ts"].values.astype(np.float64)
    edge_ids = graph_df["r"].values.astype(np.longlong)
    labels = graph_df["label"].values

    def subset(mask: np.ndarray) -> Data:
        return Data(
            src_node_ids=src_node_ids[mask],
            dst_node_ids=dst_node_ids[mask],
            node_interact_times=node_interact_times[mask],
            edge_ids=edge_ids[mask],
            labels=labels[mask],
        )

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

    val_time, test_time = list(np.quantile(graph_df["ts"], [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    # Hold out 10% of the nodes seen after val_time from training, for the inductive setting.
    random.seed(2020)
    node_set = set(src_node_ids) | set(dst_node_ids)
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    inductive_node_count = min(len(test_node_set), int(0.1 * len(node_set)))
    new_test_node_set = set(random.sample(list(test_node_set), inductive_node_count)) if inductive_node_count > 0 else set()

    new_test_source_mask = graph_df["u"].map(lambda node_id: node_id in new_test_node_set).values
    new_test_destination_mask = graph_df["i"].map(lambda node_id: node_id in new_test_node_set).values
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)
    train_data = subset(train_mask)

    new_node_set = node_set - (set(train_data.src_node_ids) | set(train_data.dst_node_ids))

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time
    edge_contains_new_node_mask = np.array(
        [(src_node_id in new_node_set or dst_node_id in new_node_set) for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)]
    )

    return {
        "full_data": full_data,
        "train_data": train_data,
        "val_data": subset(val_mask),
        "test_data": subset(test_mask),
        "new_node_val_data": subset(np.logical_and(val_mask, edge_contains_new_node_mask)),
        "new_node_test_data": subset(np.logical_and(test_mask, edge_contains_new_node_mask)),
    }
