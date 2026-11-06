"""Tensor-like feature sources used by learnable missing-modality filling."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn


class FixedFeatureSource(nn.Module):
    def __init__(self, features: np.ndarray | torch.Tensor):
        super().__init__()
        tensor = features if isinstance(features, torch.Tensor) else torch.from_numpy(features.astype(np.float32))
        self.register_buffer("features", tensor.float(), persistent=False)

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return self.features.shape[0]

    @property
    def shape(self):
        return self.features.shape

    @property
    def device(self):
        return self.features.device

    @property
    def dtype(self):
        return self.features.dtype

    def size(self, dim: int | None = None):
        return self.features.size() if dim is None else self.features.size(dim)


class LearnableMissingFeatureSource(FixedFeatureSource):
    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        missing_mask: np.ndarray | torch.Tensor,
        init_embedding: np.ndarray | torch.Tensor,
    ):
        super().__init__(features)
        mask = missing_mask if isinstance(missing_mask, torch.Tensor) else torch.from_numpy(missing_mask.astype(np.bool_))
        initial = init_embedding if isinstance(init_embedding, torch.Tensor) else torch.from_numpy(init_embedding.astype(np.float32))
        mask = mask.bool().clone()
        if len(mask) > 0:
            mask[0] = False
        self.register_buffer("missing_mask", mask, persistent=False)
        self.missing_embedding = nn.Parameter(initial.float().clone())

    def __getitem__(self, index):
        values = self.features[index]
        missing = self.missing_mask[index]
        if values.ndim == 1:
            return self.missing_embedding.to(values.dtype) if bool(missing.item()) else values
        if missing.any():
            values = values.clone()
            values[missing] = self.missing_embedding.to(values.dtype)
        return values


class ConcatFeatureSource(nn.Module):
    def __init__(self, feature_sources: Iterable[nn.Module]):
        super().__init__()
        sources = list(feature_sources)
        if not sources:
            raise ValueError("ConcatFeatureSource expects at least one feature source.")
        self.feature_sources = nn.ModuleList(sources)

    def __getitem__(self, index):
        return torch.cat([source[index] for source in self.feature_sources], dim=-1)

    def __len__(self):
        return len(self.feature_sources[0])

    @property
    def shape(self):
        return torch.Size(
            [self.feature_sources[0].shape[0], sum(source.shape[1] for source in self.feature_sources)]
        )

    @property
    def device(self):
        return self.feature_sources[0].device

    @property
    def dtype(self):
        return self.feature_sources[0].dtype

    def size(self, dim: int | None = None):
        return self.shape if dim is None else self.shape[dim]


def build_feature_sources(feature_list: list, metadata: dict | None, prefix: str) -> list[nn.Module] | None:
    if not metadata or prefix not in metadata:
        return None

    prefix_metadata = metadata[prefix]
    modality_index = prefix_metadata["modality_index"]
    sources = []
    for index, features in enumerate(feature_list):
        if index == modality_index:
            sources.append(
                LearnableMissingFeatureSource(
                    features=features,
                    missing_mask=prefix_metadata["missing_mask"],
                    init_embedding=prefix_metadata["init_embedding"],
                )
            )
        else:
            sources.append(FixedFeatureSource(features))
    return sources


def concat_or_single(feature_sources: list[nn.Module] | None):
    if feature_sources is None:
        return None
    if len(feature_sources) == 1:
        return feature_sources[0]
    return ConcatFeatureSource(feature_sources)
