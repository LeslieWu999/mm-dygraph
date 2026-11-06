from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from utils.modeling import is_memory_model


class EarlyStopping:
    def __init__(self, patience: int, save_model_folder: str, save_model_name: str, logger, model_name: str | None = None):
        self.patience = patience
        self.counter = 0
        self.best_metrics: dict[str, float] = {}
        self.best_score: tuple[float, ...] | None = None
        self.early_stop = False
        self.logger = logger
        self.model_name = model_name

        folder_path = Path(save_model_folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        self.save_model_path = folder_path / f"{save_model_name}.pkl"
        self.save_model_nonparametric_data_path = folder_path / f"{save_model_name}_nonparametric_data.pkl"

    def step(self, metrics: list[tuple[str, float, bool]], model: nn.Module):
        """Records a validation result. ``metrics`` holds (name, value, higher_is_better), compared lexicographically."""
        score = self._build_score(metrics)
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_metrics = {metric_name: metric_value for metric_name, metric_value, _ in metrics}
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience
        return self.early_stop

    def save_checkpoint(self, model: nn.Module):
        self.logger.info(f"save model {self.save_model_path}")
        torch.save(model.state_dict(), self.save_model_path)
        if is_memory_model(self.model_name or ""):
            torch.save(model[0].memory_bank.node_raw_messages, self.save_model_nonparametric_data_path)

    def load_checkpoint(self, model: nn.Module, map_location: str | None = None):
        if not self.save_model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {self.save_model_path}")
        self.logger.info(f"load model {self.save_model_path}")
        model.load_state_dict(torch.load(self.save_model_path, map_location=map_location, weights_only=False))
        if is_memory_model(self.model_name or "") and self.save_model_nonparametric_data_path.exists():
            # SALoM keeps pending messages on the host; loading them to the device would run out of
            # memory on the larger datasets.
            backbone = model[0]
            on_host = getattr(getattr(backbone, "model", backbone), "USE_HOST_MESSAGES", False)
            backbone.memory_bank.node_raw_messages = torch.load(
                self.save_model_nonparametric_data_path,
                map_location="cpu" if on_host else map_location,
                weights_only=False,
            )

    @staticmethod
    def _build_score(metrics: list[tuple[str, float, bool]]) -> tuple[float, ...]:
        return tuple(metric_value if higher_better else -metric_value for _, metric_value, higher_better in metrics)
