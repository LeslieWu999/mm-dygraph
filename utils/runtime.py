from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path


def ensure_clean_dir(directory: str | Path) -> Path:
    path = Path(directory)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(directory: str | Path) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(log_dir: str | Path, logger_name: str) -> logging.Logger:
    log_path = ensure_dir(log_dir)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path / "run.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def save_json(path: str | Path, payload: dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
    return output_path


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace("+", "_").replace(" ", "_")
