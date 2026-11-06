"""Utility helpers for configuration and logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure application-wide logging.

    Args:
        log_level: Logging level name (e.g., ``INFO``, ``DEBUG``).
        log_file: Optional path to write logs. If omitted, logs only to stdout.
    """
    log_config: Dict[str, Any] = {
        "level": getattr(logging, log_level.upper(), logging.INFO),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        log_config["filename"] = log_file
    logging.basicConfig(**log_config)


def load_config(config_path: str | Path = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f) or {}
    return config

