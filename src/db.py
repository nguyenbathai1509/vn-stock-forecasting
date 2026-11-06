"""Database utilities for SQL Server connections."""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from utils import load_config  # type: ignore
else:
    from .utils import load_config


def build_connection_string(config: Dict[str, Any]) -> str:
    """Construct SQLAlchemy connection string for SQL Server via pyodbc.

    Args:
        config: Dictionary containing SQL Server connection parameters.

    Returns:
        A SQLAlchemy connection string.
    """
    username = config.get("username", "")
    password = config.get("password", "")
    server = config.get("server", "")
    database = config.get("database", "")
    driver = config.get("driver", "ODBC Driver 17 for SQL Server").replace(" ", "+")
    return f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}"


def get_engine(config_path: str = "config/config.yaml") -> Engine:
    """Create a SQLAlchemy engine for SQL Server.

    Args:
        config_path: Path to configuration file containing SQL Server settings.

    Returns:
        A SQLAlchemy ``Engine`` instance.
    """
    cfg = load_config(config_path).get("sql_server", {})
    connection_string = build_connection_string(cfg)
    engine: Engine = create_engine(connection_string)
    return engine
