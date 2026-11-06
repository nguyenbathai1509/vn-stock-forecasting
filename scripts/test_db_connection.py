"""Quick script to test SQL Server connectivity using config."""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import text

# Add repository root to sys.path so `project` package can be imported.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.src.db import get_engine  # noqa: E402


def main(config_path: str = "config/config.yaml") -> None:
    """Attempt a simple SELECT 1 to verify connectivity."""
    engine = get_engine(config_path)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
        print(f"DB connection OK, SELECT 1 -> {result}")


if __name__ == "__main__":
    main()
