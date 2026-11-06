"""Data cleaning for price series."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import text

if __package__ in (None, ""):
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from utils import load_config, setup_logging  # type: ignore
    from db import get_engine  # type: ignore
else:
    from .utils import load_config, setup_logging
    from .db import get_engine


def get_latest_clean_date(config_path: str = "config/config.yaml") -> Optional[pd.Timestamp]:
    """Get latest trade_date already processed in prices_clean."""
    engine = get_engine(config_path)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(trade_date) FROM prices_clean")).scalar()
        if result:
            return pd.to_datetime(result)
    return None


def read_raw_from_db(config_path: str = "config/config.yaml", incremental: bool = True) -> pd.DataFrame:
    """Read raw prices from database (optionally only new dates)."""
    engine = get_engine(config_path)
    last_date = get_latest_clean_date(config_path) if incremental else None
    base_query = """
        SELECT pr.*, t.ticker_vn
        FROM prices_raw pr
        JOIN tickers t ON pr.ticker_id = t.id
        {where_clause}
    """
    where_clause = ""
    params: dict[str, Any] = {}
    if last_date is not None:
        where_clause = "WHERE pr.trade_date > :last_date"
        params["last_date"] = last_date
    query = base_query.format(where_clause=where_clause)
    with engine.connect() as conn:
        return pd.read_sql_query(text(query), conn, params=params)


def clean_data(raw_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Clean and preprocess raw price data."""
    df = raw_df.copy()
    expected_cols = {
        "ticker_vn": "ticker_vn",
        "trade_date": "trade_date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "adj_close",
        "volume": "volume",
        "source": "source",
        "ticker_id": "ticker_id",
    }
    df = df.rename(columns=expected_cols)

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    df = df.dropna(subset=["ticker_vn", "ticker_id", "trade_date"] + numeric_cols)
    df = (
        df.sort_values(["ticker_id", "trade_date"])
        .drop_duplicates(subset=["ticker_id", "trade_date"], keep="last")
    )

    df = df[df["volume"] > 0]
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df.sort_values(["ticker_id", "trade_date"]).reset_index(drop=True)
    return df


def save_clean_csv(data: pd.DataFrame, output_dir: str = "data/processed") -> Path:
    """Save cleaned data to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "prices_clean.csv"
    data.to_csv(output_path, index=False)
    return output_path


def persist_clean_to_db(data: pd.DataFrame, config_path: str = "config/config.yaml") -> None:
    """Insert cleaned data into prices_clean."""
    if data.empty:
        return
    engine = get_engine(config_path)
    rows = data[
        ["ticker_id", "trade_date", "open", "high", "low", "close", "adj_close", "volume"]
    ].to_dict("records")
    stmt = text(
        """
        INSERT INTO prices_clean (ticker_id, trade_date, [open], [high], [low], [close], adj_close, volume)
        SELECT :ticker_id, :trade_date, :open, :high, :low, :close, :adj_close, :volume
        WHERE NOT EXISTS (
            SELECT 1 FROM prices_clean WHERE ticker_id = :ticker_id AND trade_date = :trade_date
        );
        """
    )
    chunk_size = 2000
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk_size):
            conn.execute(stmt, rows[i : i + chunk_size])


def main() -> None:
    """Entry point for preprocessing raw data."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)
    raw_df = read_raw_from_db(incremental=True)
    logger.info("Loaded %s raw rows", len(raw_df))
    clean_df = clean_data(raw_df, config)
    csv_path = save_clean_csv(clean_df)
    logger.info("Saved cleaned data to %s", csv_path)
    persist_clean_to_db(clean_df)


if __name__ == "__main__":
    main()
