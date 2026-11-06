"""Load cleaned price data into SQL Server."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import bindparam, text

if __package__ in (None, ""):
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from db import get_engine  # type: ignore
    from utils import load_config, setup_logging  # type: ignore
else:
    from .db import get_engine
    from .utils import load_config, setup_logging


def read_clean_csv(clean_path: str = "data/processed/prices_clean.csv") -> pd.DataFrame:
    """Read cleaned prices from CSV."""
    return pd.read_csv(clean_path)


def read_ticker_reference(ticker_path: str = "config/tickers_vn.csv") -> pd.DataFrame:
    """Read ticker mapping file."""
    return pd.read_csv(ticker_path)


def upsert_tickers(conn, tickers_df: pd.DataFrame) -> None:
    """Upsert tickers into tickers table."""
    tickers_params = tickers_df.to_dict("records")
    conn.execute(
        text(
            """
            MERGE tickers AS tgt
            USING (VALUES (:ticker_vn, :ticker_yahoo, :name, :sector)) AS src (ticker_vn, ticker_yahoo, name, sector)
                ON tgt.ticker_vn = src.ticker_vn
            WHEN NOT MATCHED THEN
                INSERT (ticker_vn, ticker_yahoo, name, sector)
                VALUES (src.ticker_vn, src.ticker_yahoo, src.name, src.sector)
            WHEN MATCHED THEN
                UPDATE SET ticker_yahoo = src.ticker_yahoo, name = src.name, sector = src.sector;
            """
        ),
        tickers_params,
    )


def build_ticker_map(conn, ticker_vns: List[str]) -> Dict[str, int]:
    """Fetch ticker_id map for given ticker_vn list."""
    rows = conn.execute(
        text("SELECT id, ticker_vn FROM tickers WHERE ticker_vn IN :tickers").bindparams(
            bindparam("tickers", expanding=True)
        ),
        {"tickers": ticker_vns},
    ).fetchall()
    return {row.ticker_vn: row.id for row in rows}


def insert_prices_clean(conn, data: pd.DataFrame) -> None:
    """Insert cleaned price data into prices_clean with dedup."""
    rows = data[
        ["ticker_id", "trade_date", "open", "high", "low", "close", "adj_close", "volume"]
    ].to_dict("records")
    if not rows:
        return

    conn.execute(
        text(
            """
            INSERT INTO prices_clean (ticker_id, trade_date, [open], [high], [low], [close], adj_close, volume)
            SELECT :ticker_id, :trade_date, :open, :high, :low, :close, :adj_close, :volume
            WHERE NOT EXISTS (
                SELECT 1 FROM prices_clean WHERE ticker_id = :ticker_id AND trade_date = :trade_date
            );
            """
        ),
        rows,
    )


def prepare_data(clean_df: pd.DataFrame, tickers_ref: pd.DataFrame) -> pd.DataFrame:
    """Join clean data with ticker reference and prep for DB insert."""
    ref = tickers_ref[["ticker_vn", "ticker_yahoo", "name", "sector"]].copy()
    ref["ticker_yahoo"] = ref["ticker_yahoo"].fillna(ref["ticker_vn"])
    merged = clean_df.merge(ref, on="ticker_vn", how="left")
    merged = merged.dropna(subset=["ticker_yahoo"])
    merged["trade_date"] = pd.to_datetime(merged["trade_date"]).dt.date
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    merged[numeric_cols] = merged[numeric_cols].apply(pd.to_numeric, errors="coerce")
    merged = merged.dropna(subset=numeric_cols)
    return merged


def main() -> None:
    """Entry point to load cleaned CSV into SQL Server."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    clean_df = read_clean_csv()
    logger.info("Loaded %s cleaned rows", len(clean_df))

    tickers_ref = read_ticker_reference()
    prepared = prepare_data(clean_df, tickers_ref)
    if prepared.empty:
        logger.info("No data to insert after preparation")
        return

    engine = get_engine("config/config.yaml")
    with engine.begin() as conn:
        unique_tickers = prepared[["ticker_vn", "ticker_yahoo", "name", "sector"]].drop_duplicates()
        upsert_tickers(conn, unique_tickers)
        ticker_map = build_ticker_map(conn, unique_tickers["ticker_vn"].tolist())
        prepared["ticker_id"] = prepared["ticker_vn"].map(ticker_map)
        prepared = prepared.dropna(subset=["ticker_id"])
        insert_prices_clean(conn, prepared)
    logger.info("Inserted cleaned prices into database")


if __name__ == "__main__":
    main()

