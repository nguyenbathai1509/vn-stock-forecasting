"""Data crawling from Yahoo Finance."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
import numpy as np
from sqlalchemy import bindparam, text

# Allow running both as package module and as standalone script.
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


def read_tickers(ticker_path: str = "config/tickers_vn.csv") -> pd.DataFrame:
    """Load ticker mapping from CSV.

    Args:
        ticker_path: Path to ticker CSV file.

    Returns:
        DataFrame containing tickers metadata.
    """
    return pd.read_csv(ticker_path)


def get_last_trade_date(config_path: str = "config/config.yaml") -> Optional[pd.Timestamp]:
    """Fetch the latest trade_date from prices_raw to support incremental crawl."""
    engine = get_engine(config_path)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(trade_date) AS max_dt FROM prices_raw")).scalar()
        if result:
            return pd.to_datetime(result)
    return None


def fetch_price_data(tickers: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Fetch price data from Yahoo Finance for provided tickers.

    Args:
        tickers: DataFrame of tickers to request.
        config: Pipeline configuration parameters.

    Returns:
        Raw price data.
    """
    ticker_list = tickers["ticker_yahoo"].dropna().unique().tolist()
    if not ticker_list:
        return pd.DataFrame()

    ticker_map = tickers.set_index("ticker_yahoo")["ticker_vn"].to_dict()

    # Decide window: incremental by start/end if provided, else fallback to period.
    pipeline_cfg = config.get("pipeline", {})
    incremental: bool = pipeline_cfg.get("incremental_crawl", True)
    start: Optional[str] = pipeline_cfg.get("download_start")
    end: Optional[str] = pipeline_cfg.get("download_end")
    period: str = pipeline_cfg.get("download_period", "5y")
    interval: str = pipeline_cfg.get("download_interval", "1d")

    if incremental and not start:
        last_date = get_last_trade_date()
        if last_date is not None:
            start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if start or end:
        data = yf.download(
            tickers=ticker_list,
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    else:
        data = yf.download(
            tickers=ticker_list,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )

    records: List[Dict[str, Any]] = []
    for ticker in ticker_list:
        if ticker not in data.columns.get_level_values(0):
            continue
        df = data[ticker].reset_index()
        df = df.rename(
            columns={
                "Date": "trade_date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        df["ticker_vn"] = ticker_map.get(ticker, ticker)
        df["source"] = "yahoo"
        records.extend(df.to_dict("records"))

    return pd.DataFrame(records)


def save_raw_csv(data: pd.DataFrame, output_dir: str = "data/raw") -> Path:
    """Save raw data to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "prices_raw.csv"
    data.to_csv(output_path, index=False)
    return output_path


def persist_raw_to_db(
    data: pd.DataFrame,
    tickers_ref: pd.DataFrame,
    config_path: str = "config/config.yaml",
) -> None:
    """Insert raw data into prices_raw and upsert tickers."""
    if data.empty:
        return

    ref = tickers_ref[["ticker_vn", "ticker_yahoo", "name", "sector"]].copy()
    ref["ticker_yahoo"] = ref["ticker_yahoo"].fillna(ref["ticker_vn"])
    ref["name"] = ref["name"].fillna(ref["ticker_vn"])
    ref["sector"] = ref["sector"].fillna("N/A")

    df = data.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], pd.NA)
    df = df.dropna(subset=["trade_date", "ticker_vn"] + numeric_cols)
    df[numeric_cols] = df[numeric_cols].astype(float)

    engine = get_engine(config_path)
    with engine.begin() as conn:
        unique_tickers = ref.drop_duplicates(subset=["ticker_vn"])
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
            unique_tickers.to_dict("records"),
        )

        ids = conn.execute(
            text("SELECT id, ticker_vn FROM tickers WHERE ticker_vn IN :tickers").bindparams(
                bindparam("tickers", expanding=True)
            ),
            {"tickers": unique_tickers["ticker_vn"].tolist()},
        ).fetchall()
        ticker_map = {row.ticker_vn: row.id for row in ids}

        df["ticker_id"] = df["ticker_vn"].map(ticker_map)
        df = df.dropna(subset=["ticker_id", "trade_date"])
        df = (
            df.sort_values(["ticker_id", "trade_date"])
            .drop_duplicates(subset=["ticker_id", "trade_date"], keep="last")
        )

        # Replace NaN with None to satisfy DB driver for nullable columns.
        df = df.where(pd.notnull(df), None)

        rows = df[
            [
                "ticker_id",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "source",
            ]
        ].to_dict("records")

        if not rows:
            return

        stmt = text(
            """
            INSERT INTO prices_raw (ticker_id, trade_date, [open], [high], [low], [close], adj_close, volume, [source])
            SELECT :ticker_id, :trade_date, :open, :high, :low, :close, :adj_close, :volume, :source
            WHERE NOT EXISTS (
                SELECT 1 FROM prices_raw WHERE ticker_id = :ticker_id AND trade_date = :trade_date
            );
            """
        )

        # Chunk insert to avoid huge batches.
        chunk_size = 2000
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            conn.execute(stmt, chunk)


def main() -> None:
    """Entry point for crawling data."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)
    tickers = read_tickers()
    logger.info("Loaded %s tickers", len(tickers))
    raw_data = fetch_price_data(tickers, config)
    logger.info("Downloaded %s rows", len(raw_data))
    csv_path = save_raw_csv(raw_data)
    logger.info("Saved raw data to %s", csv_path)
    persist_raw_to_db(raw_data, tickers)


if __name__ == "__main__":
    main()
