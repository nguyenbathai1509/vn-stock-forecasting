"""Feature generation for model training.
Optimized for SQL Server (MS SQL) with explicit type casting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

# Allow running as module or standalone script.
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


def get_latest_feature_date(config_path: str = "config/config.yaml") -> Optional[pd.Timestamp]:
    """Fetch latest trade_date already present in features table."""
    engine = get_engine(config_path)
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT MAX(trade_date) FROM features")).scalar()
            if result:
                return pd.to_datetime(result)
    except Exception:
        return None
    return None


def read_clean_incremental(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Read clean prices from DB with a 60-day buffer for technical indicators."""
    last_feat_date = get_latest_feature_date(config_path)
    engine = get_engine(config_path)

    where_clause = ""
    params: Dict[str, Any] = {}

    if last_feat_date is not None:
        start_date = (last_feat_date - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        where_clause = "WHERE pc.trade_date >= :start_date"
        params["start_date"] = start_date

    query = f"""
        SELECT pc.trade_date, pc.[close], pc.[open], pc.[high], pc.[low], pc.adj_close, pc.volume,
               t.ticker_vn, pc.ticker_id
        FROM prices_clean pc
        JOIN tickers t ON pc.ticker_id = t.id
        {where_clause};
    """
    with engine.connect() as conn:
        return pd.read_sql_query(text(query), conn, params=params)


def create_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features and labels."""
    df = clean_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["ticker_vn", "trade_date"])

    numeric_cols = ["close", "open", "high", "low", "adj_close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    features_list: List[pd.DataFrame] = []
    for _, grp in df.groupby("ticker_vn"):
        grp = grp.sort_values("trade_date")
        grp["return_1d"] = grp["close"].pct_change()
        grp["ma_5"] = grp["close"].rolling(5).mean()
        grp["ma_20"] = grp["close"].rolling(20).mean()
        grp["vol_10"] = grp["return_1d"].rolling(10).std()
        grp["label_next_return"] = grp["return_1d"].shift(-1)
        features_list.append(grp)

    feat_df = pd.concat(features_list, ignore_index=True)
    feat_df = feat_df.dropna(subset=["ma_5", "ma_20", "return_1d", "vol_10"])
    return feat_df


def save_feature_csv(data: pd.DataFrame, output_dir: str = "data/features") -> Path:
    """Save engineered features to CSV (optional artifact)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "features.csv"
    data.to_csv(output_path, index=False)
    return output_path


def persist_features_to_db(data: pd.DataFrame, config_path: str = "config/config.yaml") -> None:
    """Persist engineered features to SQL Server with safe casting."""
    if data.empty:
        return

    def to_native_float(val: Any) -> Optional[float]:
        if pd.isna(val) or np.isinf(val):
            return None
        return float(round(val, 8))

    rows = []
    for record in data.to_dict("records"):
        rows.append(
            {
                "ticker_id": int(record["ticker_id"]),
                "trade_date": record["trade_date"].date()
                if hasattr(record["trade_date"], "date")
                else record["trade_date"],
                "close": to_native_float(record["close"]),
                "ma_5": to_native_float(record["ma_5"]),
                "ma_20": to_native_float(record["ma_20"]),
                "return_1d": to_native_float(record["return_1d"]),
                "vol_10": to_native_float(record["vol_10"]),
                "label_next_return": to_native_float(record["label_next_return"]),
            }
        )

    engine = get_engine(config_path)
    stmt = text(
        """
        MERGE features AS tgt
        USING (VALUES (:ticker_id, :trade_date, :close, :ma_5, :ma_20, :return_1d, :vol_10, :label_next_return))
            AS src (ticker_id, trade_date, [close], ma_5, ma_20, return_1d, vol_10, label_next_return)
        ON tgt.ticker_id = src.ticker_id AND tgt.trade_date = src.trade_date
        WHEN MATCHED THEN
            UPDATE SET [close] = src.[close],
                       ma_5 = src.ma_5,
                       ma_20 = src.ma_20,
                       return_1d = src.return_1d,
                       vol_10 = src.vol_10,
                       label_next_return = src.label_next_return
        WHEN NOT MATCHED THEN
            INSERT (ticker_id, trade_date, [close], ma_5, ma_20, return_1d, vol_10, label_next_return)
            VALUES (src.ticker_id, src.trade_date, src.[close], src.ma_5, src.ma_20, src.return_1d, src.vol_10, src.label_next_return);
        """
    )

    with engine.begin() as conn:
        chunk_size = 2000
        for i in range(0, len(rows), chunk_size):
            conn.execute(stmt, rows[i : i + chunk_size])


def main() -> None:
    """Main entrypoint for feature engineering."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    clean_df = read_clean_incremental()
    logger.info("Loaded %s cleaned rows (including buffer)", len(clean_df))
    if clean_df.empty:
        logger.warning("No new data to process.")
        return

    feature_df = create_features(clean_df)
    logger.info("Generated features for %s records", len(feature_df))
    csv_path = save_feature_csv(feature_df)
    logger.info("Saved features to %s", csv_path)
    persist_features_to_db(feature_df)
    logger.info("Successfully persisted features and updated labels in DB.")


if __name__ == "__main__":
    main()
