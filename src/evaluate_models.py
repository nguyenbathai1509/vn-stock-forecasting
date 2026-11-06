"""Evaluate model predictions against actuals."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sqlalchemy import text

# Allow running as module or standalone.
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


def load_pred_actual(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Load predictions with actual closes (shifted by horizon)."""
    engine = get_engine(config_path)
    query = text(
        """
        SELECT p.trade_date, p.ticker_id, t.ticker_vn, p.model_name, p.horizon,
               p.pred_return, p.pred_price,
               c.[close] AS actual_close
        FROM predictions p
        JOIN tickers t ON p.ticker_id = t.id
        LEFT JOIN prices_clean c
            ON p.ticker_id = c.ticker_id
           AND c.trade_date = DATEADD(day, p.horizon, p.trade_date)
        ORDER BY p.ticker_id, p.trade_date;
        """
    )
    with get_engine(config_path).connect() as conn:
        df = pd.read_sql_query(query, conn)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def add_actual_return(df: pd.DataFrame) -> pd.DataFrame:
    """Compute actual return per ticker based on actual_close."""
    df = df.copy()
    df["actual_return"] = np.nan
    for tid, grp in df.groupby("ticker_id"):
        grp = grp.sort_values("trade_date")
        df.loc[grp.index, "actual_return"] = grp["actual_close"].pct_change()
    return df


def compute_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Compute MAE/MAPE/RMSE for price and return."""
    def safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
        mask = (y_true != 0) & y_true.notna() & y_pred.notna()
        if mask.sum() == 0:
            return np.nan
        return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean())

    records: list[Dict[str, Any]] = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        price_mask = grp["actual_close"].notna() & grp["pred_price"].notna()
        ret_mask = grp["actual_return"].notna() & grp["pred_return"].notna()
        mae_price = (grp.loc[price_mask, "pred_price"] - grp.loc[price_mask, "actual_close"]).abs().mean()
        rmse_price = np.sqrt(((grp.loc[price_mask, "pred_price"] - grp.loc[price_mask, "actual_close"]) ** 2).mean()) if price_mask.any() else np.nan
        mape_price = safe_mape(grp.loc[price_mask, "actual_close"], grp.loc[price_mask, "pred_price"])

        mae_ret = (grp.loc[ret_mask, "pred_return"] - grp.loc[ret_mask, "actual_return"]).abs().mean()
        rmse_ret = np.sqrt(((grp.loc[ret_mask, "pred_return"] - grp.loc[ret_mask, "actual_return"]) ** 2).mean()) if ret_mask.any() else np.nan
        mape_ret = safe_mape(grp.loc[ret_mask, "actual_return"], grp.loc[ret_mask, "pred_return"])

        rec: Dict[str, Any] = {col: key for col, key in zip(group_cols, keys)}
        rec.update(
            {
                "count_price": price_mask.sum(),
                "mae_price": mae_price,
                "rmse_price": rmse_price,
                "mape_price": mape_price,
                "count_return": ret_mask.sum(),
                "mae_return": mae_ret,
                "rmse_return": rmse_ret,
                "mape_return": mape_ret,
            }
        )
        records.append(rec)
    return pd.DataFrame(records)


def save_metrics(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    df = load_pred_actual()
    if df.empty:
        logger.warning("No prediction data found for evaluation.")
        return

    df = add_actual_return(df)
    metrics_model = compute_metrics(df, ["model_name"])
    metrics_model_ticker = compute_metrics(df, ["model_name", "ticker_vn"])

    out_dir = Path("reports")
    save_metrics(metrics_model, out_dir / "metrics_by_model.csv")
    save_metrics(metrics_model_ticker, out_dir / "metrics_by_model_ticker.csv")

    logger.info("Saved metrics to %s", out_dir)
    logger.info("By model:\n%s", metrics_model)


if __name__ == "__main__":
    main()

