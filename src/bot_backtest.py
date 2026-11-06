"""Backtesting bot for trading signals based on predictions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

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


def load_predictions(config_path: str = "config/config.yaml", history: bool = False) -> pd.DataFrame:
    """Load predictions from DB. If history=True, load all for backtest."""
    engine = get_engine(config_path)
    if history:
        query = """
            SELECT p.trade_date, p.ticker_id, t.ticker_vn, p.model_name, p.horizon,
                   p.pred_return, p.pred_price,
                   c.[close] AS actual_close
            FROM predictions p
            JOIN tickers t ON p.ticker_id = t.id
            LEFT JOIN prices_clean c
                ON p.ticker_id = c.ticker_id
               AND c.trade_date = DATEADD(day, p.horizon, p.trade_date)
            ORDER BY p.trade_date, p.ticker_id;
        """
    else:
        query = """
            SELECT p.trade_date, p.ticker_id, t.ticker_vn, p.model_name, p.horizon, p.pred_return, p.pred_price
            FROM predictions p
            JOIN tickers t ON p.ticker_id = t.id
            WHERE p.trade_date = (SELECT MAX(trade_date) FROM predictions)
        """
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn)


def generate_signals(preds: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Generate trading signals based on predicted returns."""
    df = preds.copy()
    df["signal"] = "HOLD"
    df.loc[df["pred_return"] > threshold, "signal"] = "BUY"
    df.loc[df["pred_return"] < -threshold, "signal"] = "SELL"
    return df


def run_backtest(
    signals: pd.DataFrame,
    initial_cash: float = 1_000_000.0,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
    position_size_pct: float = 0.1,
) -> pd.DataFrame:
    """Portfolio simulation with stop-loss, take-profit, and position sizing over history."""
    df = signals.sort_values(["trade_date", "ticker_id"]).copy()
    cash = initial_cash
    positions: Dict[int, int] = {}  # per-ticker position
    entry_price: Dict[int, float] = {}  # entry price per ticker
    last_price: Dict[int, float] = {}  # last seen price per ticker
    records = []

    for _, row in df.iterrows():
        tid = int(row["ticker_id"])
        actual = row.get("actual_close")
        if pd.isna(actual):
            # Skip if no actual price to avoid look-ahead bias with pred_price.
            continue
        price = float(actual)
        signal = row["signal"]
        shares = 0

        pos = positions.get(tid, 0)
        ent = entry_price.get(tid)

        # Update stop-loss / take-profit for this ticker.
        if pos > 0 and price > 0 and ent:
            if price <= ent * (1 - stop_loss_pct) or price >= ent * (1 + take_profit_pct):
                shares = pos
                cash += shares * price
                positions[tid] = 0
                entry_price.pop(tid, None)
                last_price[tid] = price
                equity = cash + sum(p * last_price.get(t, 0.0) for t, p in positions.items())
                records.append(
                    {
                        "ticker_id": tid,
                        "trade_date": row["trade_date"],
                        "model_name": row["model_name"],
                        "signal": "STOP/TAKE",
                        "price": price,
                        "shares": shares,
                        "cash_after": cash,
                        "position_after": positions.get(tid, 0),
                        "equity_after": equity,
                    }
                )
                continue

        if signal == "BUY" and price > 0:
            alloc = cash * position_size_pct
            shares = int(alloc // price)
            if shares > 0:
                cash -= shares * price
                positions[tid] = positions.get(tid, 0) + shares
                entry_price[tid] = price
        elif signal == "SELL" and pos > 0 and price > 0:
            shares = pos
            cash += shares * price
            positions[tid] = 0
            entry_price.pop(tid, None)

        last_price[tid] = price
        equity = cash + sum(p * last_price.get(t, 0.0) for t, p in positions.items())
        records.append(
            {
                "ticker_id": tid,
                "trade_date": row["trade_date"],
                "model_name": row["model_name"],
                "signal": signal,
                "price": price,
                "shares": shares,
                "cash_after": cash,
                "position_after": positions.get(tid, 0),
                "equity_after": equity,
            }
        )
    return pd.DataFrame(records)


def save_backtest(results: pd.DataFrame, output_dir: str = "preds/backtest") -> Path:
    """Save backtest results to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "backtest_results.csv"
    results.to_csv(output_path, index=False)
    return output_path


def persist_backtest_to_db(results: pd.DataFrame, config_path: str = "config/config.yaml") -> None:
    """Insert backtest trades into database."""
    if results.empty:
        return
    engine = get_engine(config_path)
    rows = results.to_dict("records")
    stmt = text(
        """
        INSERT INTO backtest_trades (ticker_id, trade_date, model_name, signal, price, shares, cash_after, position_after, equity_after)
        VALUES (:ticker_id, :trade_date, :model_name, :signal, :price, :shares, :cash_after, :position_after, :equity_after);
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, rows)


def main() -> None:
    """Entry point for backtesting predictions."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)
    preds = load_predictions(history=True)
    if preds.empty:
        logger.warning("No predictions found for backtest.")
        return
    pipeline_cfg = config.get("pipeline", {})
    signals = generate_signals(preds, threshold=pipeline_cfg.get("signal_threshold", 0.0))
    results = run_backtest(
        signals,
        initial_cash=pipeline_cfg.get("initial_cash", 1_000_000.0),
        stop_loss_pct=pipeline_cfg.get("stop_loss_pct", 0.02),
        take_profit_pct=pipeline_cfg.get("take_profit_pct", 0.04),
        position_size_pct=pipeline_cfg.get("position_size_pct", 0.1),
    )
    csv_path = save_backtest(results)
    logger.info("Saved backtest results to %s", csv_path)
    persist_backtest_to_db(results)
    logger.info("Inserted backtest trades into database")


if __name__ == "__main__":
    main()
