"""Generate analysis charts for price, predictions, and backtests."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import text

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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_prices(ticker: str, config_path: str = "config/config.yaml") -> pd.DataFrame:
    engine = get_engine(config_path)
    query = text(
        """
        SELECT pc.trade_date, pc.[close], pc.volume
        FROM prices_clean pc
        JOIN tickers t ON pc.ticker_id = t.id
        WHERE t.ticker_vn = :ticker
        ORDER BY pc.trade_date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"ticker": ticker})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_predictions(ticker: str, config_path: str = "config/config.yaml") -> pd.DataFrame:
    engine = get_engine(config_path)
    query = text(
        """
        SELECT p.trade_date, p.pred_price, p.pred_return, p.model_name,
               c.[close] AS actual_close
        FROM predictions p
        JOIN tickers t ON p.ticker_id = t.id
        LEFT JOIN prices_clean c
            ON c.ticker_id = p.ticker_id AND c.trade_date = DATEADD(day, p.horizon, p.trade_date)
        WHERE t.ticker_vn = :ticker
        ORDER BY p.trade_date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"ticker": ticker})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_backtest(config_path: str = "config/config.yaml") -> pd.DataFrame:
    engine = get_engine(config_path)
    query = """
        SELECT bt.*, t.ticker_vn
        FROM backtest_trades bt
        JOIN tickers t ON bt.ticker_id = t.id
        ORDER BY bt.trade_date
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def plot_price_volume(df: pd.DataFrame, ticker: str, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["trade_date"], df["close"], label="Close", color="blue")
    ax1.plot(df["trade_date"], df["close"].rolling(5).mean(), label="MA5", color="orange")
    ax1.plot(df["trade_date"], df["close"].rolling(20).mean(), label="MA20", color="green")
    ax1.set_title(f"{ticker} Price with MA5/MA20")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.bar(df["trade_date"], df["volume"], alpha=0.2, color="gray")
    ax2.set_ylabel("Volume")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_return_hist(df: pd.DataFrame, out: Path) -> None:
    df = df.copy()
    df["return_1d"] = df["close"].pct_change()
    fig, ax = plt.subplots(figsize=(8, 4))
    df["return_1d"].dropna().hist(ax=ax, bins=50, color="steelblue")
    ax.set_title("Return Distribution (1D)")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_pred_vs_actual(df_pred: pd.DataFrame, out_line: Path, out_scatter: Path) -> None:
    if df_pred.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_pred["trade_date"], df_pred["actual_close"], label="Actual", color="black")
    ax.plot(df_pred["trade_date"], df_pred["pred_price"], label="Predicted", color="red", alpha=0.7)
    ax.set_title("Actual vs Predicted Price (Line)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_line, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df_pred["actual_close"], df_pred["pred_price"], alpha=0.4)
    maxv = max(df_pred["actual_close"].max(), df_pred["pred_price"].max())
    minv = min(df_pred["actual_close"].min(), df_pred["pred_price"].min())
    ax.plot([minv, maxv], [minv, maxv], color="red", linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (Scatter)")
    fig.tight_layout()
    fig.savefig(out_scatter, dpi=200)
    plt.close(fig)


def plot_residual(df_pred: pd.DataFrame, out: Path) -> None:
    df = df_pred.dropna(subset=["actual_close", "pred_price"]).copy()
    if df.empty:
        return
    df["residual"] = df["actual_close"] - df["pred_price"]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df["trade_date"], df["residual"], color="purple")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Residuals (Actual - Predicted)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_price_signals(df_price: pd.DataFrame, trades: pd.DataFrame, ticker: str, out: Path) -> None:
    df_price = df_price.copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_price["trade_date"], df_price["close"], label="Close", color="blue")
    buys = trades[(trades["ticker_vn"] == ticker) & (trades["signal"].str.contains("BUY"))]
    sells = trades[(trades["ticker_vn"] == ticker) & (trades["signal"].str.contains("SELL|STOP/TAKE"))]
    ax.scatter(buys["trade_date"], buys["price"], color="green", label="BUY", marker="^")
    ax.scatter(sells["trade_date"], sells["price"], color="red", label="SELL/STOP", marker="v")
    ax.set_title(f"{ticker} Price with Signals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_equity_curve(trades: pd.DataFrame, out: Path) -> None:
    if trades.empty:
        return
    df = trades.sort_values("trade_date")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["trade_date"], df["equity_after"], color="blue")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_pnl_per_ticker(trades: pd.DataFrame, out: Path) -> None:
    if trades.empty:
        return
    pnl = trades.copy()
    pnl["pnl"] = pnl.groupby("ticker_id")["equity_after"].diff().fillna(0)
    agg = pnl.groupby("ticker_vn")["pnl"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    agg.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("PnL per Ticker")
    ax.set_ylabel("PnL")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_trade_counts(trades: pd.DataFrame, out: Path) -> None:
    if trades.empty:
        return
    counts = trades.groupby(["ticker_vn", "signal"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, figsize=(8, 4))
    plt.title("Trade Counts by Ticker and Signal")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_win_rate(trades: pd.DataFrame, out: Path) -> None:
    if trades.empty:
        return
    df = trades.copy()
    df["pnl"] = df.groupby("ticker_id")["equity_after"].diff().fillna(0)
    wins = (df["pnl"] > 0).sum()
    losses = (df["pnl"] <= 0).sum()
    fig, ax = plt.subplots()
    ax.pie([wins, losses], labels=["Win", "Loss"], autopct="%1.1f%%", colors=["green", "red"])
    ax.set_title("Win Rate")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_trade_pnl_hist(trades: pd.DataFrame, out: Path) -> None:
    if trades.empty:
        return
    df = trades.copy()
    df["pnl"] = df.groupby("ticker_id")["equity_after"].diff().fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    df["pnl"].hist(ax=ax, bins=50, color="gray")
    ax.set_title("Per-Trade PnL Distribution")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_bot_vs_buy_hold(prices: pd.DataFrame, trades: pd.DataFrame, out: Path) -> None:
    if trades.empty or prices.empty:
        return
    start_price = prices["close"].iloc[0]
    end_price = prices["close"].iloc[-1]
    buy_hold_return = (end_price / start_price) - 1 if start_price > 0 else 0
    bot_return = (trades["equity_after"].iloc[-1] / trades["equity_after"].iloc[0]) - 1 if len(trades) > 1 else 0
    fig, ax = plt.subplots()
    ax.bar(["BOT", "BUY&HOLD"], [bot_return * 100, buy_hold_return * 100], color=["blue", "orange"])
    ax.set_ylabel("Return (%)")
    ax.set_title("BOT vs Buy & Hold")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    ticker = config.get("pipeline", {}).get("report_ticker", "FPT")
    out_dir = Path("reports/charts")
    ensure_dir(out_dir)

    price_df = load_prices(ticker)
    pred_df = load_predictions(ticker)
    bt_df = load_backtest()

    logger.info("Generating charts for ticker %s", ticker)
    plot_price_volume(price_df, ticker, out_dir / f"{ticker}_price_ma_volume.png")
    plot_return_hist(price_df, out_dir / f"{ticker}_return_hist.png")
    plot_pred_vs_actual(pred_df, out_dir / f"{ticker}_pred_line.png", out_dir / f"{ticker}_pred_scatter.png")
    plot_residual(pred_df, out_dir / f"{ticker}_residual.png")
    plot_price_signals(price_df, bt_df, ticker, out_dir / f"{ticker}_signals.png")
    plot_equity_curve(bt_df, out_dir / "equity_curve.png")
    plot_pnl_per_ticker(bt_df, out_dir / "pnl_per_ticker.png")
    plot_trade_counts(bt_df, out_dir / "trade_counts.png")
    plot_win_rate(bt_df, out_dir / "win_rate.png")
    plot_trade_pnl_hist(bt_df, out_dir / "trade_pnl_hist.png")
    plot_bot_vs_buy_hold(price_df, bt_df, out_dir / "bot_vs_buyhold.png")
    logger.info("Charts saved to %s", out_dir)


if __name__ == "__main__":
    main()

