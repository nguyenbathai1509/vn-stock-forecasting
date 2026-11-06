"""Prediction generation using trained models (Linear and LSTM)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from tensorflow.keras.models import load_model

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


FEATURE_COLUMNS: List[str] = ["close", "ma_5", "ma_20", "return_1d", "vol_10"]


def load_latest_features(config: dict[str, Any], lookback: int) -> pd.DataFrame:
    """Load latest features from DB sufficient for lookback window."""
    engine = get_engine("config/config.yaml")
    query = text(
        """
        SELECT f.trade_date, f.ticker_id, t.ticker_vn, f.[close], f.ma_5, f.ma_20, f.return_1d, f.vol_10
        FROM features f
        JOIN tickers t ON f.ticker_id = t.id
        WHERE f.trade_date >= (
            SELECT DATEADD(day, -:lb, MAX(trade_date)) FROM features
        )
        ORDER BY f.ticker_id, f.trade_date;
        """
    )
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params={"lb": lookback + 1})


def load_linear_model(model_path: str) -> Any:
    """Load Linear Regression model."""
    return joblib.load(model_path)


def load_lstm_model(model_path: str, scaler_path: str) -> Tuple[Any, Any]:
    """Load LSTM model and scaler."""
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_linear(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Generate linear model predictions."""
    preds = model.predict(df[FEATURE_COLUMNS])
    df = df.copy()
    df["pred_return"] = preds
    return df


def prepare_lstm_sequences(df: pd.DataFrame, lookback: int, scaler: Any) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepare lookback sequences per ticker for LSTM inference."""
    seqs = []
    meta_rows = []
    for tid, grp in df.groupby("ticker_id"):
        grp = grp.sort_values("trade_date")
        values = scaler.transform(grp[FEATURE_COLUMNS])
        if len(grp) < lookback:
            continue
        seqs.append(values[-lookback:])
        meta_rows.append(grp.iloc[-1][["ticker_id", "ticker_vn", "trade_date", "close"]])
    if not seqs:
        return pd.DataFrame(), np.array([])
    X = np.array(seqs)
    meta = pd.DataFrame(meta_rows)
    return meta, X


def predict_lstm(model: Any, scaler: Any, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Generate LSTM predictions using latest sequences."""
    meta, X = prepare_lstm_sequences(df, lookback, scaler)
    if meta.empty:
        return pd.DataFrame()
    preds = model.predict(X, verbose=0).flatten()
    meta = meta.copy()
    meta["pred_return"] = preds
    return meta


def save_predictions(preds: pd.DataFrame, output_path: str = "preds/daily_predictions.csv") -> Path:
    """Save predictions to CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output, index=False)
    return output


def persist_predictions_to_db(preds: pd.DataFrame, config_path: str = "config/config.yaml") -> None:
    """Insert predictions into database."""
    if preds.empty:
        return
    engine = get_engine(config_path)

    def to_native_float(val: Any) -> Any:
        if val is None or pd.isna(val) or np.isinf(val):
            return None
        return float(val)

    rows = []
    for rec in preds.to_dict("records"):
        rows.append(
            {
                "ticker_id": int(rec["ticker_id"]),
                "trade_date": rec["trade_date"].date() if hasattr(rec["trade_date"], "date") else rec["trade_date"],
                "model_name": rec["model_name"],
                "horizon": int(rec["horizon"]),
                "pred_return": to_native_float(rec.get("pred_return")),
                "pred_price": to_native_float(rec.get("pred_price")),
            }
        )

    stmt = text(
        """
        INSERT INTO predictions (ticker_id, trade_date, model_name, horizon, pred_return, pred_price)
        VALUES (:ticker_id, :trade_date, :model_name, :horizon, :pred_return, :pred_price);
        """
    )
    chunk_size = 2000
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk_size):
            conn.execute(stmt, rows[i : i + chunk_size])


def main() -> None:
    """Entry point for prediction step."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    lookback = config.get("pipeline", {}).get("lookback_window", 30)
    horizon = config.get("pipeline", {}).get("prediction_horizon", 1)

    feats = load_latest_features(config, lookback)
    if feats.empty:
        logger.warning("No features available for prediction.")
        return

    # Linear predictions
    linear_model_path = "models/linear_regression/linear_regression_model.pkl"
    linear_model = load_linear_model(linear_model_path)
    linear_preds = predict_linear(linear_model, feats)
    linear_preds["model_name"] = "linear_regression"
    linear_preds["horizon"] = horizon
    linear_preds["pred_price"] = linear_preds["close"] * (1.0 + linear_preds["pred_return"])

    # LSTM predictions
    lstm_model_path = "models/lstm/lstm_model.h5"
    scaler_path = "models/lstm/scaler.pkl"
    lstm_model, lstm_scaler = load_lstm_model(lstm_model_path, scaler_path)
    lstm_preds = predict_lstm(lstm_model, lstm_scaler, feats, lookback)
    lstm_preds["model_name"] = "lstm"
    lstm_preds["horizon"] = horizon
    if not lstm_preds.empty:
        lstm_preds = lstm_preds.merge(
            feats[["ticker_id", "trade_date", "close"]],
            on=["ticker_id", "trade_date"],
            how="left",
            suffixes=("", "_price"),
        )
        lstm_preds["pred_price"] = lstm_preds["close"] * (1.0 + lstm_preds["pred_return"])

    all_preds = pd.concat([linear_preds, lstm_preds], ignore_index=True, sort=False)
    csv_path = save_predictions(all_preds)
    logger.info("Saved predictions to %s", csv_path)
    persist_predictions_to_db(all_preds)
    logger.info("Inserted predictions into database")


if __name__ == "__main__":
    main()
