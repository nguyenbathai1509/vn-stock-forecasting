"""Training pipeline for LSTM model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

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
TARGET_COLUMN = "label_next_return"


def load_features(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Load feature set from database."""
    engine = get_engine(config_path)
    query = """
        SELECT trade_date, ticker_id, [close], ma_5, ma_20, return_1d, vol_10, label_next_return
        FROM features
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["ticker_id", "trade_date"])
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return df


def build_sequences(df: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences per ticker for LSTM."""
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for _, grp in df.groupby("ticker_id"):
        grp = grp.sort_values("trade_date")
        values = grp[FEATURE_COLUMNS].to_numpy()
        labels = grp[TARGET_COLUMN].to_numpy()
        for i in range(lookback, len(grp)):
            X_list.append(values[i - lookback : i])
            y_list.append(labels[i])

    return np.array(X_list), np.array(y_list)


def split_data(df: pd.DataFrame, config: dict[str, any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/val/test by time."""
    train_ratio = config.get("pipeline", {}).get("train_ratio", 0.7)
    val_ratio = config.get("pipeline", {}).get("val_ratio", 0.15)

    total = len(df)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def build_lstm_model(input_shape: Tuple[int, int]) -> Model:
    """Construct LSTM model architecture."""
    model = Sequential(
        [
            LSTM(64, input_shape=input_shape, return_sequences=False),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


def save_model(model: Model, scaler: MinMaxScaler, output_dir: str = "models/lstm") -> Tuple[Path, Path]:
    """Persist trained LSTM model and scaler to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(output_dir) / "lstm_model.h5"
    scaler_path = Path(output_dir) / "scaler.pkl"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path


def main() -> None:
    """Entry point for training LSTM model."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    lookback = config.get("pipeline", {}).get("lookback_window", 30)
    feature_df = load_features()
    if feature_df.empty:
        logger.warning("No features available for training.")
        return

    train_df, val_df, test_df = split_data(feature_df, config)
    logger.info("Split dataset into train=%s, val=%s, test=%s rows", len(train_df), len(val_df), len(test_df))

    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])

    def scale_df(df: pd.DataFrame) -> pd.DataFrame:
        scaled = df.copy()
        scaled[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])
        return scaled

    train_df_s = scale_df(train_df)
    val_df_s = scale_df(val_df)
    test_df_s = scale_df(test_df)

    X_train, y_train = build_sequences(train_df_s, lookback)
    X_val, y_val = build_sequences(val_df_s, lookback)
    X_test, y_test = build_sequences(test_df_s, lookback)

    if len(X_train) == 0 or len(X_val) == 0:
        logger.warning("Insufficient sequence data for training/validation.")
        return

    model = build_lstm_model(input_shape=(lookback, len(FEATURE_COLUMNS)))
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    model_path, scaler_path = save_model(model, scaler)
    logger.info("Saved LSTM model to %s and scaler to %s", model_path, scaler_path)


if __name__ == "__main__":
    main()

