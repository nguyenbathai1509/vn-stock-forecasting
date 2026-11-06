"""Training pipeline for Linear Regression model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    df = df.sort_values("trade_date")
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return df


def split_data(features: pd.DataFrame, config: dict[str, any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test by time."""
    train_ratio = config.get("pipeline", {}).get("train_ratio", 0.7)
    val_ratio = config.get("pipeline", {}).get("val_ratio", 0.15)

    total = len(features)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_df = features.iloc[:train_end]
    val_df = features.iloc[train_end:val_end]
    test_df = features.iloc[val_end:]
    return train_df, val_df, test_df


def train_model(train_df: pd.DataFrame) -> Pipeline:
    """Train Linear Regression model with standard scaling."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model: Pipeline, output_dir: str = "models/linear_regression") -> Path:
    """Persist trained model to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "linear_regression_model.pkl"
    joblib.dump(model, output_path)
    return output_path


def main() -> None:
    """Entry point for training Linear Regression model."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)
    feature_df = load_features()
    if feature_df.empty:
        logger.warning("No features available for training.")
        return
    train_df, val_df, test_df = split_data(feature_df, config)
    logger.info("Dataset split into train=%s, val=%s, test=%s", len(train_df), len(val_df), len(test_df))
    model = train_model(train_df)
    model_path = save_model(model)
    logger.info("Saved Linear Regression model to %s", model_path)


if __name__ == "__main__":
    main()

