"""Run full pipeline sequentially: crawl -> preprocess -> feature -> train -> predict -> backtest."""

from __future__ import annotations

import logging

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from importlib import import_module

    crawl_data = import_module("crawl_data")
    preprocess = import_module("preprocess")
    load_clean = import_module("load_clean_to_db")
    feature_engineering = import_module("feature_engineering")
    train_linear = import_module("train_linear")
    train_lstm = import_module("train_lstm")
    predict = import_module("predict")
    bot_backtest = import_module("bot_backtest")
    from utils import load_config, setup_logging
else:
    from . import (
        bot_backtest,
        crawl_data,
        feature_engineering,
        load_clean_to_db,
        predict,
        preprocess,
        train_linear,
        train_lstm,
    )
    from .utils import load_config, setup_logging


def main() -> None:
    """Run pipeline end-to-end."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    logger.info("Starting crawl step")
    crawl_data.main()

    logger.info("Starting preprocess step")
    preprocess.main()

    logger.info("Loading clean data to DB (if any)")
    load_clean_to_db.main()

    logger.info("Starting feature engineering step")
    feature_engineering.main()

    logger.info("Starting Linear Regression training")
    train_linear.main()

    logger.info("Starting LSTM training")
    train_lstm.main()

    logger.info("Starting prediction step")
    predict.main()

    logger.info("Starting backtest step")
    bot_backtest.main()


if __name__ == "__main__":
    main()

