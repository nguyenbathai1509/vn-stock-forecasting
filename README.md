# Stock Price Prediction & Trading Bot Skeleton

Pipeline skeleton for downloading Vietnamese tickers from Yahoo Finance, cleaning data, generating features, training Linear Regression and LSTM models, producing predictions, and backtesting trading signals with SQL Server storage.

## Pipeline Steps
1. **crawl_data.py**: Read tickers from `config/tickers_vn.csv`, fetch prices from Yahoo Finance, save to `data/raw/`, insert into `prices_raw`.
2. **preprocess.py**: Clean raw data, handle missing/duplicate rows, save to `data/processed/`, insert into `prices_clean`.
3. **feature_engineering.py**: Build features (`return_1d`, `ma_5`, `ma_20`, `vol_10`, `label_next_return`), save to `data/features/`, insert into `features`.
4. **train_linear.py**: Time-based split, train `LinearRegression`, save to `models/linear_regression/`.
5. **train_lstm.py**: Prepare sequences, build LSTM (Keras/TensorFlow), save to `models/lstm/`.
6. **predict.py**: Load latest features and trained models, generate next-day predictions, save CSV to `preds/`, insert into `predictions`.
7. **bot_backtest.py**: Generate signals, simulate portfolio, save results to `preds/backtest/`, insert into `backtest_trades`.
8. **run_pipeline.py**: Orchestrates all steps in sequence.

## SQL Server Setup
1. Install SQL Server and the **ODBC Driver 17 for SQL Server**.
2. Update credentials in `config/config.yaml`.
3. Run schema: `sqlcmd -S <server> -U <user> -P <password> -d <database> -i database/db_init.sql`.

## Environment & Dependencies
- Python 3.10+
- Requirements listed in `requirements.txt`. Install via:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline
1. Update `config/config.yaml` and `config/tickers_vn.csv` with actual values.
2. Ensure database is initialized.
3. Execute:

```bash
python -m project.src.run_pipeline
```

Each module can also be run individually for debugging.

## Data & Outputs
- Raw/processed/features CSVs stored under `data/`.
- Models stored under `models/`.
- Predictions saved to `preds/` and backtests to `preds/backtest/`.
- Database tables mirror each stage for durability and analysis.
