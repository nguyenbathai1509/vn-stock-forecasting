-- Database for stock prediction pipeline
CREATE DATABASE StockDB;
GO

USE StockDB;
GO

-- Danh mục mã cổ phiếu, mapping mã VN và mã Yahoo
CREATE TABLE tickers (
    id INT IDENTITY PRIMARY KEY,
    ticker_vn NVARCHAR(50) NOT NULL,
    ticker_yahoo NVARCHAR(50) NOT NULL,
    [name] NVARCHAR(255) NOT NULL,
    sector NVARCHAR(255) NULL,
    CONSTRAINT uq_tickers_yahoo UNIQUE (ticker_yahoo),
    CONSTRAINT uq_tickers_vn UNIQUE (ticker_vn)
);
GO
-- Bảng giá raw (dữ liệu chưa làm sạch, cho phép null)
CREATE TABLE prices_raw (
    id BIGINT IDENTITY PRIMARY KEY,
    ticker_id INT NULL,
    trade_date DATE NULL,
    [open] FLOAT NULL,
    [high] FLOAT NULL,
    [low] FLOAT NULL,
    [close] FLOAT NULL,
    adj_close FLOAT NULL,
    volume BIGINT NULL,
    [source] NVARCHAR(50) NULL,
    inserted_at DATETIME NOT NULL DEFAULT GETDATE()
);
GO
-- Bảng giá đã làm sạch (không lưu raw)
CREATE TABLE prices_clean (
    id BIGINT IDENTITY PRIMARY KEY,
    ticker_id INT NOT NULL,
    trade_date DATE NOT NULL,
    [open] FLOAT NOT NULL,
    [high] FLOAT NOT NULL,
    [low] FLOAT NOT NULL,
    [close] FLOAT NOT NULL,
    adj_close FLOAT NOT NULL,
    volume BIGINT NOT NULL,
    cleaned_at DATETIME NOT NULL DEFAULT GETDATE(),
    CONSTRAINT uq_prices_clean UNIQUE (ticker_id, trade_date)
);
GO

-- Bảng lưu features kỹ thuật và nhãn cho training
CREATE TABLE features (
    id BIGINT IDENTITY PRIMARY KEY,
    ticker_id INT NOT NULL,
    trade_date DATE NOT NULL,
    [close] FLOAT NOT NULL,
    ma_5 FLOAT NOT NULL,
    ma_20 FLOAT NOT NULL,
    return_1d FLOAT NOT NULL,
    vol_10 FLOAT NOT NULL,
    label_next_return FLOAT NULL,
    generated_at DATETIME NOT NULL DEFAULT GETDATE(),
    CONSTRAINT uq_features UNIQUE (ticker_id, trade_date)
);
GO

-- Bảng lưu kết quả dự đoán (return/price) theo mô hình
CREATE TABLE predictions (
    id BIGINT IDENTITY PRIMARY KEY,
    ticker_id INT NOT NULL,
    trade_date DATE NOT NULL,
    model_name NVARCHAR(50) NOT NULL,
    horizon INT NOT NULL,
    pred_return FLOAT NOT NULL,
    pred_price FLOAT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT GETDATE()
);
GO

-- Bảng lưu giao dịch mô phỏng/backtest
CREATE TABLE backtest_trades (
    id BIGINT IDENTITY PRIMARY KEY,
    ticker_id INT NOT NULL,
    trade_date DATE NOT NULL,
    model_name NVARCHAR(50) NOT NULL,
    [signal] NVARCHAR(10) NOT NULL,
    [price] FLOAT NOT NULL,
    shares INT NOT NULL,
    cash_after FLOAT NOT NULL,
    position_after INT NOT NULL,
    equity_after FLOAT NOT NULL
);
GO

-- Foreign keys
ALTER TABLE prices_clean
    ADD CONSTRAINT fk_prices_clean_ticker FOREIGN KEY (ticker_id) REFERENCES tickers (id);
GO

ALTER TABLE features
    ADD CONSTRAINT fk_features_ticker FOREIGN KEY (ticker_id) REFERENCES tickers (id);
GO

ALTER TABLE predictions
    ADD CONSTRAINT fk_predictions_ticker FOREIGN KEY (ticker_id) REFERENCES tickers (id);
GO

ALTER TABLE backtest_trades
    ADD CONSTRAINT fk_backtest_trades_ticker FOREIGN KEY (ticker_id) REFERENCES tickers (id);
GO
SELECT p.trade_date AS pred_date, t.ticker_vn, p.model_name,
       p.pred_return, p.pred_price,
       c.[close] AS actual_close_next_day,
       c.trade_date AS actual_date
FROM predictions p
JOIN tickers t ON p.ticker_id = t.id
JOIN prices_clean c ON c.ticker_id = p.ticker_id
                  AND c.trade_date = DATEADD(day, 1, p.trade_date)
WHERE t.ticker_vn = 'FPT' AND p.trade_date = '2025-12-23';

-- Indexes for query performance
CREATE INDEX ix_prices_clean_ticker_date ON prices_clean (ticker_id, trade_date);
GO

CREATE INDEX ix_features_ticker_date ON features (ticker_id, trade_date);
GO

CREATE INDEX ix_predictions_ticker_date ON predictions (ticker_id, trade_date);
GO

CREATE INDEX ix_backtest_trades_ticker_date ON backtest_trades (ticker_id, trade_date);
GO
-- use master
-- ALTER DATABASE StockDB SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
-- DROP DATABASE StockDB;
