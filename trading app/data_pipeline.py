"""
data_pipeline.py
----------------
Fetches OHLCV candle data from Delta Exchange and computes a rich set of
technical indicators used as model features.
"""

import requests
import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume
import time
from datetime import datetime, timedelta


DELTA_BASE_URL = "https://api.delta.exchange"
DELTA_TESTNET_URL = "https://testnet-api.delta.exchange"


def get_base_url(mode="testnet"):
    return DELTA_TESTNET_URL if mode == "testnet" else DELTA_BASE_URL


def fetch_candles(symbol: str, interval: str = "5m", limit: int = 500, mode: str = "testnet") -> pd.DataFrame:
    """
    Fetch OHLCV candles from Delta Exchange public API.

    Args:
        symbol:   Trading pair, e.g. 'BTCUSDT'
        interval: Candle size — '1m','5m','15m','1h','4h','1d'
        limit:    Number of candles to fetch (max 500)
        mode:     'testnet' or 'live'

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    resolution_map = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "1d": 1440
    }
    resolution = resolution_map.get(interval)
    if resolution is None:
        raise ValueError(f"Unsupported interval: {interval}. Choose from {list(resolution_map.keys())}")

    end_time = int(time.time())
    start_time = end_time - (resolution * 60 * limit)

    url = f"{get_base_url(mode)}/v2/history/candles"
    params = {
        "resolution": resolution,
        "symbol": symbol,
        "start": start_time,
        "end": end_time
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to fetch candles from Delta Exchange: {e}")

    if not data.get("success") or not data.get("result"):
        raise ValueError(f"Empty or error response from Delta Exchange: {data}")

    candles = data["result"]
    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a comprehensive set of technical indicators and append them as
    columns to the OHLCV DataFrame.

    Feature groups:
      - Trend:      EMA-9, EMA-21, EMA-50, MACD, ADX
      - Momentum:   RSI-14, Stochastic %K/%D, Williams %R, ROC
      - Volatility: Bollinger Bands (upper/mid/lower/width/pct), ATR-14
      - Volume:     OBV, VWAP (rolling), Volume ratio
      - Price:      Log returns, Price relative to EMAs
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # --- Trend ---
    df["ema_9"]  = trend.EMAIndicator(close, window=9).ema_indicator()
    df["ema_21"] = trend.EMAIndicator(close, window=21).ema_indicator()
    df["ema_50"] = trend.EMAIndicator(close, window=50).ema_indicator()

    macd_obj = trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"]   = macd_obj.macd_diff()

    df["adx"] = trend.ADXIndicator(high, low, close, window=14).adx()

    # --- Momentum ---
    df["rsi"] = momentum.RSIIndicator(close, window=14).rsi()

    stoch = momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["williams_r"] = momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    df["roc"]        = momentum.ROCIndicator(close, window=10).roc()

    # --- Volatility ---
    bb = volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_mid"]    = bb.bollinger_mavg()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_width"]  = bb.bollinger_wband()
    df["bb_pct"]    = bb.bollinger_pband()

    df["atr"] = volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # --- Volume ---
    df["obv"] = volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()

    # Rolling VWAP (20-period approximation)
    typical_price  = (high + low + close) / 3
    df["vwap"]     = (typical_price * vol).rolling(20).sum() / vol.rolling(20).sum()

    # Volume relative to 20-period average
    df["volume_ratio"] = vol / vol.rolling(20).mean()

    # --- Price-derived ---
    df["log_return"]      = np.log(close / close.shift(1))
    df["price_vs_ema9"]   = (close - df["ema_9"])  / df["ema_9"]
    df["price_vs_ema21"]  = (close - df["ema_21"]) / df["ema_21"]
    df["price_vs_vwap"]   = (close - df["vwap"])   / df["vwap"]
    df["high_low_spread"] = (high - low) / close

    # Target: 1 if next close > current close, 0 otherwise (classification)
    # Also store next close for regression head
    df["target_direction"] = (close.shift(-1) > close).astype(int)
    df["target_next_close"] = close.shift(-1)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "ema_9", "ema_21", "ema_50",
    "macd", "macd_signal", "macd_diff", "adx",
    "rsi", "stoch_k", "stoch_d", "williams_r", "roc",
    "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct", "atr",
    "obv", "vwap", "volume_ratio",
    "log_return", "price_vs_ema9", "price_vs_ema21",
    "price_vs_vwap", "high_low_spread"
]
