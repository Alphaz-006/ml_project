# Delta Exchange LSTM Trading Bot

A production-structured ML trading bot for Delta Exchange using a dual-head LSTM model.

---

## Architecture

```
data_pipeline.py   →  Fetches candles, computes 30+ technical indicators
model.py           →  Dual-head LSTM (direction + price prediction)
signal_engine.py   →  Multi-gate signal filter (confidence + confirmations + risk)
delta_client.py    →  Authenticated Delta Exchange REST API client
bot.py             →  Orchestrator — train / backtest / run
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your environment

```bash
cp .env.example .env
# Edit .env and fill in your API credentials
```

Get your API key from:  
**Delta Exchange → Account → API Management → Create API Key**

Set `TRADING_MODE=testnet` first. Delta testnet: https://testnet.delta.exchange

### 3. Train the model

```bash
python bot.py --train
```

This fetches 500 candles, computes indicators, runs 5-fold walk-forward cross-validation, and saves the model to `./saved_model/`.

Expected output:
```
Mean CV direction accuracy: 0.58 ± 0.03
Training final model on full dataset...
Model saved to saved_model/
```

### 4. Backtest

```bash
python bot.py --backtest
```

Simulates the signal engine on historical data and prints win rate + final equity.

### 5. Run live (testnet first!)

```bash
python bot.py --run
```

The bot will:
- Tick once per candle interval (e.g., every 5 minutes for `5m`)
- Fetch fresh data → run model → generate signal
- Place market orders with automatic stop-loss and take-profit brackets
- Log everything to `bot.log`

---

## Configuration Reference (`.env`)

| Variable               | Default    | Description                                                        |
|------------------------|------------|--------------------------------------------------------------------|
| `DELTA_API_KEY`        | —          | Your Delta Exchange API key                                        |
| `DELTA_API_SECRET`     | —          | Your Delta Exchange API secret                                     |
| `TRADING_MODE`         | `testnet`  | `testnet` for paper trading, `live` for real money                 |
| `SYMBOL`               | `BTCUSDT`  | Trading pair (check Delta for exact symbol names)                  |
| `CANDLE_INTERVAL`      | `5m`       | `1m` `5m` `15m` `1h` `4h` `1d`                                    |
| `TRADE_SIZE_USD`       | `50`       | USD notional per trade                                             |
| `MAX_OPEN_TRADES`      | `2`        | Maximum simultaneous positions                                     |
| `STOP_LOSS_PCT`        | `2.0`      | Stop loss distance (%)                                             |
| `TAKE_PROFIT_PCT`      | `4.0`      | Take profit distance (%)                                           |
| `LOOKBACK`             | `60`       | Candles fed into LSTM per prediction                               |
| `CONFIDENCE_THRESHOLD` | `0.65`     | Minimum model confidence to consider trading (0–1)                 |

---

## Model Details

**Input:** 32 features × 60 candles = (60, 32) sequence per sample

**Features include:**
- Raw OHLCV
- EMA 9/21/50, MACD, ADX (trend)
- RSI, Stochastic %K/%D, Williams %R, ROC (momentum)
- Bollinger Bands (width, %B), ATR (volatility)
- OBV, VWAP, volume ratio (volume)
- Log returns, price vs. EMA/VWAP (price-derived)

**Architecture:**
```
Input (60, 32) → LSTM(128) → Dropout(0.3) → LSTM(128) → Dropout(0.3)
             → Dense(64) → BatchNorm
             ├── Dense(32) → Dense(1, sigmoid)  ← direction (UP/DOWN)
             └── Dense(32) → Dense(1, linear)   ← next close price
```

**Training:** 5-fold TimeSeriesSplit cross-validation + early stopping + LR scheduling

---

## Signal Gate Logic

A trade is only placed when ALL three gates pass:

```
Gate 1: Model confidence ≥ CONFIDENCE_THRESHOLD (default 0.65)
Gate 2: Open positions < MAX_OPEN_TRADES
Gate 3: At least 2 of 4 technical indicators confirm direction:
        - EMA alignment (9 > 21 > 50 for UP)
        - RSI zone (< 40 → UP, > 60 → DOWN)
        - MACD histogram direction
        - Bollinger %B (< 0.15 → UP, > 0.85 → DOWN)
```

This multi-gate system reduces false signals significantly.

---

## Risk Warning

- **Past performance does not guarantee future results.**
- No ML model can reliably predict crypto prices with high accuracy.
- Always test on testnet before using real funds.
- Never trade money you cannot afford to lose.
- Start with very small position sizes when going live.
- Set hard daily loss limits and monitor the bot actively.

---

## File Structure

```
delta_trading_bot/
├── bot.py              Main entry point
├── data_pipeline.py    OHLCV fetch + indicator computation
├── model.py            LSTM model definition + training
├── signal_engine.py    Signal filtering + risk management
├── delta_client.py     Delta Exchange API client
├── requirements.txt    Python dependencies
├── .env.example        Config template
└── saved_model/        Created after training
    ├── model.keras
    └── scaler.pkl
```
