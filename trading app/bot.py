"""
bot.py
------
Main trading bot loop.

Usage:
  # Step 1: Train the model (first time only)
  python bot.py --train

  # Step 2: Run the live bot
  python bot.py --run

  # Optional: Run a backtest
  python bot.py --backtest
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv

from data_pipeline import fetch_candles, add_technical_indicators
from model import PricePredictor
from delta_client import DeltaClient
from signal_engine import SignalEngine

load_dotenv()

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config from .env
# ------------------------------------------------------------------
API_KEY              = os.getenv("DELTA_API_KEY", "")
API_SECRET           = os.getenv("DELTA_API_SECRET", "")
TRADING_MODE         = os.getenv("TRADING_MODE", "testnet")
SYMBOL               = os.getenv("SYMBOL", "BTCUSDT")
CANDLE_INTERVAL      = os.getenv("CANDLE_INTERVAL", "5m")
TRADE_SIZE_USD       = float(os.getenv("TRADE_SIZE_USD", "50"))
MAX_OPEN_TRADES      = int(os.getenv("MAX_OPEN_TRADES", "2"))
STOP_LOSS_PCT        = float(os.getenv("STOP_LOSS_PCT", "2.0"))
TAKE_PROFIT_PCT      = float(os.getenv("TAKE_PROFIT_PCT", "4.0"))
LOOKBACK             = int(os.getenv("LOOKBACK", "60"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

# Candle interval → seconds between bot ticks
INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400
}


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def run_training():
    log.info("=" * 60)
    log.info("TRAINING MODE")
    log.info("=" * 60)

    log.info(f"Fetching 500 candles for {SYMBOL} ({CANDLE_INTERVAL})…")
    df = fetch_candles(SYMBOL, interval=CANDLE_INTERVAL, limit=500, mode=TRADING_MODE)
    df = add_technical_indicators(df)
    log.info(f"Dataset shape after indicators: {df.shape}")

    predictor = PricePredictor(lookback=LOOKBACK)
    results   = predictor.train(df, epochs=100, batch_size=32)

    log.info("\n" + "=" * 40)
    log.info(f"Training complete.")
    log.info(f"CV direction accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
    log.info("Model saved to ./saved_model/")


# ------------------------------------------------------------------
# Backtest (simple walk-forward simulation)
# ------------------------------------------------------------------
def run_backtest():
    log.info("=" * 60)
    log.info("BACKTEST MODE (walk-forward simulation)")
    log.info("=" * 60)

    df = fetch_candles(SYMBOL, interval=CANDLE_INTERVAL, limit=500, mode=TRADING_MODE)
    df = add_technical_indicators(df)

    predictor = PricePredictor(lookback=LOOKBACK)
    predictor.load()

    engine = SignalEngine(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        stop_loss_pct=STOP_LOSS_PCT,
        take_profit_pct=TAKE_PROFIT_PCT,
        trade_size_usd=TRADE_SIZE_USD
    )

    wins, losses, holds = 0, 0, 0
    equity = 1000.0  # start with $1000 paper

    for i in range(LOOKBACK + 100, len(df) - 1):
        slice_df   = df.iloc[:i].copy()
        prediction = predictor.predict(slice_df)
        signal     = engine.generate_signal(prediction, slice_df, open_trades=0)

        if signal.action == "HOLD":
            holds += 1
            continue

        next_close = df["close"].iloc[i + 1]
        if signal.action == "BUY":
            pnl = (next_close - signal.entry_price) / signal.entry_price * TRADE_SIZE_USD
        else:
            pnl = (signal.entry_price - next_close) / signal.entry_price * TRADE_SIZE_USD

        equity += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1

    total_trades = wins + losses
    win_rate     = wins / total_trades if total_trades > 0 else 0
    log.info(f"Backtest complete: {total_trades} trades | Win rate: {win_rate:.1%} | Final equity: ${equity:.2f}")
    log.info(f"(HOLD signals: {holds})")
    return {"wins": wins, "losses": losses, "win_rate": win_rate, "equity": equity}


# ------------------------------------------------------------------
# Live trading loop
# ------------------------------------------------------------------
def run_live():
    log.info("=" * 60)
    log.info(f"LIVE BOT MODE — {TRADING_MODE.upper()}")
    log.info(f"Symbol: {SYMBOL} | Interval: {CANDLE_INTERVAL}")
    log.info("=" * 60)

    if not API_KEY or not API_SECRET:
        log.error("DELTA_API_KEY / DELTA_API_SECRET not set in .env — cannot connect.")
        sys.exit(1)

    client = DeltaClient(API_KEY, API_SECRET, mode=TRADING_MODE)
    engine = SignalEngine(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        stop_loss_pct=STOP_LOSS_PCT,
        take_profit_pct=TAKE_PROFIT_PCT,
        trade_size_usd=TRADE_SIZE_USD,
        max_open_trades=MAX_OPEN_TRADES
    )

    # Load model
    predictor = PricePredictor(lookback=LOOKBACK)
    try:
        predictor.load()
    except FileNotFoundError:
        log.error("No saved model found. Run `python bot.py --train` first.")
        sys.exit(1)

    # Get product_id once
    product_id = client.get_product_id(SYMBOL)
    if product_id is None:
        log.error(f"Could not find product_id for {SYMBOL}. Check the symbol name.")
        sys.exit(1)
    log.info(f"Product ID for {SYMBOL}: {product_id}")

    tick_seconds = INTERVAL_SECONDS.get(CANDLE_INTERVAL, 300)
    log.info(f"Bot will tick every {tick_seconds}s. Press Ctrl+C to stop.\n")

    while True:
        try:
            tick_start = time.time()
            log.info(f"--- Tick @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            # 1. Fetch fresh data
            df = fetch_candles(SYMBOL, interval=CANDLE_INTERVAL, limit=300, mode=TRADING_MODE)
            df = add_technical_indicators(df)

            # 2. Run model prediction
            prediction = predictor.predict(df)
            log.info(
                f"Model → {prediction['direction']} | "
                f"Confidence: {prediction['confidence']:.2%} | "
                f"Predicted next close: {prediction['next_price']:.2f}"
            )

            # 3. Count open trades
            open_orders  = client.get_open_orders(SYMBOL)
            open_count   = len(open_orders)

            # 4. Generate signal
            signal = engine.generate_signal(prediction, df, open_trades=open_count)
            log.info(f"Signal: {signal.action} — {signal.reason}")

            # 5. Execute trade
            if signal.action in ("BUY", "SELL"):
                side = "buy" if signal.action == "BUY" else "sell"
                log.info(
                    f"Placing {side.upper()} order: "
                    f"{signal.contracts} contract(s) @ ~{signal.entry_price:.2f} | "
                    f"SL={signal.stop_loss} | TP={signal.take_profit}"
                )
                order = client.place_order(
                    product_id=product_id,
                    side=side,
                    size=signal.contracts,
                    order_type="market_order",
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
                log.info(f"Order placed: {order.get('id')} | State: {order.get('state')}")
            else:
                log.info("Holding. No trade this tick.")

            # 6. Sleep until next candle
            elapsed = time.time() - tick_start
            sleep_time = max(0, tick_seconds - elapsed)
            log.info(f"Sleeping {sleep_time:.0f}s until next tick...\n")
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            log.info("Bot stopped by user.")
            break
        except Exception as e:
            log.error(f"Error in main loop: {e}", exc_info=True)
            log.info("Retrying in 60 seconds...")
            time.sleep(60)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delta Exchange LSTM Trading Bot")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train",     action="store_true", help="Train model on historical data")
    group.add_argument("--run",       action="store_true", help="Run the live trading bot")
    group.add_argument("--backtest",  action="store_true", help="Run a walk-forward backtest")
    args = parser.parse_args()

    if args.train:
        run_training()
    elif args.run:
        run_live()
    elif args.backtest:
        run_backtest()
