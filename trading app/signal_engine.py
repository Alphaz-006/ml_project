"""
signal_engine.py
----------------
Converts raw model output into a trading signal, applying:
  - Confidence threshold gate
  - Technical indicator confirmation (multi-confirmation system)
  - Position sizing (fixed fractional)
  - Stop-loss / take-profit calculation

A trade is only entered when BOTH the model AND at least 2 of 3
confirmatory indicator checks agree on direction.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeSignal:
    action:        str          # "BUY", "SELL", or "HOLD"
    direction:     str          # "UP" or "DOWN"
    confidence:    float
    entry_price:   float
    stop_loss:     float
    take_profit:   float
    contracts:     int
    reason:        str          # Human-readable rationale


class SignalEngine:
    def __init__(
        self,
        confidence_threshold: float = 0.65,
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
        trade_size_usd: float = 50.0,
        max_open_trades: int = 2
    ):
        self.confidence_threshold = confidence_threshold
        self.stop_loss_pct        = stop_loss_pct / 100
        self.take_profit_pct      = take_profit_pct / 100
        self.trade_size_usd       = trade_size_usd
        self.max_open_trades      = max_open_trades

    # ------------------------------------------------------------------
    # Indicator confirmation checks
    # ------------------------------------------------------------------

    def _check_ema_trend(self, df: pd.DataFrame) -> Optional[str]:
        """
        EMA alignment: ema9 > ema21 > ema50 → bullish
                       ema9 < ema21 < ema50 → bearish
        """
        row = df.iloc[-1]
        if row["ema_9"] > row["ema_21"] > row["ema_50"]:
            return "UP"
        if row["ema_9"] < row["ema_21"] < row["ema_50"]:
            return "DOWN"
        return None

    def _check_rsi(self, df: pd.DataFrame) -> Optional[str]:
        """
        RSI below 40 → potential oversold (bullish signal)
        RSI above 60 → potential overbought (bearish signal)
        """
        rsi = df["rsi"].iloc[-1]
        if rsi < 40:
            return "UP"
        if rsi > 60:
            return "DOWN"
        return None

    def _check_macd(self, df: pd.DataFrame) -> Optional[str]:
        """
        MACD histogram turning positive → bullish
        MACD histogram turning negative → bearish
        """
        diff_now  = df["macd_diff"].iloc[-1]
        diff_prev = df["macd_diff"].iloc[-2]
        if diff_now > 0 and diff_prev <= 0:
            return "UP"
        if diff_now < 0 and diff_prev >= 0:
            return "DOWN"
        if diff_now > 0:
            return "UP"
        if diff_now < 0:
            return "DOWN"
        return None

    def _check_bollinger(self, df: pd.DataFrame) -> Optional[str]:
        """
        Price touching lower band → potential bounce (UP)
        Price touching upper band → potential reversal (DOWN)
        """
        row = df.iloc[-1]
        bb_pct = row["bb_pct"]
        if bb_pct < 0.15:
            return "UP"
        if bb_pct > 0.85:
            return "DOWN"
        return None

    def _count_confirmations(self, df: pd.DataFrame, model_direction: str) -> tuple[int, list[str]]:
        """
        Run all indicator checks, count how many agree with the model direction.

        Returns:
            (confirmation_count, list_of_confirming_indicators)
        """
        checks = {
            "EMA trend":       self._check_ema_trend(df),
            "RSI":             self._check_rsi(df),
            "MACD":            self._check_macd(df),
            "Bollinger Bands": self._check_bollinger(df)
        }
        confirmed = [name for name, direction in checks.items() if direction == model_direction]
        return len(confirmed), confirmed

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _calculate_contracts(self, entry_price: float) -> int:
        """
        Simple fixed-USD sizing: how many contracts = trade_size_usd / entry_price
        Delta perpetuals are typically 1 contract = 1 USD notional.
        Adjust this to match the specific contract spec.
        """
        contracts = max(1, int(self.trade_size_usd / entry_price))
        return contracts

    # ------------------------------------------------------------------
    # Main signal generation
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        prediction: dict,
        df: pd.DataFrame,
        open_trades: int = 0
    ) -> TradeSignal:
        """
        Combine model prediction + indicator confirmation into a trade signal.

        Args:
            prediction:  Output dict from PricePredictor.predict()
            df:          DataFrame with computed indicators (latest rows)
            open_trades: Current number of open positions

        Returns:
            TradeSignal dataclass
        """
        direction   = prediction["direction"]
        confidence  = prediction["confidence"]
        entry_price = df["close"].iloc[-1]

        # --- Gate 1: Confidence threshold ---
        if confidence < self.confidence_threshold:
            return TradeSignal(
                action="HOLD", direction=direction, confidence=confidence,
                entry_price=entry_price, stop_loss=0, take_profit=0, contracts=0,
                reason=f"Low confidence: {confidence:.2%} < {self.confidence_threshold:.0%} threshold"
            )

        # --- Gate 2: Max open trades ---
        if open_trades >= self.max_open_trades:
            return TradeSignal(
                action="HOLD", direction=direction, confidence=confidence,
                entry_price=entry_price, stop_loss=0, take_profit=0, contracts=0,
                reason=f"Max trades reached ({open_trades}/{self.max_open_trades})"
            )

        # --- Gate 3: Indicator confirmation (need at least 2/4) ---
        n_confirmed, confirmed_by = self._count_confirmations(df, direction)
        if n_confirmed < 2:
            return TradeSignal(
                action="HOLD", direction=direction, confidence=confidence,
                entry_price=entry_price, stop_loss=0, take_profit=0, contracts=0,
                reason=f"Insufficient confirmation: only {n_confirmed}/4 indicators agree ({direction})"
            )

        # --- Compute SL / TP ---
        if direction == "UP":
            stop_loss   = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
            action      = "BUY"
        else:
            stop_loss   = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
            action      = "SELL"

        contracts = self._calculate_contracts(entry_price)
        reason = (
            f"Model: {direction} ({confidence:.2%}). "
            f"Confirmed by: {', '.join(confirmed_by)}. "
            f"SL={stop_loss:.2f}, TP={take_profit:.2f}"
        )

        return TradeSignal(
            action=action, direction=direction, confidence=confidence,
            entry_price=entry_price, stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2), contracts=contracts, reason=reason
        )
