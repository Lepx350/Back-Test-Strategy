"""
intraday_vwap_strategy.py
Intraday VWAP Rejection Fade Strategy.

Logic:
  1. Calculate session VWAP (anchored at 09:30 ET each day)
  2. When price extends >= N*ATR from VWAP → overextended
  3. Wait for rejection candle (confirms exhaustion)
  4. Enter counter-direction (fade back to VWAP)
  5. Fixed SL (points), Fixed TP (points)
  6. Force-close at EOD (3:55 PM ET)
  7. Max trades per day cap
  8. Cooldown after each trade

Designed for ES with 4pt SL / 6pt TP (1.5:1 R/R).
Works on any futures instrument by adjusting point values.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class IntradayVWAPConfig:
    # Entry signal
    extension_atr_mult: float = 2.5   # How far from VWAP triggers setup
    atr_period: int = 14              # Rolling ATR on 5-min basis

    # Rejection confirmation
    require_rejection: bool = True    # Wait for opposing candle
    require_volume: bool = False      # Optional: volume spike on rejection

    # Risk management (POINTS — instrument-specific)
    stop_points: float = 4.0          # Hard stop
    target_points: float = 6.0        # Take profit
    # R/R = target/stop (4/6 = 1.5:1)

    # Direction
    long_only: bool = False           # Fade both sides (buy lows, sell highs)
    short_only: bool = False

    # Trade management
    max_trades_per_day: int = 3
    cooldown_bars: int = 12           # Wait 12 bars (60 min on 5m) between trades
    eod_exit_bar: str = "15:50"       # Force close near session close

    # Session window
    session_start: str = "09:30"
    session_end: str = "16:00"
    trading_start: str = "10:00"      # Don't trade first 30 min chaos
    trading_end: str = "15:30"        # Stop entering near close

    # Filters
    avoid_news_first_30: bool = True  # Skip first 30 min (implied by trading_start)
    min_vwap_distance: float = 2.0    # Min points from VWAP to even consider

    # Execution
    instrument: str = "ES"
    commission_per_side: float = 2.50
    tick_size: float = 0.25
    tick_value: float = 12.50
    multiplier: float = 50.0
    slippage_ticks: float = 1.0


def atr_rolling(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def session_vwap(df: pd.DataFrame, session_start: str) -> pd.Series:
    """Session-anchored VWAP. Resets at session_start each day."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    # Build a "session group" id — increments at each session_start
    t = df.index.time
    start_t = pd.to_datetime(session_start).time()
    is_start = pd.Series([x == start_t for x in t], index=df.index)
    session_id = is_start.cumsum()
    cum_pv = pv.groupby(session_id).cumsum()
    cum_v = df["volume"].groupby(session_id).cumsum()
    vwap = cum_pv / cum_v.replace(0, np.nan)
    return vwap.ffill()


def generate_intraday_vwap_signals(df: pd.DataFrame, cfg: IntradayVWAPConfig) -> pd.DataFrame:
    """
    Generate VWAP rejection fade signals.

    Returns DataFrame with:
      - vwap, atr, dist_from_vwap
      - is_extended_high, is_extended_low
      - signal_long (fade high), signal_short (fade low)  [NAMING: 'long' means we buy = fading a low]
      - eod_exit
    """
    out = df.copy()

    # Session window flags
    t = out.index.time
    open_t = pd.to_datetime(cfg.session_start).time()
    close_t = pd.to_datetime(cfg.session_end).time()
    trade_start_t = pd.to_datetime(cfg.trading_start).time()
    trade_end_t = pd.to_datetime(cfg.trading_end).time()
    eod_t = pd.to_datetime(cfg.eod_exit_bar).time()

    in_session = pd.Series([(x >= open_t) and (x <= close_t) for x in t], index=out.index)
    in_trading_window = pd.Series([(x >= trade_start_t) and (x <= trade_end_t) for x in t],
                                  index=out.index)
    out["in_session"] = in_session

    # VWAP + ATR
    out["vwap"] = session_vwap(out, cfg.session_start)
    out["atr"] = atr_rolling(out, cfg.atr_period)

    # Distance from VWAP
    out["dist_from_vwap"] = out["close"] - out["vwap"]

    # Extension thresholds
    threshold = out["atr"] * cfg.extension_atr_mult
    out["extended_high"] = (out["dist_from_vwap"] >= threshold) & \
                          (out["dist_from_vwap"].abs() >= cfg.min_vwap_distance)
    out["extended_low"] = (out["dist_from_vwap"] <= -threshold) & \
                         (out["dist_from_vwap"].abs() >= cfg.min_vwap_distance)

    # Rejection candle patterns
    # For fading HIGH (going short): prev bar was extended high AND current bar is bearish
    body = out["close"] - out["open"]
    upper_wick = out["high"] - out[["close", "open"]].max(axis=1)
    lower_wick = out[["close", "open"]].min(axis=1) - out["low"]

    # Bearish rejection: close < open AND upper wick > body (rejection from above)
    bearish_rejection = (body < 0) & (upper_wick > body.abs() * 0.5)
    # Bullish rejection: close > open AND lower wick > body
    bullish_rejection = (body > 0) & (lower_wick > body * 0.5)

    # Entry conditions
    # SHORT (fade extended HIGH): prev bar extended_high AND bearish rejection now
    short_setup = (out["extended_high"].shift(1).fillna(False)) & bearish_rejection
    # LONG (fade extended LOW): prev bar extended_low AND bullish rejection now
    long_setup = (out["extended_low"].shift(1).fillna(False)) & bullish_rejection

    if not cfg.require_rejection:
        # Just fade the extension directly
        short_setup = out["extended_high"]
        long_setup = out["extended_low"]

    # Apply trading window filter
    short_setup = short_setup & in_trading_window
    long_setup = long_setup & in_trading_window

    # Apply direction filters
    if cfg.long_only:
        short_setup[:] = False
    if cfg.short_only:
        long_setup[:] = False

    out["signal_long"] = long_setup
    out["signal_short"] = short_setup

    # EOD exit flag
    is_eod = pd.Series([x >= eod_t for x in t], index=out.index) & in_session
    out["eod_exit"] = is_eod

    return out


if __name__ == "__main__":
    print("Intraday VWAP strategy module. Import generate_intraday_vwap_signals.")
