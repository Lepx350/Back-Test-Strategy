"""
meanrev_strategy.py
Mean Reversion Strategy — Connors RSI-2 style, adapted for futures/ETFs.

Logic:
  1. Compute 2-period RSI (Wilder's)
  2. Long when RSI(2) < oversold_thresh AND price > long-term MA (regime)
  3. Exit when RSI(2) > exit_thresh OR after max_hold_days
  4. Short when RSI(2) > overbought_thresh AND price < long-term MA
  5. Risk: ATR-based stop loss as safety net

Variations tested:
  - RSI threshold: 5, 10, 15, 20
  - Regime MA: 50, 100, 200 days
  - Hold period: 3, 5, 10 days
  - Exit: RSI-based vs time-based

Originally documented by Larry Connors for SPY/QQQ.
Application to futures is experimental.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class MeanRevConfig:
    # Entry signal
    rsi_period: int = 2              # Connors uses 2
    rsi_oversold: float = 10         # Enter long when RSI < this
    rsi_overbought: float = 90       # Enter short when RSI > this
    rsi_exit: float = 65             # Exit long when RSI > this (for longs)
    rsi_exit_short: float = 35       # Exit short when RSI < this

    # Regime filter
    use_regime_ma: bool = True
    regime_ma_days: int = 200        # Long only above, short only below

    # Risk management
    max_hold_days: int = 10          # Force exit after N days
    use_atr_stop: bool = True
    atr_stop_mult: float = 2.0       # Stop at entry ± 2×ATR
    atr_period: int = 14

    # Direction
    long_only: bool = True           # Connors original was long-only
    short_only: bool = False

    # Session / timing
    session_start: str = "09:30"     # Compute daily bars from these RTH prices
    session_end: str = "16:00"
    entry_time: str = "15:45"        # Enter near session close (next-bar open)

    # Execution
    instrument: str = "ES"
    commission_per_side: float = 2.50
    tick_size: float = 0.25
    tick_value: float = 12.50
    multiplier: float = 50.0


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Classic Wilder's RSI computation."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def daily_atr(df_daily: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df_daily["high"], df_daily["low"], df_daily["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def generate_meanrev_signals(df: pd.DataFrame, cfg: MeanRevConfig) -> pd.DataFrame:
    """
    Generate mean reversion entries/exits.

    Input:  intraday OHLCV DataFrame with datetime index (ET tz)
    Output: same DataFrame augmented with mr_long / mr_short / exit_signal columns

    Strategy runs on DAILY timeframe but fires entries/exits at session boundaries.
    """
    out = df.copy()

    # Collapse intraday to daily bars (based on session close)
    open_t = pd.to_datetime(cfg.session_start).time()
    close_t = pd.to_datetime(cfg.session_end).time()
    t = out.index.time
    in_session = pd.Series([(x >= open_t) and (x <= close_t) for x in t], index=out.index)
    out["in_session"] = in_session

    # Build daily OHLC from session bars
    sess = out.loc[in_session]
    day = sess.index.normalize()
    daily = pd.DataFrame({
        "open":   sess["open"].groupby(day).first(),
        "high":   sess["high"].groupby(day).max(),
        "low":    sess["low"].groupby(day).min(),
        "close":  sess["close"].groupby(day).last(),
        "volume": sess["volume"].groupby(day).sum(),
    })

    # Compute daily indicators
    daily["rsi"] = wilder_rsi(daily["close"], cfg.rsi_period)
    daily["regime_ma"] = daily["close"].rolling(cfg.regime_ma_days, min_periods=20).mean()
    daily["atr"] = daily_atr(daily, cfg.atr_period)

    # Broadcast daily to intraday (shift 1 to use yesterday's data, no lookahead)
    out["daily_rsi"] = out.index.normalize().map(daily["rsi"].shift(1))
    out["daily_ma"] = out.index.normalize().map(daily["regime_ma"].shift(1))
    out["daily_atr"] = out.index.normalize().map(daily["atr"].shift(1))
    out["daily_close_prev"] = out.index.normalize().map(daily["close"].shift(1))

    # Regime bias
    if cfg.use_regime_ma:
        bull_regime = out["daily_close_prev"] > out["daily_ma"]
        bear_regime = out["daily_close_prev"] < out["daily_ma"]
    else:
        bull_regime = pd.Series(True, index=out.index)
        bear_regime = pd.Series(True, index=out.index)

    # Entry times — fire at the entry_time bar of each session (one entry per day)
    entry_t = pd.to_datetime(cfg.entry_time).time()
    is_entry_bar = pd.Series([x == entry_t for x in t], index=out.index) & in_session

    # Long setup: RSI oversold + above regime MA
    long_setup = (
        is_entry_bar
        & (out["daily_rsi"] < cfg.rsi_oversold)
        & bull_regime
    )

    # Short setup: RSI overbought + below regime MA
    short_setup = (
        is_entry_bar
        & (out["daily_rsi"] > cfg.rsi_overbought)
        & bear_regime
    )

    # Apply direction gate
    if cfg.long_only:
        short_setup[:] = False
    if cfg.short_only:
        long_setup[:] = False

    out["mr_long"] = long_setup
    out["mr_short"] = short_setup

    # Exit conditions (evaluated at entry_bar each day while in position)
    # Long exit: RSI recovered OR held too long
    long_exit = is_entry_bar & (out["daily_rsi"] > cfg.rsi_exit)
    short_exit = is_entry_bar & (out["daily_rsi"] < cfg.rsi_exit_short)

    out["long_exit_signal"] = long_exit
    out["short_exit_signal"] = short_exit

    # End-of-session bars (for forced time-based exits)
    eod_bars_per_day = sess.groupby(sess.index.normalize()).apply(lambda g: g.index.max())
    eod_bar_set = set(eod_bars_per_day.values)
    out["eod_exit"] = out.index.isin(eod_bar_set)

    return out


if __name__ == "__main__":
    print("Mean reversion strategy module. Import generate_meanrev_signals.")
