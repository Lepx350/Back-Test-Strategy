"""
orb_strategy.py
Opening Range Breakout — one of the most documented edges in index futures.

Logic:
  1. At RTH open (09:30 ET), record the high/low of the first N minutes
     (default 15 min = the "opening range")
  2. After the OR forms, watch for a breakout above the OR high (long)
     or below the OR low (short)
  3. Exit at either:
     - opposite side of the OR (breakeven-ish to loss)
     - fixed profit target (e.g. 1x OR size, 2x OR size)
     - end of session
  4. Only one trade per day (first break wins or loses)

Variations tested:
  - OR period: 5, 15, 30, 60 minutes
  - Target: 1x OR range, 2x OR range, or ATR-based
  - Stop: opposite side of OR, half-OR, or fixed
  - Filters: trend (prev day close vs today's open), VIX regime

This is well-documented edge on ES/NQ/RTY. See Zarattini & Aziz paper (2023).
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

# Reuse instrument registry from volman_strategy.py
from volman_strategy import INSTRUMENTS, apply_instrument


@dataclass
class ORBConfig:
    # Opening range definition
    or_minutes: int = 15              # 5, 15, 30, 60 typical
    session_open: str = "09:30"       # RTH start (ET)
    session_close: str = "16:00"      # Force exit by this time

    # Breakout entry
    breakout_buffer_ticks: float = 0  # Extra ticks beyond OR to confirm break
    allow_both_sides: bool = True     # If False, only trade one direction based on bias
    long_only: bool = False           # Take ORB longs only
    short_only: bool = False          # Take ORB shorts only

    # Trend filter (optional)
    use_trend_filter: bool = True     # Require gap direction to align with break
    gap_size_min_atr: float = 0.0     # Minimum gap size to require directional bias

    # Regime filter (NEW)
    use_regime_filter: bool = False   # Only trade longs above N-day MA, shorts below
    regime_ma_days: int = 200         # Lookback for regime MA

    # Risk
    target_mode: str = "or_multiple"  # "or_multiple" | "atr" | "opposite_side"
    target_mult: float = 1.0          # 1x OR range, or 1x ATR
    stop_mode: str = "opposite_side"  # "opposite_side" | "half_or" | "atr"
    stop_mult: float = 1.0

    # Time stop
    exit_by_eod: bool = True          # Force close at session_close

    # Execution
    instrument: str = "ES"
    commission_per_side: float = 2.50
    tick_size: float = 0.25
    tick_value: float = 12.50
    multiplier: float = 50.0


def generate_orb_signals(df: pd.DataFrame, cfg: ORBConfig) -> pd.DataFrame:
    """
    Generate ORB entry signals.

    Returns DataFrame with additional columns:
      - or_high, or_low: opening range boundaries (filled throughout day)
      - orb_long, orb_short: entry bars (bool)
      - eod_exit: bar at which any open position must close
    """
    out = df.copy()

    open_t = pd.to_datetime(cfg.session_open).time()
    close_t = pd.to_datetime(cfg.session_close).time()

    # Mark session bars
    t = out.index.time
    is_session = pd.Series([(x >= open_t) and (x <= close_t) for x in t], index=out.index)
    out["in_session"] = is_session

    # Minutes since session open (per bar)
    day = out.index.normalize()

    def _minutes_since_open(ts):
        open_dt = pd.Timestamp.combine(ts.date(), open_t).tz_localize(ts.tz)
        return (ts - open_dt).total_seconds() / 60.0

    mins_since = pd.Series([_minutes_since_open(ts) if ts.time() >= open_t else -1
                            for ts in out.index], index=out.index)
    out["mins_since_open"] = mins_since

    # Bars inside the opening range (first or_minutes after open)
    in_or = is_session & (mins_since >= 0) & (mins_since < cfg.or_minutes)
    out["in_or"] = in_or

    # Per-day opening range high/low
    # Build per-day OR from the in-OR subset; then broadcast back
    subset = out.loc[in_or]
    subset_day = subset.index.normalize()
    or_high_by_day = subset["high"].groupby(subset_day).max()
    or_low_by_day = subset["low"].groupby(subset_day).min()
    out["or_high"] = out.index.normalize().map(or_high_by_day)
    out["or_low"] = out.index.normalize().map(or_low_by_day)

    # Flag bars eligible for breakout (in session, after OR complete)
    out["after_or"] = is_session & (mins_since >= cfg.or_minutes)

    # Track if we've already entered today (only first break per day)
    tick = cfg.tick_size
    buf = cfg.breakout_buffer_ticks * tick

    long_break = out["after_or"] & (out["close"] > out["or_high"] + buf)
    short_break = out["after_or"] & (out["close"] < out["or_low"] - buf)

    # Only first break per day
    day_idx = out.index.normalize()
    first_long = long_break & ~long_break.groupby(day_idx).cumsum().shift(1, fill_value=0).astype(bool)
    first_short = short_break & ~short_break.groupby(day_idx).cumsum().shift(1, fill_value=0).astype(bool)

    # Trend filter: only take long if today opened above yesterday's close (gap up)
    if cfg.use_trend_filter:
        # Previous day's last close
        session_mask = is_session
        daily_close = out.loc[session_mask, "close"].groupby(out.index[session_mask].normalize()).last()
        prev_close = daily_close.shift(1)
        out["prev_close"] = out.index.normalize().map(prev_close)
        # Today's open = first bar of session
        session_opens = out.loc[session_mask, "open"].groupby(out.index[session_mask].normalize()).first()
        out["today_open"] = out.index.normalize().map(session_opens)
        gap_up = out["today_open"] > out["prev_close"]
        gap_dn = out["today_open"] < out["prev_close"]
        first_long = first_long & gap_up
        first_short = first_short & gap_dn

    # Regime filter: only long above N-day MA, only short below
    if cfg.use_regime_filter:
        session_mask = is_session
        # Daily close at session close
        daily_close_series = out.loc[session_mask, "close"].groupby(
            out.index[session_mask].normalize()
        ).last()
        # Daily MA, shifted 1 day to avoid lookahead
        daily_ma = daily_close_series.rolling(cfg.regime_ma_days, min_periods=20).mean().shift(1)
        out["regime_ma"] = out.index.normalize().map(daily_ma)
        # Today's open compared to yesterday's MA
        above_ma = out["today_open"] > out["regime_ma"] if "today_open" in out.columns else (
            out["open"] > out["regime_ma"]
        )
        below_ma = out["today_open"] < out["regime_ma"] if "today_open" in out.columns else (
            out["open"] < out["regime_ma"]
        )
        first_long = first_long & above_ma.fillna(False)
        first_short = first_short & below_ma.fillna(False)

    # Ensure we don't double-trade same day (long AND short)
    any_break = first_long | first_short
    break_cum = any_break.groupby(day_idx).cumsum()
    actually_long = first_long & (break_cum == 1)
    actually_short = first_short & (break_cum == 1)

    # Apply direction filters
    if cfg.long_only:
        actually_short[:] = False
    if cfg.short_only:
        actually_long[:] = False

    out["orb_long"] = actually_long
    out["orb_short"] = actually_short

    # End-of-day flag (last bar of session per day)
    sess_bars = out[is_session]
    eod_bars = sess_bars.groupby(sess_bars.index.normalize()).apply(lambda g: g.index.max())
    eod_bar_set = set(eod_bars.values)
    out["eod_exit"] = out.index.isin(eod_bar_set)

    return out


if __name__ == "__main__":
    print("ORB strategy module. Import generate_orb_signals from orb_strategy.")
