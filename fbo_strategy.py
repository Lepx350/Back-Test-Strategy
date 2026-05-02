"""
fbo_strategy.py

Failed Breakout Reversal — fade the first OR break that fails to hold.

Hypothesis (orthogonal to ORB long-trend edge in this repo):
On range/chop days, the first ORB break is "premature" — when price re-enters
the opening range within N minutes, the trend setup has aborted and a sweep
to the OPPOSITE OR boundary is high probability.

Logic:
  1. Compute opening range over first OR_MINUTES of RTH (default 15m).
  2. Identify FIRST break of that range between (open + or_minutes) and 11:00 ET.
  3. Track the post-break extreme (highest-high after a long break, lowest-low
     after a short break).
  4. Trigger an entry on the bar whose close re-enters the OR within
     FAILURE_WINDOW_MIN of the break. Direction = opposite of original break.
  5. Stop  = post-break extreme + 1 tick beyond.
  6. Target = opposite OR boundary (≈1× OR width).
  7. Force flat by 15:55 ET.
  8. Day filter: trade only if 0.25 ≤ OR_width / ATR14_daily ≤ 2.0.
  9. Max 1 trade per day.

Parameters (4): or_minutes, failure_window_min, or_atr_ratio_min, or_atr_ratio_max.
"""

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

# Reuse instrument registry from volman_strategy.py
from volman_strategy import INSTRUMENTS, apply_instrument


@dataclass
class FBOConfig:
    # --- Strategy parameters (4) ---
    or_minutes: int = 15
    failure_window_min: int = 30
    or_atr_ratio_min: float = 0.25
    or_atr_ratio_max: float = 2.0

    # --- Session ---
    session_open: str = "09:30"
    session_close: str = "16:00"
    breakout_window_end: str = "11:00"  # latest time first break is valid
    force_flat_time: str = "15:55"      # exit before settlement liquidity dies

    # --- Execution / instrument ---
    instrument: str = "ES"
    commission_per_side: float = 2.50
    tick_size: float = 0.25
    tick_value: float = 12.50
    multiplier: float = 50.0


def _t(s: str) -> time:
    return pd.to_datetime(s).time()


def generate_fbo_signals(df: pd.DataFrame, cfg: FBOConfig) -> pd.DataFrame:
    """
    Produce per-bar columns consumed by backtest_fbo.py:
        fbo_long, fbo_short        bool — entry trigger bars
        fbo_stop_price             float — absolute SL price (NaN unless signal)
        fbo_target_price           float — absolute TP price (NaN unless signal)
        or_high, or_low            float — OR boundaries (broadcast to all bars of day)
        in_session                 bool
        eod_exit                   bool — force-close bar (15:55 ET)
    """
    out = df.copy()

    open_t  = _t(cfg.session_open)
    close_t = _t(cfg.session_close)
    brk_end = _t(cfg.breakout_window_end)
    flat_t  = _t(cfg.force_flat_time)

    idx_t = out.index.time
    day   = out.index.normalize()

    in_session = pd.Series([(x >= open_t) and (x <= close_t) for x in idx_t],
                           index=out.index)
    out["in_session"] = in_session

    # ---- Minutes since session open (per bar; NaN before 09:30) ----
    def _mins(ts):
        if ts.time() < open_t:
            return np.nan
        open_dt = pd.Timestamp.combine(ts.date(), open_t)
        if ts.tz is not None:
            open_dt = open_dt.tz_localize(ts.tz)
        return (ts - open_dt).total_seconds() / 60.0

    mins_since = pd.Series([_mins(ts) for ts in out.index], index=out.index)
    out["mins_since_open"] = mins_since

    # ---- Opening range (first or_minutes of session) ----
    in_or = in_session & (mins_since >= 0) & (mins_since < cfg.or_minutes)
    or_subset = out.loc[in_or]
    or_high_by_day = or_subset["high"].groupby(or_subset.index.normalize()).max()
    or_low_by_day  = or_subset["low" ].groupby(or_subset.index.normalize()).min()

    out["or_high"] = day.map(or_high_by_day)
    out["or_low"]  = day.map(or_low_by_day)
    out["or_width"] = out["or_high"] - out["or_low"]

    # ---- Daily ATR14 (computed at daily resolution, shifted to avoid look-ahead) ----
    daily_close = out.loc[in_session, "close"].groupby(
        out.index[in_session].normalize()
    ).last()
    daily_high = out.loc[in_session, "high"].groupby(
        out.index[in_session].normalize()
    ).max()
    daily_low = out.loc[in_session, "low"].groupby(
        out.index[in_session].normalize()
    ).min()
    prev_close = daily_close.shift(1)
    tr = pd.concat([
        daily_high - daily_low,
        (daily_high - prev_close).abs(),
        (daily_low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=10).mean().shift(1)  # shift => uses prior days only
    out["atr14"] = day.map(atr14)
    out["valid_day"] = (
        (out["or_width"] / out["atr14"])
        .between(cfg.or_atr_ratio_min, cfg.or_atr_ratio_max)
    ).fillna(False)

    # ---- First OR break (close beyond OR), restricted to (or_end, breakout_window_end] ----
    after_or = in_session & (mins_since >= cfg.or_minutes)
    in_brk_window = pd.Series([x <= brk_end for x in idx_t], index=out.index)
    eligible = after_or & in_brk_window

    long_brk_raw  = eligible & (out["close"] > out["or_high"])
    short_brk_raw = eligible & (out["close"] < out["or_low"])

    # First-break-only-per-day mask (mirrors orb_strategy.py idiom)
    def _first_per_day(mask: pd.Series) -> pd.Series:
        cs = mask.groupby(day).cumsum().shift(1, fill_value=0).astype(bool)
        return mask & ~cs

    first_long_brk  = _first_per_day(long_brk_raw)
    first_short_brk = _first_per_day(short_brk_raw)

    # ---- Per-day: bar index of first break, direction, level ----
    # Use cummax/cummin to track the post-break extreme cleanly.
    brk_dir = pd.Series(0, index=out.index, dtype=int)
    brk_dir[first_long_brk]  = 1
    brk_dir[first_short_brk] = -1
    # Forward-fill direction within the day (after the first break)
    brk_dir_day = brk_dir.replace(0, np.nan).groupby(day).ffill().fillna(0).astype(int)

    # Time of break (minutes since open), forward-filled within day
    brk_time = pd.Series(np.where(brk_dir != 0, mins_since, np.nan), index=out.index)
    brk_time = brk_time.groupby(day).ffill()

    # ---- Post-break extremes (running) ----
    # After a long break: track running max(high) since break => stop ref
    # After a short break: track running min(low) since break => stop ref
    post_break = brk_dir_day != 0

    # Reset-on-day cummax/cummin of high/low restricted to post-break bars
    high_after = out["high"].where(post_break)
    low_after  = out["low" ].where(post_break)
    run_max_h = high_after.groupby(day).cummax()
    run_min_l = low_after .groupby(day).cummin()

    # ---- Failure trigger: bar closes back inside OR within failure_window ----
    bars_since_brk = mins_since - brk_time
    in_failure_win = (bars_since_brk > 0) & (bars_since_brk <= cfg.failure_window_min)
    inside_or = (out["close"] >= out["or_low"]) & (out["close"] <= out["or_high"])

    fbo_short_raw = in_failure_win & inside_or & (brk_dir_day ==  1) & out["valid_day"]
    fbo_long_raw  = in_failure_win & inside_or & (brk_dir_day == -1) & out["valid_day"]

    # Only the FIRST failure per day fires
    any_sig = fbo_long_raw | fbo_short_raw
    first_sig = _first_per_day(any_sig)
    fbo_long  = fbo_long_raw  & first_sig
    fbo_short = fbo_short_raw & first_sig

    out["fbo_long"]  = fbo_long
    out["fbo_short"] = fbo_short

    # ---- Pre-compute SL / TP absolute prices on signal bars ----
    tick = cfg.tick_size
    sl_short = run_max_h + tick   # stop above the post-break high
    sl_long  = run_min_l - tick   # stop below the post-break low
    tp_short = out["or_low"]      # opposite OR boundary
    tp_long  = out["or_high"]

    out["fbo_stop_price"] = np.where(fbo_long,  sl_long,
                              np.where(fbo_short, sl_short, np.nan))
    out["fbo_target_price"] = np.where(fbo_long,  tp_long,
                                np.where(fbo_short, tp_short, np.nan))

    # ---- EOD force-flat bar (15:55 ET) ----
    out["eod_exit"] = pd.Series([x == flat_t for x in idx_t], index=out.index)

    return out


if __name__ == "__main__":
    print("FBO strategy module. Import generate_fbo_signals from fbo_strategy.")
