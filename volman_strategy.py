"""
volman_strategy.py
Python port of the Volman Pine Script v3.

Signal logic only — no order execution here. This module produces a DataFrame
with boolean columns for each setup + final long_signal/short_signal columns
after all filters are applied.

Key design choices (matching the Pine script):
- ATR-based risk (default 1.0 × ATR for both TP and SL)
- 25 EMA trend filter
- VWAP with slope alignment
- Higher-timeframe (15m) trend filter
- Session window + skip-open bars
- Optional round-number avoidance
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


# ============================================================================
# INSTRUMENT REGISTRY
# ============================================================================
# Add new instruments here. tick_value = tick_size * multiplier.
# Commission is per side per contract, round-trip ≈ 2×.

INSTRUMENTS = {
    # S&P 500
    "MES": {"tick_size": 0.25, "tick_value": 1.25,  "multiplier": 5,   "commission": 0.50, "round_grid": 10.0},
    "ES":  {"tick_size": 0.25, "tick_value": 12.50, "multiplier": 50,  "commission": 2.50, "round_grid": 25.0},
    # Nasdaq 100
    "MNQ": {"tick_size": 0.25, "tick_value": 0.50,  "multiplier": 2,   "commission": 0.50, "round_grid": 25.0},
    "NQ":  {"tick_size": 0.25, "tick_value": 5.00,  "multiplier": 20,  "commission": 2.50, "round_grid": 50.0},
    # Russell 2000
    "M2K": {"tick_size": 0.10, "tick_value": 0.50,  "multiplier": 5,   "commission": 0.50, "round_grid": 5.0},
    "RTY": {"tick_size": 0.10, "tick_value": 5.00,  "multiplier": 50,  "commission": 2.50, "round_grid": 10.0},
    # Dow
    "MYM": {"tick_size": 1.0,  "tick_value": 0.50,  "multiplier": 0.5, "commission": 0.50, "round_grid": 50.0},
    "YM":  {"tick_size": 1.0,  "tick_value": 5.00,  "multiplier": 5,   "commission": 2.50, "round_grid": 100.0},
    # Commodities (common scalping targets)
    "GC":  {"tick_size": 0.10, "tick_value": 10.00, "multiplier": 100, "commission": 2.50, "round_grid": 10.0},
    "MGC": {"tick_size": 0.10, "tick_value": 1.00,  "multiplier": 10,  "commission": 0.50, "round_grid": 10.0},
    "CL":  {"tick_size": 0.01, "tick_value": 10.00, "multiplier": 1000,"commission": 2.50, "round_grid": 0.50},
    "MCL": {"tick_size": 0.01, "tick_value": 1.00,  "multiplier": 100, "commission": 0.50, "round_grid": 0.50},
}


def apply_instrument(cfg: "VolmanConfig", symbol: str) -> "VolmanConfig":
    """Update a config in-place with the specs for the given instrument."""
    sym = symbol.upper()
    if sym not in INSTRUMENTS:
        avail = ", ".join(sorted(INSTRUMENTS.keys()))
        raise ValueError(f"Unknown instrument '{symbol}'. Available: {avail}")
    spec = INSTRUMENTS[sym]
    cfg.tick_size = spec["tick_size"]
    cfg.tick_value = spec["tick_value"]
    cfg.multiplier = spec["multiplier"]
    cfg.commission_per_side = spec["commission"]
    cfg.instrument = sym
    # Only overwrite round_grid if user hasn't explicitly set one (i.e. still 0)
    if cfg.round_grid == 0.0:
        cfg.round_grid = spec["round_grid"]
    return cfg


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class VolmanConfig:
    # Trend & range
    use_ema_filter: bool = True
    ema_len: int = 25
    atr_len: int = 14
    range_bars: int = 6
    range_atr_mult: float = 2.0

    # Higher timeframe
    use_htf_filter: bool = True
    htf_minutes: int = 15        # HTF in minutes

    # Setups
    use_rb: bool = True
    use_pb: bool = True
    use_sb: bool = True
    use_dd: bool = True
    use_fb: bool = False
    use_bb: bool = True
    sb_max_bars: int = 8
    block_bars: int = 8

    # Risk
    tp_atr_mult: float = 1.0
    sl_atr_mult: float = 1.0

    # VWAP
    use_vwap: bool = True
    use_vwap_slope: bool = True
    vwap_slope_bars: int = 3

    # Round numbers
    round_grid: float = 0.0      # 0 = disabled. MES: 10 or 25
    round_buf_atr: float = 0.25

    # Session
    use_session: bool = True
    session_start: str = "09:30"
    session_end: str = "16:00"
    skip_open_mins: int = 5

    # Execution realism (filled in by apply_instrument)
    instrument: str = "MES"
    commission_per_side: float = 0.50
    slippage_ticks: float = 1.0
    tick_size: float = 0.25
    tick_value: float = 1.25
    multiplier: float = 5.0


# ============================================================================
# INDICATORS
# ============================================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    # Wilder's smoothing (matches TradingView ta.atr)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def rolling_vwap_session(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP. Resets at the start of each trading day."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    # Anchor by date (ET)
    day = df.index.normalize()
    cum_pv = pv.groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum()
    vwap = cum_pv / cum_v.replace(0, np.nan)
    return vwap.ffill()


# ============================================================================
# HIGHER TIMEFRAME
# ============================================================================

def htf_trend(df: pd.DataFrame, htf_minutes: int, ema_len: int) -> pd.Series:
    """
    Resample to HTF, compute close > ema, forward-fill to original index.
    Returns +1 (long bias), -1 (short bias), or 0 (flat).
    """
    rule = f"{htf_minutes}min"
    htf = df["close"].resample(rule, label="right", closed="right").last().dropna()
    htf_ema = ema(htf, ema_len)
    bias = np.where(htf > htf_ema, 1, np.where(htf < htf_ema, -1, 0))
    bias_s = pd.Series(bias, index=htf.index, dtype=int)
    # Reindex to base timeframe; shift 1 bar to avoid lookahead
    out = bias_s.reindex(df.index, method="ffill").shift(1).fillna(0).astype(int)
    return out


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_signals(df: pd.DataFrame, cfg: VolmanConfig) -> pd.DataFrame:
    """
    Input:  DataFrame with columns [open, high, low, close, volume] and datetime index (ET tz).
    Output: same DataFrame augmented with signal + filter columns.
    """
    out = df.copy()

    # Core indicators
    out["ema25"] = ema(out["close"], cfg.ema_len)
    out["atr"]   = atr(out, cfg.atr_len)
    out["vwap"]  = rolling_vwap_session(out)

    # Trend filters
    long_trend  = (~cfg.use_ema_filter) | (out["close"] > out["ema25"])
    short_trend = (~cfg.use_ema_filter) | (out["close"] < out["ema25"])
    out["long_trend"]  = long_trend
    out["short_trend"] = short_trend

    # HTF
    if cfg.use_htf_filter:
        htf_bias = htf_trend(out, cfg.htf_minutes, cfg.ema_len)
        out["htf_long"]  = htf_bias > 0
        out["htf_short"] = htf_bias < 0
    else:
        out["htf_long"] = True
        out["htf_short"] = True

    # VWAP filter
    if cfg.use_vwap:
        vwap_rising  = out["vwap"] > out["vwap"].shift(cfg.vwap_slope_bars)
        vwap_falling = out["vwap"] < out["vwap"].shift(cfg.vwap_slope_bars)
        long_vwap = (out["close"] > out["vwap"])
        short_vwap = (out["close"] < out["vwap"])
        if cfg.use_vwap_slope:
            long_vwap  &= vwap_rising
            short_vwap &= vwap_falling
        out["long_vwap"] = long_vwap
        out["short_vwap"] = short_vwap
    else:
        out["long_vwap"] = True
        out["short_vwap"] = True

    # Session filter
    if cfg.use_session:
        t = out.index.time
        start = pd.to_datetime(cfg.session_start).time()
        end   = pd.to_datetime(cfg.session_end).time()
        in_sess = pd.Series([(t_ >= start) and (t_ <= end) for t_ in t], index=out.index)
        out["in_session"] = in_sess
        # Skip first N minutes
        sess_start_flag = in_sess & (~in_sess.shift(1, fill_value=False))
        bars_into = np.zeros(len(out), dtype=int)
        counter = 0
        start_arr = sess_start_flag.values
        in_arr = in_sess.values
        for i in range(len(out)):
            if start_arr[i]:
                counter = 0
            elif in_arr[i]:
                counter += 1
            else:
                counter = 0
            bars_into[i] = counter
        # Compute bars to skip based on inferred timeframe
        if len(out) > 1:
            tf_mins = (out.index[1] - out.index[0]).total_seconds() / 60
        else:
            tf_mins = 1
        bars_skip = int(cfg.skip_open_mins / max(tf_mins, 1e-9))
        out["past_open"] = bars_into >= bars_skip
    else:
        out["in_session"] = True
        out["past_open"]  = True

    # Range detection
    r_high = out["high"].rolling(cfg.range_bars).max()
    r_low  = out["low"].rolling(cfg.range_bars).min()
    r_size = r_high - r_low
    out["range_high"] = r_high
    out["range_low"]  = r_low
    out["tight_range"] = r_size <= (out["atr"] * cfg.range_atr_mult)

    # Doji
    body = (out["close"] - out["open"]).abs()
    avg_body = body.rolling(20).mean()
    out["is_doji"] = (body < avg_body * 0.3) & ((out["high"] - out["low"]) > 0)

    # Round number
    if cfg.round_grid > 0:
        nearest = (out["close"] / cfg.round_grid).round() * cfg.round_grid
        out["near_round"] = (out["close"] - nearest).abs() < (out["atr"] * cfg.round_buf_atr)
    else:
        out["near_round"] = False

    # =====================
    # SETUPS (shifted .shift(1) where needed to match Pine's prior-bar semantics)
    # =====================
    rh_prev = r_high.shift(1)
    rl_prev = r_low.shift(1)
    tight_prev = out["tight_range"].shift(1).fillna(False)

    # RB
    out["rb_long"]  = cfg.use_rb & tight_prev & (out["close"] > rh_prev) & long_trend
    out["rb_short"] = cfg.use_rb & tight_prev & (out["close"] < rl_prev) & short_trend

    # PB — pullback to EMA within last 3 bars
    low3  = out["low"].rolling(3).min()
    high3 = out["high"].rolling(3).max()
    pulled = (low3 <= out["ema25"]) & (high3 >= out["ema25"])
    out["pb_long"]  = cfg.use_pb & long_trend  & pulled & (out["close"] > rh_prev)
    out["pb_short"] = cfg.use_pb & short_trend & pulled & (out["close"] < rl_prev)

    # SB — second break after a failed break
    h1 = out["high"].shift(1); c1 = out["close"].shift(1)
    rh2 = r_high.shift(2);     rl2 = r_low.shift(2)
    failed_long  = (h1 > rh2) & (c1 < rh2)
    failed_short = (out["low"].shift(1) < rl2) & (c1 > rl2)
    # bars since last failure (vectorized using groupby trick)
    since_fail_long  = (~failed_long).groupby(failed_long.cumsum()).cumcount()
    since_fail_short = (~failed_short).groupby(failed_short.cumsum()).cumcount()
    out["sb_long"]  = cfg.use_sb & long_trend  & (out["close"] > rh_prev) & (since_fail_long  < cfg.sb_max_bars)
    out["sb_short"] = cfg.use_sb & short_trend & (out["close"] < rl_prev) & (since_fail_short < cfg.sb_max_bars)

    # DD — double doji break
    two_doji = out["is_doji"].shift(1).fillna(False) & out["is_doji"].shift(2).fillna(False)
    max2 = pd.concat([out["high"].shift(1), out["high"].shift(2)], axis=1).max(axis=1)
    min2 = pd.concat([out["low"].shift(1),  out["low"].shift(2)],  axis=1).min(axis=1)
    out["dd_long"]  = cfg.use_dd & two_doji & (out["close"] > max2) & long_trend
    out["dd_short"] = cfg.use_dd & two_doji & (out["close"] < min2) & short_trend

    # FB — false break (fade)
    out["fb_short"] = cfg.use_fb & (out["high"] > rh_prev) & (out["close"] < rh_prev) & short_trend
    out["fb_long"]  = cfg.use_fb & (out["low"]  < rl_prev) & (out["close"] > rl_prev) & long_trend

    # BB — block break
    b_high = out["high"].rolling(cfg.block_bars).max()
    b_low  = out["low"].rolling(cfg.block_bars).min()
    is_block = (b_high - b_low) <= (out["atr"] * cfg.range_atr_mult * 1.2)
    is_block_prev = is_block.shift(1).fillna(False)
    bh_prev = b_high.shift(1)
    bl_prev = b_low.shift(1)
    out["bb_long"]  = cfg.use_bb & is_block_prev & (out["close"] > bh_prev) & long_trend
    out["bb_short"] = cfg.use_bb & is_block_prev & (out["close"] < bl_prev) & short_trend

    # Raw signals
    long_raw  = out[["rb_long",  "pb_long",  "sb_long",  "dd_long",  "fb_long",  "bb_long"]].any(axis=1)
    short_raw = out[["rb_short", "pb_short", "sb_short", "dd_short", "fb_short", "bb_short"]].any(axis=1)

    # Final signals with all filters
    out["long_signal"]  = (long_raw  & out["long_vwap"]  & out["htf_long"]
                           & out["in_session"] & out["past_open"] & ~out["near_round"])
    out["short_signal"] = (short_raw & out["short_vwap"] & out["htf_short"]
                           & out["in_session"] & out["past_open"] & ~out["near_round"])

    # Setup label (which setup fired, priority: RB, PB, SB, DD, FB, BB)
    def _label(row):
        if row["long_signal"]:
            for k in ["rb_long", "pb_long", "sb_long", "dd_long", "fb_long", "bb_long"]:
                if row[k]: return k.split("_")[0].upper()
        if row["short_signal"]:
            for k in ["rb_short", "pb_short", "sb_short", "dd_short", "fb_short", "bb_short"]:
                if row[k]: return k.split("_")[0].upper()
        return ""
    out["setup"] = out.apply(_label, axis=1)

    return out
