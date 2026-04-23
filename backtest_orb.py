"""
backtest_orb.py
Run ORB (Opening Range Breakout) strategy on historical data.

Usage:
    python backtest_orb.py --data data/ES_ohlcv-1m_2020-01-01_2026-04-22.parquet --or-mins 15
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from orb_strategy import ORBConfig, generate_orb_signals
from volman_strategy import INSTRUMENTS, apply_instrument


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class ORBStrategy(Strategy):
    cfg: ORBConfig = ORBConfig()
    signals_df: pd.DataFrame = None

    def init(self):
        self.long_sig = self.I(lambda x: x,
                               self.signals_df["orb_long"].astype(int).values,
                               name="Long", overlay=False)
        self.short_sig = self.I(lambda x: x,
                                self.signals_df["orb_short"].astype(int).values,
                                name="Short", overlay=False)

    def next(self):
        i = len(self.data) - 1
        if i >= len(self.signals_df):
            return

        row = self.signals_df.iloc[i]

        # Force-exit at EOD
        if self.position and row["eod_exit"]:
            self.position.close()
            return

        # Force-exit if we leave the session
        if self.position and not row["in_session"]:
            self.position.close()
            return

        # Already in a position — let TP/SL manage it
        if self.position:
            return

        price = self.data.Close[-1]
        or_high = row["or_high"]
        or_low = row["or_low"]
        if pd.isna(or_high) or pd.isna(or_low):
            return

        or_range = or_high - or_low
        if or_range <= 0:
            return

        # Build TP and SL based on config
        cfg = self.cfg
        if cfg.target_mode == "or_multiple":
            tp_dist = or_range * cfg.target_mult
        elif cfg.target_mode == "atr":
            # Use OR range as ATR proxy for simplicity
            tp_dist = or_range * cfg.target_mult
        else:
            tp_dist = or_range

        if cfg.stop_mode == "opposite_side":
            # Stop at the opposite side of OR
            if row["orb_long"]:
                sl_price = or_low
            else:
                sl_price = or_high
        elif cfg.stop_mode == "half_or":
            sl_dist = or_range * 0.5
            sl_price = price - sl_dist if row["orb_long"] else price + sl_dist
        else:
            sl_dist = or_range * cfg.stop_mult
            sl_price = price - sl_dist if row["orb_long"] else price + sl_dist

        if row["orb_long"]:
            tp_price = price + tp_dist
            if sl_price < price < tp_price:
                self.buy(sl=sl_price, tp=tp_price, tag="ORB_L")

        elif row["orb_short"]:
            tp_price = price - tp_dist
            if tp_price < price < sl_price:
                self.sell(sl=sl_price, tp=tp_price, tag="ORB_S")


def run_orb_backtest(data_path: Path, cfg: ORBConfig, cash: float = 25_000, plot: bool = False,
                     realistic: bool = False, slippage_ticks: float = 0.0):
    print(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} bars, {df.index.min()} → {df.index.max()}")

    print(f"Generating ORB signals (OR={cfg.or_minutes}m, target={cfg.target_mult}x OR)...")
    sig_df = generate_orb_signals(df, cfg)
    n_long = int(sig_df["orb_long"].sum())
    n_short = int(sig_df["orb_short"].sum())
    print(f"  {n_long} long breakouts, {n_short} short breakouts")

    bt_df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                               "close": "Close", "volume": "Volume"})
    notional_est = bt_df["Close"].mean() * cfg.multiplier

    # Add slippage to commission so it's reflected in P&L
    slippage_dollars = slippage_ticks * cfg.tick_value
    total_friction = cfg.commission_per_side + slippage_dollars
    commission_frac = total_friction / notional_est

    mode = "REALISTIC (next-bar open + slippage)" if realistic else "OPTIMISTIC (close fill)"
    print(f"  Execution mode: {mode}")
    print(f"  Friction: ${cfg.commission_per_side:.2f} comm + ${slippage_dollars:.2f} slip = ${total_friction:.2f}/side")

    ORBStrategy.cfg = cfg
    ORBStrategy.signals_df = sig_df.reset_index(drop=True)

    # Realistic: fill at next bar's open (standard backtesting.py default).
    # Optimistic: fill at current bar's close (unrealistic but faster to converge).
    bt = Backtest(bt_df, ORBStrategy, cash=cash, commission=commission_frac,
                  exclusive_orders=True,
                  trade_on_close=not realistic,
                  hedging=False)

    print("Running backtest...")
    stats = bt.run()
    print("\n=== RESULTS ===")
    print(stats)

    trades = stats["_trades"]
    if len(trades) > 0 and "Tag" in trades.columns:
        print("\n=== BY DIRECTION ===")
        by_tag = trades.groupby("Tag").agg(
            count=("PnL", "count"),
            total_pnl=("PnL", "sum"),
            avg_pnl=("PnL", "mean"),
            win_rate=("PnL", lambda x: (x > 0).mean()),
        )
        print(by_tag)

    if plot:
        out_html = RESULTS_DIR / f"{data_path.stem}_ORB{cfg.or_minutes}m.html"
        bt.plot(filename=str(out_html), open_browser=False)
        print(f"\nPlot saved → {out_html}")

    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--instrument", default=None)
    p.add_argument("--or-mins", type=int, default=15,
                   help="Opening range in minutes (5, 15, 30, 60)")
    p.add_argument("--target", type=float, default=1.0,
                   help="TP = this × OR range")
    p.add_argument("--stop-mode", choices=["opposite_side", "half_or", "or_mult"],
                   default="opposite_side")
    p.add_argument("--no-trend-filter", action="store_true")
    p.add_argument("--long-only", action="store_true", help="Only take ORB longs")
    p.add_argument("--short-only", action="store_true", help="Only take ORB shorts")
    p.add_argument("--realistic", action="store_true",
                   help="Fill at next-bar open (realistic) vs current-bar close (optimistic)")
    p.add_argument("--slippage-ticks", type=float, default=1.0,
                   help="Slippage in ticks per side (default: 1). Only applied with --realistic.")
    p.add_argument("--cash", type=float, default=25_000)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    # Infer instrument
    stem = Path(args.data).stem
    first = stem.split("_")[0].upper()
    instrument = args.instrument or (first if first in INSTRUMENTS else "ES")
    print(f"Instrument: {instrument}")

    cfg = ORBConfig(
        or_minutes=args.or_mins,
        target_mult=args.target,
        stop_mode=args.stop_mode,
        use_trend_filter=not args.no_trend_filter,
        long_only=args.long_only,
        short_only=args.short_only,
    )
    # Apply instrument specs
    spec = INSTRUMENTS[instrument]
    cfg.tick_size = spec["tick_size"]
    cfg.tick_value = spec["tick_value"]
    cfg.multiplier = spec["multiplier"]
    cfg.commission_per_side = spec["commission"]
    cfg.instrument = instrument

    slip = args.slippage_ticks if args.realistic else 0.0
    run_orb_backtest(Path(args.data), cfg, cash=args.cash, plot=args.plot,
                     realistic=args.realistic, slippage_ticks=slip)


if __name__ == "__main__":
    main()
