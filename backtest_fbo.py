"""
backtest_fbo.py

Run FBO (Failed Breakout Reversal) strategy on historical data.

Usage:
    python backtest_fbo.py --data data/ES_ohlcv-1m_2020-01-01_2026-04-22.parquet \
        --or-mins 15 --failure-window 30 --realistic --slippage-ticks 1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting import Backtest, Strategy

from fbo_strategy import FBOConfig, generate_fbo_signals
from volman_strategy import INSTRUMENTS, apply_instrument

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class FBOStrategy(Strategy):
    cfg: FBOConfig = FBOConfig()
    signals_df: pd.DataFrame = None

    def init(self):
        self.long_sig = self.I(
            lambda x: x,
            self.signals_df["fbo_long"].astype(int).values,
            name="Long", overlay=False,
        )
        self.short_sig = self.I(
            lambda x: x,
            self.signals_df["fbo_short"].astype(int).values,
            name="Short", overlay=False,
        )

    def next(self):
        i = len(self.data) - 1
        if i >= len(self.signals_df):
            return
        row = self.signals_df.iloc[i]

        # Force-exit at EOD
        if self.position and row["eod_exit"]:
            self.position.close()
            return

        # Force-exit if outside session (overnight gap protection)
        if self.position and not row["in_session"]:
            self.position.close()
            return

        # Library handles SL/TP exits passed at order entry
        if self.position:
            return

        price = self.data.Close[-1]
        sl_price = row["fbo_stop_price"]
        tp_price = row["fbo_target_price"]

        if pd.isna(sl_price) or pd.isna(tp_price):
            return

        if row["fbo_long"]:
            # Sanity: SL below price, TP above price
            if sl_price < price < tp_price:
                self.buy(sl=float(sl_price), tp=float(tp_price), tag="FBO_L")
        elif row["fbo_short"]:
            if tp_price < price < sl_price:
                self.sell(sl=float(sl_price), tp=float(tp_price), tag="FBO_S")


def run_fbo_backtest(
    data_path: Path,
    cfg: FBOConfig,
    cash: float = 25_000,
    plot: bool = False,
    realistic: bool = True,
    slippage_ticks: float = 1.0,
    return_stats: bool = True,
    df_override: pd.DataFrame = None,
):
    if df_override is not None:
        df = df_override
        print(f"Using provided df: {len(df):,} bars, {df.index.min()} → {df.index.max()}")
    else:
        print(f"Loading {data_path}...")
        df = pd.read_parquet(data_path)
        print(f"  {len(df):,} bars, {df.index.min()} → {df.index.max()}")

    print(f"Generating FBO signals "
          f"(OR={cfg.or_minutes}m, fail_win={cfg.failure_window_min}m, "
          f"or/atr=[{cfg.or_atr_ratio_min}, {cfg.or_atr_ratio_max}])...")
    sig_df = generate_fbo_signals(df, cfg)
    n_long  = int(sig_df["fbo_long"].sum())
    n_short = int(sig_df["fbo_short"].sum())
    print(f"  {n_long} long fades, {n_short} short fades")

    if n_long + n_short == 0:
        print("  No signals — aborting.")
        return None

    bt_df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })

    notional_est = bt_df["Close"].mean() * cfg.multiplier
    slippage_dollars = slippage_ticks * cfg.tick_value
    total_friction = cfg.commission_per_side + slippage_dollars
    commission_frac = total_friction / notional_est

    mode = "REALISTIC (next-bar open + slippage)" if realistic else "OPTIMISTIC (close fill)"
    print(f"  Execution mode: {mode}")
    print(f"  Friction: ${cfg.commission_per_side:.2f} comm + "
          f"${slippage_dollars:.2f} slip = ${total_friction:.2f}/side "
          f"({commission_frac*1e4:.2f} bps)")

    FBOStrategy.cfg = cfg
    FBOStrategy.signals_df = sig_df.reset_index(drop=True)

    bt = Backtest(
        bt_df, FBOStrategy,
        cash=cash,
        commission=commission_frac,
        exclusive_orders=True,
        trade_on_close=not realistic,
        hedging=False,
    )

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
        out_html = RESULTS_DIR / (
            f"{Path(data_path).stem if data_path else 'fbo'}"
            f"_FBO_or{cfg.or_minutes}_fw{cfg.failure_window_min}.html"
        )
        bt.plot(filename=str(out_html), open_browser=False)
        print(f"\nPlot saved → {out_html}")

    return stats if return_stats else bt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--instrument", default=None)
    p.add_argument("--or-mins", type=int, default=15)
    p.add_argument("--failure-window", type=int, default=30,
                   help="Minutes after first break to consider failure trigger")
    p.add_argument("--or-atr-min", type=float, default=0.25)
    p.add_argument("--or-atr-max", type=float, default=2.0)
    p.add_argument("--realistic", action="store_true",
                   help="Fill at next-bar open + slippage (recommended)")
    p.add_argument("--slippage-ticks", type=float, default=1.0,
                   help="Slippage in ticks per side (default: 1)")
    p.add_argument("--cash", type=float, default=25_000)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    stem = Path(args.data).stem
    first = stem.split("_")[0].upper()
    instrument = args.instrument or (first if first in INSTRUMENTS else "ES")
    print(f"Instrument: {instrument}")

    cfg = FBOConfig(
        or_minutes=args.or_mins,
        failure_window_min=args.failure_window,
        or_atr_ratio_min=args.or_atr_min,
        or_atr_ratio_max=args.or_atr_max,
    )

    spec = INSTRUMENTS[instrument]
    cfg.tick_size           = spec["tick_size"]
    cfg.tick_value          = spec["tick_value"]
    cfg.multiplier          = spec["multiplier"]
    cfg.commission_per_side = spec["commission"]
    cfg.instrument          = instrument

    slip = args.slippage_ticks if args.realistic else 0.0
    run_fbo_backtest(
        Path(args.data), cfg,
        cash=args.cash, plot=args.plot,
        realistic=args.realistic, slippage_ticks=slip,
    )


if __name__ == "__main__":
    main()
