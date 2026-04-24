"""
backtest_intraday_vwap.py
Runs the intraday VWAP rejection fade strategy.
Uses FIXED POINTS for stop/target (not ATR-based) — matches how you'd trade via Tradovate brackets.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from intraday_vwap_strategy import IntradayVWAPConfig, generate_intraday_vwap_signals
from volman_strategy import INSTRUMENTS


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class IntradayVWAPStrategy(Strategy):
    cfg: IntradayVWAPConfig = IntradayVWAPConfig()
    signals_df: pd.DataFrame = None

    def init(self):
        self.long_sig = self.I(lambda x: x,
                               self.signals_df["signal_long"].astype(int).values,
                               name="Long", overlay=False)
        self.short_sig = self.I(lambda x: x,
                                self.signals_df["signal_short"].astype(int).values,
                                name="Short", overlay=False)
        self.last_trade_bar = -9999
        self.trades_today = 0
        self.current_day = None

    def next(self):
        i = len(self.data) - 1
        if i >= len(self.signals_df):
            return

        row = self.signals_df.iloc[i]
        bar_time = self.data.index[-1]
        bar_day = pd.Timestamp(bar_time).normalize()

        # Reset trade count at start of new day
        if self.current_day != bar_day:
            self.current_day = bar_day
            self.trades_today = 0

        # Force-exit at EOD
        if self.position and row["eod_exit"]:
            self.position.close()
            return

        # Force-exit if we leave session
        if self.position and not row["in_session"]:
            self.position.close()
            return

        # Already in a position — let SL/TP manage
        if self.position:
            return

        # Check max trades per day
        if self.trades_today >= self.cfg.max_trades_per_day:
            return

        # Check cooldown
        if (i - self.last_trade_bar) < self.cfg.cooldown_bars:
            return

        price = self.data.Close[-1]
        cfg = self.cfg

        if row["signal_long"]:
            sl = price - cfg.stop_points
            tp = price + cfg.target_points
            if sl < price < tp:
                self.buy(sl=sl, tp=tp, tag="VWAP_L")
                self.last_trade_bar = i
                self.trades_today += 1
        elif row["signal_short"]:
            sl = price + cfg.stop_points
            tp = price - cfg.target_points
            if tp < price < sl:
                self.sell(sl=sl, tp=tp, tag="VWAP_S")
                self.last_trade_bar = i
                self.trades_today += 1


def run_intraday_vwap_backtest(data_path: Path, cfg: IntradayVWAPConfig,
                                cash: float = 25_000, plot: bool = False,
                                realistic: bool = True,
                                slippage_ticks: float = 1.0):
    print(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} bars, {df.index.min()} → {df.index.max()}")

    print(f"Generating signals (ext {cfg.extension_atr_mult}×ATR, SL {cfg.stop_points}pt, TP {cfg.target_points}pt)...")
    sig_df = generate_intraday_vwap_signals(df, cfg)
    n_long = int(sig_df["signal_long"].sum())
    n_short = int(sig_df["signal_short"].sum())
    print(f"  {n_long} long setups, {n_short} short setups")

    bt_df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                               "close": "Close", "volume": "Volume"})

    notional_est = bt_df["Close"].mean() * cfg.multiplier
    slippage_dollars = slippage_ticks * cfg.tick_value
    total_friction = cfg.commission_per_side + slippage_dollars
    commission_frac = total_friction / notional_est

    mode = "REALISTIC" if realistic else "OPTIMISTIC"
    print(f"  Execution: {mode}  |  Friction ${total_friction:.2f}/side")
    print(f"  Risk per trade (1 contract): ${cfg.stop_points * cfg.multiplier:.2f}")
    print(f"  Reward per trade (1 contract): ${cfg.target_points * cfg.multiplier:.2f}")

    IntradayVWAPStrategy.cfg = cfg
    IntradayVWAPStrategy.signals_df = sig_df.reset_index(drop=True)

    bt = Backtest(bt_df, IntradayVWAPStrategy, cash=cash, commission=commission_frac,
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
        out_html = RESULTS_DIR / f"{data_path.stem}_IntradayVWAP.html"
        bt.plot(filename=str(out_html), open_browser=False)
        print(f"\nPlot saved → {out_html}")

    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--instrument", default=None)
    p.add_argument("--stop-pts", type=float, default=4.0)
    p.add_argument("--target-pts", type=float, default=6.0)
    p.add_argument("--ext-atr", type=float, default=2.5)
    p.add_argument("--max-trades", type=int, default=3)
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--short-only", action="store_true")
    p.add_argument("--no-rejection", action="store_true")
    p.add_argument("--realistic", action="store_true", default=True)
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--cash", type=float, default=25_000)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    stem = Path(args.data).stem
    first = stem.split("_")[0].upper()
    instrument = args.instrument or (first if first in INSTRUMENTS else "ES")
    print(f"Instrument: {instrument}")

    cfg = IntradayVWAPConfig(
        stop_points=args.stop_pts,
        target_points=args.target_pts,
        extension_atr_mult=args.ext_atr,
        max_trades_per_day=args.max_trades,
        long_only=args.long_only,
        short_only=args.short_only,
        require_rejection=not args.no_rejection,
    )
    spec = INSTRUMENTS[instrument]
    cfg.tick_size = spec["tick_size"]
    cfg.tick_value = spec["tick_value"]
    cfg.multiplier = spec["multiplier"]
    cfg.commission_per_side = spec["commission"]
    cfg.instrument = instrument

    run_intraday_vwap_backtest(Path(args.data), cfg, cash=args.cash, plot=args.plot,
                                realistic=args.realistic,
                                slippage_ticks=args.slippage_ticks)


if __name__ == "__main__":
    main()
