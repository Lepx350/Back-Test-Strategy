"""
backtest_meanrev.py
Run Connors-style mean reversion strategy on historical data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from meanrev_strategy import MeanRevConfig, generate_meanrev_signals
from volman_strategy import INSTRUMENTS


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class MeanRevStrategy(Strategy):
    cfg: MeanRevConfig = MeanRevConfig()
    signals_df: pd.DataFrame = None
    max_hold_bars: int = 10 * 390  # default: ~10 RTH days of minute bars

    def init(self):
        self.long_sig = self.I(lambda x: x,
                               self.signals_df["mr_long"].astype(int).values,
                               name="Long", overlay=False)
        self.short_sig = self.I(lambda x: x,
                                self.signals_df["mr_short"].astype(int).values,
                                name="Short", overlay=False)
        self.bars_in_trade = 0

    def next(self):
        i = len(self.data) - 1
        if i >= len(self.signals_df):
            return

        row = self.signals_df.iloc[i]

        # Track days held
        if self.position:
            self.bars_in_trade += 1
        else:
            self.bars_in_trade = 0

        # Force-exit at max hold
        if self.position and self.bars_in_trade > self.max_hold_bars:
            self.position.close()
            return

        # RSI-based exit
        if self.position.is_long and row["long_exit_signal"]:
            self.position.close()
            return
        if self.position.is_short and row["short_exit_signal"]:
            self.position.close()
            return

        if self.position:
            return  # stay in trade; TP/SL handles it

        # Entries
        price = self.data.Close[-1]
        atr = row["daily_atr"]
        if pd.isna(atr) or atr <= 0:
            return

        cfg = self.cfg
        if row["mr_long"]:
            sl = price - cfg.atr_stop_mult * atr if cfg.use_atr_stop else None
            # Mean-reversion target: daily MA or exit via RSI
            # We rely on RSI exit; use a loose TP as safety
            tp = price + 3 * atr
            try:
                self.buy(sl=sl, tp=tp, tag="MR_L")
            except Exception:
                pass

        elif row["mr_short"]:
            sl = price + cfg.atr_stop_mult * atr if cfg.use_atr_stop else None
            tp = price - 3 * atr
            try:
                self.sell(sl=sl, tp=tp, tag="MR_S")
            except Exception:
                pass


def run_meanrev_backtest(data_path: Path, cfg: MeanRevConfig, cash: float = 25_000,
                         plot: bool = False, realistic: bool = False,
                         slippage_ticks: float = 0.0):
    print(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} bars, {df.index.min()} → {df.index.max()}")

    print(f"Generating MR signals (RSI<{cfg.rsi_oversold}, MA={cfg.regime_ma_days}d)...")
    sig_df = generate_meanrev_signals(df, cfg)
    n_long = int(sig_df["mr_long"].sum())
    n_short = int(sig_df["mr_short"].sum())
    print(f"  {n_long} long setups, {n_short} short setups")

    bt_df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                               "close": "Close", "volume": "Volume"})

    notional_est = bt_df["Close"].mean() * cfg.multiplier
    slippage_dollars = slippage_ticks * cfg.tick_value
    total_friction = cfg.commission_per_side + slippage_dollars
    commission_frac = total_friction / notional_est

    mode = "REALISTIC" if realistic else "OPTIMISTIC"
    print(f"  Execution: {mode}  |  "
          f"Friction ${total_friction:.2f}/side")

    MeanRevStrategy.cfg = cfg
    MeanRevStrategy.signals_df = sig_df.reset_index(drop=True)
    # approx minute bars per day
    if len(df) > 1:
        tf_mins = (df.index[1] - df.index[0]).total_seconds() / 60
        bars_per_day = int(390 / max(tf_mins, 1))
        MeanRevStrategy.max_hold_bars = cfg.max_hold_days * bars_per_day

    bt = Backtest(bt_df, MeanRevStrategy, cash=cash, commission=commission_frac,
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
        out_html = RESULTS_DIR / f"{data_path.stem}_MR.html"
        bt.plot(filename=str(out_html), open_browser=False)
        print(f"\nPlot saved → {out_html}")

    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--instrument", default=None)
    p.add_argument("--rsi-oversold", type=float, default=10)
    p.add_argument("--regime-ma", type=int, default=200)
    p.add_argument("--max-hold", type=int, default=10)
    p.add_argument("--atr-stop", type=float, default=2.0)
    p.add_argument("--long-only", action="store_true", default=True)
    p.add_argument("--short-only", action="store_true")
    p.add_argument("--no-regime", action="store_true")
    p.add_argument("--realistic", action="store_true")
    p.add_argument("--slippage-ticks", type=float, default=1.0)
    p.add_argument("--cash", type=float, default=25_000)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    stem = Path(args.data).stem
    first = stem.split("_")[0].upper()
    instrument = args.instrument or (first if first in INSTRUMENTS else "ES")
    print(f"Instrument: {instrument}")

    cfg = MeanRevConfig(
        rsi_oversold=args.rsi_oversold,
        regime_ma_days=args.regime_ma,
        max_hold_days=args.max_hold,
        atr_stop_mult=args.atr_stop,
        long_only=args.long_only and not args.short_only,
        short_only=args.short_only,
        use_regime_ma=not args.no_regime,
    )
    spec = INSTRUMENTS[instrument]
    cfg.tick_size = spec["tick_size"]
    cfg.tick_value = spec["tick_value"]
    cfg.multiplier = spec["multiplier"]
    cfg.commission_per_side = spec["commission"]
    cfg.instrument = instrument

    slip = args.slippage_ticks if args.realistic else 0.0
    run_meanrev_backtest(Path(args.data), cfg, cash=args.cash, plot=args.plot,
                         realistic=args.realistic, slippage_ticks=slip)


if __name__ == "__main__":
    main()
