"""
backtest.py
Run the Volman strategy on historical data with realistic execution.

Usage:
    python backtest.py --data data/MES_ohlcv-1m_2023-01-01_2025-12-31.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting import Backtest, Strategy

from volman_strategy import VolmanConfig, generate_signals, apply_instrument, INSTRUMENTS


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class VolmanStrategy(Strategy):
    """
    backtesting.py adapter around our pre-computed signals.
    Signals are calculated once up-front; this class just executes them.
    """
    # Class-level config injection points
    cfg: VolmanConfig = VolmanConfig()
    signals_df: pd.DataFrame = None

    def init(self):
        # Pre-computed signals are attached to self.signals_df
        # We just index them as indicators for visualization
        self.long_sig  = self.I(lambda x: x, self.signals_df["long_signal"].astype(int).values,
                                name="Long",  overlay=False)
        self.short_sig = self.I(lambda x: x, self.signals_df["short_signal"].astype(int).values,
                                name="Short", overlay=False)
        self.atr_series = self.I(lambda x: x, self.signals_df["atr"].values,
                                 name="ATR", overlay=False)

    def next(self):
        i = len(self.data) - 1
        if i >= len(self.signals_df):
            return

        # Only one position at a time
        if self.position:
            return

        row = self.signals_df.iloc[i]
        atr_now = row["atr"]
        if pd.isna(atr_now) or atr_now <= 0:
            return

        price = self.data.Close[-1]
        tp_dist = self.cfg.tp_atr_mult * atr_now
        sl_dist = self.cfg.sl_atr_mult * atr_now

        if row["long_signal"]:
            sl = price - sl_dist
            tp = price + tp_dist
            self.buy(sl=sl, tp=tp, tag=row["setup"])
        elif row["short_signal"]:
            sl = price + sl_dist
            tp = price - tp_dist
            self.sell(sl=sl, tp=tp, tag=row["setup"])


def run_backtest(data_path: Path, cfg: VolmanConfig, cash: float = 25_000,
                 plot: bool = False) -> dict:
    print(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} bars, {df.index.min()} → {df.index.max()}")

    print("Generating signals...")
    sig_df = generate_signals(df, cfg)
    n_long  = int(sig_df["long_signal"].sum())
    n_short = int(sig_df["short_signal"].sum())
    print(f"  {n_long:,} long signals, {n_short:,} short signals")

    # backtesting.py expects OHLCV columns with capitalized names
    bt_df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                               "close": "Close", "volume": "Volume"})

    # Commission: backtesting.py expects relative (fraction), but we approximate
    # by converting $/contract to a fraction of typical contract notional.
    # Uses instrument multiplier from cfg (set via --instrument flag).
    notional_est = bt_df["Close"].mean() * cfg.multiplier
    commission_frac = cfg.commission_per_side / notional_est
    print(f"  Instrument: {cfg.instrument}  |  multiplier: {cfg.multiplier}  |  "
          f"tick: ${cfg.tick_value}  |  comm/side: ${cfg.commission_per_side}")

    VolmanStrategy.cfg = cfg
    VolmanStrategy.signals_df = sig_df.reset_index(drop=True)

    bt = Backtest(
        bt_df,
        VolmanStrategy,
        cash=cash,
        commission=commission_frac,
        exclusive_orders=True,
        trade_on_close=True,    # fill at close price of signal bar (realistic for close-based signals)
        hedging=False,
    )

    print("Running backtest...")
    stats = bt.run()
    print("\n=== RESULTS ===")
    print(stats)

    # Per-setup breakdown
    trades = stats["_trades"]
    if len(trades) > 0:
        print("\n=== PER-SETUP PERFORMANCE ===")
        if "Tag" in trades.columns:
            setup_col = "Tag"
        else:
            setup_col = None
        if setup_col:
            by_setup = trades.groupby(setup_col).agg(
                count=("PnL", "count"),
                total_pnl=("PnL", "sum"),
                avg_pnl=("PnL", "mean"),
                win_rate=("PnL", lambda x: (x > 0).mean()),
            )
            print(by_setup)

    if plot:
        out_html = RESULTS_DIR / f"{data_path.stem}_backtest.html"
        bt.plot(filename=str(out_html), open_browser=False)
        print(f"\nPlot saved → {out_html}")

    return {
        "stats": stats,
        "trades": trades,
        "signals": sig_df,
    }


def _infer_instrument(data_path: Path) -> str:
    """Try to pull the symbol from the filename, e.g. ES_ohlcv-1m_...parquet -> 'ES'."""
    stem = data_path.stem
    first = stem.split("_")[0].upper()
    return first if first in INSTRUMENTS else "MES"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to parquet file from fetch_data.py")
    p.add_argument("--instrument", default=None,
                   help=f"Instrument spec. Options: {', '.join(sorted(INSTRUMENTS.keys()))}. "
                        "Defaults to auto-detect from filename.")
    p.add_argument("--cash", type=float, default=25_000)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--tp", type=float, default=1.0, help="TP ATR multiplier")
    p.add_argument("--sl", type=float, default=1.0, help="SL ATR multiplier")
    p.add_argument("--no-htf",  action="store_true")
    p.add_argument("--no-vwap", action="store_true")
    # Per-setup toggles. Use --setups rb,sb to enable only specific ones.
    p.add_argument("--setups", type=str, default=None,
                   help="Comma-separated setups to enable (rb,pb,sb,dd,fb,bb). "
                        "Default: all except fb. Example: --setups rb,sb")
    p.add_argument("--round-grid", type=float, default=None,
                   help="Override round-number grid (default: instrument preset)")
    args = p.parse_args()

    instrument = args.instrument or _infer_instrument(Path(args.data))
    print(f"Using instrument profile: {instrument}")

    # Build setup toggles from --setups or use defaults
    if args.setups:
        enabled = {s.strip().lower() for s in args.setups.split(",")}
        setup_cfg = dict(
            use_rb = "rb" in enabled,
            use_pb = "pb" in enabled,
            use_sb = "sb" in enabled,
            use_dd = "dd" in enabled,
            use_fb = "fb" in enabled,
            use_bb = "bb" in enabled,
        )
        print(f"Enabled setups: {sorted(enabled)}")
    else:
        setup_cfg = {}   # use dataclass defaults

    cfg = VolmanConfig(
        tp_atr_mult=args.tp,
        sl_atr_mult=args.sl,
        use_htf_filter=not args.no_htf,
        use_vwap=not args.no_vwap,
        round_grid=args.round_grid if args.round_grid is not None else 0.0,
        **setup_cfg,
    )
    apply_instrument(cfg, instrument)

    run_backtest(Path(args.data), cfg, cash=args.cash, plot=args.plot)


if __name__ == "__main__":
    main()
