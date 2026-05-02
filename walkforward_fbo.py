"""
walkforward_fbo.py

Walk-forward validation harness for the FBO strategy.

Methodology:
  - Roll 5 folds of 18-month IS / 6-month OOS, stepping 6 months each fold.
  - On each IS slice, sweep the 4-parameter grid (108 cells).
  - Pick the MEDIAN of the top-quartile cells by Sharpe — NOT the peak.
    Rationale: peak picking is curve-fitting; median-of-plateau is robust.
  - Evaluate that one config OOS.
  - Apply the kill criteria from the spec; print PASS / FAIL per fold.

Usage:
    python walkforward_fbo.py --data data/ES_ohlcv-1m_2020-01-01_2026-04-22.parquet
"""

import argparse
import itertools
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from fbo_strategy import FBOConfig
from backtest_fbo import run_fbo_backtest
from volman_strategy import INSTRUMENTS

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---- Parameter grid (sensitivity, not optimization) ----
PARAM_GRID = {
    "or_minutes":         [5, 15, 30],
    "failure_window_min": [15, 30, 45, 60],
    "or_atr_ratio_min":   [0.15, 0.25, 0.40],
    "or_atr_ratio_max":   [1.5, 2.0, 3.0],
}

# ---- Walk-forward folds (anchor dates) ----
# Fold i: IS [start, start+18mo), OOS [start+18mo, start+24mo)
FOLDS = [
    ("2020-01-01", "2021-07-01", "2022-01-01"),
    ("2020-07-01", "2022-01-01", "2022-07-01"),
    ("2021-01-01", "2022-07-01", "2023-01-01"),
    ("2021-07-01", "2023-01-01", "2023-07-01"),
    ("2022-01-01", "2023-07-01", "2024-01-01"),
    # Holdout (true OOS, never used for selection):
    ("2022-07-01", "2024-01-01", "2026-04-22"),
]


def _make_cfg(params: dict, instrument: str = "ES") -> FBOConfig:
    cfg = FBOConfig(**params)
    spec = INSTRUMENTS[instrument]
    cfg.tick_size           = spec["tick_size"]
    cfg.tick_value          = spec["tick_value"]
    cfg.multiplier          = spec["multiplier"]
    cfg.commission_per_side = spec["commission"]
    cfg.instrument          = instrument
    return cfg


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index < end)].copy()


def _summary(stats) -> dict:
    if stats is None:
        return {"n_trades": 0, "sharpe": np.nan, "pf": np.nan,
                "ret_pct": np.nan, "max_dd_pct": np.nan, "win_rate": np.nan}
    return {
        "n_trades":   int(stats.get("# Trades", 0)),
        "sharpe":     float(stats.get("Sharpe Ratio", np.nan)),
        "pf":         float(stats.get("Profit Factor", np.nan)),
        "ret_pct":    float(stats.get("Return [%]", np.nan)),
        "max_dd_pct": float(stats.get("Max. Drawdown [%]", np.nan)),
        "win_rate":   float(stats.get("Win Rate [%]", np.nan)),
    }


def _grid_iter():
    keys = list(PARAM_GRID.keys())
    for vals in itertools.product(*[PARAM_GRID[k] for k in keys]):
        yield dict(zip(keys, vals))


def run_fold(df: pd.DataFrame, is_start: str, oos_start: str, oos_end: str,
             instrument: str = "ES") -> dict:
    print("\n" + "=" * 70)
    print(f"FOLD: IS [{is_start} → {oos_start})  |  OOS [{oos_start} → {oos_end})")
    print("=" * 70)

    df_is  = _slice(df, is_start,  oos_start)
    df_oos = _slice(df, oos_start, oos_end)
    print(f"IS bars: {len(df_is):,}   OOS bars: {len(df_oos):,}")

    # 1) Sweep IS grid
    is_rows = []
    for params in _grid_iter():
        cfg = _make_cfg(params, instrument)
        try:
            stats = run_fbo_backtest(
                data_path=None, cfg=cfg, cash=25_000,
                realistic=True, slippage_ticks=1.0,
                df_override=df_is, plot=False,
            )
        except Exception as e:
            print(f"  IS fail {params}: {e}")
            continue
        s = _summary(stats)
        s.update(params)
        is_rows.append(s)

    is_df = pd.DataFrame(is_rows)
    if is_df.empty or is_df["sharpe"].isna().all():
        print("IS sweep produced no valid runs — fold rejected.")
        return {"is_start": is_start, "oos_start": oos_start,
                "oos_end": oos_end, "verdict": "REJECT_NO_IS"}

    # 2) Plateau pick: median of top-quartile cells
    is_df_clean = is_df.dropna(subset=["sharpe"])
    q75 = is_df_clean["sharpe"].quantile(0.75)
    plateau = is_df_clean[is_df_clean["sharpe"] >= q75]
    if len(plateau) < 3:
        plateau = is_df_clean.nlargest(3, "sharpe")

    chosen = {}
    for k in PARAM_GRID:
        # pick the median value of the plateau, snap to nearest grid value
        med = plateau[k].median()
        grid_vals = np.array(PARAM_GRID[k], dtype=float)
        chosen[k] = type(PARAM_GRID[k][0])(grid_vals[np.argmin(np.abs(grid_vals - med))])

    # Plateau quality: stddev of Sharpe across plateau (lower = more robust)
    plateau_std = float(plateau["sharpe"].std())
    plateau_med = float(plateau["sharpe"].median())
    print(f"\nIS plateau: median Sharpe {plateau_med:.2f}, std {plateau_std:.2f}, "
          f"n={len(plateau)} cells")
    print(f"Chosen params: {chosen}")

    # 3) Evaluate chosen params on IS (full window, for comparison) and OOS
    cfg = _make_cfg(chosen, instrument)

    is_stats = run_fbo_backtest(
        data_path=None, cfg=cfg, cash=25_000,
        realistic=True, slippage_ticks=1.0,
        df_override=df_is, plot=False,
    )
    oos_stats = run_fbo_backtest(
        data_path=None, cfg=cfg, cash=25_000,
        realistic=True, slippage_ticks=1.0,
        df_override=df_oos, plot=False,
    )

    is_s  = _summary(is_stats)
    oos_s = _summary(oos_stats)

    # 4) Kill criteria
    fail_reasons = []
    if oos_s["n_trades"] < 30:
        fail_reasons.append(f"OOS trade count {oos_s['n_trades']} < 30")
    if not np.isnan(is_s["sharpe"]) and not np.isnan(oos_s["sharpe"]):
        if is_s["sharpe"] > 0 and oos_s["sharpe"] < 0.6 * is_s["sharpe"]:
            fail_reasons.append(
                f"OOS Sharpe {oos_s['sharpe']:.2f} < 60% of IS {is_s['sharpe']:.2f}"
            )
    if not np.isnan(is_s["max_dd_pct"]) and not np.isnan(oos_s["max_dd_pct"]):
        # Both are negative percentages; "worse" = more negative
        if abs(oos_s["max_dd_pct"]) > 2.0 * abs(is_s["max_dd_pct"]):
            fail_reasons.append(
                f"OOS max DD {oos_s['max_dd_pct']:.1f}% > 2x IS {is_s['max_dd_pct']:.1f}%"
            )
    if oos_s["pf"] is not None and not np.isnan(oos_s["pf"]) and oos_s["pf"] < 1.15:
        fail_reasons.append(f"OOS profit factor {oos_s['pf']:.2f} < 1.15")
    if plateau_std > 0.5 * abs(plateau_med if plateau_med != 0 else 1):
        fail_reasons.append(
            f"Plateau Sharpe std {plateau_std:.2f} too wide vs median {plateau_med:.2f} "
            f"(suggests no real plateau)"
        )

    verdict = "PASS" if not fail_reasons else "FAIL"
    print(f"\n>>> Fold verdict: {verdict}")
    if fail_reasons:
        for r in fail_reasons:
            print(f"    - {r}")

    return {
        "is_start": is_start, "oos_start": oos_start, "oos_end": oos_end,
        "chosen_params": chosen,
        "plateau_median_sharpe": plateau_med,
        "plateau_std": plateau_std,
        "plateau_n": len(plateau),
        **{f"is_{k}": v for k, v in is_s.items()},
        **{f"oos_{k}": v for k, v in oos_s.items()},
        "verdict": verdict,
        "fail_reasons": "; ".join(fail_reasons),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--instrument", default=None)
    args = p.parse_args()

    stem = Path(args.data).stem
    first = stem.split("_")[0].upper()
    instrument = args.instrument or (first if first in INSTRUMENTS else "ES")
    print(f"Instrument: {instrument}")

    print(f"Loading {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"  {len(df):,} bars, {df.index.min()} → {df.index.max()}")

    rows = []
    for is_start, oos_start, oos_end in FOLDS:
        try:
            row = run_fold(df, is_start, oos_start, oos_end, instrument)
            rows.append(row)
        except Exception as e:
            print(f"\nFold {is_start} → {oos_end} crashed: {e}")
            rows.append({"is_start": is_start, "oos_start": oos_start,
                         "oos_end": oos_end, "verdict": "ERROR",
                         "fail_reasons": str(e)})

    summary_df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "fbo_walkforward_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\n\n=== WALK-FORWARD SUMMARY ===")
    print(summary_df.to_string())
    print(f"\nSaved → {out_csv}")

    n_pass = (summary_df["verdict"] == "PASS").sum()
    n_total = len(summary_df)
    print(f"\nPassing folds: {n_pass} / {n_total}")
    if n_pass >= n_total - 1:
        print("VERDICT: Strategy survives walk-forward. Candidate for paper trading.")
    elif n_pass >= n_total // 2:
        print("VERDICT: Mixed. Re-examine failing fold regimes before deploying.")
    else:
        print("VERDICT: REJECT. Strategy does not generalize. Document & move on.")


if __name__ == "__main__":
    main()
