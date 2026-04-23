"""
fetch_data.py
Pulls MES/ES 1-minute bars from Databento and saves to parquet.

Usage:
    1. Set your Databento API key as env var: export DATABENTO_API_KEY=db-xxxxx
    2. Run: python fetch_data.py --symbol MES --start 2023-01-01 --end 2025-12-31

Notes:
    - Uses the CME Globex MDP 3.0 dataset (GLBX.MDP3)
    - Pulls continuous front-month via parent symbology (MES.c.0)
    - Saves as parquet for fast reload
    - Cost estimate: ~$2-5 for 2 years of 1-minute MES data
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

try:
    import databento as db
except ImportError:
    print("ERROR: databento not installed. Run: pip install databento")
    sys.exit(1)


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_bars(symbol: str, start: str, end: str, schema: str = "ohlcv-1m") -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV bars for a continuous futures contract.

    Args:
        symbol: 'MES', 'ES', 'NQ', 'MNQ', etc.
        start: ISO date 'YYYY-MM-DD'
        end:   ISO date 'YYYY-MM-DD'
        schema: 'ohlcv-1m' (1-min bars) or 'ohlcv-5m' (5-min bars)

    Returns:
        DataFrame with datetime index and OHLCV columns
    """
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY environment variable not set.")
        print("  Windows:  set DATABENTO_API_KEY=db-xxxxx")
        print("  Mac/Lin:  export DATABENTO_API_KEY=db-xxxxx")
        sys.exit(1)

    client = db.Historical(api_key)

    # Check cost first so we don't burn credits by accident
    cost_est = client.metadata.get_cost(
        dataset="GLBX.MDP3",
        symbols=[f"{symbol}.c.0"],   # continuous front-month
        stype_in="continuous",
        schema=schema,
        start=start,
        end=end,
    )
    print(f"Estimated cost: ${cost_est:.2f}")

    resp = input("Proceed with download? [y/N]: ").strip().lower()
    if resp != "y":
        print("Aborted.")
        sys.exit(0)

    print(f"Fetching {symbol} {schema} from {start} to {end}...")
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[f"{symbol}.c.0"],
        stype_in="continuous",
        schema=schema,
        start=start,
        end=end,
    )

    df = data.to_df()
    # Normalize column names + index
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df.index.name = "datetime"
    keep = ["open", "high", "low", "close", "volume"]
    df = df[keep].astype({"open": float, "high": float, "low": float,
                          "close": float, "volume": float})
    return df


def save(df: pd.DataFrame, symbol: str, schema: str) -> Path:
    fname = f"{symbol}_{schema}_{df.index.min().date()}_{df.index.max().date()}.parquet"
    path = DATA_DIR / fname
    df.to_parquet(path)
    print(f"Saved {len(df):,} bars → {path}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="MES",
                   help="MES, ES, NQ, MNQ (default: MES)")
    p.add_argument("--start", default="2023-01-01", help="YYYY-MM-DD")
    p.add_argument("--end",   default="2025-12-31", help="YYYY-MM-DD")
    p.add_argument("--schema", default="ohlcv-1m",
                   choices=["ohlcv-1m", "ohlcv-5m", "ohlcv-1h"])
    args = p.parse_args()

    df = fetch_bars(args.symbol, args.start, args.end, args.schema)
    save(df, args.symbol, args.schema)

    # Quick sanity summary
    print("\n=== Data Sanity Check ===")
    print(f"Rows:           {len(df):,}")
    print(f"Trading days:   {df.index.normalize().nunique():,}")
    print(f"Avg bars/day:   {len(df) / df.index.normalize().nunique():.0f}")
    print(f"First 3 rows:")
    print(df.head(3))
    print(f"Last 3 rows:")
    print(df.tail(3))


if __name__ == "__main__":
    main()
