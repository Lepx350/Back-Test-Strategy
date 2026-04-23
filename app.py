"""
app.py — Volman Backtest Cloud Dashboard
Mobile-first Streamlit app for running backtests on the go.
"""
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Make sure we can import our strategy modules
sys.path.insert(0, str(Path(__file__).parent))

from volman_strategy import VolmanConfig, generate_signals, apply_instrument, INSTRUMENTS
from orb_strategy import ORBConfig, generate_orb_signals
from backtest import run_backtest as run_volman_backtest
from backtest_orb import run_orb_backtest

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Volman Cloud Backtest",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Mobile-friendly CSS
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stButton>button { width: 100%; height: 3rem; font-size: 1rem; }
    .metric-container { background: #1a1a1a; padding: 0.5rem; border-radius: 8px; }
    .stSelectbox>div>div { font-size: 1rem; }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================
st.title("📊 Volman Cloud Backtest")
st.caption("Systematic strategy backtesting, anywhere 📱☁️")

# ================================================================
# DATA LOADING
# ================================================================
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


@st.cache_data
def list_datasets():
    """Find all parquet files in the data folder."""
    return sorted([p.name for p in DATA_DIR.glob("*.parquet")])


@st.cache_data
def load_dataset(fname: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / fname)


datasets = list_datasets()

# ================================================================
# FETCH DATA — always available, essential when no data exists
# ================================================================
with st.expander("📥 Fetch Data from Databento", expanded=not datasets):
    st.markdown(
        "Pull historical futures data directly to the cloud. "
        "Previously-purchased data re-downloads at no extra cost."
    )

    has_api_key = bool(os.getenv("DATABENTO_API_KEY"))
    if not has_api_key:
        st.warning(
            "⚠️ `DATABENTO_API_KEY` not set. On Railway: "
            "**Settings → Variables → New Variable** → add `DATABENTO_API_KEY` = `db-xxxx`"
        )

    fc1, fc2 = st.columns(2)
    with fc1:
        fetch_symbol = st.selectbox(
            "Symbol", ["ES", "MES", "NQ", "MNQ", "RTY", "M2K", "YM", "MYM", "GC", "MGC", "CL", "MCL"],
            index=0, key="fetch_symbol"
        )
        fetch_schema = st.selectbox("Bar Size", ["ohlcv-1m", "ohlcv-5m", "ohlcv-1h"], index=0)
    with fc2:
        fetch_start = st.date_input("Start Date", value=pd.Timestamp("2020-01-01"),
                                    min_value=pd.Timestamp("2010-01-01"),
                                    max_value=pd.Timestamp("2026-04-22"))
        fetch_end = st.date_input("End Date", value=pd.Timestamp("2026-04-22"),
                                  min_value=pd.Timestamp("2010-01-01"),
                                  max_value=pd.Timestamp("2026-04-22"))

    fetch_btn = st.button("🚀 Fetch Data", type="primary",
                           disabled=not has_api_key, key="fetch_btn")

    if fetch_btn:
        try:
            import databento as db
            client = db.Historical(os.getenv("DATABENTO_API_KEY"))

            with st.spinner(f"Estimating cost..."):
                cost = client.metadata.get_cost(
                    dataset="GLBX.MDP3",
                    symbols=[f"{fetch_symbol}.c.0"],
                    stype_in="continuous",
                    schema=fetch_schema,
                    start=str(fetch_start),
                    end=str(fetch_end),
                )
            st.info(f"💰 Estimated cost: **${cost:.2f}** "
                    f"(Data previously purchased = $0.00)")

            with st.spinner(f"Downloading {fetch_symbol} from Databento... (30-120s for large ranges)"):
                data = client.timeseries.get_range(
                    dataset="GLBX.MDP3",
                    symbols=[f"{fetch_symbol}.c.0"],
                    stype_in="continuous",
                    schema=fetch_schema,
                    start=str(fetch_start),
                    end=str(fetch_end),
                )
                df = data.to_df()
                df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
                df.index.name = "datetime"
                keep = ["open", "high", "low", "close", "volume"]
                df = df[keep].astype({c: float for c in keep})

                fname = f"{fetch_symbol}_{fetch_schema}_{df.index.min().date()}_{df.index.max().date()}.parquet"
                out_path = DATA_DIR / fname
                df.to_parquet(out_path)

            st.success(f"✅ Downloaded {len(df):,} bars → `{fname}`")
            st.info("♻️ Refresh the page or pick the new dataset above")
            st.cache_data.clear()  # Refresh the dataset list

        except Exception as e:
            st.error(f"❌ Fetch failed: {e}")

# Re-check datasets after potential fetch
datasets = list_datasets()
if not datasets:
    st.warning("📭 No data yet. Use the Fetch Data panel above to download some.")
    st.stop()

# ================================================================
# DATA MANAGEMENT — download to phone, delete from server
# ================================================================
with st.expander("💾 Manage Data Files"):
    st.caption("Download a dataset to your phone, then upload to GitHub for permanence.")
    for ds in datasets:
        fpath = DATA_DIR / ds
        size_mb = fpath.stat().st_size / (1024 * 1024)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**{ds}**  \n`{size_mb:.1f} MB`")
        with c2:
            with open(fpath, "rb") as f:
                st.download_button(
                    label="⬇️ Download",
                    data=f.read(),
                    file_name=ds,
                    mime="application/octet-stream",
                    key=f"dl_{ds}",
                    use_container_width=True,
                )

# ================================================================
# CONTROLS (top-level, tap-friendly)
# ================================================================
col1, col2 = st.columns(2)

with col1:
    dataset = st.selectbox("📁 Dataset", datasets, index=0)

with col2:
    strategy = st.selectbox("🎯 Strategy", ["ORB", "Volman"], index=0)

# Load data
with st.spinner("Loading data..."):
    df = load_dataset(dataset)
    st.caption(f"📊 {len(df):,} bars · {df.index.min().date()} → {df.index.max().date()}")

# ================================================================
# DATE RANGE FILTER (for split-half validation & regime testing)
# ================================================================
with st.expander("📅 Date Range Filter (test sub-periods)"):
    dmin = df.index.min().date()
    dmax = df.index.max().date()
    dc1, dc2 = st.columns(2)
    with dc1:
        filter_start = st.date_input("From", value=dmin, min_value=dmin, max_value=dmax, key="filter_start")
    with dc2:
        filter_end = st.date_input("To", value=dmax, min_value=dmin, max_value=dmax, key="filter_end")

    # Apply filter
    mask = (df.index.date >= filter_start) & (df.index.date <= filter_end)
    filtered_bars = mask.sum()
    if filtered_bars < len(df):
        df = df.loc[mask]
        st.info(f"🔍 Filtered to {filtered_bars:,} bars ({filter_start} → {filter_end})")
    else:
        st.caption("Using full dataset")

# ================================================================
# STRATEGY-SPECIFIC CONTROLS
# ================================================================
if strategy == "ORB":
    st.subheader("⚙️ ORB Parameters")
    c1, c2 = st.columns(2)
    with c1:
        or_mins = st.select_slider("OR Minutes", [5, 15, 30, 60], value=15)
        target_mult = st.slider("Target × OR", 0.5, 3.0, 1.0, 0.1)
    with c2:
        direction = st.radio("Direction", ["Both", "Long only", "Short only"], index=1, horizontal=True)
        realistic = st.toggle("Realistic execution", value=True, help="Next-bar open fills + slippage")

    with st.expander("Advanced"):
        slippage_ticks = st.slider("Slippage (ticks/side)", 0.0, 3.0, 1.0, 0.5)
        use_trend_filter = st.checkbox("Gap-direction filter", value=True)

else:  # Volman
    st.subheader("⚙️ Volman Parameters")
    c1, c2 = st.columns(2)
    with c1:
        tp_mult = st.slider("TP × ATR", 0.5, 3.0, 1.0, 0.1)
        sl_mult = st.slider("SL × ATR", 0.5, 3.0, 1.0, 0.1)
    with c2:
        use_htf = st.toggle("HTF trend filter", value=True)
        use_vwap = st.toggle("VWAP filter", value=True)

    st.caption("Setups to enable:")
    setup_cols = st.columns(6)
    setups = {}
    for i, s in enumerate(["RB", "PB", "SB", "DD", "FB", "BB"]):
        with setup_cols[i]:
            setups[s.lower()] = st.checkbox(s, value=(s in ["RB", "PB", "SB", "DD", "BB"]))

# ================================================================
# INSTRUMENT
# ================================================================
c1, c2 = st.columns(2)
with c1:
    # Infer from filename
    inferred = dataset.split("_")[0].upper()
    default_inst = inferred if inferred in INSTRUMENTS else "ES"
    instrument = st.selectbox("📈 Instrument", sorted(INSTRUMENTS.keys()),
                              index=sorted(INSTRUMENTS.keys()).index(default_inst))
with c2:
    cash = st.number_input("💰 Capital ($)", 5000, 1_000_000, 25000, 1000)

# ================================================================
# RUN BUTTON
# ================================================================
st.markdown("---")
run = st.button("🚀 Run Backtest", type="primary")

# ================================================================
# EXECUTE
# ================================================================
if run:
    progress = st.progress(0, "Starting...")
    start_time = datetime.now()

    try:
        progress.progress(20, "Configuring strategy...")

        if strategy == "ORB":
            cfg = ORBConfig(
                or_minutes=or_mins,
                target_mult=target_mult,
                long_only=(direction == "Long only"),
                short_only=(direction == "Short only"),
                use_trend_filter=use_trend_filter,
            )
            spec = INSTRUMENTS[instrument]
            cfg.tick_size = spec["tick_size"]
            cfg.tick_value = spec["tick_value"]
            cfg.multiplier = spec["multiplier"]
            cfg.commission_per_side = spec["commission"]
            cfg.instrument = instrument

            progress.progress(50, "Running ORB backtest...")
            # Write filtered df to temp file for the runner
            tmp_path = DATA_DIR / f"_tmp_orb_{os.getpid()}.parquet"
            df.to_parquet(tmp_path)
            try:
                stats = run_orb_backtest(
                    tmp_path, cfg,
                    cash=cash, plot=False,
                    realistic=realistic,
                    slippage_ticks=slippage_ticks if realistic else 0.0,
                )
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        else:
            cfg = VolmanConfig(
                tp_atr_mult=tp_mult,
                sl_atr_mult=sl_mult,
                use_htf_filter=use_htf,
                use_vwap=use_vwap,
                use_rb=setups["rb"], use_pb=setups["pb"], use_sb=setups["sb"],
                use_dd=setups["dd"], use_fb=setups["fb"], use_bb=setups["bb"],
            )
            apply_instrument(cfg, instrument)
            progress.progress(50, "Running Volman backtest...")
            tmp_path = DATA_DIR / f"_tmp_volman_{os.getpid()}.parquet"
            df.to_parquet(tmp_path)
            try:
                result = run_volman_backtest(tmp_path, cfg, cash=cash, plot=False)
                stats = result["stats"]
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        progress.progress(90, "Processing results...")
        elapsed = (datetime.now() - start_time).total_seconds()
        progress.progress(100, f"Done in {elapsed:.1f}s ✅")

        # ============================================================
        # RESULTS — MOBILE-FRIENDLY LAYOUT
        # ============================================================
        st.markdown("---")
        st.subheader("📊 Results")

        # Top metrics in 2x3 grid
        r1c1, r1c2, r1c3 = st.columns(3)
        r2c1, r2c2, r2c3 = st.columns(3)

        ret = stats.get("Return [%]", 0)
        sharpe = stats.get("Sharpe Ratio", 0)
        dd = stats.get("Max. Drawdown [%]", 0)
        pf = stats.get("Profit Factor", 0)
        n_trades = int(stats.get("# Trades", 0))
        win_rate = stats.get("Win Rate [%]", 0)
        equity_final = stats.get("Equity Final [$]", cash)
        commissions = stats.get("Commissions [$]", 0)

        with r1c1:
            st.metric("Return", f"{ret:+.2f}%",
                     delta=f"${equity_final-cash:+,.0f}",
                     delta_color="normal")
        with r1c2:
            st.metric("Sharpe", f"{sharpe:.2f}",
                     delta_color="off")
        with r1c3:
            st.metric("Max DD", f"{dd:.2f}%",
                     delta_color="inverse")
        with r2c1:
            st.metric("# Trades", f"{n_trades:,}")
        with r2c2:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with r2c3:
            st.metric("Profit Factor", f"{pf:.2f}")

        # Commission detail
        st.caption(f"💸 Commissions: ${commissions:,.2f}  |  "
                   f"Gross: ${equity_final - cash + commissions:+,.2f}  |  "
                   f"Net: ${equity_final - cash:+,.2f}")

        # ============================================================
        # EQUITY CURVE (Plotly — interactive, mobile-friendly)
        # ============================================================
        eq_curve = stats.get("_equity_curve")
        if eq_curve is not None and len(eq_curve) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_curve.index, y=eq_curve["Equity"],
                mode="lines", name="Equity",
                line=dict(color="#00d4ff", width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 212, 255, 0.1)",
            ))
            fig.add_hline(y=cash, line_dash="dash", line_color="gray",
                          annotation_text=f"Start: ${cash:,}")
            fig.update_layout(
                height=400, margin=dict(l=0, r=0, t=30, b=0),
                template="plotly_dark",
                xaxis_title=None, yaxis_title="$",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # TRADES + PER-SETUP BREAKDOWN
        # ============================================================
        trades = stats.get("_trades")
        if trades is not None and len(trades) > 0:
            with st.expander("📋 Per-setup breakdown"):
                if "Tag" in trades.columns:
                    by_tag = trades.groupby("Tag").agg(
                        count=("PnL", "count"),
                        total_pnl=("PnL", "sum"),
                        avg_pnl=("PnL", "mean"),
                        win_rate=("PnL", lambda x: (x > 0).mean() * 100),
                    ).round(2)
                    by_tag.columns = ["# Trades", "Total P&L ($)", "Avg P&L ($)", "Win %"]
                    st.dataframe(by_tag, use_container_width=True)
                else:
                    st.info("No tag column in trades.")

            with st.expander("📜 Last 20 trades"):
                show_cols = [c for c in ["EntryTime", "ExitTime", "Size", "EntryPrice",
                                         "ExitPrice", "PnL", "ReturnPct", "Tag"]
                             if c in trades.columns]
                st.dataframe(trades[show_cols].tail(20), use_container_width=True)

    except Exception as e:
        progress.empty()
        st.error(f"❌ Backtest failed: {e}")
        with st.expander("Traceback"):
            import traceback
            st.code(traceback.format_exc())

# ================================================================
# FOOTER
# ================================================================
with st.expander("ℹ️ About"):
    st.markdown("""
    **Systematic Backtest Dashboard**

    Strategies:
    - **Volman** — Price action setups (RB, PB, SB, DD, FB, BB) from *Understanding Price Action*
    - **ORB** — Opening Range Breakout (Zarattini & Aziz, 2023)

    Data: Databento CME futures 1-minute bars.
    Engine: `backtesting.py` with realistic execution options.

    Deployed on Railway. Code: [GitHub](https://github.com/)
    """)
