# Volman Cloud Backtest 📊☁️

Mobile-first systematic trading strategy backtester. Deploy to Railway, access from any phone.

## Features

- 📱 **Mobile-first UI** — all controls thumb-friendly
- 🎯 **Multiple strategies** — Volman price action + ORB (Opening Range Breakout)
- ⚙️ **Parameter sliders** — tune TP/SL, setups, filters with taps
- 📊 **Interactive charts** — Plotly equity curves, per-setup breakdowns
- ☁️ **Cloud-hosted** — runs on Railway, accessible via URL
- 🔒 **Private** — your data, your dashboard

## Quick Start

### 1. Local development
```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

### 2. Deploy to Railway
See [DEPLOY.md](DEPLOY.md) for step-by-step instructions.

## Strategies

### Volman Price Action
Based on *Understanding Price Action* by Bob Volman. Setups: RB, PB, SB, DD, FB, BB.
**Note:** Backtesting showed this fails on ES futures across 6 years (see research notes).

### ORB (Opening Range Breakout)
Based on Zarattini & Aziz (2023). Trade the break of the first N minutes' range, in gap direction.
**Status:** Shows real asymmetric edge on ES — longs profitable, shorts not.

## Project Structure

```
volman_cloud/
├── app.py                  # Streamlit dashboard
├── volman_strategy.py      # Volman signal logic
├── orb_strategy.py         # ORB signal logic
├── backtest.py             # Volman backtest runner
├── backtest_orb.py         # ORB backtest runner
├── fetch_data.py           # Databento data pull
├── requirements.txt        # Python deps
├── railway.toml            # Railway deploy config
├── nixpacks.toml           # Build config
├── .streamlit/config.toml  # Streamlit theming
├── data/                   # Parquet files (historical bars)
└── DEPLOY.md               # Step-by-step cloud guide
```

## Credits

- Built with [Streamlit](https://streamlit.io), [backtesting.py](https://kernc.github.io/backtesting.py/), [Plotly](https://plotly.com/python/)
- Data from [Databento](https://databento.com) (CME Globex MDP 3.0)
- Strategy research: Bob Volman, Zarattini & Aziz
