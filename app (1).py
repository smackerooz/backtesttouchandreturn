import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Touch & Turn Scalping (Shared Capital)", layout="wide")
st.title("💰 Touch & Turn Scalping – Shared $5,000 Capital, Max $750 per trade")

# ... (same sidebar inputs, same DEFAULT_TICKERS list) ...

@st.cache_data(ttl=3600)
def get_daily_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [col.lower().capitalize() for col in df.columns]
    return df[["Open", "High", "Low", "Close"]]

@st.cache_data(ttl=3600)
def get_15min_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval="15m", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [col.lower().capitalize() for col in df.columns]
    return df[["Open", "High", "Low", "Close"]]

def compute_atr(daily_df, period=14):
    high, low, close = daily_df["High"], daily_df["Low"], daily_df["Close"]
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def extract_setups(ticker, daily_df, intraday_df, atr_period, range_threshold):
    """Return list of potential trade setups (no execution, no position sizing)."""
    if daily_df is None or intraday_df is None:
        return []
    
    atr_series = compute_atr(daily_df, atr_period)
    daily_df = daily_df.copy()
    daily_df["ATR"] = atr_series
    
    setups = []
    intraday_df = intraday_df.copy()
    intraday_df["Date"] = intraday_df.index.date
    grouped = intraday_df.groupby("Date")
    
    for date, bars in grouped:
        if date not in daily_df.index.date:
            continue
        daily_row = daily_df[daily_df.index.date == date]
        if daily_row.empty:
            continue
        atr_val = daily_row["ATR"].iloc[0]
        if pd.isna(atr_val):
            continue
        
        first_bar = bars.iloc[0]
        first_open, first_high, first_low, first_close = first_bar[["Open", "High", "Low", "Close"]]
        if first_close >= first_open:
            continue
        
        candle_range = first_high - first_low
        if candle_range <= range_threshold * atr_val:
            continue
        
        entry_price = first_low
        target_price = entry_price + 0.382 * candle_range
        stop_price = entry_price - 0.191 * candle_range  # half target distance
        
        # Scan subsequent bars to determine if setup would have been filled and its exit
        filled = False
        entry_time = None
        exit_time = None
        exit_price = None
        exit_reason = None
        
        for idx in range(1, len(bars)):
            bar = bars.iloc[idx]
            bar_high, bar_low, bar_time = bar["High"], bar["Low"], bar.name
            if not filled:
                if bar_low <= entry_price <= bar_high:
                    filled = True
                    entry_time = bar_time
                    if bar_high >= target_price:
                        exit_price = target_price
                        exit_reason = "Target"
                        exit_time = bar_time
                        break
                    elif bar_low <= stop_price:
                        exit_price = stop_price
                        exit_reason = "Stop"
                        exit_time = bar_time
                        break
                continue
            # filled
            if bar_high >= target_price:
                exit_price = target_price
                exit_reason = "Target"
                exit_time = bar_time
                break
            elif bar_low <= stop_price:
                exit_price = stop_price
                exit_reason = "Stop"
                exit_time = bar_time
                break
        
        if filled and exit_time is None:
            last_bar = bars.iloc[-1]
            exit_price = last_bar["Close"]
            exit_reason = "EOD"
            exit_time = last_bar.name
        
        if filled:
            setups.append({
                "Time": entry_time,  # for sorting
                "Ticker": ticker,
                "EntryPrice": entry_price,
                "ExitPrice": exit_price,
                "ExitReason": exit_reason,
                "Target": target_price,
                "Stop": stop_price
            })
    return setups

# ------------------- Main -------------------
if run_btn:
    if not tickers:
        st.error("Enter tickers")
        st.stop()
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    all_setups = []
    for i, ticker in enumerate(tickers):
        status.text(f"Fetching {ticker}...")
        daily = get_daily_data(ticker, start_date, end_date)
        intra = get_15min_data(ticker, start_date, end_date)
        setups = extract_setups(ticker, daily, intra, atr_period, range_threshold)
        all_setups.extend(setups)
        progress_bar.progress((i+1)/len(tickers))
    
    status.text("Simulating sequential trades with shared capital...")
    
    # Sort setups by entry time
    all_setups.sort(key=lambda x: x["Time"])
    
    # Simulate with $5000 capital, $750 max per trade
    equity = 5000.0
    trades = []
    for setup in all_setups:
        max_notional = min(750, equity)
        if max_notional <= 0:
            break
        shares = max_notional / setup["EntryPrice"]
        pnl = shares * (setup["ExitPrice"] - setup["EntryPrice"])
        equity += pnl
        trades.append({
            "Time": setup["Time"],
            "Ticker": setup["Ticker"],
            "EntryPrice": setup["EntryPrice"],
            "ExitPrice": setup["ExitPrice"],
            "Shares": round(shares, 4),
            "PnL_USD": pnl,
            "Equity_After": equity,
            "ExitReason": setup["ExitReason"]
        })
    
    if not trades:
        st.warning("No trades executed.")
    else:
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        wins = trades_df[trades_df["PnL_USD"] > 0]
        losses = trades_df[trades_df["PnL_USD"] <= 0]
        win_rate = len(wins)/total_trades*100 if total_trades>0 else 0
        total_pnl = trades_df["PnL_USD"].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Total PnL (USD)", f"${total_pnl:,.2f}")
        col4.metric("Final Equity", f"${equity:,.2f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trades_df["Time"], y=trades_df["Equity_After"],
                                 mode="lines+markers", name="Equity"))
        fig.update_layout(title="Equity Curve (USD)", xaxis_title="Trade Time")
        st.plotly_chart(fig)
        
        st.dataframe(trades_df)
        csv = trades_df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "trades.csv", "text/csv")