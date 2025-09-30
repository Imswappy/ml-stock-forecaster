# pages/Stock_Analysis.py
import datetime
import io
import os

import pandas as pd
import requests
import streamlit as st

# optional imports
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# local utils
from pages.utils.plotly_figure import (MACD, RSI, Moving_average, candlestick,
                                       close_chart, plotly_table)

# Page config
st.set_page_config(
    page_title="Stock Analysis",
    page_icon="📊",
    layout="wide",
)
st.title("Stock Analysis")

# Controls
col1, col2, col3 = st.columns(3)
today = datetime.date.today()

with col1:
    ticker = st.text_input("Stock Ticker", "AAPL").strip().upper()
with col2:
    start_date = st.date_input(
        "Start Date", datetime.date(today.year - 1, today.month, today.day)
    )
with col3:
    end_date = st.date_input("End Date", datetime.date(today.year, today.month, today.day))

uploaded_csv = st.file_uploader(
    "Optional: Upload CSV (Date + Close or Adj Close)", type=["csv"]
)

# --- CSV Helper
def parse_uploaded_csv_bytes(b):
    try:
        df = pd.read_csv(io.BytesIO(b))
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(b.decode()))
        except Exception:
            return pd.DataFrame()
    # detect date column
    date_col = None
    for c in df.columns:
        if c.lower() in ("date", "timestamp", "time"):
            date_col = c
            break
    if not date_col:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col).sort_index()
    # close column
    close_col = None
    for c in df.columns:
        if c.lower() in ("adj close", "adjusted_close", "adjusted close"):
            close_col = c
            break
    if not close_col:
        for c in df.columns:
            if c.lower() == "close":
                close_col = c
                break
    if not close_col:
        return pd.DataFrame()
    out_cols = [close_col]
    for k in ["Open", "High", "Low", "Volume"]:
        if k in df.columns:
            out_cols.append(k)
    return df[out_cols].rename(columns={close_col: "Close"})

# --- Price fetcher
@st.cache_data(ttl=3600)
def fetch_history(ticker_sym, start_dt, end_dt, uploaded_bytes=None):
    # 1. Uploaded CSV
    if uploaded_bytes:
        parsed = parse_uploaded_csv_bytes(uploaded_bytes)
        if not parsed.empty:
            return parsed

    # 2. yfinance
    if HAVE_YF:
        try:
            df = yf.download(ticker_sym, start=start_dt, end=end_dt, progress=False)
            if df is not None and not df.empty:
                if "Adj Close" in df.columns and "Close" not in df.columns:
                    df = df.rename(columns={"Adj Close": "Close"})
                cols = [
                    c for c in ["Close", "Open", "High", "Low", "Volume"] if c in df.columns
                ]
                return df[cols].dropna()
        except Exception:
            pass

    # 3. Alpha Vantage fallback
    av_key = (
        st.secrets.get("ALPHAVANTAGE_API_KEY", None)
        if hasattr(st, "secrets")
        else os.environ.get("ALPHAVANTAGE_API_KEY")
    )
    if av_key:
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker_sym,
                "outputsize": "full",
                "apikey": av_key,
            }
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                j = r.json()
                ts = j.get("Time Series (Daily)") or j.get("Time Series (Daily Adjusted)")
                if ts:
                    df = pd.DataFrame.from_dict(ts, orient="index")
                    df.index = pd.to_datetime(df.index)
                    df = df.rename(
                        columns={
                            "1. open": "Open",
                            "2. high": "High",
                            "3. low": "Low",
                            "4. close": "Close",
                            "5. adjusted close": "Adj Close",
                            "6. volume": "Volume",
                        }
                    )
                    # ensure numeric
                    for c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    cols = [
                        c
                        for c in ["Close", "Open", "High", "Low", "Volume"]
                        if c in df.columns
                    ]
                    out = df[cols].sort_index().loc[start_dt:end_dt]
                    return out.dropna()
        except Exception as e:
            st.warning(f"Alpha Vantage fetch failed: {e}")

    return pd.DataFrame()

# --- Metadata fetcher
def fetch_company_metadata(ticker_sym):
    meta = {
        "summary": None,
        "sector": None,
        "employees": None,
        "website": None,
        "ratios": {},
    }

    av_key = (
        st.secrets.get("ALPHAVANTAGE_API_KEY", None)
        if hasattr(st, "secrets")
        else os.environ.get("ALPHAVANTAGE_API_KEY")
    )
    if av_key:
        try:
            url = "https://www.alphavantage.co/query"
            params = {"function": "OVERVIEW", "symbol": ticker_sym, "apikey": av_key}
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                j = r.json()
                if j and "Symbol" in j:
                    meta["summary"] = j.get("Description")
                    meta["sector"] = j.get("Sector")
                    meta["employees"] = j.get("FullTimeEmployees")
                    meta["website"] = j.get("Website")
                    meta["ratios"] = {
                        "Market Cap": j.get("MarketCapitalization"),
                        "Beta": j.get("Beta"),
                        "EPS (trailing)": j.get("EPS"),
                        "PE Ratio (trailing)": j.get("PERatio"),
                        "Quick Ratio": j.get("QuickRatio"),
                        "Revenue per share": j.get("RevenuePerShareTTM"),
                        "Profit Margins": j.get("ProfitMargin"),
                        "Debt to Equity": j.get("DebtToEquity"),
                        "Return on Equity": j.get("ReturnOnEquityTTM"),
                    }
                    return meta
        except Exception:
            pass

    if HAVE_YF:
        try:
            info = yf.Ticker(ticker_sym).get_info()
            meta["summary"] = info.get("longBusinessSummary")
            meta["sector"] = info.get("sector")
            meta["employees"] = info.get("fullTimeEmployees")
            meta["website"] = info.get("website")
        except Exception:
            pass

    return meta

# --- Render
st.header(f"{ticker} — Overview")

meta = fetch_company_metadata(ticker)

if meta["summary"]:
    st.subheader("Company Summary")
    st.write(meta["summary"])
else:
    st.info("Company summary not available from APIs.")

meta_cols = st.columns(3)
with meta_cols[0]:
    st.write("**Sector**")
    st.write(meta["sector"] or "—")
with meta_cols[1]:
    st.write("**Employees**")
    st.write(meta["employees"] or "—")
with meta_cols[2]:
    st.write("**Website**")
    st.write(meta["website"] or "—")

# Ratios
left_df = pd.DataFrame(
    index=["Market Cap", "Beta", "EPS (trailing)", "PE Ratio (trailing)"]
)
left_df["value"] = [
    meta["ratios"].get("Market Cap"),
    meta["ratios"].get("Beta"),
    meta["ratios"].get("EPS (trailing)"),
    meta["ratios"].get("PE Ratio (trailing)"),
]
right_df = pd.DataFrame(
    index=[
        "Quick Ratio",
        "Revenue per share",
        "Profit Margins",
        "Debt to Equity",
        "Return on Equity",
    ]
)
right_df["value"] = [
    meta["ratios"].get("Quick Ratio"),
    meta["ratios"].get("Revenue per share"),
    meta["ratios"].get("Profit Margins"),
    meta["ratios"].get("Debt to Equity"),
    meta["ratios"].get("Return on Equity"),
]

col_left, col_right = st.columns(2)
with col_left:
    st.plotly_chart(plotly_table(left_df), use_container_width=True)
with col_right:
    st.plotly_chart(plotly_table(right_df), use_container_width=True)

st.markdown("---")

# Fetch prices
uploaded_bytes = uploaded_csv.read() if uploaded_csv else None
data = fetch_history(
    ticker, start_date.isoformat(), end_date.isoformat(), uploaded_bytes
)

if data.empty:
    st.error(
        "No historical price data available. Check ticker/network, upload CSV, or set ALPHAVANTAGE_API_KEY."
    )
else:
    try:
        daily_change = data["Close"].iloc[-1] - data["Close"].iloc[-2]
    except Exception:
        daily_change = None

    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.metric(
            "Latest Close",
            f"{data['Close'].iloc[-1]:.2f}" if "Close" in data.columns else "—",
            f"{daily_change:.2f}" if daily_change is not None else "--",
        )
    with metrics_cols[1]:
        st.write("### Head")
        st.dataframe(data.head().round(3))
    with metrics_cols[2]:
        st.write("### Tail")
        st.dataframe(data.tail().round(3))

    # Chart controls
    c1, c2 = st.columns([1, 3])
    with c1:
        chart_type = st.selectbox("Chart type", ("Candle", "Line"))
    with c2:
        if chart_type == "Candle":
            indicator = st.selectbox("Indicator", ("RSI", "MACD"))
        else:
            indicator = st.selectbox(
                "Indicator", ("RSI", "Moving Average", "MACD")
            )

    def safe_plot(func):
        try:
            fig = func(data, "1y")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotting failed: {e}")

    if chart_type == "Candle":
        safe_plot(candlestick)
        if indicator == "RSI":
            safe_plot(RSI)
        elif indicator == "MACD":
            safe_plot(MACD)
    else:
        safe_plot(close_chart)
        if indicator == "RSI":
            safe_plot(RSI)
        elif indicator == "Moving Average":
            safe_plot(Moving_average)
        elif indicator == "MACD":
            safe_plot(MACD)

st.caption(
    "Notes: Uses Alpha Vantage for metadata and price fallback. "
    "Set `ALPHAVANTAGE_API_KEY` in Streamlit secrets or environment."
)
