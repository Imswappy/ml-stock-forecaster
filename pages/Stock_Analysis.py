# pages/Stock_Analysis.py
import datetime
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# optional imports (graceful fallback)
try:
    import ta
    HAVE_TA = True
except Exception:
    HAVE_TA = False

try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# local utilities (plot helpers)
from pages.utils.plotly_figure import (MACD, RSI, Moving_average, candlestick,
                                       close_chart, plotly_table)

# Page config
st.set_page_config(
    page_title="Stock Analysis",
    page_icon="page_with_curl",
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

st.markdown("---")

st.header(f"{ticker} — Overview")

# Optional upload (CSV fallback) — useful for offline / blocked network
uploaded_csv = st.file_uploader("Optional: upload CSV (Date + Close or Adj Close) to use instead of network", type=["csv"])

# Helper: parse uploaded CSV into same shape returned by yfinance
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
    if date_col is None:
        date_col = df.columns[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    except Exception:
        return pd.DataFrame()
    df = df.set_index(date_col).sort_index()
    # find close column
    close_col = None
    for c in df.columns:
        if c.lower() in ("adj close", "adjusted_close", "adjusted close"):
            close_col = c
            break
    if close_col is None:
        for c in df.columns:
            if c.lower() == "close":
                close_col = c
                break
    if close_col is None:
        return pd.DataFrame()
    out_cols = [close_col]
    for k in ["Open", "High", "Low", "Volume"]:
        if k in df.columns:
            out_cols.append(k)
    out = df[out_cols].rename(columns={close_col: "Close"})
    return out

# Safe fetch helper (uses uploaded CSV first, then yfinance)
@st.cache_data(ttl=3600)
def fetch_history(ticker_sym, start_dt, end_dt, uploaded_bytes=None, show_diag=False):
    # uploaded CSV first
    if uploaded_bytes is not None:
        parsed = parse_uploaded_csv_bytes(uploaded_bytes)
        if not parsed.empty:
            if show_diag:
                st.info("Using uploaded CSV for historical prices.")
            return parsed

    if not HAVE_YF:
        return pd.DataFrame()

    try:
        df = yf.download(ticker_sym, start=start_dt, end=end_dt, progress=False)
        if df is None or df.empty:
            # try yf.Ticker.history fallback
            t = yf.Ticker(ticker_sym)
            df = t.history(start=start_dt, end=end_dt, actions=False)
        # rename Adj Close if necessary
        if df is None:
            return pd.DataFrame()
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        # keep relevant columns and drop NA rows
        cols = [c for c in ["Close", "Open", "High", "Low", "Volume"] if c in df.columns]
        out = df[cols].dropna()
        return out
    except Exception as e:
        # return empty df to caller; caller will show message
        return pd.DataFrame()

# --- Company metadata (safe)
company_summary = None
sector = None
employees = None
website = None
ratios = {}

if HAVE_YF:
    try:
        ticker_obj = yf.Ticker(ticker)
    except Exception:
        ticker_obj = None
else:
    ticker_obj = None

if ticker_obj is not None:
    # try get_info (safer to wrap)
    try:
        info = ticker_obj.get_info()  # wrapped call
    except Exception as e:
        # fallback to fast_info
        info = {}
        try:
            fast = ticker_obj.fast_info or {}
            info.update(fast)
        except Exception:
            info = {}

    # Get summary and fields safely
    company_summary = info.get("longBusinessSummary") or info.get("longBusinessSummary", None)
    sector = info.get("sector") or info.get("industry") or None
    employees = info.get("fullTimeEmployees") or info.get("employees") or None
    website = info.get("website") or info.get("websiteAddress") or info.get("url") or None

    # collect ratios/values without direct indexing
    def safe_get(k):
        return info.get(k, None)

    ratios = {
        "Market Cap": safe_get("marketCap"),
        "Beta": safe_get("beta"),
        "EPS (trailing)": safe_get("trailingEps"),
        "PE Ratio (trailing)": safe_get("trailingPE"),
        "Quick Ratio": safe_get("quickRatio"),
        "Revenue per share": safe_get("revenuePerShare"),
        "Profit Margins": safe_get("profitMargins"),
        "Debt to Equity": safe_get("debtToEquity"),
        "Return on Equity": safe_get("returnOnEquity"),
    }
else:
    company_summary = None

# Display metadata with graceful fallbacks
if company_summary:
    st.subheader("Company Summary")
    st.write(company_summary)
else:
    st.info("Company summary not available from yfinance. Showing available metadata (if any).")

meta_cols = st.columns(3)
with meta_cols[0]:
    st.write("**Sector**")
    st.write(sector or "—")
with meta_cols[1]:
    st.write("**Employees**")
    st.write(employees or "—")
with meta_cols[2]:
    st.write("**Website**")
    if website:
        st.write(website)
    else:
        st.write("—")

# Show ratio tables (only present keys)
left_df = pd.DataFrame(index=["Market Cap", "Beta", "EPS (trailing)", "PE Ratio (trailing)"])
left_df["value"] = [
    ratios.get("Market Cap"),
    ratios.get("Beta"),
    ratios.get("EPS (trailing)"),
    ratios.get("PE Ratio (trailing)"),
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
    ratios.get("Quick Ratio"),
    ratios.get("Revenue per share"),
    ratios.get("Profit Margins"),
    ratios.get("Debt to Equity"),
    ratios.get("Return on Equity"),
]

col_left, col_right = st.columns(2)
with col_left:
    st.plotly_chart(plotly_table(left_df), use_container_width=True)
with col_right:
    st.plotly_chart(plotly_table(right_df), use_container_width=True)

st.markdown("---")

# Fetch historical data
uploaded_bytes = uploaded_csv.read() if uploaded_csv is not None else None
data = fetch_history(ticker, start_date.isoformat(), end_date.isoformat(), uploaded_bytes=uploaded_bytes, show_diag=True)

if data is None or data.empty:
    st.error(
        "No historical price data available. Check ticker/network, or upload a CSV with Date + Close column."
    )
else:
    # display small summary metrics
    try:
        daily_change = data["Close"].iloc[-1] - data["Close"].iloc[-2]
    except Exception:
        daily_change = None

    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.metric("Latest Close", f"{data['Close'].iloc[-1]:.2f}" if "Close" in data.columns else "—", f"{daily_change:.2f}" if daily_change is not None else "--")
    with metrics_cols[1]:
        st.write("### Head (first rows)")
        st.dataframe(data.head().round(6))
    with metrics_cols[2]:
        st.write("### Tail (last rows)")
        st.dataframe(data.tail().round(6))

    st.markdown("---")

    # Quick buttons for ranges (these affect interactive charts below)
    btn_cols = st.columns([1] * 7)
    num_period = ""
    labels = ["5D", "1M", "6M", "YTD", "1Y", "5Y", "MAX"]
    vals = ["5d", "1mo", "6mo", "ytd", "1y", "5y", "max"]
    for c, lab, v in zip(btn_cols, labels, vals):
        with c:
            if st.button(lab):
                num_period = v

    # Chart controls
    c1, c2 = st.columns([1, 3])
    with c1:
        chart_type = st.selectbox("Chart type", ("Candle", "Line"))
    with c2:
        if chart_type == "Candle":
            indicator = st.selectbox("Indicator", ("RSI", "MACD"))
        else:
            indicator = st.selectbox("Indicator", ("RSI", "Moving Average", "MACD"))

    # Prepare a ticker history for the chosen period (use yf.Ticker to get 'max' window easily)
    ticker_obj = yf.Ticker(ticker) if HAVE_YF else None
    try:
        hist_all = ticker_obj.history(period="max") if ticker_obj is not None else data
    except Exception:
        hist_all = data

    df_for_plot = hist_all.copy()

    # Use num_period if set, otherwise use selected user start/end range
    if num_period:
        # use period-based plotting
        plot_src = df_for_plot
        period_arg = num_period
    else:
        plot_src = data
        period_arg = None

    # Helper to call plotting functions safely
    def safe_plot(func, df_in, period_arg_local):
        try:
            if period_arg_local:
                fig = func(df_in, period_arg_local)
            else:
                fig = func(df_in, "1y")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotting failed: {e}")

    # Render charts
    if chart_type == "Candle":
        safe_plot(candlestick, plot_src, num_period or "1y")
        if indicator == "RSI":
            safe_plot(RSI, plot_src, num_period or "1y")
        elif indicator == "MACD":
            safe_plot(MACD, plot_src, num_period or "1y")
    else:  # Line chart
        safe_plot(close_chart, plot_src, num_period or "1y")
        if indicator == "RSI":
            safe_plot(RSI, plot_src, num_period or "1y")
        elif indicator == "Moving Average":
            safe_plot(Moving_average, plot_src, num_period or "1y")
        elif indicator == "MACD":
            safe_plot(MACD, plot_src, num_period or "1y")

st.markdown("---")
st.caption("Notes: yfinance metadata (`.info`) can be unstable; the app uses guarded calls and fallbacks (fast_info / uploaded CSV) to avoid crashes.")
