# pages/Stock_Analysis.py
import datetime
import io
import os

import pandas as pd
import requests
import streamlit as st

# optional imports
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

# local plot helpers
from pages.utils.plotly_figure import (MACD, RSI, Moving_average, candlestick,
                                       close_chart, plotly_table)

# --- Page config
st.set_page_config(
    page_title="Stock Analysis",
    page_icon="page_with_curl",
    layout="wide",
)
st.title("Stock Analysis")

# --- Controls
col1, col2, col3 = st.columns(3)
today = datetime.date.today()
with col1:
    ticker = st.text_input("Stock Ticker", "AAPL").strip().upper()
with col2:
    start_date = st.date_input(
        "Start Date", datetime.date(today.year - 1, today.month, today.day)
    )
with col3:
    end_date = st.date_input(
        "End Date", datetime.date(today.year, today.month, today.day)
    )

st.markdown("---")
st.header(f"{ticker} — Overview")

# --- Optional CSV upload
uploaded_csv = st.file_uploader(
    "Optional: upload CSV (Date + Close or Adj Close)",
    type=["csv"],
)

# --- CSV parser
def parse_uploaded_csv_bytes(b):
    try:
        df = pd.read_csv(io.BytesIO(b))
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(b.decode()))
        except Exception:
            return pd.DataFrame()
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

# --- History fetcher
@st.cache_data(ttl=3600)
def fetch_history(ticker_sym, start_dt, end_dt, uploaded_bytes=None):
    if uploaded_bytes is not None:
        parsed = parse_uploaded_csv_bytes(uploaded_bytes)
        if not parsed.empty:
            return parsed

    if not HAVE_YF:
        return pd.DataFrame()

    try:
        df = yf.download(ticker_sym, start=start_dt, end=end_dt, progress=False)
        if df is None or df.empty:
            t = yf.Ticker(ticker_sym)
            df = t.history(start=start_dt, end=end_dt, actions=False)
        if df is None:
            return pd.DataFrame()
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        cols = [c for c in ["Close", "Open", "High", "Low", "Volume"] if c in df.columns]
        return df[cols].dropna()
    except Exception:
        return pd.DataFrame()

# --- Metadata fetch (Alpha Vantage -> yfinance fallback)
company_summary, sector, employees, website = None, None, None, None
ratios = {}

# Alpha Vantage key
av_key = None
try:
    av_key = st.secrets.get("ALPHAVANTAGE_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    av_key = None
if not av_key:
    av_key = os.environ.get("ALPHAVANTAGE_API_KEY")

if av_key:
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function": "OVERVIEW", "symbol": ticker, "apikey": av_key}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            ov = r.json()
            if ov and "Name" in ov:
                company_summary = ov.get("Description")
                sector = ov.get("Sector")
                employees = ov.get("FullTimeEmployees")
                website = ov.get("Website")
                ratios = {
                    "Market Cap": ov.get("MarketCapitalization"),
                    "Beta": ov.get("Beta"),
                    "EPS (trailing)": ov.get("EPS"),
                    "PE Ratio (trailing)": ov.get("PERatio"),
                    "Quick Ratio": ov.get("QuickRatio"),
                    "Revenue per share": ov.get("RevenuePerShareTTM"),
                    "Profit Margins": ov.get("ProfitMargin"),
                    "Debt to Equity": ov.get("DebtToEquity"),
                    "Return on Equity": ov.get("ReturnOnEquityTTM"),
                }
    except Exception as e:
        st.warning(f"AlphaVantage metadata failed: {e}")

# yfinance fallback if missing
if not company_summary and HAVE_YF:
    try:
        info = yf.Ticker(ticker).get_info()
        company_summary = info.get("longBusinessSummary")
        sector = sector or info.get("sector")
        employees = employees or info.get("fullTimeEmployees")
        website = website or info.get("website")
        if not ratios:
            ratios = {
                "Market Cap": info.get("marketCap"),
                "Beta": info.get("beta"),
                "EPS (trailing)": info.get("trailingEps"),
                "PE Ratio (trailing)": info.get("trailingPE"),
                "Quick Ratio": info.get("quickRatio"),
                "Revenue per share": info.get("revenuePerShare"),
                "Profit Margins": info.get("profitMargins"),
                "Debt to Equity": info.get("debtToEquity"),
                "Return on Equity": info.get("returnOnEquity"),
            }
    except Exception:
        pass

# --- Display metadata
if company_summary:
    st.subheader("Company Summary")
    st.write(company_summary)
else:
    st.info("Company summary not available. Showing available metadata (if any).")

mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.write("**Sector**")
    st.write(sector or "—")
with mc2:
    st.write("**Employees**")
    st.write(employees or "—")
with mc3:
    st.write("**Website**")
    st.write(website or "—")

# ratios table, dropping Nones
left = pd.DataFrame({
    "Metric": ["Market Cap", "Beta", "EPS (trailing)", "PE Ratio (trailing)"],
    "Value": [ratios.get("Market Cap"), ratios.get("Beta"),
              ratios.get("EPS (trailing)"), ratios.get("PE Ratio (trailing)")],
}).dropna()
right = pd.DataFrame({
    "Metric": ["Quick Ratio", "Revenue per share", "Profit Margins",
               "Debt to Equity", "Return on Equity"],
    "Value": [ratios.get("Quick Ratio"), ratios.get("Revenue per share"),
              ratios.get("Profit Margins"), ratios.get("Debt to Equity"),
              ratios.get("Return on Equity")],
}).dropna()

c1, c2 = st.columns(2)
with c1:
    if not left.empty:
        st.plotly_chart(plotly_table(left.set_index("Metric")), use_container_width=True)
with c2:
    if not right.empty:
        st.plotly_chart(plotly_table(right.set_index("Metric")), use_container_width=True)

st.markdown("---")

# --- Historical prices
uploaded_bytes = uploaded_csv.read() if uploaded_csv else None
data = fetch_history(ticker, start_date.isoformat(), end_date.isoformat(), uploaded_bytes)

if data.empty:
    st.error("No historical price data available. Check ticker/network, or upload a CSV.")
else:
    try:
        daily_change = data["Close"].iloc[-1] - data["Close"].iloc[-2]
    except Exception:
        daily_change = None
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Latest Close",
                  f"{data['Close'].iloc[-1]:.2f}" if "Close" in data else "—",
                  f"{daily_change:.2f}" if daily_change is not None else "--")
    with m2:
        st.write("### Head")
        st.dataframe(data.head().round(6))
    with m3:
        st.write("### Tail")
        st.dataframe(data.tail().round(6))

    st.markdown("---")

    # buttons
    num_period = ""
    btn_cols = st.columns([1]*7)
    labels = ["5D","1M","6M","YTD","1Y","5Y","MAX"]
    vals   = ["5d","1mo","6mo","ytd","1y","5y","max"]
    for c, lab, v in zip(btn_cols, labels, vals):
        with c:
            if st.button(lab):
                num_period = v

    # chart controls
    c1, c2 = st.columns([1,3])
    with c1:
        chart_type = st.selectbox("Chart type", ("Candle","Line"))
    with c2:
        if chart_type=="Candle":
            indicator = st.selectbox("Indicator", ("RSI","MACD"))
        else:
            indicator = st.selectbox("Indicator", ("RSI","Moving Average","MACD"))

    # get max history for plotting
    try:
        hist_all = yf.Ticker(ticker).history(period="max") if HAVE_YF else data
    except Exception:
        hist_all = data
    plot_src = hist_all if num_period else data

    # safe plot helper
    def safe_plot(func, df_in, per):
        try:
            fig = func(df_in, per or "1y")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotting failed: {e}")

    # render
    if chart_type=="Candle":
        safe_plot(candlestick, plot_src, num_period or "1y")
        if indicator=="RSI": safe_plot(RSI, plot_src, num_period or "1y")
        if indicator=="MACD": safe_plot(MACD, plot_src, num_period or "1y")
    else:
        safe_plot(close_chart, plot_src, num_period or "1y")
        if indicator=="RSI": safe_plot(RSI, plot_src, num_period or "1y")
        if indicator=="Moving Average": safe_plot(Moving_average, plot_src, num_period or "1y")
        if indicator=="MACD": safe_plot(MACD, plot_src, num_period or "1y")

st.caption("Notes: Metadata uses AlphaVantage if API key set, else yfinance (less reliable). Price data from yfinance unless CSV uploaded.")
