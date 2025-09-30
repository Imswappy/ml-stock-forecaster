# pages/Stock_Analysis.py
import datetime
import io
import os
from typing import Optional

import pandas as pd
import requests
import streamlit as st

# optional imports
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# local plot helpers (must exist)
from pages.utils.plotly_figure import (MACD, RSI, Moving_average, candlestick,
                                       close_chart, plotly_table)

# Page config
st.set_page_config(page_title="Stock Analysis", page_icon="📊", layout="wide")
st.title("📊 Stock Analysis")

# -----------------------
# Controls
# -----------------------
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
    "Optional: Upload CSV (Date + Close or Adj Close) to use instead of network", type=["csv"]
)

show_diag = st.checkbox("Show fetch diagnostics (useful for debugging)", value=True)
force_twelvedata = st.checkbox("Force use Twelve Data (if key available)", value=False)

# -----------------------------
# CSV Parser
# -----------------------------
def parse_uploaded_csv_bytes(b: bytes) -> pd.DataFrame:
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

    # detect close column
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
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(how="all")
    return out

# -----------------------------
# Twelve Data helper
# -----------------------------
def fetch_from_twelvedata(symbol: str, start_dt: str, end_dt: str, show_diag: bool = False) -> pd.DataFrame:
    td_key = None
    try:
        td_key = st.secrets.get("TWELVEDATA_API_KEY")
    except Exception:
        td_key = None
    if not td_key:
        td_key = os.environ.get("TWELVEDATA_API_KEY")

    if not td_key:
        if show_diag:
            st.info("No TWELVEDATA_API_KEY found in secrets or environment.")
        return pd.DataFrame()

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "start_date": start_dt,
        "end_date": end_dt,
        "outputsize": 5000,
        "format": "JSON",
        "apikey": td_key,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
    except Exception as e:
        if show_diag:
            st.warning(f"Twelve Data request failed: {e}")
        return pd.DataFrame()

    if show_diag:
        st.write(f"Twelve Data HTTP {r.status_code} for {symbol}")
        snippet = (r.text or "")[:1000]
        st.code(snippet + ("... (truncated)" if len(r.text or "") > 1000 else ""))

    if r.status_code != 200:
        return pd.DataFrame()

    try:
        j = r.json()
    except Exception:
        return pd.DataFrame()

    # check for API error
    if isinstance(j, dict) and j.get("status") == "error":
        if show_diag:
            st.error(f"Twelve Data error: {j.get('message') or j}")
        return pd.DataFrame()

    values = j.get("values") if isinstance(j, dict) else None
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "close":
            col_map[c] = "Close"
        elif lc == "open":
            col_map[c] = "Open"
        elif lc == "high":
            col_map[c] = "High"
        elif lc == "low":
            col_map[c] = "Low"
        elif lc == "volume":
            col_map[c] = "Volume"
    df = df.rename(columns=col_map)
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [c for c in ["Close", "Open", "High", "Low", "Volume"] if c in df.columns]
    out = df[keep].dropna()
    if show_diag and not out.empty:
        st.success(f"Twelve Data returned {len(out)} rows for {symbol}")
    return out

# -----------------------------
# Alpha Vantage fallback
# -----------------------------
def fetch_from_alphavantage(symbol: str, start_dt: str, end_dt: str, show_diag: bool = False) -> pd.DataFrame:
    av_key = None
    try:
        av_key = st.secrets.get("ALPHAVANTAGE_API_KEY")
    except Exception:
        av_key = None
    if not av_key:
        av_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not av_key:
        return pd.DataFrame()

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": av_key,
        "datatype": "json",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
    except Exception:
        return pd.DataFrame()

    if r.status_code != 200:
        return pd.DataFrame()

    try:
        j = r.json()
    except Exception:
        return pd.DataFrame()

    ts = j.get("Time Series (Daily)") or j.get("Time Series (Daily Adjusted)")
    if not ts:
        if show_diag:
            st.warning("AlphaVantage returned no time series (might be rate-limited).")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "adjusted close" in lc or "5. adjusted close" in lc:
            col_map[c] = "Adj Close"
        elif "close" in lc:
            col_map[c] = "Close"
        elif "open" in lc:
            col_map[c] = "Open"
        elif "high" in lc:
            col_map[c] = "High"
        elif "low" in lc:
            col_map[c] = "Low"
        elif "volume" in lc:
            col_map[c] = "Volume"
    df = df.rename(columns=col_map)
    for col in ["Close", "Adj Close", "Open", "High", "Low", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    cols = [c for c in ["Close", "Open", "High", "Low", "Volume"] if c in df.columns]
    out = df[cols].sort_index().loc[start_dt:end_dt].dropna()
    if show_diag and not out.empty:
        st.success(f"AlphaVantage returned {len(out)} rows for {symbol}")
    return out

# -----------------------------
# High-level fetch (cached)
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_history(ticker_sym: str, start_dt: str, end_dt: str, uploaded_bytes: Optional[bytes], show_diag: bool = False, force_td: bool = False) -> pd.DataFrame:
    # 0) uploaded CSV
    if uploaded_bytes is not None:
        parsed = parse_uploaded_csv_bytes(uploaded_bytes)
        if not parsed.empty:
            if show_diag:
                st.info("Using uploaded CSV for historical prices.")
            return parsed

    # if forced, try Twelve Data first
    if force_td:
        td = fetch_from_twelvedata(ticker_sym, start_dt, end_dt, show_diag=show_diag)
        if not td.empty:
            return td

    # 1) yfinance.download (if available)
    if HAVE_YF and not force_td:
        try:
            df = yf.download(ticker_sym, start=start_dt, end=end_dt, progress=False)
            if (df is None) or df.empty:
                t = yf.Ticker(ticker_sym)
                df = t.history(start=start_dt, end=end_dt, actions=False)
            if df is not None and not df.empty:
                if "Adj Close" in df.columns and "Close" not in df.columns:
                    df = df.rename(columns={"Adj Close": "Close"})
                cols = [c for c in ["Close", "Open", "High", "Low", "Volume"] if c in df.columns]
                out = df[cols].dropna()
                if show_diag and not out.empty:
                    st.success(f"yfinance returned {len(out)} rows for {ticker_sym}")
                return out
            else:
                if show_diag:
                    st.warning("yfinance returned empty data.")
        except Exception as e:
            if show_diag:
                st.warning(f"yfinance exception: {e}")

    # 2) Twelve Data (default second)
    td = fetch_from_twelvedata(ticker_sym, start_dt, end_dt, show_diag=show_diag)
    if not td.empty:
        return td

    # 3) AlphaVantage fallback
    av = fetch_from_alphavantage(ticker_sym, start_dt, end_dt, show_diag=show_diag)
    if not av.empty:
        return av

    return pd.DataFrame()

# -----------------------------
# Company metadata (tries TD -> AV -> yfinance)
# -----------------------------
def fetch_company_metadata(ticker_sym: str, show_diag: bool = False) -> dict:
    meta = {"summary": None, "sector": None, "employees": None, "website": None, "ratios": {}}

    # try Twelve Data profile endpoint
    td_key = None
    try:
        td_key = st.secrets.get("TWELVEDATA_API_KEY")
    except Exception:
        td_key = None
    if not td_key:
        td_key = os.environ.get("TWELVEDATA_API_KEY")

    if td_key:
        try:
            url = "https://api.twelvedata.com/company_profile"
            r = requests.get(url, params={"symbol": ticker_sym, "apikey": td_key}, timeout=10)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, dict) and j.get("name"):
                    meta["summary"] = j.get("description")
                    meta["sector"] = j.get("industry") or j.get("sector")
                    meta["employees"] = j.get("employees")
                    meta["website"] = j.get("website")
                    if show_diag:
                        st.success("Fetched company profile from Twelve Data")
                    return meta
        except Exception:
            if show_diag:
                st.warning("Twelve Data profile attempt failed.")

    # AlphaVantage overview
    av_key = None
    try:
        av_key = st.secrets.get("ALPHAVANTAGE_API_KEY")
    except Exception:
        av_key = None
    if not av_key:
        av_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if av_key:
        try:
            url = "https://www.alphavantage.co/query"
            params = {"function": "OVERVIEW", "symbol": ticker_sym, "apikey": av_key}
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                j = r.json()
                if j and j.get("Symbol"):
                    meta["summary"] = j.get("Description")
                    meta["sector"] = j.get("Sector")
                    meta["employees"] = j.get("FullTimeEmployees")
                    meta["website"] = j.get("Website")
                    meta["ratios"] = {
                        "Market Cap": j.get("MarketCapitalization"),
                        "Beta": j.get("Beta"),
                        "EPS (trailing)": j.get("EPS"),
                        "PE Ratio (trailing)": j.get("PERatio"),
                    }
                    if show_diag:
                        st.success("Fetched company metadata from AlphaVantage")
                    return meta
        except Exception:
            if show_diag:
                st.warning("AlphaVantage overview attempt failed.")

    # yfinance fallback
    if HAVE_YF:
        try:
            t = yf.Ticker(ticker_sym)
            try:
                info = t.get_info()
            except Exception:
                info = {}
            try:
                fast = t.fast_info or {}
            except Exception:
                fast = {}
            meta["summary"] = info.get("longBusinessSummary") or meta["summary"]
            meta["sector"] = info.get("sector") or info.get("industry") or fast.get("industry")
            meta["employees"] = info.get("fullTimeEmployees") or fast.get("employees")
            meta["website"] = info.get("website") or info.get("websiteAddress")
            if info:
                meta["ratios"].update(
                    {
                        "Market Cap": info.get("marketCap"),
                        "Beta": info.get("beta"),
                        "EPS (trailing)": info.get("trailingEps"),
                        "PE Ratio (trailing)": info.get("trailingPE"),
                    }
                )
            if show_diag:
                st.info("Used yfinance for metadata fallback.")
        except Exception:
            if show_diag:
                st.warning("yfinance metadata attempt failed.")

    return meta

# -----------------------------
# Render
# -----------------------------
st.header(f"{ticker} — Overview")

meta = fetch_company_metadata(ticker, show_diag=show_diag)

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
    if meta["website"]:
        st.write(meta["website"])
    else:
        st.write("—")

# Ratios tables
left_df = pd.DataFrame(
    {
        "value": [
            meta["ratios"].get("Market Cap"),
            meta["ratios"].get("Beta"),
            meta["ratios"].get("EPS (trailing)"),
            meta["ratios"].get("PE Ratio (trailing)"),
        ]
    },
    index=["Market Cap", "Beta", "EPS (trailing)", "PE Ratio (trailing)"],
)
right_df = pd.DataFrame(
    {
        "value": [
            meta["ratios"].get("Quick Ratio"),
            meta["ratios"].get("Revenue per share"),
            meta["ratios"].get("Profit Margins"),
            meta["ratios"].get("Debt to Equity"),
            meta["ratios"].get("Return on Equity"),
        ]
    },
    index=["Quick Ratio", "Revenue per share", "Profit Margins", "Debt to Equity", "Return on Equity"],
)

col_left, col_right = st.columns(2)
with col_left:
    st.plotly_chart(plotly_table(left_df), use_container_width=True)
with col_right:
    st.plotly_chart(plotly_table(right_df), use_container_width=True)

st.markdown("---")

# -----------------------------
# Test Twelve Data key button (visible to user)
# -----------------------------
td_test_col1, td_test_col2 = st.columns([3, 1])
with td_test_col1:
    st.caption("If you set `TWELVEDATA_API_KEY`, press the test button to validate it from this environment.")
with td_test_col2:
    if st.button("Test Twelve Data key"):
        td_key = None
        try:
            td_key = st.secrets.get("TWELVEDATA_API_KEY")
        except Exception:
            td_key = None
        if not td_key:
            td_key = os.environ.get("TWELVEDATA_API_KEY")
        if not td_key:
            st.error("No TWELVEDATA_API_KEY found in Streamlit secrets or environment.")
        else:
            r = requests.get("https://api.twelvedata.com/time_series", params={"symbol": "AAPL", "interval": "1day", "apikey": td_key, "outputsize": 1})
            st.write("HTTP status:", r.status_code)
            try:
                st.json(r.json())
            except Exception:
                st.text(r.text[:1000])

# -----------------------------
# Fetch prices and render charts/head/tail
# -----------------------------
uploaded_bytes = uploaded_csv.read() if uploaded_csv else None
data = fetch_history(
    ticker,
    start_date.isoformat(),
    end_date.isoformat(),
    uploaded_bytes,
    show_diag=show_diag,
    force_td=(force_twelvedata or False),
)

if data is None or data.empty:
    st.error(
        "No historical price data available. Check ticker/network, upload CSV, or set TWELVEDATA_API_KEY / ALPHAVANTAGE_API_KEY in Streamlit secrets or environment."
    )
    if show_diag:
        st.caption("Diagnostics were enabled — check the messages above for HTTP / response details.")
else:
    # small summary metrics
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
        st.write("### Head (first rows)")
        st.dataframe(data.head().round(6))
    with metrics_cols[2]:
        st.write("### Tail (last rows)")
        st.dataframe(data.tail().round(6))

    st.markdown("---")

    # Quick range buttons
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

    def safe_plot(func, df_in, period_arg="1y"):
        try:
            fig = func(df_in, period_arg)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotting failed: {e}")

    if chart_type == "Candle":
        safe_plot(candlestick, data, num_period or "1y")
        if indicator == "RSI":
            safe_plot(RSI, data, num_period or "1y")
        elif indicator == "MACD":
            safe_plot(MACD, data, num_period or "1y")
    else:
        safe_plot(close_chart, data, num_period or "1y")
        if indicator == "RSI":
            safe_plot(RSI, data, num_period or "1y")
        elif indicator == "Moving Average":
            safe_plot(Moving_average, data, num_period or "1y")
        elif indicator == "MACD":
            safe_plot(MACD, data, num_period or "1y")

st.markdown("---")
st.caption(
    "Notes: Price fetching order (by default): uploaded CSV → yfinance → Twelve Data → AlphaVantage. "
    "Tick the 'Force use Twelve Data' box to try Twelve Data first. "
    "Set `TWELVEDATA_API_KEY` and/or `ALPHAVANTAGE_API_KEY` in Streamlit secrets or environment variables."
)
