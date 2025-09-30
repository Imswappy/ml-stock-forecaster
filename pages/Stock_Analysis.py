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

# local utils (plot helpers)
from pages.utils.plotly_figure import (MACD, RSI, Moving_average, candlestick,
                                       close_chart, plotly_table)

# Page config
st.set_page_config(
    page_title="Stock Analysis",
    page_icon="📊",
    layout="wide",
)
st.title("📊 Stock Analysis")

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
    "Optional: Upload CSV (Date + Close or Adj Close). If provided, CSV will be used instead of network.",
    type=["csv"],
)

show_diagnostics = st.checkbox("Show fetch diagnostics / trace", value=False)


# -----------------------------
# CSV Parser
# -----------------------------
def parse_uploaded_csv_bytes(b: bytes) -> pd.DataFrame:
    """Parse user-uploaded CSV into DataFrame with index datetime and 'Close' column (plus OHLC/Volume if present)."""
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

    out = df[out_cols].rename(columns={close_col: "Close"})
    return out


# -----------------------------
# Twelve Data fetch helper
# -----------------------------
# Diagnostic Twelve Data fetch + robust fetch_history — paste in pages/Stock_Analysis.py
def fetch_from_twelvedata(symbol: str, start_dt: str, end_dt: str, show_diag: bool = False) -> pd.DataFrame:
    """
    Diagnostic Twelve Data fetch. Returns DataFrame or empty.
    Shows diagnostics in Streamlit when show_diag True.
    """
    td_key = None
    try:
        td_key = st.secrets.get("TWELVEDATA_API_KEY")
    except Exception:
        td_key = None
    if not td_key:
        td_key = os.environ.get("TWELVEDATA_API_KEY")

    if not td_key:
        if show_diag:
            st.warning("No TWELVEDATA_API_KEY found in secrets or environment.")
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
            st.error(f"Twelve Data request exception: {e}")
        return pd.DataFrame()

    if show_diag:
        st.write(f"Twelve Data HTTP {r.status_code} for {symbol}")
        # show small portion of response for debugging but avoid huge dumps
        txt = r.text
        st.code(txt[:1000] + ("... (truncated)" if len(txt) > 1000 else ""))

    try:
        j = r.json()
    except Exception as e:
        if show_diag:
            st.error(f"Failed to parse JSON from Twelve Data: {e}")
        return pd.DataFrame()

    # Common failure cases
    if isinstance(j, dict) and (j.get("status") == "error" or "message" in j):
        if show_diag:
            st.error(f"Twelve Data error: {j.get('message') or j}")
        return pd.DataFrame()

    values = j.get("values") if isinstance(j, dict) else None
    if not values:
        if show_diag:
            st.warning("Twelve Data returned no 'values' field or empty list.")
        return pd.DataFrame()

    df = pd.DataFrame(values)
    # unify index
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    # map numeric columns
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("close",):
            col_map[c] = "Close"
        elif lc in ("open",):
            col_map[c] = "Open"
        elif lc in ("high",):
            col_map[c] = "High"
        elif lc in ("low",):
            col_map[c] = "Low"
        elif lc in ("volume",):
            col_map[c] = "Volume"
    df = df.rename(columns=col_map)
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [c for c in ["Close", "Open", "High", "Low", "Volume"] if c in df.columns]
    out = df[keep].dropna()
    if show_diag:
        st.success(f"Twelve Data returned {len(out)} rows for {symbol}")
    return out

@st.cache_data(ttl=3600)
def fetch_history(ticker_sym: str, start_dt: str, end_dt: str, uploaded_bytes: Optional[bytes]) -> pd.DataFrame:
    """
    Diagnostic fetch_history with explicit diagnostics (uploaded CSV -> yfinance -> Twelve Data).
    """
    # 1) uploaded CSV
    if uploaded_bytes:
        parsed = parse_uploaded_csv_bytes(uploaded_bytes)
        if not parsed.empty:
            if show_diagnostics:
                st.success("Using uploaded CSV (parsed).")
            return parsed
        else:
            if show_diagnostics:
                st.warning("Uploaded CSV present but parsing produced empty DataFrame.")

    # 2) yfinance
    if HAVE_YF:
        try:
            df = yf.download(ticker_sym, start=start_dt, end=end_dt, progress=False)
            if df is None or df.empty:
                t = yf.Ticker(ticker_sym)
                df = t.history(start=start_dt, end=end_dt, actions=False)
            if df is not None and not df.empty:
                if "Adj Close" in df.columns and "Close" not in df.columns:
                    df = df.rename(columns={"Adj Close": "Close"})
                cols = [c for c in ["Close", "Open", "High", "Low", "Volume"] if c in df.columns]
                out = df[cols].dropna()
                if not out.empty:
                    if show_diagnostics:
                        st.success(f"yfinance returned {len(out)} rows for {ticker_sym}")
                    return out
                else:
                    if show_diagnostics:
                        st.warning("yfinance returned rows but after selecting columns & dropping NA the result is empty.")
            else:
                if show_diagnostics:
                    st.warning("yfinance returned no data (empty).")
        except Exception as e:
            if show_diagnostics:
                st.error(f"yfinance exception: {e}")

    # 3) Twelve Data fallback
    td_df = fetch_from_twelvedata(ticker_sym, start_dt, end_dt, show_diag=show_diagnostics)
    if not td_df.empty:
        return td_df

    # nothing worked
    if show_diagnostics:
        st.error("All data sources failed (uploaded / yfinance / Twelve Data).")
    return pd.DataFrame()



# -----------------------------
# Metadata (Alpha Vantage + yfinance fallback)
# -----------------------------
def fetch_company_metadata(ticker_sym: str) -> dict:
    meta = {"summary": None, "sector": None, "employees": None, "website": None, "ratios": {}}

    # try Alpha Vantage (if key present)
    av_key = None
    try:
        av_key = st.secrets.get("ALPHAVANTAGE_API_KEY", None)
    except Exception:
        av_key = os.environ.get("ALPHAVANTAGE_API_KEY")

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
            # ignore and fallback
            pass

    # yfinance fallback for basic metadata
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


# -----------------------------
# Render page
# -----------------------------
st.header(f"{ticker} — Overview")

meta = fetch_company_metadata(ticker)

if meta["summary"]:
    st.subheader("Company Summary")
    st.write(meta["summary"])
else:
    st.info("Company summary not available from APIs. (Use Alpha Vantage in secrets or upload a CSV for metadata.)")

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

# Ratios (may be empty)
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

# Fetch price data (uploaded CSV preferred)
uploaded_bytes = uploaded_csv.read() if uploaded_csv else None
data = fetch_history(ticker, start_date.isoformat(), end_date.isoformat(), uploaded_bytes)

if data.empty:
    # helpful diagnostics/hints
    msg = "No historical price data available. Check ticker/network, upload CSV, or set TWELVEDATA_API_KEY or ALPHAVANTAGE_API_KEY in Streamlit secrets / environment."
    st.error(msg)
    if show_diagnostics:
        st.info("Diagnostics tips:")
        st.write("- Is TWELVEDATA_API_KEY present in Streamlit secrets (.streamlit/secrets.toml) or environment variable?")
        st.write("- yfinance can sometimes be blocked by hosting; try uploading a CSV.")
        st.write("- For Twelve Data: key and allowed symbols (example: AAPL).")
else:
    # metrics & head/tail
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

    # Chart controls
    btn_cols = st.columns([1] * 7)
    num_period = ""
    labels = ["5D", "1M", "6M", "YTD", "1Y", "5Y", "MAX"]
    vals = ["5d", "1mo", "6mo", "ytd", "1y", "5y", "max"]
    for c, lab, v in zip(btn_cols, labels, vals):
        with c:
            if st.button(lab):
                num_period = v

    c1, c2 = st.columns([1, 3])
    with c1:
        chart_type = st.selectbox("Chart type", ("Candle", "Line"))
    with c2:
        if chart_type == "Candle":
            indicator = st.selectbox("Indicator", ("RSI", "MACD"))
        else:
            indicator = st.selectbox("Indicator", ("RSI", "Moving Average", "MACD"))

    # Choose which DF to plot: if num_period requested, attempt using built-in 'history' from yfinance (if available),
    # otherwise plot 'data' which corresponds to selected date range / uploaded CSV / Twelve Data.
    plot_src = data

    def safe_plot(func, df_in, period_arg="1y"):
        try:
            fig = func(df_in, period_arg)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotting failed: {e}")

    if chart_type == "Candle":
        safe_plot(candlestick, plot_src, num_period or "1y")
        if indicator == "RSI":
            safe_plot(RSI, plot_src, num_period or "1y")
        elif indicator == "MACD":
            safe_plot(MACD, plot_src, num_period or "1y")
    else:
        safe_plot(close_chart, plot_src, num_period or "1y")
        if indicator == "RSI":
            safe_plot(RSI, plot_src, num_period or "1y")
        elif indicator == "Moving Average":
            safe_plot(Moving_average, plot_src, num_period or "1y")
        elif indicator == "MACD":
            safe_plot(MACD, plot_src, num_period or "1y")

st.caption(
    "Notes: Historical prices prefer uploaded CSV -> yfinance -> Twelve Data (if TWELVEDATA_API_KEY present). "
    "Set keys in Streamlit secrets or as environment variables."
)
