# pages/CAPM_Beta.py
import importlib
import io
import math
import os
import traceback

import numpy as np
import pandas as pd
import streamlit as st

# yfinance fallback (import here but safe if missing)
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    yf = None
    HAVE_YF = False

from sklearn.linear_model import LinearRegression

# Streamlit page config (first Streamlit call)
st.set_page_config(page_title="CAPM Beta", page_icon="📈", layout="wide")
st.title("CAPM — Single-stock Beta & Expected Return")

st.markdown(
    """
This page computes a stock's **beta** by regressing its returns against a market index's returns.

Features:
- Tries multiple data sources (yfinance, pandas_datareader→stooq, Alpha Vantage) in order.
- Allows uploading CSV files for stock/index as a fallback.
- Uses `st.secrets['ALPHAVANTAGE_API_KEY']` or environment variable `ALPHAVANTAGE_API_KEY` if present.
"""
)

# --- Inputs & UI ---
col1, col2 = st.columns([1.5, 1])
with col1:
    ticker = st.text_input("Stock ticker", value="AAPL").strip().upper()
with col2:
    index_symbol = st.text_input("Market index symbol", value="^GSPC").strip().upper()

# File upload fallback
st.markdown("**Optional:** Upload CSV files if network/data APIs are unavailable.")
upload_col1, upload_col2 = st.columns(2)
with upload_col1:
    stock_csv = st.file_uploader("Upload stock CSV (columns: Date, Close or Adj Close)", type=["csv"], key="stock_csv")
with upload_col2:
    index_csv = st.file_uploader("Upload index CSV (columns: Date, Close or Adj Close)", type=["csv"], key="index_csv")

period_col1, period_col2 = st.columns(2)
with period_col1:
    years = st.slider("Years of history", 1, 10, 3)
with period_col2:
    freq = st.selectbox("Return frequency", options=["daily", "weekly", "monthly"], index=0)

rf_col1, rf_col2 = st.columns([1, 1])
with rf_col1:
    rf_input = st.number_input("Risk-free rate (annual %, default 0)", value=0.0, step=0.01)
    Rf = float(rf_input) / 100.0
with rf_col2:
    show_plot = st.checkbox("Show regression plot", value=True)

start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
end = pd.Timestamp.today().strftime("%Y-%m-%d")
st.markdown(f"**Data range:** {start} → {end}")

# Helper: parse uploaded CSV to DataFrame with 'Close' column and datetime index
def read_price_csv(uploaded_file):
    try:
        content = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(content))
        # standardize column names
        cols = [c.lower() for c in df.columns]
        date_col = None
        for c in df.columns:
            if c.lower() in ("date", "timestamp"):
                date_col = c
                break
        if date_col is None:
            # try first column as date
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        # find close column
        close_col = None
        for c in df.columns:
            if c.lower() in ("adj close", "adjusted_close", "adjusted close", "adjusted_close"):
                close_col = c
                break
        if close_col is None:
            for c in df.columns:
                if c.lower() == "close":
                    close_col = c
                    break
        if close_col is None:
            st.warning("Uploaded CSV did not contain a recognizable 'Close' or 'Adj Close' column.")
            return pd.DataFrame()
        out = df[[close_col]].rename(columns={close_col: "Close"})
        return out
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")
        st.text(traceback.format_exc())
        return pd.DataFrame()

# Multi-source fetcher: yfinance, pandas_datareader->stooq, Alpha Vantage (if key), plus CSV fallback
def fetch_prices(sym, start, end, verbose=True):
    """
    Try multiple data sources to fetch daily Close prices for `sym` between start/end.
    Order:
      1) Uploaded CSV (if provided by caller)
      2) yfinance
      3) pandas_datareader -> stooq
      4) Alpha Vantage (if API key in st.secrets or env)
    Returns DataFrame with 'Close' column or empty DataFrame.
    """
    errors = []

    # ---------- 1) yfinance ----------
    try:
        if HAVE_YF:
            df = yf.download(sym, start=start, end=end, progress=False)
            if df is not None and not df.empty:
                if "Adj Close" in df.columns:
                    out = df[["Adj Close"]].rename(columns={"Adj Close": "Close"})
                else:
                    out = df[["Close"]]
                if verbose:
                    st.success(f"Fetched {len(out)} rows for {sym} from yfinance")
                return out
            else:
                errors.append(("yfinance", "empty or no data returned"))
        else:
            errors.append(("yfinance", "not installed"))
    except Exception as e:
        errors.append(("yfinance", str(e)))
        if verbose:
            st.warning(f"yfinance attempt for {sym} failed: {e}")
            st.text(traceback.format_exc())

    # ---------- 2) pandas_datareader -> stooq ----------
    try:
        pdr_spec = importlib.util.find_spec("pandas_datareader")
        if pdr_spec is not None:
            try:
                from pandas_datareader import data as pdr
                df = pdr.DataReader(sym, "stooq", start, end)
                if df is not None and not df.empty:
                    df = df.sort_index()
                    # find close column
                    if "Close" in df.columns:
                        out = df[["Close"]]
                    else:
                        col = [c for c in df.columns if c.lower() == "close"]
                        if col:
                            out = df[[col[0]]].rename(columns={col[0]: "Close"})
                        else:
                            out = pd.DataFrame()
                    if not out.empty:
                        if verbose:
                            st.success(f"Fetched {len(out)} rows for {sym} from stooq (pandas_datareader)")
                        return out
                    else:
                        errors.append(("pandas_datareader-stooq", "no Close column"))
                else:
                    errors.append(("pandas_datareader-stooq", "empty or no data"))
            except Exception as e:
                errors.append(("pandas_datareader-stooq", str(e)))
                if verbose:
                    st.warning(f"pandas_datareader (stooq) attempt failed for {sym}: {e}")
                    st.text(traceback.format_exc())
        else:
            errors.append(("pandas_datareader", "not installed"))
    except Exception as e:
        errors.append(("pandas_datareader_probe", str(e)))
        if verbose:
            st.warning(f"pandas_datareader probe failed: {e}")
            st.text(traceback.format_exc())

    # ---------- 3) Alpha Vantage ----------
    av_key = None
    try:
        # prefer Streamlit secrets, then env var
        av_key = st.secrets.get("ALPHAVANTAGE_API_KEY") if hasattr(st, "secrets") else None
    except Exception:
        av_key = None
    if not av_key:
        av_key = os.environ.get("ALPHAVANTAGE_API_KEY")

    if av_key:
        try:
            import requests
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": sym,
                "outputsize": "full",
                "apikey": av_key,
                "datatype": "csv",
            }
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200 and r.text:
                from io import StringIO
                df = pd.read_csv(StringIO(r.text), parse_dates=["timestamp"], index_col="timestamp")
                # prefer adjusted_close
                if "adjusted_close" in df.columns:
                    out = df[["adjusted_close"]].rename(columns={"adjusted_close": "Close"})
                elif "close" in df.columns:
                    out = df[["close"]].rename(columns={"close": "Close"})
                else:
                    out = pd.DataFrame()
                if not out.empty:
                    out = out.sort_index()
                    out = out.loc[(out.index >= pd.to_datetime(start)) & (out.index <= pd.to_datetime(end))]
                    if not out.empty:
                        if verbose:
                            st.success(f"Fetched {len(out)} rows for {sym} from Alpha Vantage")
                        return out
                errors.append(("alphavantage", "no usable data in CSV"))
            else:
                errors.append(("alphavantage", f"HTTP {r.status_code}"))
        except Exception as e:
            errors.append(("alphavantage", str(e)))
            if verbose:
                st.warning(f"Alpha Vantage attempt failed: {e}")
                st.text(traceback.format_exc())
    else:
        errors.append(("alphavantage", "no API key provided"))

    # ---------- Nothing worked ----------
    if verbose:
        st.error(f"All data sources failed for {sym}. See attempted sources and errors below:")
        for src, msg in errors:
            st.write(f"- {src}: {msg}")
        st.info(
            "Workarounds: upload CSVs, use SPY as market proxy, set ALPHAVANTAGE_API_KEY in Streamlit secrets or environment, or run locally with network access."
        )
    return pd.DataFrame()

# --- Compute Beta on button press ---
if st.button("Compute Beta"):
    # If CSVs provided, prefer them for that symbol
    if stock_csv is not None:
        stock_df = read_price_csv(stock_csv)
        if stock_df.empty:
            st.warning("Uploaded stock CSV couldn't be used; attempting remote fetch.")
            stock_df = fetch_prices(ticker, start, end)
    else:
        stock_df = fetch_prices(ticker, start, end)

    if index_csv is not None:
        idx_df = read_price_csv(index_csv)
        if idx_df.empty:
            st.warning("Uploaded index CSV couldn't be used; attempting remote fetch.")
            idx_df = fetch_prices(index_symbol, start, end)
    else:
        idx_df = fetch_prices(index_symbol, start, end)

    # If market index failed and symbol looks like ^GSPC, try SPY as fallback
    if (idx_df is None or idx_df.empty) and index_symbol.upper() in ("^GSPC", "GSPC"):
        st.warning("Primary index (^GSPC) failed to fetch — trying SPY (ETF) as market proxy.")
        idx_df = fetch_prices("SPY", start, end)
        if not idx_df.empty:
            st.success("Fetched SPY as market proxy.")

    if stock_df is None or stock_df.empty:
        st.error(f"Failed to fetch price data for {ticker}. Check symbol, CSV, or network.")
        st.stop()
    if idx_df is None or idx_df.empty:
        st.error(f"Failed to fetch price data for market index {index_symbol}. Check symbol, CSV, or network.")
        st.stop()

    # align on dates (inner join)
    df = stock_df.join(idx_df, how="inner", lsuffix="_stock", rsuffix="_idx")
    df.columns = ["Close_stock", "Close_idx"]

    if df.empty or len(df) < 10:
        st.error("Not enough overlapping data between stock and index to compute Beta (need more data).")
        st.stop()

    # compute returns according to frequency
    if freq == "daily":
        ret_stock = df["Close_stock"].pct_change().dropna()
        ret_idx = df["Close_idx"].pct_change().dropna()
    elif freq == "weekly":
        ret_stock = df["Close_stock"].pct_change(periods=5).dropna()
        ret_idx = df["Close_idx"].pct_change(periods=5).dropna()
    else:  # monthly
        ret_stock = df["Close_stock"].pct_change(periods=21).dropna()
        ret_idx = df["Close_idx"].pct_change(periods=21).dropna()

    # align returns
    ret_df = pd.concat([ret_stock, ret_idx], axis=1).dropna()
    ret_df.columns = ["R_i", "R_m"]

    if ret_df.empty:
        st.error("No overlapping returns after resampling. Try a different frequency or longer history.")
        st.stop()

    # convert to excess returns using Rf (annual -> period)
    if freq == "daily":
        periods_per_year = 252.0
    elif freq == "weekly":
        periods_per_year = 52.0
    else:
        periods_per_year = 12.0

    period_rf = (1 + Rf) ** (1.0 / periods_per_year) - 1.0
    ret_df["Ri_ex"] = ret_df["R_i"] - period_rf
    ret_df["Rm_ex"] = ret_df["R_m"] - period_rf

    # Linear regression for beta
    X = ret_df["Rm_ex"].values.reshape(-1, 1)
    y = ret_df["Ri_ex"].values
    lr = LinearRegression()
    lr.fit(X, y)
    beta = float(lr.coef_[0])
    alpha = float(lr.intercept_)

    # alternative slope via polyfit
    slope, intercept = np.polyfit(ret_df["Rm_ex"].values, ret_df["Ri_ex"].values, 1)

    st.metric("Estimated Beta (LinearRegression slope)", f"{beta:.4f}")
    st.write(f"Intercept (alpha): {alpha:.6f}")
    st.write(f"Slope (polyfit): {slope:.6f} — should be similar to beta above")

    # annualize beta-based expected return via CAPM
    mean_rm = ret_df["R_m"].mean() * periods_per_year
    expected_return = Rf + beta * (mean_rm - Rf)
    st.write("Estimated market mean return (annualized):", f"{mean_rm:.3%}")
    st.write("CAPM expected return (annualized):", f"{expected_return:.3%}")

    # Interpretation
    st.markdown("**Beta interpretation:**")
    if beta > 1.0:
        st.info("β > 1 : stock is more volatile than the market.")
    elif beta < 1.0:
        st.info("β < 1 : stock is less volatile than the market.")
    else:
        st.info("β = 1 : stock moves in line with the market.")

    # Plot scatter + regression line
    if show_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(ret_df["Rm_ex"], ret_df["Ri_ex"], alpha=0.6, label="observations")
        xs = np.linspace(ret_df["Rm_ex"].min(), ret_df["Rm_ex"].max(), 100)
        ys = slope * xs + intercept
        ax.plot(xs, ys, color="red", label=f"fit: slope={slope:.3f}")
        ax.set_xlabel("Market excess return")
        ax.set_ylabel("Stock excess return")
        ax.set_title(f"{ticker} vs {index_symbol} — Beta regression")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Sample of returns used")
    st.dataframe(
        ret_df.tail(50)
        .assign(Ri=lambda d: d["R_i"].round(5), Rm=lambda d: d["R_m"].round(5))[["Ri", "Rm"]]
    )
