# pages/CAPM_Beta.py
import datetime
import math

import numpy as np
import pandas as pd
import streamlit as st

# yfinance fallback
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    yf = None
    HAVE_YF = False

from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="CAPM Beta", page_icon="📈", layout="wide")
st.title("CAPM — Single-stock Beta & Expected Return")

st.markdown(
    """
This page computes a stock's **beta** by regressing its returns against a market index's returns.

Implementation notes:
- The app **does not** import `pandas_datareader` at module load time (that package may fail to import on Python >=3.12).
- If `pandas_datareader` can be imported successfully *at runtime*, it will be used; otherwise the app falls back to `yfinance`.
"""
)

# --- Inputs
col1, col2 = st.columns([1.5, 1])
with col1:
    ticker = st.text_input("Stock ticker", value="AAPL").strip().upper()
with col2:
    index_symbol = st.text_input("Market index symbol", value="^GSPC").strip().upper()

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

# --- Data fetch helper (lazy import of pandas_datareader) ---
def fetch_prices(sym, start, end):
    """
    Robust fetch that tries pandas_datareader (if importable) then yfinance.
    On any failure it returns an empty DataFrame *and* shows the exception details in Streamlit.
    """
    import importlib
    import traceback

    # Try pandas_datareader safely
    try:
        pdr_spec = importlib.util.find_spec("pandas_datareader")
        if pdr_spec is not None:
            try:
                from pandas_datareader import data as web
                df = web.DataReader(sym, "yahoo", start, end)
                if "Adj Close" in df.columns:
                    df = df[["Adj Close"]].rename(columns={"Adj Close": "Close"})
                else:
                    df = df[["Close"]]
                if df.empty:
                    st.warning(f"pandas_datareader returned empty DataFrame for {sym}")
                return df
            except Exception as e:
                st.warning(f"pandas_datareader attempt for {sym} failed: {e}")
                st.text(traceback.format_exc())
    except Exception:
        # any unexpected issue when probing pdr -> continue
        pass

    # Try yfinance and show errors if any
    if HAVE_YF:
        try:
            df = yf.download(sym, start=start, end=end, progress=False)
            if df is None or df.empty:
                st.warning(f"yfinance returned no data for {sym} (empty dataframe).")
                return pd.DataFrame()
            if "Adj Close" in df.columns:
                df = df[["Adj Close"]].rename(columns={"Adj Close": "Close"})
            else:
                df = df[["Close"]]
            return df
        except Exception as e:
            st.error(f"yfinance.download for {sym} raised an exception: {e}")
            st.text(traceback.format_exc())
            return pd.DataFrame()

    # Nothing available
    st.error("No data backend available (pandas_datareader nor yfinance).")
    return pd.DataFrame()
