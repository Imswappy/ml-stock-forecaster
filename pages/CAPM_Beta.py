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

# --- Compute Beta on button press ---
if st.button("Compute Beta"):
    if ticker == "" or index_symbol == "":
        st.error("Please enter both a stock ticker and an index symbol.")
    else:
        with st.spinner("Fetching data..."):
            stock_df = fetch_prices(ticker, start, end)
            idx_df = fetch_prices(index_symbol, start, end)

        if stock_df.empty:
            st.error(f"Failed to fetch price data for {ticker}. Check symbol or network.")
            st.stop()
        if idx_df.empty:
            st.error(f"Failed to fetch price data for market index {index_symbol}. Check symbol or network.")
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

# If neither backend available
if not HAVE_YF:
    st.error(
        "No price-fetch backend available. Install `yfinance` in your environment.\n\n"
        "Quick fix (shell):\n"
        "  pip install yfinance\n\n"
        "Then restart the Streamlit app."
    )
