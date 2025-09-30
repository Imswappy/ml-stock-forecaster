# pages/CAPM_Return.py
import datetime
import io
import os
import traceback

import numpy as np
import pandas as pd
import streamlit as st

# use yfinance when available; fallback behavior if not installed
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    yf = None
    HAVE_YF = False

from pages.utils import capm_functions

st.set_page_config(
    page_title="Calculate CAPM Return",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

st.title("Capital Asset Pricing Model — Multi-stock Return 📈")
st.markdown(
    """
Select stocks and a history window; the app computes beta for each stock vs the market and
estimates expected return using CAPM: \\(E[R_i] = R_f + \\beta_i (E[R_m] - R_f)\\).

Notes:
- By default the market proxy used is **SPY** (ETF) for robust availability; you may change it below.
- If network fetch fails you can upload CSVs (Date + Close/Adj Close) for stocks and/or market.
"""
)

# ----------------------------
# Inputs
# ----------------------------
col1, col2 = st.columns([1.5, 1])
with col1:
    stocks_list = st.multiselect(
        "Choose stocks (pick up to 8)", 
        options=['TSLA', 'AAPL','NFLX','MGM','MSFT','AMZN','NVDA','GOOGL'],
        default=['TSLA', 'AAPL','MSFT','NFLX'],
        help="Choose one or more stocks to compute CAPM returns for."
    )
with col2:
    years = st.number_input("Number of Years", min_value=1, max_value=10, value=3, help="Historical window length (years)")

market_col1, market_col2 = st.columns([1, 1])
with market_col1:
    market_symbol = st.text_input("Market proxy symbol (ETF or Index)", value="SPY", help="Use SPY (ETF) by default. If you want to use S&P500 index ticker (e.g. ^GSPC) note some hosts may block index scraping.")
with market_col2:
    rf_pct = st.number_input("Risk-free rate (annual %, default 0)", value=0.0, step=0.01)
    Rf = float(rf_pct) / 100.0

st.markdown("**Optional:** upload CSVs (Date + Close/Adj Close) to avoid network fetch problems.")
upload_col1, upload_col2 = st.columns(2)
with upload_col1:
    uploaded_stocks_zip = st.file_uploader("Upload a single CSV with all stocks (Date + Close columns named by ticker) OR individual stock CSVs zipped (not required)", type=["csv"], help="If you upload a single CSV it should contain Date and one column per ticker named exactly as the tickers selected.")
with upload_col2:
    uploaded_market_csv = st.file_uploader("Upload market CSV (Date + Close/Adj Close)", type=["csv"])

# Diagnostics toggle
show_diagnostics = st.checkbox("Show diagnostics (detailed fetch errors)", value=False)

# ----------------------------
# Helpers
# ----------------------------
def parse_price_csv_bytes(bytes_io):
    """
    Parse CSV bytes, try to detect date column and close/adj close.
    Return DataFrame with index as datetime and single 'Close' column (or empty DF on failure).
    """
    try:
        df = pd.read_csv(io.BytesIO(bytes_io))
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(bytes_io.decode()))
        except Exception:
            return pd.DataFrame()
    # find date-like column
    date_col = None
    for c in df.columns:
        if c.lower() in ("date", "timestamp", "time"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    except Exception:
        return pd.DataFrame()
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
        return pd.DataFrame()
    return df[[close_col]].rename(columns={close_col: "Close"})

def fetch_prices_yf(sym, years):
    """Fetch using yfinance (simple). Returns df or empty df and error message."""
    if not HAVE_YF:
        return pd.DataFrame(), "yfinance not installed"
    try:
        df = yf.download(sym, period=f"{years}y", progress=False)
        if df is None or df.empty:
            return pd.DataFrame(), "yfinance returned empty data"
        # prefer Adj Close
        if "Adj Close" in df.columns:
            out = df[["Adj Close"]].rename(columns={"Adj Close":"Close"})
        else:
            out = df[["Close"]]
        return out, None
    except Exception as e:
        return pd.DataFrame(), f"yfinance error: {e}"

# ----------------------------
# Validation & Compute
# ----------------------------
if st.button("Compute CAPM returns"):
    # basic validation
    if not stocks_list:
        st.error("Please select at least one stock to analyze.")
        st.stop()
    if years < 1:
        st.error("Please choose at least 1 year of history.")
        st.stop()

    start_date = (datetime.date.today() - datetime.timedelta(days=365*years)).isoformat()
    end_date = datetime.date.today().isoformat()

    # 1) Try to construct stocks_df from uploaded CSV (if provided)
    stocks_df = pd.DataFrame()
    source_stocks = None
    if uploaded_stocks_zip is not None:
        # If user uploaded a CSV with Date + columns for tickers, try to use it
        try:
            uploaded_bytes = uploaded_stocks_zip.read()
            candidate = parse_price_csv_bytes(uploaded_bytes)
            if not candidate.empty:
                # If file contains multiple columns (named tickers), use them
                # candidate has single Close column — so assume this was individual file -> skip
                # Instead try reading the CSV again and keep all columns except Date
                df_all = pd.read_csv(io.BytesIO(uploaded_bytes))
                date_col = next((c for c in df_all.columns if c.lower() in ("date","timestamp","time")), df_all.columns[0])
                df_all[date_col] = pd.to_datetime(df_all[date_col], errors='coerce')
                df_all = df_all.set_index(date_col).sort_index()
                # ensure we have requested tickers as columns
                missing = [t for t in stocks_list if t not in df_all.columns]
                if not missing:
                    stocks_df = df_all[stocks_list].rename(columns=lambda c: c)
                    stocks_df.index = pd.to_datetime(stocks_df.index)
                    source_stocks = "uploaded_csv_multi"
                else:
                    # fallback: maybe it's a single-ticker CSV (handled below)
                    pass
        except Exception:
            if show_diagnostics:
                st.text("Uploaded stocks CSV parse traceback:")
                st.text(traceback.format_exc())

    # 2) If not constructed, fetch each stock from yfinance
    if stocks_df.empty:
        tmp = {}
        any_fail = False
        for t in stocks_list:
            df_t, err = fetch_prices_yf(t, years)
            if df_t is None or df_t.empty:
                any_fail = True
                if show_diagnostics:
                    st.warning(f"Failed to fetch {t} via yfinance: {err}")
            else:
                tmp[t] = df_t["Close"]
        if tmp:
            stocks_df = pd.DataFrame(tmp).sort_index()
            source_stocks = "yfinance"
        else:
            st.error("Failed to fetch any stock prices via yfinance. Try uploading CSVs or check network.")
            st.stop()

    # 3) Market prices: prefer uploaded market CSV, else fetch market_symbol via yfinance
    market_df = pd.DataFrame()
    source_market = None
    if uploaded_market_csv is not None:
        try:
            uploaded_bytes = uploaded_market_csv.read()
            mdf = parse_price_csv_bytes(uploaded_bytes)
            if not mdf.empty:
                market_df = mdf.rename(columns={"Close":"market_close"})
                source_market = "uploaded_market_csv"
        except Exception:
            if show_diagnostics:
                st.text("Uploaded market CSV parse traceback:")
                st.text(traceback.format_exc())

    if market_df.empty:
        mdf, merr = fetch_prices_yf(market_symbol, years)
        if mdf is None or mdf.empty:
            # try SPY fallback if requested market failed
            if market_symbol.upper() not in ("SPY",):
                if show_diagnostics:
                    st.warning(f"Market fetch for {market_symbol} failed ({merr}). Trying SPY as fallback.")
                mdf2, merr2 = fetch_prices_yf("SPY", years)
                if mdf2 is not None and not mdf2.empty:
                    market_df = mdf2.rename(columns={"Close":"market_close"})
                    source_market = "yfinance:SPY"
                    market_symbol = "SPY"
                else:
                    st.error("Failed to fetch market data (both requested market and SPY). Upload CSV or check network.")
                    st.stop()
            else:
                st.error("Failed to fetch market data for SPY. Upload CSV or check network.")
                st.stop()
        else:
            market_df = mdf.rename(columns={"Close":"market_close"})
            source_market = f"yfinance:{market_symbol}"

    # Merge stocks and market on dates
    # ensure datetime index and alignment
    stocks_df.index = pd.to_datetime(stocks_df.index)
    market_df.index = pd.to_datetime(market_df.index)
    combined = stocks_df.join(market_df, how="inner")
    if combined.empty:
        st.error("After aligning dates, no overlapping data remains between stocks and market. Try expanding years or upload CSVs with matching date ranges.")
        st.stop()

    # show dataframes head/tail
    colh1, colh2 = st.columns([1,1])
    with colh1:
        st.markdown("### Sample data (head)")
        st.dataframe(combined.head().round(4))
    with colh2:
        st.markdown("### Sample data (tail)")
        st.dataframe(combined.tail().round(4))

    # compute daily returns (percentage) similar to your capm_functions.daily_return
    stocks_daily_return = combined.copy()
    # We'll compute pct change *as fractions* (not percent), to match math in capm_functions (which uses percent),
    # but we can adapt: capm_functions expects percent values (it multiplies by 100). To reuse existing functions,
    # create a DataFrame with percent returns.
    ret_df = stocks_daily_return.copy()
    for col in ret_df.columns:
        ret_df[col] = ret_df[col].pct_change().fillna(0) * 100.0  # percent

    # rename market column to 'sp500' to reuse capm_functions.calculate_beta and later code
    # choose 'sp500' label because capm_functions expects that
    ret_df = ret_df.rename(columns={"market_close": "sp500"})

    # compute beta & alpha for all selected stocks
    beta = {}
    alpha = {}
    for col in stocks_list:
        if col in ret_df.columns:
            try:
                b, a = capm_functions.calculate_beta(ret_df.reset_index(drop=True), col)
                beta[col] = b
                alpha[col] = a
            except Exception:
                # fallback: compute linear regression slope directly (robust)
                try:
                    x = ret_df["sp500"].values
                    y = ret_df[col].values
                    slope, inter = np.polyfit(x, y, 1)
                    beta[col] = slope
                    alpha[col] = inter
                except Exception:
                    beta[col] = float("nan")
                    alpha[col] = float("nan")

    # Present results
    beta_df = pd.DataFrame({
        "Stock": list(beta.keys()),
        "Beta Value": [round(v, 4) if not np.isnan(v) else None for v in beta.values()]
    })

    st.markdown("### Calculated Beta values")
    st.dataframe(beta_df, use_container_width=True)

    # compute expected returns via CAPM (annualized)
    # first compute market mean return (note: ret_df 'sp500' is percent per period)
    if freq := "daily":
        periods_per_year = 252.0
    rm = (ret_df["sp500"].mean() / 100.0) * periods_per_year  # convert percent -> fraction then annualize

    return_rows = []
    for stock_name, b in beta.items():
        if b is None or (isinstance(b, float) and np.isnan(b)):
            expected = None
        else:
            expected = Rf + b * (rm - Rf)
        return_rows.append({"Stock": stock_name, "CAPM Return (annual %)": round(expected * 100.0, 3) if expected is not None else None})

    return_df = pd.DataFrame(return_rows)
    st.markdown("### CAPM Expected Returns (annualized)")
    st.dataframe(return_df, use_container_width=True)

    # show a simple price chart and normalized chart using your util
    col_chart1, col_chart2 = st.columns([1, 1])
    with col_chart1:
        st.markdown("### Price of selected stocks")
        st.plotly_chart(capm_functions.interactive_plot(combined.reset_index().rename(columns={combined.reset_index().columns[0]:"Date"})), use_container_width=True)
    with col_chart2:
        st.markdown("### Normalized prices (start = 1)")
        try:
            normalized = capm_functions.normalize(combined.reset_index().rename(columns={combined.reset_index().columns[0]:"Date"}))
            st.plotly_chart(capm_functions.interactive_plot(normalized), use_container_width=True)
        except Exception:
            if show_diagnostics:
                st.text("Normalization plotting traceback:")
                st.text(traceback.format_exc())

    # final diagnostics / info
    if show_diagnostics:
        st.write(f"Data sources used: stocks ← {source_stocks}, market ← {source_market}")
    else:
        st.info(f"Data sources used: stocks ← {source_stocks}, market ← {source_market}")

