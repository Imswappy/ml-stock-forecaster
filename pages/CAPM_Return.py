# pages/CAPM_Return.py
import datetime
import importlib
import io
import os
import traceback

import numpy as np
import pandas as pd
import streamlit as st

# try yfinance
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    yf = None
    HAVE_YF = False

from pages.utils import capm_functions

st.set_page_config(page_title="CAPM Return (multi-stock)", page_icon="📊", layout="wide")
st.title("CAPM Return — Multi-stock CAPM")

st.markdown(
    """
Compute Beta for multiple stocks vs a market proxy and estimate expected returns by CAPM.

**Behavior**
- Data sources tried (per symbol): uploaded CSV → yfinance → pandas_datareader (stooq) → Alpha Vantage (if key present).
- Set `ALPHAVANTAGE_API_KEY` in Streamlit secrets or as an environment variable to enable AlphaVantage fallback.
- Use "Show diagnostics" to reveal detailed fetch errors/traces.
"""
)

# -------------------------
# UI inputs
# -------------------------
col1, col2 = st.columns([2, 1])
with col1:
    stocks_list = st.multiselect(
        "Choose stocks (1–8)", 
        options=['TSLA','AAPL','MSFT','NFLX','AMZN','NVDA','GOOGL','SPY'],
        default=['TSLA','AAPL','MSFT','NFLX'],
    )
with col2:
    years = st.number_input("Years of history", min_value=1, max_value=10, value=3)

market_col1, market_col2 = st.columns([1, 1])
with market_col1:
    market_symbol = st.text_input("Market proxy symbol", value="SPY")
with market_col2:
    rf_pct = st.number_input("Risk-free rate (annual %)", value=0.0, step=0.01)
    Rf = float(rf_pct) / 100.0

st.markdown("Optional: upload CSVs (Date + Close or Adj Close). You can upload separate CSV per stock or one market CSV.")
upload_stock_files = st.file_uploader("Upload individual stock CSVs (multiple allowed)", type="csv", accept_multiple_files=True, help="Each CSV should have Date and Close/Adj Close column. Name does not need to match ticker; uploaded files are matched by asking the user below.")
uploaded_market_csv = st.file_uploader("Upload market CSV (single)", type="csv")

show_diagnostics = st.checkbox("Show diagnostics (detailed errors)", value=False)

# helper: parse uploaded CSV (single file)
def parse_uploaded_csv_bytes(b):
    try:
        df = pd.read_csv(io.BytesIO(b))
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(b.decode()))
        except Exception:
            return pd.DataFrame()
    # find date col
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
    # find close col
    close_col = None
    for c in df.columns:
        if c.lower() in ("adj close","adjusted_close","adjusted close","adjusted_close"):
            close_col = c
            break
    if close_col is None:
        for c in df.columns:
            if c.lower() == "close":
                close_col = c
                break
    if close_col is None:
        return pd.DataFrame()
    out = df[[close_col]].rename(columns={close_col: "Close"})
    return out

# multi-source fetch function (returns df, source_name)
def fetch_symbol(sym, start, end, uploaded_files_map=None):
    """
    Attempt fetching symbol in order:
    - uploaded_files_map: dict mapping symbol->file bytes (prefer these)
    - yfinance
    - pandas_datareader->stooq
    - AlphaVantage (if API key available)
    Returns (df, source_name) where df has 'Close' column.
    """
    errors = []
    # 0) uploaded
    if uploaded_files_map and sym in uploaded_files_map:
        try:
            df = parse_uploaded_csv_bytes(uploaded_files_map[sym])
            if not df.empty:
                return df, "uploaded_csv"
        except Exception as e:
            errors.append(("uploaded", str(e)))
            if show_diagnostics:
                st.text(traceback.format_exc())

    # 1) yfinance
    try:
        if HAVE_YF:
            df = yf.download(sym, start=start, end=end, progress=False)
            if df is not None and not df.empty:
                if "Adj Close" in df.columns:
                    out = df[["Adj Close"]].rename(columns={"Adj Close":"Close"})
                else:
                    out = df[["Close"]]
                return out, "yfinance"
            else:
                errors.append(("yfinance", "empty"))
        else:
            errors.append(("yfinance", "not installed"))
    except Exception as e:
        errors.append(("yfinance", str(e)))
        if show_diagnostics:
            st.write(f"yfinance error for {sym}:")
            st.text(traceback.format_exc())

    # 2) pandas_datareader -> stooq
    try:
        pdr_spec = importlib.util.find_spec("pandas_datareader")
        if pdr_spec is not None:
            try:
                from pandas_datareader import data as pdr
                df = pdr.DataReader(sym, "stooq", start, end)
                if df is not None and not df.empty:
                    df = df.sort_index()
                    if "Close" in df.columns:
                        out = df[["Close"]]
                    else:
                        col = [c for c in df.columns if c.lower()=="close"]
                        if col:
                            out = df[[col[0]]].rename(columns={col[0]:"Close"})
                        else:
                            out = pd.DataFrame()
                    if not out.empty:
                        return out, "pandas_datareader(stooq)"
                    else:
                        errors.append(("pdr-stooq", "no Close col"))
                else:
                    errors.append(("pdr-stooq", "empty"))
            except Exception as e:
                errors.append(("pdr-stooq", str(e)))
                if show_diagnostics:
                    st.write(f"pandas_datareader(stooq) error for {sym}:")
                    st.text(traceback.format_exc())
        else:
            errors.append(("pandas_datareader", "not installed"))
    except Exception as e:
        errors.append(("pdr-probe", str(e)))
        if show_diagnostics:
            st.text(traceback.format_exc())

    # 3) Alpha Vantage (CSV mode) if key present in secrets or env
    av_key = None
    try:
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
                "function":"TIME_SERIES_DAILY_ADJUSTED",
                "symbol": sym,
                "outputsize":"full",
                "apikey": av_key,
                "datatype":"csv"
            }
            r = requests.get(url, params=params, timeout=15)
            if r.status_code==200 and r.text:
                # robust parse
                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO(r.text))
                except Exception:
                    # fallback: try generic parse
                    df = pd.read_csv(io.StringIO(r.text))
                if df.empty:
                    errors.append(("alphavantage","empty CSV"))
                else:
                    # find date col and close/adjusted_close
                    # prefer adjusted_close then close
                    date_col = None
                    for c in df.columns:
                        if c.lower() in ("timestamp","date","time"):
                            date_col = c
                            break
                    if date_col is None:
                        date_col = df.columns[0]
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                    df = df.set_index(date_col).sort_index()
                    if "adjusted_close" in [c.lower() for c in df.columns]:
                        col = next(c for c in df.columns if c.lower()=="adjusted_close")
                        out = df[[col]].rename(columns={col:"Close"})
                        return out, "alphavantage"
                    elif "close" in [c.lower() for c in df.columns]:
                        col = next(c for c in df.columns if c.lower()=="close")
                        out = df[[col]].rename(columns={col:"Close"})
                        return out, "alphavantage"
                    else:
                        errors.append(("alphavantage","no close col"))
            else:
                errors.append(("alphavantage", f"HTTP {r.status_code}"))
        except Exception as e:
            errors.append(("alphavantage", str(e)))
            if show_diagnostics:
                st.write(f"AlphaVantage error for {sym}:")
                st.text(traceback.format_exc())
    else:
        errors.append(("alphavantage","no API key"))

    # All failed
    if show_diagnostics:
        st.error(f"All sources failed for {sym}. Details:")
        for s, m in errors:
            st.write(f"- {s}: {m}")
    return pd.DataFrame(), None

# -------------------------
# Main compute flow
# -------------------------
if st.button("Compute CAPM"):
    if not stocks_list:
        st.error("Please select at least one stock.")
        st.stop()
    if years < 1:
        st.error("Please specify at least 1 year.")
        st.stop()

    # prepare uploaded files mapping by asking user to name them optionally
    uploaded_map = {}
    if upload_stock_files:
        # try to map filenames that equal ticker.csv or user may have uploaded in same order
        for f in upload_stock_files:
            name = os.path.splitext(f.name)[0].upper()
            try:
                uploaded_map[name] = f.read()
            except Exception:
                if show_diagnostics:
                    st.text(f"Failed reading uploaded file {f.name}")
                    st.text(traceback.format_exc())

    end = datetime.date.today().isoformat()
    start = (datetime.date.today() - datetime.timedelta(days=365*years)).isoformat()

    # fetch each stock
    stock_series = {}
    stock_sources = {}
    for t in stocks_list:
        df, src = fetch_symbol(t, start, end, uploaded_files_map=uploaded_map)
        if df is None or df.empty:
            if show_diagnostics:
                st.warning(f"{t}: no data found (source tried: {src})")
        else:
            stock_series[t] = df["Close"]
            stock_sources[t] = src

    if not stock_series:
        st.error("Failed to fetch any stock prices via available data sources. Try uploading CSVs or check network / AlphaVantage key.")
        st.stop()

    # market series
    market_df = pd.DataFrame()
    market_src = None
    if uploaded_market_csv is not None:
        try:
            b = uploaded_market_csv.read()
            mdf = parse_uploaded_csv_bytes(b)
            if not mdf.empty:
                market_df = mdf.rename(columns={"Close":"market_close"})
                market_src = "uploaded_market_csv"
        except Exception:
            if show_diagnostics:
                st.text("Uploaded market CSV parse traceback:")
                st.text(traceback.format_exc())

    if market_df.empty:
        mdf, msrc = fetch_symbol(market_symbol, start, end, uploaded_files_map=uploaded_map)
        if mdf is None or mdf.empty:
            # try SPY fallback
            if market_symbol.upper() not in ("SPY",):
                mdf2, msrc2 = fetch_symbol("SPY", start, end, uploaded_files_map=uploaded_map)
                if mdf2 is not None and not mdf2.empty:
                    market_df = mdf2.rename(columns={"Close":"market_close"})
                    market_src = "yfinance:SPY" if msrc2=="yfinance" else msrc2
                    market_symbol = "SPY"
                else:
                    st.error("Failed to fetch market data. Upload CSV or check network / API keys.")
                    st.stop()
            else:
                st.error("Failed to fetch SPY. Upload CSV or check network / API keys.")
                st.stop()
        else:
            market_df = mdf.rename(columns={"Close":"market_close"})
            market_src = msrc

    # combine into DataFrame
    combined = pd.DataFrame(stock_series).join(market_df, how="inner")
    if combined.empty:
        st.error("After aligning dates, no overlapping data remains between the selected stocks and market. Try expanding years or upload CSVs with matching ranges.")
        st.stop()

    # show small sample
    st.markdown("### Sample (head)")
    st.dataframe(combined.head().round(4))
    st.markdown("### Sample (tail)")
    st.dataframe(combined.tail().round(4))

    # compute percent returns per period (capm_functions expects percent)
    returns = combined.copy()
    for col in returns.columns:
        returns[col] = returns[col].pct_change().fillna(0) * 100.0
    returns = returns.rename(columns={"market_close":"sp500"})

    # compute beta for each stock using capm_functions.calculate_beta if available (it expects a DataFrame)
    results = []
    for s in stocks_list:
        if s not in returns.columns:
            results.append({"Stock": s, "Beta": None})
            continue
        try:
            b, a = capm_functions.calculate_beta(returns.reset_index(drop=True), s)
            results.append({"Stock": s, "Beta": b})
        except Exception:
            # fallback polyfit
            try:
                slope, inter = np.polyfit(returns["sp500"].values, returns[s].values, 1)
                results.append({"Stock": s, "Beta": float(slope)})
            except Exception:
                results.append({"Stock": s, "Beta": None})

    results_df = pd.DataFrame(results)
    st.markdown("### Beta estimates")
    st.dataframe(results_df, use_container_width=True)

    # expected returns CAPM
    periods_per_year = 252.0
    mean_rm = (returns["sp500"].mean() / 100.0) * periods_per_year  # percent->fraction then annualize
    ers = []
    for row in results:
        b = row["Beta"]
        if b is None:
            er = None
        else:
            er = Rf + b * (mean_rm - Rf)
        ers.append({"Stock": row["Stock"], "CAPM_Return_annual_%": round(er*100.0, 3) if er is not None else None})
    ers_df = pd.DataFrame(ers)
    st.markdown("### CAPM Expected Returns (annualized)")
    st.dataframe(ers_df, use_container_width=True)

    # plot normalized prices
    try:
        plot_df = combined.copy()
        norm = capm_functions.normalize(plot_df.reset_index().rename(columns={plot_df.reset_index().columns[0]:"Date"}))
        st.plotly_chart(capm_functions.interactive_plot(norm), use_container_width=True)
    except Exception:
        if show_diagnostics:
            st.text("Plotting traceback:")
            st.text(traceback.format_exc())

    # show data sources summary
    if show_diagnostics:
        st.write("Data sources (per stock):")
        for k, v in stock_sources.items():
            st.write(f"- {k}: {v}")
        st.write(f"Market source: {market_src}")
    else:
        st.info(f"Data sources: market ← {market_src}; stocks ← multiple (open diagnostics to view).")

