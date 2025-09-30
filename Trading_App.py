# Trading_App.py (with worked example + ML prefill link)
import datetime

import streamlit as st

st.set_page_config(
    page_title="Trading App",
    page_icon=":heavy_dollar_sign:",
    layout="wide",
)

st.title("Trading Guide App :bar_chart:")
st.subheader("A compact financial analysis & forecasting playground (Stock Info, Technicals, CAPM & Forecasting)")

st.markdown(
    """
This application contains multiple pages:

- **Stock Information** — company fundamentals, sector info, quick financial ratios and recent historical prices.  
- **Stock Prediction** — statistical time-series forecasting (ARIMA) to produce a 30-day forecast.  
- **ML Style Forecasting** — machine-learning and deep-learning forecasting (Random Forest, XGBoost/GradientBoosting, LSTM) with tuning and model comparison.  
- **CAPM Return** — multi-stock CAPM calculations and visualizations.  
- **CAPM Beta** — single-stock Beta and CAPM expected return calculator.

Use the left sidebar to navigate between pages. If you prefer, the quick navigation buttons below attempt to set a `?page=...` query param (some Streamlit deployments/platforms ignore programmatic navigation — if that happens, please click the page from the sidebar).
"""
)

st.markdown("---")

# Quick navigation buttons (best-effort; Streamlit's sidebar pages are authoritative)
st.markdown("### Quick navigation")
cols = st.columns(5)
pages_map = {
    "Stock Analysis": "Stock_Analysis",
    "Stock Prediction": "Stock_Prediction",
    "ML Style Forecasting": "ML_Forecasting",
    "CAPM Return": "CAPM_Return",
    "CAPM Beta": "CAPM_Beta",
}

for i, (label, slug) in enumerate(pages_map.items()):
    with cols[i]:
        if st.button(f"Open {label}"):
            # best-effort: set query params (works on many deployments)
            st.experimental_set_query_params(page=slug)
            st.experimental_rerun()

st.markdown(
    """
**If a button didn't switch pages:** open the page from the *left sidebar* (Streamlit's built-in multi-page navigation).  
Below each section I also provide a clickable link that attempts to set the `?page=...` query param.
"""
)

st.markdown("---")

##############################################
# Section: Stock Information (Stock Analysis)
##############################################
st.header("Stock Information — What you'll find on the Stock Analysis page")
st.markdown(
    """
**What this page does (summary):**
- Fetches company metadata (sector, website, headcount) via `yfinance`.
- Shows **fundamental ratios** (market cap, PE, EPS, profit margins, ROE, Debt/Equity).
- Displays recent historical table + interactive charts (candlesticks, line charts).
- Computes technical indicators (RSI, MACD, SMA) used by technical analysts.
"""
)

st.markdown("**Technical indicators (formulas & intuition):**")

st.subheader("1. Simple Moving Average (SMA)")
st.markdown(
    "The SMA over window *N* averages the last *N* close prices. Intuition: smooths price noise — short SMA responds quickly, long SMA shows trend."
)
st.latex(r"\mathrm{SMA}_N[t] \;=\; \frac{1}{N}\sum_{i=0}^{N-1} P_{t-i}")

st.subheader("2. Relative Strength Index (RSI)")
st.markdown(
    "Typical window is 14 periods. Compute average gains and average losses; RSI maps relative gain to [0,100]. Intuition: RSI near 70 ⇒ overbought, near 30 ⇒ oversold."
)
st.latex(r"\mathrm{RSI} \;=\; 100 - \frac{100}{1 + \mathrm{RS}}")

st.subheader("3. MACD (Moving Average Convergence Divergence)")
st.markdown(
    "MACD uses exponential moving averages: MACD line = EMA_fast − EMA_slow; signal line is an EMA of the MACD line."
)
st.markdown("[Open Stock Analysis page](?page=Stock_Analysis)")

st.markdown("---")

################################################
# Section: Stock Prediction (Forecasting / Model)
################################################
st.header("Stock Prediction — Model, math & intuition")
st.markdown(
    """
**What this page does (summary):**
- Downloads historical close prices (via `yfinance`), applies a 7-day rolling mean for noise reduction.
- Performs stationarity check (ADF test) to determine differencing order d.
- Fits an ARIMA(p, d, q) model and forecasts the next 30 days of closing prices.
"""
)

st.latex(r"\phi(B)(1 - B)^d y_t = \theta(B)\varepsilon_t")
st.markdown("[Open Stock Prediction page](?page=Stock_Prediction)")

st.markdown("---")

#########################################
# Section: ML Style Forecasting (math & intuition)
#########################################
st.header("ML Style Forecasting — math & data-science intuition")
st.markdown(
    """
This page contains *machine-learning* and *deep-learning* approaches to forecasting:
- Feature engineering (lagging, technical indicators)
- Supervised models (Random Forest, XGBoost/GradientBoosting)
- Sequence models (LSTM)
- Tuning & model selection (RandomizedSearch, Optuna)
- Evaluation (RMSE, MAE, R²)
"""
)

st.subheader("Worked example — building lag features (quick)")
st.markdown(
    "Given a short closing-price series: `[100, 101, 102, 101, 103, 104]` we build lag features with p=3. The supervised rows (target is current close) look like:"
)
# build example table
import pandas as _pd

series = _pd.Series([100, 101, 102, 101, 103, 104], name="Close")
p = 3
df_ex = _pd.DataFrame({"Close": series})
for lag in range(1, p + 1):
    df_ex[f"lag_{lag}"] = df_ex["Close"].shift(lag)
df_ex = df_ex.dropna().reset_index(drop=True)
# target y_t is Close, features are lag_1..lag_p
st.write("Example price series (last rows used to form supervised dataset):")
st.table(df_ex)

st.latex(r"X_t = [y_{t-1}, y_{t-2}, \dots, y_{t-p}] \quad \text{target } y_t")

st.markdown(
    """
**Notes on the example:**  
- Each row's `lag_1` is the previous day's close, `lag_2` the day before that, etc.  
- You can add SMA/RSI as extra columns to the feature matrix.  
- For multi-step forecasting we use **recursive forecasting**: predict next value, append it to the lag features, and repeat.
"""
)

# ML link with prefilled ticker example
st.markdown("[Open ML Style Forecasting for AAPL](?page=ML_Forecasting&ticker=AAPL)")

st.markdown("---")

#########################################
# Section: CAPM Return (multi-stock CAPM)
#########################################
st.header("CAPM Return — Theory & implementation")
st.latex(r"E[R_i] = R_f + \beta_i \left(E[R_m] - R_f\right)")
st.markdown(
    """
Interpretation of Beta (β):
- β > 1 : more volatile than market  
- β < 1 : less volatile than market  
- β = 1 : moves with market

In this app, β is estimated via linear regression (slope of stock returns vs market returns).
"""
)
st.markdown("[Open CAPM Return page](?page=CAPM_Return) • [Open CAPM Beta page](?page=CAPM_Beta)")

st.markdown("---")

st.markdown("**Files present in the app (for reference):**")
st.code(
    """
pages/
  ├─ Stock_Analysis.py
  ├─ Stock_Prediction.py
  ├─ ML_Forecasting.py
  ├─ CAPM_Return.py
  ├─ CAPM_Beta.py
pages/utils/
  ├─ plotly_figure.py
  ├─ model_train.py
  ├─ capm_functions.py
"""
)
