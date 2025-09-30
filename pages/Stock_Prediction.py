# pages/Stock_Prediction.py

import pandas as pd
import streamlit as st

from pages.utils.model_train import (evaluate_model, get_data,
                                     get_differencing_order, get_forecast,
                                     get_rolling_mean, inverse_scaling,
                                     scaling)
from pages.utils.plotly_figure import Moving_average_forecast, plotly_table

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Stock Prediction")

# -----------------------------
# Controls
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    ticker = st.text_input("Stock Ticker", "AAPL").strip().upper()

with col2:
    uploaded_csv = st.file_uploader("Optional: Upload CSV (Date + Close)", type=["csv"])

st.subheader(f"Predicting Next 30 days Close Price for: {ticker}")

# -----------------------------
# Fetch data safely
# -----------------------------
uploaded_bytes = uploaded_csv.read() if uploaded_csv else None

try:
    # call get_data with safest signature: ticker and uploaded bytes only
    # some versions of your utils may accept years/uploaded_csv_bytes or just ticker,
    # so pass uploaded bytes positionally by name to be robust.
    close_price = get_data(ticker, uploaded_csv_bytes=uploaded_bytes)
except TypeError:
    # fallback if the utils use a different kw name or signature: try positional
    try:
        close_price = get_data(ticker, uploaded_bytes)
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        st.stop()
except Exception as e:
    st.error(f"Failed to fetch data for {ticker}: {e}")
    st.stop()

# validate returned object
if not isinstance(close_price, pd.Series):
    st.error("Ticker lookup failed — get_data did not return a pandas Series. Check your pages/utils/model_train.py implementation.")
    st.stop()

if close_price.empty:
    st.error("No data available for this ticker. Try another ticker, upload CSV, or check network/API keys.")
    st.stop()

# -----------------------------
# Preprocessing & Model
# -----------------------------
# compute rolling mean (smoothing) — function should handle short series
rolling_price = get_rolling_mean(close_price)
if not isinstance(rolling_price, pd.Series) or rolling_price.empty:
    st.error("Rolling-smoothed series is empty. Cannot proceed — try a different ticker or upload more history.")
    st.stop()

# differencing order via ADF
try:
    differencing_order = get_differencing_order(rolling_price)
except Exception as e:
    st.warning(f"ADF differencing selection failed, defaulting to d=1: {e}")
    differencing_order = 1

# scaling
try:
    scaled_data, scaler = scaling(rolling_price)
except Exception as e:
    st.error(f"Scaling failed: {e}")
    st.stop()

# evaluate model
try:
    rmse = evaluate_model(scaled_data, differencing_order)
except Exception as e:
    st.warning(f"Model evaluation failed: {e}")
    rmse = float("nan")

st.write("**Model RMSE Score:**", rmse if rmse == rmse else "N/A")  # show N/A for NaN

# forecast
try:
    # pass last_date so forecast index follows original series
    last_date = rolling_price.index[-1] if len(rolling_price.index) > 0 else None
    forecast = get_forecast(scaled_data, differencing_order, steps=30, last_date=last_date)
except TypeError:
    # some implementations may not accept last_date or steps named, try simpler call
    try:
        forecast = get_forecast(scaled_data, differencing_order)
    except Exception as e:
        st.error(f"Forecast generation failed: {e}")
        st.stop()
except Exception as e:
    st.error(f"Forecast generation failed: {e}")
    st.stop()

if forecast is None or forecast.empty:
    st.error("Forecast failed — not enough data or model failed to fit.")
    st.stop()

# inverse scale forecast
try:
    # forecast expected to have column 'Close' in scaled units
    if "Close" not in forecast.columns:
        st.warning("Forecast does not contain 'Close' column; trying to interpret single-column output.")
        # if the forecast is a Series-like, convert to DataFrame
        if isinstance(forecast, pd.Series):
            forecast = forecast.to_frame(name="Close")
        else:
            # try to coerce first numeric column
            numeric_cols = forecast.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                st.error("Forecast has no numeric columns to inverse scale.")
                st.stop()
            forecast = forecast[[numeric_cols[0]]].rename(columns={numeric_cols[0]: "Close"})

    forecast["Close"] = inverse_scaling(scaler, forecast["Close"].values)
except Exception as e:
    st.error(f"Inverse scaling of forecast failed: {e}")
    st.stop()

# -----------------------------
# Display forecast
# -----------------------------
st.write("##### Forecast Data (Next 30 days)")
try:
    fig_tail = plotly_table(forecast.sort_index(ascending=True).round(3))
    fig_tail.update_layout(height=220)
    st.plotly_chart(fig_tail, use_container_width=True)
except Exception:
    # fallback to table display
    st.dataframe(forecast.round(3))

# concat for plotting (align types)
try:
    # ensure rolling_price is a DataFrame/Series with name 'Close'
    if isinstance(rolling_price, pd.Series):
        rolling_df = rolling_price.to_frame(name="Close")
    else:
        rolling_df = rolling_price

    forecast_full = pd.concat([rolling_df, forecast], axis=0, sort=False)
    # try to plot last 200 rows
    plot_slice = forecast_full.iloc[-200:]
    st.plotly_chart(Moving_average_forecast(plot_slice), use_container_width=True)
except Exception as e:
    st.warning(f"Plotting combined forecast failed: {e}")
    try:
        st.line_chart(forecast["Close"])
    except Exception:
        pass
