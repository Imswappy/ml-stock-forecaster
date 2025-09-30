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

rmse = 0
st.subheader(f"Predicting Next 30 days Close Price for: {ticker}")

# -----------------------------
# Fetch data safely
# -----------------------------
uploaded_bytes = uploaded_csv.read() if uploaded_csv else None

try:
    close_price = get_data(ticker, years=2, uploaded_csv_bytes=uploaded_bytes)
    if not isinstance(close_price, pd.Series):
        st.error("Ticker lookup failed. No valid price series returned.")
        st.stop()
    if close_price.empty:
        st.error("No data available for this ticker. Try another ticker or upload CSV.")
        st.stop()
except Exception as e:
    st.error(f"Failed to fetch data for {ticker}: {e}")
    st.stop()

# -----------------------------
# Preprocessing & Model
# -----------------------------
rolling_price = get_rolling_mean(close_price)

# differencing order via ADF
differencing_order = get_differencing_order(rolling_price)

# scaling
scaled_data, scaler = scaling(rolling_price)

# evaluate model
rmse = evaluate_model(scaled_data, differencing_order)
st.write("**Model RMSE Score:**", rmse if rmse == rmse else "N/A")  # check NaN

# forecast
forecast = get_forecast(
    scaled_data, differencing_order, steps=30, last_date=rolling_price.index[-1]
)
if forecast.empty:
    st.error("Forecast failed — not enough data to build model.")
    st.stop()

# inverse scale forecast
forecast["Close"] = inverse_scaling(scaler, forecast["Close"])

# -----------------------------
# Display forecast
# -----------------------------
st.write("##### Forecast Data (Next 30 days)")
fig_tail = plotly_table(forecast.sort_index(ascending=True).round(3))
fig_tail.update_layout(height=220)
st.plotly_chart(fig_tail, use_container_width=True)

# concat for plotting
forecast_full = pd.concat([rolling_price, forecast])

st.plotly_chart(Moving_average_forecast(forecast_full.iloc[-200:]), use_container_width=True)
