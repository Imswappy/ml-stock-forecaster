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
ticker = st.text_input("Stock Ticker", "AAPL").strip().upper()
st.subheader(f"Predicting Next 30 days Close Price for: {ticker}")

# -----------------------------
# Fetch data
# -----------------------------
try:
    close_price = get_data(ticker)
except Exception as e:
    st.error(f"Failed to fetch data for {ticker}: {e}")
    st.stop()

if not isinstance(close_price, pd.Series) or close_price.empty:
    st.error("No data available for this ticker. Try another ticker.")
    st.stop()

# -----------------------------
# Preprocessing & Model
# -----------------------------
rolling_price = get_rolling_mean(close_price)

if rolling_price.empty:
    st.error("Rolling-smoothed series is empty. Cannot proceed.")
    st.stop()

# differencing order
differencing_order = get_differencing_order(rolling_price)

# scaling
scaled_data, scaler = scaling(rolling_price)

# evaluate model
rmse = evaluate_model(scaled_data, differencing_order)
st.write("**Model RMSE Score:**", rmse if rmse == rmse else "N/A")

# forecast
try:
    last_date = rolling_price.index[-1] if len(rolling_price.index) > 0 else None
    forecast = get_forecast(scaled_data, differencing_order, steps=30, last_date=last_date)
except Exception as e:
    st.error(f"Forecast generation failed: {e}")
    st.stop()

if forecast.empty:
    st.error("Forecast failed — not enough data to build model.")
    st.stop()

# inverse scale forecast
forecast["Close"] = inverse_scaling(scaler, forecast["Close"].values)

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
