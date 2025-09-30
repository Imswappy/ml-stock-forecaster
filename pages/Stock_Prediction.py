import pandas as pd
import streamlit as st

from pages.utils.model_train import (evaluate_model, get_data,
                                     get_differencing_order, get_forecast,
                                     get_rolling_mean, inverse_scaling,
                                     scaling)
from pages.utils.plotly_figure import Moving_average_forecast, plotly_table

st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📉",
    layout="wide",
)

st.title("Stock Prediction")

# Input
col1, _, _ = st.columns(3)
with col1:
    ticker = st.text_input("Stock Ticker", "AAPL").strip().upper()

st.subheader(f"Predicting Next 30 days Close Price for: {ticker}")

# --- Fetch data
close_price = get_data(ticker, years=2)

if close_price.empty:
    st.error("No data available for this ticker. Try uploading CSV or another ticker.")
    st.stop()

# --- Rolling mean
rolling_price = get_rolling_mean(close_price)

# --- Differencing order
differencing_order = get_differencing_order(rolling_price)

# --- Scaling
scaled_data, scaler = scaling(rolling_price)

# --- Evaluate model
rmse = evaluate_model(scaled_data, differencing_order)
st.write("**Model RMSE Score:**", f"{rmse:.4f}" if rmse == rmse else "N/A")

# --- Forecast
last_date = rolling_price.index[-1]
forecast_scaled = get_forecast(scaled_data, differencing_order, steps=30, last_date=last_date)

if forecast_scaled.empty:
    st.warning("Forecast could not be generated (not enough data).")
    st.stop()

forecast_scaled["Close"] = inverse_scaling(scaler, forecast_scaled["Close"].values)

st.write("##### Forecast Data (Next 30 days)")
fig_tail = plotly_table(forecast_scaled.round(3))
fig_tail.update_layout(height=220)
st.plotly_chart(fig_tail, use_container_width=True)

# Combine historical + forecast
forecast_all = pd.concat([rolling_price, forecast_scaled["Close"]])

# Plot moving average forecast chart
st.plotly_chart(Moving_average_forecast(forecast_all.iloc[-150:]), use_container_width=True)
