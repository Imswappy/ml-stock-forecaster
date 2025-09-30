# pages/utils/model_train.py
"""
Robust model training & forecasting helpers used by pages/Stock_Prediction.py.

Functions provided:
- get_data(ticker, years=2, uploaded_csv_bytes=None): returns pd.Series of Close prices (datetime index)
- get_rolling_mean(close_series, window=None): returns rolling mean series (dynamically chooses window)
- stationary_check(close_price): returns adf p-value (guarded)
- get_differencing_order(close_price): returns 0 or 1
- scaling(series): returns (scaled_array, scaler)
- inverse_scaling(scaler, array_or_series): returns inverse-transformed series
- evaluate_model(scaled_data, d): fits a light ARIMA and returns RMSE (guarded)
- get_forecast(scaled_data, d, steps=30): returns DataFrame with forecasted 'Close' (scaled units), indexed by next business days
"""

import io
import math
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Fetching
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

from sklearn.metrics import mean_squared_error
# preprocessing / metrics
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
# stats & models
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


def _parse_csv_bytes_to_close_series(b: bytes) -> pd.Series:
    """Parse uploaded CSV bytes into a Close price series (datetime index)."""
    try:
        df = pd.read_csv(io.BytesIO(b))
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(b.decode()))
        except Exception:
            return pd.Series(dtype=float)

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
        return pd.Series(dtype=float)

    df = df.set_index(date_col).sort_index()
    # find close col preferring adj close
    close_col = None
    for c in df.columns:
        if c.lower() in ("adj close", "adjusted_close", "adjusted close"):
            close_col = c
            break
    if close_col is None:
        for c in df.columns:
            if c.lower() == "close":
                close_col = c
                break
    if close_col is None:
        return pd.Series(dtype=float)

    s = df[close_col].astype(float).dropna()
    s.name = "Close"
    return s


def get_data(ticker: str, years: int = 2, uploaded_csv_bytes: Optional[bytes] = None) -> pd.Series:
    """
    Returns a pd.Series of Close prices indexed by datetime for `ticker`.
    Attempts, in order:
        1) uploaded CSV (if provided)
        2) yfinance.download(period=f"{years}y")
        3) yf.Ticker.history() fallback
    Returns empty Series on failure (caller must handle).
    """
    # 0) uploaded CSV
    if uploaded_csv_bytes is not None:
        try:
            s = _parse_csv_bytes_to_close_series(uploaded_csv_bytes)
            if not s.empty:
                return s
        except Exception:
            pass

    # 1) yfinance.download
    if HAVE_YF:
        try:
            df = yf.download(ticker, period=f"{years}y", progress=False, threads=False)
            if df is not None and not df.empty:
                if "Adj Close" in df.columns and "Close" not in df.columns:
                    df = df.rename(columns={"Adj Close": "Close"})
                if "Close" in df.columns:
                    s = df["Close"].dropna()
                    s.name = "Close"
                    if not s.empty:
                        return s
        except Exception:
            # fallback to history below
            pass

        # 2) yf.Ticker.history fallback
        try:
            t = yf.Ticker(ticker)
            df2 = t.history(period=f"{years}y", actions=False)
            if df2 is not None and not df2.empty:
                if "Adj Close" in df2.columns and "Close" not in df2.columns:
                    df2 = df2.rename(columns={"Adj Close": "Close"})
                if "Close" in df2.columns:
                    s = df2["Close"].dropna()
                    s.name = "Close"
                    if not s.empty:
                        return s
        except Exception:
            pass

    # If everything failed return empty series
    return pd.Series(dtype=float)


def get_rolling_mean(close_series: pd.Series, window: Optional[int] = None) -> pd.Series:
    """
    Return a rolling mean to smooth noise.
    Chooses a safe window automatically if not provided:
      window = min(20, max(2, len(series) // 5))
    Ensures result is not constant and drops leading NaNs before returning.
    """
    if close_series is None or close_series.empty:
        return pd.Series(dtype=float)

    n = len(close_series)
    if window is None:
        # dynamic window but at least 2 and at most 20
        window = int(min(20, max(2, max(2, n // 5))))
    window = max(2, int(window))

    # if window larger than series length, reduce it
    if window >= n:
        window = max(2, n // 2)

    rolled = close_series.rolling(window=window, min_periods=1).mean()
    # if rolling produced constant due to small variance, return original clipped to dropna
    if rolled.dropna().empty or (rolled.max() == rolled.min()):
        # fallback: simple EWMA or original
        try:
            ewma = close_series.ewm(span=min(5, max(2, n // 10))).mean()
            if not ewma.dropna().empty and ewma.max() != ewma.min():
                return ewma.dropna()
        except Exception:
            pass
        return close_series.dropna()
    return rolled.dropna()


def stationary_check(close_price: pd.Series) -> float:
    """
    Run ADF on the series (dropna first). Returns p-value.
    Guarded: returns p-value 1.0 if series is empty/constant or if ADF fails.
    """
    if close_price is None:
        return 1.0
    s = close_price.dropna()
    if s.empty:
        return 1.0
    # If constant series -> not stationary (return high p-value)
    try:
        if s.max() == s.min():
            return 1.0
    except Exception:
        return 1.0

    try:
        res = adfuller(s, autolag="AIC", maxlag=None)
        p_value = float(res[1])
        return p_value
    except Exception:
        return 1.0


def get_differencing_order(close_price: pd.Series) -> int:
    """
    Decide differencing order (0 or 1) using ADF test.
    If p < 0.05 -> stationary -> d=0, else d=1.
    """
    p_value = stationary_check(close_price)
    return 0 if p_value < 0.05 else 1


def scaling(series: pd.Series) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale a 1-D pandas Series to numpy array and return (scaled_array, scaler).
    """
    arr = np.array(series).reshape(-1, 1).astype(float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(arr)
    return scaled, scaler


def inverse_scaling(scaler: StandardScaler, arr: np.ndarray) -> np.ndarray:
    """
    Inverse transform scaled numpy array back to original scale.
    Accepts 1D or 2D arrays.
    """
    if arr is None:
        return np.array([])
    arr2 = np.array(arr)
    # ensure 2D for scaler
    if arr2.ndim == 1:
        arr2 = arr2.reshape(-1, 1)
    inv = scaler.inverse_transform(arr2)
    # return flattened 1d
    return inv.flatten()


def evaluate_model(scaled_data: np.ndarray, d: int, n_forecast_test: int = 30) -> float:
    """
    Fit a light ARIMA model on scaled_data (which is numpy 2d shaped [-1,1]).
    Returns RMSE computed on last n_forecast_test points (walk-forward / in-sample forecast).
    Guarded: returns math.nan if fitting fails or insufficient data.
    """
    # convert to 1d
    if scaled_data is None:
        return float("nan")
    arr = np.asarray(scaled_data).reshape(-1)
    n = len(arr)
    if n < 10:
        return float("nan")

    # choose small p/q to avoid overfitting / long fit times
    p = min(5, max(1, n // 20))
    q = min(5, max(0, n // 20))

    # ensure we have some test portion
    train_end = max(int(n * 0.7), n - n_forecast_test)
    train_data = arr[:train_end]
    test_data = arr[train_end:]

    try:
        # small ARIMA
        model = ARIMA(train_data, order=(p, d, q))
        fitted = model.fit(method_kwargs={"warn_convergence": False})
        # forecast same length as test_data
        preds = fitted.forecast(steps=len(test_data))
        rmse = math.sqrt(mean_squared_error(test_data, preds))
        return float(rmse)
    except Exception:
        # fallback simple persistence baseline (use last value)
        try:
            preds = np.repeat(train_data[-1], len(test_data))
            rmse = math.sqrt(mean_squared_error(test_data, preds))
            return float(rmse)
        except Exception:
            return float("nan")


def get_forecast(scaled_data: np.ndarray, d: int, steps: int = 30, last_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Produce a steps-day forecast using ARIMA trained on full scaled_data.
    Returns a DataFrame with index set to next business days and column 'Close' (in scaled units).
    Caller should inverse_scale the 'Close' column if needed.
    """
    if scaled_data is None:
        return pd.DataFrame(columns=["Close"])

    arr = np.asarray(scaled_data).reshape(-1)
    n = len(arr)
    if n < 3:
        return pd.DataFrame(columns=["Close"])

    # model hyperparams (small)
    p = min(5, max(1, n // 20))
    q = min(5, max(0, n // 20))

    try:
        model = ARIMA(arr, order=(p, d, q))
        fitted = model.fit(method_kwargs={"warn_convergence": False})
        preds = fitted.forecast(steps=steps)
        preds = np.asarray(preds).flatten()
    except Exception:
        # fallback: use last value repeated
        preds = np.repeat(arr[-1], steps)

    # make date index for next business days
    if last_date is None:
        # if caller has original series, they can pass last_date; otherwise use today
        last_date = pd.Timestamp.today()
    future_idx = pd.bdate_range(start=(pd.Timestamp(last_date) + pd.Timedelta(days=1)), periods=steps)
    out = pd.DataFrame({"Close": preds}, index=future_idx)
    return out
