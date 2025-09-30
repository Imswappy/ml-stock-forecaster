# pages/ML_Forecasting.py
import importlib
import io
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# optional libraries (graceful)
try:
    import xgboost as xgb
    HAVE_XGBOOST = True
except Exception:
    HAVE_XGBOOST = False

try:
    import optuna
    HAVE_OPTUNA = True
except Exception:
    HAVE_OPTUNA = False

try:
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    HAVE_TF = True
except Exception:
    HAVE_TF = False

# ---------- page config ----------
st.set_page_config(page_title="ML Style Forecasting (Tuning + Save)", page_icon="🤖", layout="wide")
st.title("ML-Style Forecasting — Tuning, Train & Save")

# ---------- model storage ----------
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

saved_models = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
selected_saved = st.selectbox("Or load a saved model", ["(none)"] + saved_models)
LOADED_MODEL = None
if selected_saved and selected_saved != "(none)":
    try:
        bundle = joblib.load(os.path.join(models_dir, selected_saved))
        LOADED_MODEL = {"model": bundle.get("model"), "scaler": bundle.get("scaler"), "features": bundle.get("features")}
        st.success(f"Loaded {selected_saved}")
        st.write("Model expects features:", LOADED_MODEL.get("features"))
    except Exception as e:
        st.error(f"Failed loading saved model: {e}")

# ---------- inputs ----------
params = st.experimental_get_query_params()
prefill_ticker = params.get("ticker", [None])[0]
if prefill_ticker:
    prefill_ticker = str(prefill_ticker).upper()

popular = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "JPM", "UNH"]
options = popular + ["Other"]

if prefill_ticker and prefill_ticker in popular:
    default_index = popular.index(prefill_ticker)
elif prefill_ticker and prefill_ticker not in popular:
    default_index = len(popular)
else:
    default_index = 0

col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    sel = st.selectbox("Ticker", options=options, index=default_index)
    if sel == "Other":
        custom_value = prefill_ticker if (prefill_ticker and prefill_ticker not in popular) else ""
        custom_ticker = st.text_input("Enter ticker (e.g. AAPL)", value=custom_value)
        ticker = custom_ticker.strip().upper()
    else:
        ticker = sel
with col2:
    years = st.slider("Years of data", 1, 10, 2)
with col3:
    n_lags = st.slider("Number of lag features", 1, 10, 5)

add_tech = st.checkbox("Add simple technical indicators (SMA, RSI)", value=True)
train_frac = st.slider("Train fraction", 0.6, 0.95, 0.9, 0.01)
csv_upload = st.file_uploader("Optional: upload CSV (Date + Close or Adj Close)", type=["csv"])

st.markdown("### Tuning")
use_tuning = st.checkbox("Enable hyperparameter tuning?", value=False)
tuning_algo = st.selectbox("Tuning algorithm", ["RandomizedSearchCV", "Optuna (if installed)"])
max_evals = st.number_input("Max trials/iterations (small recommended)", min_value=5, max_value=200, value=20, step=5)

st.markdown("### Model choices")
default_models = ["RandomForest"]
available = ["RandomForest", "GradientBoosting"] + (["XGBoost"] if HAVE_XGBOOST else [])
models_to_train = st.multiselect("Select models to train", available, default=default_models)

run_button = st.button("Fetch + Train (may take time)")

# ---------- helper functions ----------
def parse_csv_bytes(b):
    try:
        df = pd.read_csv(io.BytesIO(b))
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(b.decode()))
        except Exception:
            return pd.DataFrame()
    # detect date column
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
        return pd.DataFrame()
    cols = []
    for k in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]:
        if k in df.columns:
            cols.append(k)
    if close_col not in cols:
        cols.append(close_col)
    out = df[cols].rename(columns={close_col: "Close"})
    out = out[["Close"] + [c for c in ["Open", "High", "Low", "Volume"] if c in out.columns]]
    return out

@st.cache_data(ttl=3600)
def fetch_data(ticker, years, uploaded_csv_bytes=None, show_diagnostics=False):
    """
    Tries (in order): uploaded CSV, yfinance.download, yf.Ticker.history, pandas_datareader(stooq), AlphaVantage JSON.
    Returns DataFrame with Close (and optional Open/High/Low/Volume), or empty DataFrame.
    """
    # 0) uploaded
    if uploaded_csv_bytes is not None:
        parsed = parse_csv_bytes(uploaded_csv_bytes)
        if not parsed.empty:
            if show_diagnostics:
                st.info("Using uploaded CSV for data.")
            return parsed

    # 1) yfinance.download
    try:
        df = yf.download(ticker, period=f"{years}y", progress=False)
        if df is not None and not df.empty:
            if {"Close", "Open", "High", "Low", "Volume"}.intersection(df.columns):
                df = df[[c for c in ['Close','Open','High','Low','Volume'] if c in df.columns]].dropna()
            if not df.empty:
                if show_diagnostics:
                    st.info(f"yfinance.download returned {len(df)} rows")
                return df
    except Exception as e:
        if show_diagnostics:
            st.warning(f"yfinance.download error: {e}")

    # 2) yf.Ticker.history
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=f"{years}y", actions=False)
        if df is not None and not df.empty:
            if "Adj Close" in df.columns and "Close" not in df.columns:
                df = df.rename(columns={"Adj Close": "Close"})
            if "Close" in df.columns:
                out = df[['Close'] + [c for c in ['Open','High','Low','Volume'] if c in df.columns]].dropna()
                if not out.empty:
                    if show_diagnostics:
                        st.info(f"yf.Ticker.history returned {len(out)} rows")
                    return out
    except Exception as e:
        if show_diagnostics:
            st.warning(f"yf.Ticker.history error: {e}")

    # 3) pandas_datareader -> stooq (if installed)
    try:
        pdr_spec = importlib.util.find_spec("pandas_datareader")
        if pdr_spec is not None:
            try:
                from pandas_datareader import data as pdr
                start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
                end = pd.Timestamp.today().strftime("%Y-%m-%d")
                df = pdr.DataReader(ticker, "stooq", start, end)
                if df is not None and not df.empty:
                    if "Close" in df.columns:
                        out = df[["Close"] + [c for c in ["Open","High","Low"] if c in df.columns]].sort_index().dropna()
                        if show_diagnostics:
                            st.info(f"pandas_datareader(stooq) returned {len(out)} rows")
                        return out
            except Exception as e:
                if show_diagnostics:
                    st.warning(f"pandas_datareader(stooq) error: {e}")
    except Exception:
        # pandas_datareader not available or error probing it
        pass

    # 4) AlphaVantage JSON fallback if key
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
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "outputsize": "full",
                "apikey": av_key,
                "datatype": "json"
            }
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                j = r.json()
                if "Note" in j and show_diagnostics:
                    st.warning("AlphaVantage Note: " + str(j.get("Note")))
                if "Error Message" in j and show_diagnostics:
                    st.warning("AlphaVantage Error: " + str(j.get("Error Message")))
                ts = j.get("Time Series (Daily)") or j.get("Time Series (Daily Adjusted)")
                if ts:
                    df = pd.DataFrame.from_dict(ts, orient="index")
                    df.index = pd.to_datetime(df.index)
                    # map columns to readable names
                    col_map = {}
                    for c in df.columns:
                        lc = c.lower()
                        if "close" in lc:
                            if "adjusted" in lc or lc.startswith("5.") or "adjusted" in lc:
                                col_map[c] = "Close"
                            else:
                                col_map[c] = col_map.get(c, "Close")
                        elif "open" in lc:
                            col_map[c] = "Open"
                        elif "high" in lc:
                            col_map[c] = "High"
                        elif "low" in lc:
                            col_map[c] = "Low"
                        elif "volume" in lc:
                            col_map[c] = "Volume"
                    df = df.rename(columns=col_map)
                    keep = [c for c in ["Close","Open","High","Low","Volume"] if c in df.columns]
                    out = df[keep].sort_index().astype(float).dropna()
                    if not out.empty:
                        if show_diagnostics:
                            st.info(f"AlphaVantage returned {len(out)} rows")
                        return out
            else:
                if show_diagnostics:
                    st.warning(f"AlphaVantage HTTP {r.status_code}")
        except Exception as e:
            if show_diagnostics:
                st.warning(f"AlphaVantage request failed: {e}")

    # nothing worked
    return pd.DataFrame()

def add_lag_features(df, n_lags):
    df2 = df.copy()
    for lag in range(1, n_lags + 1):
        df2[f"lag_{lag}"] = df2["Close"].shift(lag)
    return df2

def add_technical_indicators(df):
    df2 = df.copy()
    df2["SMA_10"] = df2["Close"].rolling(10).mean()
    df2["SMA_50"] = df2["Close"].rolling(50).mean()
    delta = df2["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df2["RSI_14"] = 100 - (100 / (1 + rs))
    return df2

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

# parameter grids
rf_param_dist = {"n_estimators": [50, 100, 200, 300], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}
gb_param_dist = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]}
xgb_param_dist = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]}

def run_random_search(estimator, param_dist, Xtr, ytr, iters=20):
    rs = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=min(iters, 20), cv=3,
                            scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=42, verbose=0)
    rs.fit(Xtr, ytr)
    return rs.best_estimator_, rs.best_params_

def run_optuna_rf(Xtr, ytr, trials=20):
    import optuna
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split, n_jobs=-1, random_state=42)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, Xtr, ytr, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
        return -scores.mean()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=min(trials, 50))
    best = study.best_params
    model = RandomForestRegressor(**best, n_jobs=-1, random_state=42)
    model.fit(Xtr, ytr)
    return model, best

# ---------- AlphaVantage diagnostic ----------
st.markdown("---")
st.subheader("AlphaVantage diagnostic (optional)")
av_key = None
try:
    av_key = st.secrets.get("ALPHAVANTAGE_API_KEY") if hasattr(st, "secrets") else None
except Exception:
    av_key = None
if not av_key:
    av_key = os.environ.get("ALPHAVANTAGE_API_KEY")
st.write("ALPHAVANTAGE key present:", bool(av_key))
if av_key:
    av_test_symbol = st.text_input("AlphaVantage test symbol", value="AAPL", key="av_test_sym")
    if st.button("Run AlphaVantage test"):
        import requests
        url = "https://www.alphavantage.co/query"
        params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": av_test_symbol, "outputsize": "compact",
                  "apikey": av_key, "datatype": "json"}
        with st.spinner("Calling AlphaVantage..."):
            try:
                r = requests.get(url, params=params, timeout=20)
                st.write("HTTP status:", r.status_code)
                try:
                    j = r.json()
                    st.write("Returned top-level keys:", list(j.keys())[:10])
                    if "Note" in j:
                        st.error("Rate limit Note: " + str(j["Note"]))
                    if "Error Message" in j:
                        st.error("Error Message: " + str(j["Error Message"]))
                except Exception:
                    st.text("Response text (truncated):")
                    st.text(r.text[:2000])
            except Exception as e:
                st.error(f"AlphaVantage request failed: {e}")
st.markdown("---")

# ---------- Quick inference from loaded model ----------
if LOADED_MODEL is not None:
    st.subheader("Quick inference from loaded model")
    st.write("Auto-forecast 7 business days using the loaded model (builds features from recent data).")
    do_quick = st.button("Auto-forecast 7 days with loaded model")
    if do_quick:
        uploaded_bytes = csv_upload.read() if csv_upload is not None else None
        df_recent = fetch_data(ticker, 1, uploaded_csv_bytes=uploaded_bytes, show_diagnostics=True)
        if df_recent.empty:
            st.error("Failed to fetch recent data for quick inference. Try uploading CSV or check network/AlphaVantage.")
        else:
            df_recent = df_recent.tail(max(60, n_lags * 3)).copy()
            df_feat_recent = add_lag_features(df_recent, n_lags)
            if add_tech:
                df_feat_recent = add_technical_indicators(df_feat_recent)
            df_feat_recent = df_feat_recent.dropna()
            if df_feat_recent.empty:
                st.error("Not enough recent data to build features for quick inference.")
            else:
                feature_names = LOADED_MODEL.get("features", [])
                model_obj = LOADED_MODEL.get("model")
                scaler_obj = LOADED_MODEL.get("scaler")
                missing = [f for f in feature_names if f not in df_feat_recent.columns]
                if missing:
                    st.warning(f"Missing features: {missing}. Filling with zeros.")
                    for m in missing:
                        df_feat_recent[m] = 0.0
                last_row = df_feat_recent.iloc[-1:].reindex(columns=feature_names, fill_value=0.0)
                model_module = type(model_obj).__module__ if model_obj is not None else ""
                if "tensorflow" in model_module or "keras" in model_module:
                    st.warning("Loaded model is TF/Keras — quick recursive routine currently supports sklearn/xgboost only.")
                else:
                    forecasts = []
                    cur_row = last_row.copy()
                    for _ in range(7):
                        X_cur = cur_row.values.astype(float)
                        Xs = scaler_obj.transform(X_cur)
                        pred = float(model_obj.predict(Xs)[0])
                        forecasts.append(pred)
                        lag_cols = sorted([c for c in feature_names if c.startswith("lag_")],
                                          key=lambda x: int(x.split("_")[1])) if any(c.startswith("lag_") for c in feature_names) else []
                        if lag_cols:
                            for i in range(len(lag_cols) - 1, 0, -1):
                                cur_row[lag_cols[i]] = cur_row[lag_cols[i - 1]].values
                            cur_row[lag_cols[0]] = pred
                    last_date = df_recent.index[-1]
                    future_idx = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=7)
                    forecast_df = pd.DataFrame({"Forecast": forecasts}, index=future_idx)
                    plot_hist = pd.concat([df_recent["Close"].tail(60), forecast_df["Forecast"]])
                    st.line_chart(plot_hist)
                    st.table(forecast_df.round(3))

# ---------- Training pipeline ----------
if run_button:
    if not ticker or str(ticker).strip() == "":
        st.error("Enter a ticker.")
    else:
        uploaded_bytes = csv_upload.read() if csv_upload is not None else None
        data = fetch_data(ticker, years, uploaded_csv_bytes=uploaded_bytes, show_diagnostics=True)
        if data.empty:
            av_key_present = False
            try:
                av_key_present = bool(st.secrets.get("ALPHAVANTAGE_API_KEY")) if hasattr(st, "secrets") else bool(os.environ.get("ALPHAVANTAGE_API_KEY"))
            except Exception:
                av_key_present = bool(os.environ.get("ALPHAVANTAGE_API_KEY"))
            if av_key_present:
                st.error("No data fetched. AlphaVantage key present but request failed (rate limit or symbol). Upload CSV or try another ticker.")
            else:
                st.error("No data fetched. Check ticker, network, or upload CSV. To enable AlphaVantage fallback, set ALPHAVANTAGE_API_KEY.")
        else:
            st.success(f"Fetched {len(data)} rows for {ticker}")
            st.markdown("#### Dataframe head")
            st.dataframe(data.head().round(6))
            st.markdown("#### Dataframe tail")
            st.dataframe(data.tail().round(6))

            df_feat = add_lag_features(data, n_lags)
            if add_tech:
                df_feat = add_technical_indicators(df_feat)
            df_feat = df_feat.dropna()
            if df_feat.empty:
                st.error("No features after lag/indicator creation. Adjust lags or data length.")
            else:
                X = df_feat.drop(columns=["Close"])
                y = df_feat["Close"]
                split_idx = int(len(X) * train_frac)
                if split_idx < 1 or split_idx >= len(X):
                    st.error("Train fraction produced invalid split. Adjust it.")
                else:
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    trained = {}
                    results = {}

                    for mname in models_to_train:
                        if mname == "RandomForest":
                            st.info("Training RandomForest...")
                            base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                            if use_tuning and tuning_algo == "RandomizedSearchCV":
                                est, best = run_random_search(base, rf_param_dist, X_train_s, y_train, iters=int(max_evals))
                                st.write("Best RF params:", best)
                            elif use_tuning and tuning_algo.startswith("Optuna") and HAVE_OPTUNA:
                                est, best = run_optuna_rf(X_train_s, y_train, trials=int(max_evals))
                                st.write("Optuna RF params:", best)
                            else:
                                est = base
                            est.fit(X_train_s, y_train)
                            y_pred = est.predict(X_test_s)
                            trained["RandomForest"] = (est, scaler, X.columns.tolist())
                            results["RandomForest"] = compute_metrics(y_test, y_pred)
                            st.write("RF metrics:", results["RandomForest"])

                        elif mname == "XGBoost":
                            if HAVE_XGBOOST:
                                st.info("Training XGBoost...")
                                base = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                                if use_tuning and tuning_algo == "RandomizedSearchCV":
                                    est, best = run_random_search(base, xgb_param_dist, X_train_s, y_train, iters=int(max_evals))
                                    st.write("Best XGB params:", best)
                                else:
                                    est = base
                                est.fit(X_train_s, y_train)
                                y_pred = est.predict(X_test_s)
                                trained["XGBoost"] = (est, scaler, X.columns.tolist())
                                results["XGBoost"] = compute_metrics(y_test, y_pred)
                                st.write("XGBoost metrics:", results["XGBoost"])
                            else:
                                st.info("XGBoost not available; skipping.")

                        elif mname == "GradientBoosting":
                            st.info("Training GradientBoosting...")
                            base = GradientBoostingRegressor(n_estimators=100, random_state=42)
                            if use_tuning and tuning_algo == "RandomizedSearchCV":
                                est, best = run_random_search(base, gb_param_dist, X_train_s, y_train, iters=int(max_evals))
                                st.write("Best GB params:", best)
                            else:
                                est = base
                            est.fit(X_train_s, y_train)
                            y_pred = est.predict(X_test_s)
                            trained["GradientBoosting"] = (est, scaler, X.columns.tolist())
                            results["GradientBoosting"] = compute_metrics(y_test, y_pred)
                            st.write("GradientBoosting metrics:", results["GradientBoosting"])
                        else:
                            st.warning(f"Unknown model: {mname}")

                    if results:
                        metrics_df = pd.DataFrame(results).T
                        st.subheader("Model comparison")
                        st.dataframe(metrics_df[["RMSE", "MAE", "R2"]].sort_values(by="RMSE"))

                    if trained:
                        sel = st.selectbox("Select trained model to inspect or save", list(trained.keys()))
                        if sel:
                            model_obj, model_scaler, feat_list = trained[sel]
                            y_pred = model_obj.predict(model_scaler.transform(X_test))
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(X_test.index, y_test.values, label="Actual")
                            ax.plot(X_test.index, y_pred, label="Predicted")
                            ax.set_title(f"{ticker} — {sel} — Actual vs Predicted")
                            ax.legend()
                            st.pyplot(fig)

                            if st.button(f"Save {sel} to disk"):
                                fname = f"{ticker}_{sel}.joblib"
                                outpath = os.path.join(models_dir, fname)
                                joblib.dump({"model": model_obj, "scaler": model_scaler, "features": feat_list}, outpath)
                                st.success(f"Saved {outpath}")

st.info("Configure inputs and click 'Fetch + Train'. Keep tuning small in interactive mode.")
