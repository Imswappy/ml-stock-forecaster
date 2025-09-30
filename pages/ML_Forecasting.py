# pages/ML_Forecasting.py
import os
import time

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

# OPTIONAL LIBS (graceful fallback)
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

# --------- Streamlit page config MUST be the first Streamlit call ----------
st.set_page_config(page_title="ML Style Forecasting (Tuning + Save)", page_icon="🤖", layout="wide")
st.title("ML-Style Forecasting — Tuning, Train & Save")

# --------- Saved model loader (uses models/ folder) ----------
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)  # ensure folder exists

saved_models = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
selected_saved = st.selectbox("Or load a saved model", ["(none)"] + saved_models)
LOADED_MODEL = None
if selected_saved and selected_saved != "(none)":
    bundle_path = os.path.join(models_dir, selected_saved)
    try:
        bundle = joblib.load(bundle_path)
        LOADED_MODEL = {"model": bundle.get("model"), "scaler": bundle.get("scaler"), "features": bundle.get("features")}
        st.success(f"Loaded {selected_saved} — ready for inference")
        st.write("Features expected by this model:", LOADED_MODEL["features"])
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# --------- UI inputs ----------
# support prefill via query params
params = st.experimental_get_query_params()
prefill_ticker = params.get("ticker", [None])[0]
if prefill_ticker:
    prefill_ticker = str(prefill_ticker).upper()

# Popular tickers list (you can edit this)
popular = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "JPM", "UNH"]
options = popular + ["Other"]

# determine default index based on prefill
if prefill_ticker and prefill_ticker in popular:
    default_index = popular.index(prefill_ticker)
elif prefill_ticker and prefill_ticker not in popular:
    default_index = len(popular)  # "Other"
else:
    default_index = 0

col1, col2, col3 = st.columns([1.5, 1, 1])
with col1:
    sel = st.selectbox("Ticker", options=options, index=default_index)
    if sel == "Other":
        # prefill custom input if query param provided and not in popular
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

st.markdown("### Tuning options (compute heavy)")
use_tuning = st.checkbox("Enable hyperparameter tuning?", value=False)
tuning_algo = st.selectbox("Tuning algorithm", options=["RandomizedSearchCV", "Optuna (if installed)"])
max_evals = st.number_input("Max trials / iterations (keep small for demo)", min_value=5, max_value=200, value=20, step=5)

st.markdown("### Model choices")
default_models = ["RandomForest"]
if HAVE_XGBOOST:
    available = ["RandomForest", "GradientBoosting", "XGBoost"]
else:
    available = ["RandomForest", "GradientBoosting"]
models_to_train = st.multiselect("Train these models", available, default=default_models)

run_button = st.button("Fetch + Train (may take time)")

# --------- Helpers ----------
@st.cache_data(ttl=3600)
def fetch_data(ticker, years):
    df = yf.download(ticker, period=f"{years}y", progress=False)
    if df.empty:
        return df
    df = df[['Close', 'Open', 'High', 'Low', 'Volume']].dropna()
    return df

def add_lag_features(df, n_lags):
    df2 = df.copy()
    for lag in range(1, n_lags+1):
        df2[f'lag_{lag}'] = df2['Close'].shift(lag)
    return df2

def add_technical_indicators(df):
    df2 = df.copy()
    df2['SMA_10'] = df2['Close'].rolling(10).mean()
    df2['SMA_50'] = df2['Close'].rolling(50).mean()
    delta = df2['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df2['RSI_14'] = 100 - (100 / (1 + rs))
    return df2

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# param grids
rf_param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
}
gb_param_dist = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
}
xgb_param_dist = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
}

def run_random_search(estimator, param_dist, Xtr, ytr, iters=20):
    rs = RandomizedSearchCV(estimator, param_distributions=param_dist,
                            n_iter=min(iters, 20), cv=3, scoring='neg_root_mean_squared_error',
                            n_jobs=-1, random_state=42, verbose=0)
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
        scores = cross_val_score(model, Xtr, ytr, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
        return -scores.mean()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=min(trials, 50))
    best = study.best_params
    model = RandomForestRegressor(**best, n_jobs=-1, random_state=42)
    model.fit(Xtr, ytr)
    return model, best

# ---------- Auto-forecast & quick inference (if a saved model is loaded) ----------
if LOADED_MODEL is not None:
    st.markdown("---")
    st.subheader("Quick inference from loaded model")
    st.write("This will fetch recent data, build features matching the saved model, and do a 7-day recursive forecast.")
    if 'features' not in LOADED_MODEL or LOADED_MODEL['features'] is None:
        st.warning("Loaded model does not contain a `features` list — cannot auto-build features.")
    else:
        do_quick = st.button("Auto-forecast 7 days with loaded model")
        if do_quick:
            with st.spinner("Fetching recent market data and running forecast..."):
                try:
                    # fetch a bit more history to compute indicators / lags
                    recent_days = max(60, n_lags * 3)
                    df_recent = fetch_data(ticker, 1)  # 1 year is fine; we'll pick tail
                    if df_recent.empty:
                        st.error("Failed to fetch recent data for ticker.")
                    else:
                        df_recent = df_recent.tail(recent_days).copy()
                        # build features same way we do for training
                        df_feat_recent = add_lag_features(df_recent, n_lags)
                        if add_tech:
                            df_feat_recent = add_technical_indicators(df_feat_recent)
                        df_feat_recent = df_feat_recent.dropna()
                        if df_feat_recent.empty:
                            st.error("Not enough recent data to build features. Try increasing 'Years of data' or reducing lag count.")
                        else:
                            feature_names = LOADED_MODEL['features']
                            model_obj = LOADED_MODEL['model']
                            scaler_obj = LOADED_MODEL['scaler']

                            # Ensure all saved features are present (if not, try to fill missing with zeros)
                            missing = [f for f in feature_names if f not in df_feat_recent.columns]
                            if missing:
                                st.warning(f"Some features from saved model are missing in recent data: {missing}. Filling with zeros.")
                                for m in missing:
                                    df_feat_recent[m] = 0.0
                            # Prepare last feature row
                            last_row = df_feat_recent.iloc[-1:].copy()
                            # order columns as saved
                            last_row = last_row.reindex(columns=feature_names, fill_value=0.0)

                            # Determine model type (sklearn/xgboost vs TF)
                            model_module = type(model_obj).__module__ if model_obj is not None else ""
                            if 'tensorflow' in model_module or 'keras' in model_module:
                                st.warning("Loaded model appears to be a TF/Keras model. This quick recursive routine currently supports sklearn/xgboost tree models. For LSTM saved-model inference, use a dedicated routine.")
                            else:
                                # recursive 7-day forecast
                                forecasts = []
                                cur_row = last_row.copy()
                                for step in range(7):
                                    X_cur = cur_row.values.astype(float)
                                    # scale
                                    Xs = scaler_obj.transform(X_cur)
                                    pred = model_obj.predict(Xs)[0]
                                    forecasts.append(pred)

                                    # update cur_row: shift lag columns if present
                                    lag_cols = [c for c in feature_names if c.startswith('lag_')]
                                    lag_cols_sorted = sorted(lag_cols, key=lambda x: int(x.split('_')[1])) if lag_cols else []
                                    if lag_cols_sorted:
                                        # shift: lag_n = lag_{n-1}, lag_1 = pred
                                        for i in range(len(lag_cols_sorted)-1, 0, -1):
                                            cur_row[lag_cols_sorted[i]] = cur_row[lag_cols_sorted[i-1]].values
                                        cur_row[lag_cols_sorted[0]] = pred

                                    # for SMA/RSI features we keep them unchanged (approx) for short horizon
                                    # if stored features include date-like columns, ignore them (we only used numeric features)

                                # Build forecast DataFrame with dates
                                last_date = df_recent.index[-1]
                                future_idx = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=7)  # business days
                                forecast_df = pd.DataFrame({'Forecast': forecasts}, index=future_idx)

                                # Plot last 60 days and forecast
                                plot_hist = pd.concat([df_recent['Close'].tail(60), forecast_df['Forecast']])
                                st.line_chart(plot_hist)

                                st.write("7-day forecast (business days):")
                                st.table(forecast_df.round(3))

                                st.success("Quick forecast completed.")
                except Exception as e:
                    st.error(f"Quick forecast failed: {e}")

# --------- Training pipeline ----------
if run_button:
    data = fetch_data(ticker, years)
    if data.empty:
        st.error("No data fetched. Check ticker or network.")
        st.stop()
    st.success(f"Fetched {len(data)} rows for {ticker}")
    st.write(data.tail())

    df_feat = add_lag_features(data, n_lags)
    if add_tech:
        df_feat = add_technical_indicators(df_feat)
    df_feat = df_feat.dropna()
    X = df_feat.drop(columns=['Close'])
    y = df_feat['Close']

    split_idx = int(len(X) * train_frac)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    trained = {}
    results = {}

    for mname in models_to_train:
        if mname == "RandomForest":
            st.info("Preparing RandomForest...")
            base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            if use_tuning and tuning_algo == "RandomizedSearchCV":
                st.info("Running RandomizedSearchCV for RF (limited iterations)...")
                est, best = run_random_search(base, rf_param_dist, X_train_s, y_train, iters=int(max_evals))
                st.write("Best RF params:", best)
            elif use_tuning and tuning_algo == "Optuna (if installed)" and HAVE_OPTUNA:
                st.info("Running Optuna for RF (may take some time)...")
                est, best = run_optuna_rf(X_train_s, y_train, trials=int(max_evals))
                st.write("Optuna RF best params:", best)
            else:
                est = base
            est.fit(X_train_s, y_train)
            y_pred = est.predict(X_test_s)
            trained['RandomForest'] = (est, scaler, X.columns.tolist())
            results['RandomForest'] = compute_metrics(y_test, y_pred)
            st.write("RF metrics:", results['RandomForest'])

        elif mname == "XGBoost":
            if HAVE_XGBOOST:
                st.info("Preparing XGBoost...")
                base = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                if use_tuning and tuning_algo == "RandomizedSearchCV":
                    est, best = run_random_search(base, xgb_param_dist, X_train_s, y_train, iters=int(max_evals))
                    st.write("Best XGB params:", best)
                else:
                    est = base
                est.fit(X_train_s, y_train)
                y_pred = est.predict(X_test_s)
                trained['XGBoost'] = (est, scaler, X.columns.tolist())
                results['XGBoost'] = compute_metrics(y_test, y_pred)
                st.write("XGBoost metrics:", results['XGBoost'])
            else:
                st.info("XGBoost not available; skipping XGBoost.")

        elif mname == "GradientBoosting":
            st.info("Preparing GradientBoosting...")
            base = GradientBoostingRegressor(n_estimators=100, random_state=42)
            if use_tuning and tuning_algo == "RandomizedSearchCV":
                est, best = run_random_search(base, gb_param_dist, X_train_s, y_train, iters=int(max_evals))
                st.write("Best GB params:", best)
            else:
                est = base
            est.fit(X_train_s, y_train)
            y_pred = est.predict(X_test_s)
            trained['GradientBoosting'] = (est, scaler, X.columns.tolist())
            results['GradientBoosting'] = compute_metrics(y_test, y_pred)
            st.write("GradientBoosting metrics:", results['GradientBoosting'])
        else:
            st.warning(f"Unknown model requested: {mname}")

    # Show comparison table
    if results:
        metrics_df = pd.DataFrame(results).T
        st.subheader("Model comparison")
        st.dataframe(metrics_df[['RMSE', 'MAE', 'R2']].sort_values(by='RMSE'))

    # Inspect & save
    if trained:
        sel = st.selectbox("Select a trained model to inspect or save", list(trained.keys()))
        if sel:
            model_obj, model_scaler, feat_list = trained[sel]
            y_pred = model_obj.predict(model_scaler.transform(X_test))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(X_test.index, y_test.values, label='Actual')
            ax.plot(X_test.index, y_pred, label='Predicted')
            ax.set_title(f"{ticker} — {sel} — Actual vs Predicted")
            ax.legend()
            st.pyplot(fig)

            if st.button(f"Save {sel} (sklearn/xgboost) to disk"):
                fname = f"{ticker}_{sel}.joblib"
                outpath = os.path.join(models_dir, fname)
                joblib.dump({"model": model_obj, "scaler": model_scaler, "features": feat_list}, outpath)
                st.success(f"Saved model bundle as {outpath}")
                st.info("You can now use the FastAPI server to load this .joblib and serve predictions.")

    st.success("Training & tuning completed (interactive demo). For large tuning, run offline or on cloud instances.")
else:
    st.info("Configure inputs and click `Fetch + Train` to run training. If enabling tuning, keep max trials small in interactive mode.")
