
#SARIMAX and LSTM models for food inflation forecasting.

#Import necessary libraries
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

tf.get_logger().setLevel(40)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#-------------------------------
#create sequences for LSTM model
def make_sequences(values, target_values, lookback_value):
    X, y = [], []
    for i in range(len(values) - lookback_value + 1):
        X.append(values[i : i + lookback_value])
        y.append(target_values[i + lookback_value - 1])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)

#-------------------------------
#
def check_stationarity(series):
    #Run ADF test to check if the series needs differencing.
    result = adfuller(series.dropna(), autolag="AIC")
    return float(result[0]), float(result[1])


def pick_sarimax_order(train_series, exog_train, max_p=2, max_q=2, max_d=1):
    # Compare a few SARIMAX orders and keep the one with the lowest AIC
    best_aic = 1e10
    best_order = (0, 1, 0)

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            for d in range(0, max_d + 1):
                try:
                    model = SARIMAX(
                        train_series,
                        exog=exog_train,
                        order=(p, d, q),
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = model.fit(disp=False, maxiter=200)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    return best_order, best_aic


def walk_forward_metrics(train, test, features, order):
    """
    Walk-forward validation.
    """
    hist = train.copy()
    preds = []
    actuals = []

    for i in range(len(test)):
        train_x = hist[features]
        train_y = hist["food_price_inflation"]

        try:
            model = SARIMAX(
                train_y,
                exog=train_x,
                order=order,
                seasonal_order=(0, 1, 0, 12),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False, maxiter=200)

            next_exog = test[features].iloc[[i]]
            fc = res.get_forecast(steps=1, exog=next_exog)
            pred = float(fc.predicted_mean.iloc[0])
        except Exception:
            # If the model fails, just use the last known value
            pred = float(train_y.iloc[-1])

        preds.append(pred)
        actuals.append(float(test["food_price_inflation"].iloc[i]))
        hist = pd.concat([hist, test.iloc[[i]]])

    preds_arr = np.array(preds)
    actuals_arr = np.array(actuals)

    return {
        "mae": float(mean_absolute_error(actuals_arr, preds_arr)),
        "rmse": float(mean_squared_error(actuals_arr, preds_arr) ** 0.5),
        "mape": float(mean_absolute_percentage_error(actuals_arr, preds_arr) * 100),
    }


def train_sarimax(train_data, test_data, features=None):
    # Train SARIMAX model
    if features is None:
        from src.preprocessing import FEATURE_COLS
        features = [c for c in FEATURE_COLS if c in train_data.columns]

    train_x = train_data[features]
    train_y = train_data["food_price_inflation"]
    test_x = test_data[features]

    adf_stat, adf_p = check_stationarity(train_y)

    t0 = time.time()
    order, best_aic = pick_sarimax_order(train_y, train_x)
    print(f"SARIMAX order selected: {order} (AIC={best_aic:.2f}, time={time.time()-t0:.1f}s)")

    seasonal_order = (0, 1, 0, 12)

    model = SARIMAX(
        train_y,
        exog=train_x,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=300)

    forecast = result.get_forecast(steps=len(test_data), exog=test_x)
    predictions = forecast.predicted_mean
    conf_int = forecast.conf_int()
    actual = test_data["food_price_inflation"]
    residuals = actual - predictions

    # Ljung-Box test on residuals - checks if there's remaining autocorrelation
    ljung_box = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)

    wf_metrics = walk_forward_metrics(train_data, test_data, features, order)

    return {
        "model": result,
        "forecast_obj": forecast,
        "predictions": predictions,
        "actual": actual,
        "conf_int": conf_int,
        "residuals": residuals,
        "order": order,
        "seasonal_order": seasonal_order,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "adf_statistic": adf_stat,
        "adf_pvalue": adf_p,
        "ljung_box": {
            "lb_stat": float(ljung_box["lb_stat"].iloc[0]),
            "lb_pvalue": float(ljung_box["lb_pvalue"].iloc[0]),
        },
        "metrics": {
            "mae": float(mean_absolute_error(actual, predictions)),
            "rmse": float(mean_squared_error(actual, predictions) ** 0.5),
            "mape": float(mean_absolute_percentage_error(actual, predictions) * 100),
        },
        "walk_forward_metrics": wf_metrics,
        "features": features,
    }


def forecast_future_sarimax(data, horizon=12, order=None, features=None):
    # Forecast horizon months
    if features is None:
        from src.preprocessing import FEATURE_COLS
        features = [c for c in FEATURE_COLS if c in data.columns]

    frame = data[features + ["food_price_inflation"]].astype(float).copy()

    if order is None:
        order, _ = pick_sarimax_order(frame["food_price_inflation"], frame[features])
    seasonal_order = (0, 1, 0, 12)

    last_exog = frame[features].iloc[[-1]]
    future_exog = pd.concat([last_exog] * horizon, ignore_index=True)

    model = SARIMAX(
        frame["food_price_inflation"],
        exog=frame[features],
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=300)
    forecast = result.get_forecast(steps=horizon, exog=future_exog)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return pred_mean, conf_int


def train_lstm(train_data, test_data, lookback=12, features=None):
    # Train LSTM model
    if features is None:
        from src.preprocessing import FEATURE_COLS
        features = [c for c in FEATURE_COLS if c in train_data.columns]

    train_features = train_data[features].to_numpy(dtype=np.float64)
    train_target = train_data["food_price_inflation"].to_numpy(dtype=np.float64)
    test_features = test_data[features].to_numpy(dtype=np.float64)
    test_target = test_data["food_price_inflation"].to_numpy(dtype=np.float64)

    # Scale features and target separately
    # This lets us inverse-transform predictions back to original units
    feature_scaler = MinMaxScaler()
    scaled_train = feature_scaler.fit_transform(train_features)
    scaled_test = feature_scaler.transform(test_features)

    target_scaler = MinMaxScaler()
    scaled_train_target = target_scaler.fit_transform(train_target.reshape(-1, 1)).flatten()
    scaled_test_target = target_scaler.transform(test_target.reshape(-1, 1)).flatten()

    X_train, y_train = make_sequences(scaled_train, scaled_train_target, lookback)

    # For test, prepend the last training rows so the first test month has context
    combined_values = np.concatenate([scaled_train[-lookback:], scaled_test], axis=0)
    combined_targets = np.concatenate([scaled_train_target[-lookback:], scaled_test_target], axis=0)
    X_test_full, y_test_full_scaled = make_sequences(combined_values, combined_targets, lookback)
    X_test = X_test_full[-len(test_data):]
    y_test_scaled = y_test_full_scaled[-len(test_data):]

    tf.random.set_seed(42)
    np.random.seed(42)

    # Two LSTM layers with dropout to prevent overfitting on small data
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    checkpoint_path = "best_lstm_model.keras"

    # Stop training if validation loss stops improving
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=8,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0,
    )

    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        os.remove(checkpoint_path)

    # Predict and convert back to original units
    test_pred_scaled = model.predict(X_test, verbose=0).ravel()
    test_pred = target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    y_test_inv = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    full_predictions = np.full(len(test_data), np.nan)
    full_predictions[-len(test_pred):] = test_pred

    return {
        "model": model,
        "predictions": full_predictions,
        "actual": y_test_inv,
        "history": history,
        "metrics": {
            "mae": float(mean_absolute_error(y_test_inv, test_pred)),
            "rmse": float(mean_squared_error(y_test_inv, test_pred) ** 0.5),
            "mape": float(mean_absolute_percentage_error(y_test_inv, test_pred) * 100),
        },
        "scaler": target_scaler,
        "feature_scaler": feature_scaler,
    }


def forecast_future_lstm(data, horizon=12, lookback=12, features=None):
    # Forecast horizon months ahead
    if features is None:
        from src.preprocessing import FEATURE_COLS
        features = [c for c in FEATURE_COLS if c in data.columns]

    all_features = data[features].to_numpy(dtype=np.float64)
    all_target = data["food_price_inflation"].to_numpy(dtype=np.float64)

    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(all_features)

    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(all_target.reshape(-1, 1)).flatten()

    X_train, y_train = make_sequences(scaled_features, scaled_target, lookback)

    tf.random.set_seed(42)
    np.random.seed(42)

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    checkpoint_path = "best_lstm_model.keras"
    callbacks = [
        EarlyStopping(monitor="loss", patience=10, restore_best_weights=True, verbose=0),
        ModelCheckpoint(checkpoint_path, monitor="loss", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
    ]

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=8,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0,
    )

    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        os.remove(checkpoint_path)

    # Forecast horizon steps ahead by rolling predictions
    last_sequence = scaled_features[-lookback:].copy()
    forecasts_scaled = []

    for _ in range(horizon):
        input_seq = last_sequence.reshape(1, lookback, -1)
        pred_scaled = model.predict(input_seq, verbose=0).ravel()[0]
        forecasts_scaled.append(pred_scaled)

        # Append prediction and slide the window
        new_row = last_sequence[-1].copy()
        last_sequence = np.vstack([last_sequence[1:], new_row])

    forecasts = target_scaler.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1)).flatten()
    return forecasts


if __name__ == "__main__":
    from src.preprocessing import add_features, load_data, split_train_test

    data = load_data()
    data = add_features(data)
    train, test = split_train_test(data)

    sarimax_result = train_sarimax(train, test)
    print("SARIMAX order:", sarimax_result["order"])
    print("SARIMAX metrics")
    for k, v in sarimax_result["metrics"].items():
        print(f"  {k}: {v:.3f}")

    lstm_result = train_lstm(train, test)
    if lstm_result is not None:
        print("LSTM metrics")
        for k, v in lstm_result["metrics"].items():
            print(f"  {k}: {v:.3f}")
