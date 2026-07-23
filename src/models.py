import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential
except ImportError:
    EarlyStopping = None
    Dense = None
    Dropout = None
    LSTM = None
    Sequential = None


def check_stationarity(series):
    # Run ADF test to check if the series needs differencing
    adf_result = adfuller(series.dropna(), autolag="AIC")
    return adf_result


def pick_sarimax_order(train_series, exog_train, max_p=2, max_q=2, max_d=1):
    # Small grid search to pick (p,d,q) based on AIC
    # A student would try a few combinations, not just guess
    best_aic = 1e10
    best_order = (1, 0, 1)

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
                    print(f"Skipping SARIMAX order ({p},{d},{q}): fit failed")
                    continue

    return best_order, best_aic


def train_sarimax(train_data, test_data):
    features = [
        "brent_price",
        "general_cpi",
        "food_cpi",
        "bdi_mean",
        "bdi_std",
        "bdi_range",
        "bdi_momentum_3",
        "bdi_momentum_6",
        "bdi_trend",
        "bdi_extreme_days",
        "inflation_lag_1",
        "inflation_lag_3",
        "inflation_lag_6",
        "inflation_lag_12",
        "brent_lag_1",
        "bdi_lag_1",
        "inflation_roll3_mean",
        "inflation_roll6_mean",
        "inflation_roll3_std",
    ]

    train_x = train_data[features]
    train_y = train_data["food_price_inflation"]
    test_x = test_data[features]

    # Check stationarity
    adf_result = check_stationarity(train_y)
    adf_stat = adf_result[0]
    adf_p = adf_result[1]

    # Let the grid search decide d
    order, best_aic = pick_sarimax_order(train_y, train_x)

    model = SARIMAX(
        train_y,
        exog=train_x,
        order=order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)

    forecast = result.get_forecast(steps=len(test_data), exog=test_x)
    predictions = forecast.predicted_mean
    actual = test_data["food_price_inflation"]
    residuals = actual - predictions

    return {
        "predictions": predictions,
        "actual": actual,
        "order": order,
        "aic": result.aic,
        "diagnostics": {
            "adf_statistic": adf_stat,
            "adf_pvalue": adf_p,
            "residual_mean": residuals.mean(),
            "acf_value": residuals.autocorr(lag=1),
            "bic": result.bic,
        },
        "metrics": {
            "mae": mean_absolute_error(actual, predictions),
            "rmse": mean_squared_error(actual, predictions) ** 0.5,
            "mape": mean_absolute_percentage_error(actual, predictions) * 100,
        },
    }


def forecast_future_sarimax(data, horizon=12):
    features = [
        "brent_price",
        "general_cpi",
        "food_cpi",
        "bdi_mean",
        "bdi_std",
        "bdi_range",
        "bdi_momentum_3",
        "bdi_momentum_6",
        "bdi_trend",
        "bdi_extreme_days",
        "inflation_lag_1",
        "inflation_lag_3",
        "inflation_lag_6",
        "inflation_lag_12",
        "brent_lag_1",
        "bdi_lag_1",
        "inflation_roll3_mean",
        "inflation_roll6_mean",
        "inflation_roll3_std",
    ]

    frame = data[features + ["food_price_inflation"]].astype(float).copy()
    order, _ = pick_sarimax_order(frame["food_price_inflation"], frame[features])

    # The submission horizon is fixed to a full year into the future.
    # We repeat the latest observed exogenous values for the forecast window.
    last_exog = frame[features].iloc[[-1]]
    future_exog = pd.concat([last_exog] * horizon, ignore_index=True)

    model = SARIMAX(
        frame["food_price_inflation"],
        exog=frame[features],
        order=order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    forecast = result.get_forecast(steps=horizon, exog=future_exog)
    return forecast.predicted_mean


def train_lstm(train_data, test_data, lookback=12):
    features = [
        "brent_price",
        "general_cpi",
        "food_cpi",
        "bdi_mean",
        "bdi_std",
        "bdi_range",
        "bdi_momentum_3",
        "bdi_momentum_6",
        "bdi_trend",
        "bdi_extreme_days",
        "inflation_lag_1",
        "inflation_lag_3",
        "inflation_lag_6",
        "inflation_lag_12",
        "brent_lag_1",
        "bdi_lag_1",
        "inflation_roll3_mean",
        "inflation_roll6_mean",
        "inflation_roll3_std",
    ]

    train_features = train_data[features].to_numpy()
    train_target = train_data["food_price_inflation"].to_numpy()
    test_features = test_data[features].to_numpy()
    test_target = test_data["food_price_inflation"].to_numpy()

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_test = scaler.transform(test_features)

    # make sequences: each sample uses `lookback` months of features to predict next month
    def make_sequences(values, target_values, lookback_value):
        X, y = [], []
        for i in range(len(values) - lookback_value + 1):
            X.append(values[i : i + lookback_value])
            y.append(target_values[i + lookback_value - 1])
        return np.array(X), np.array(y)

    X_train, y_train = make_sequences(scaled_train, train_target, lookback)

    # For test, we need the last `lookback` training rows so the first test month has context
    combined_values = np.concatenate([scaled_train[-lookback:], scaled_test], axis=0)
    combined_targets = np.concatenate([train_target[-lookback:], test_target], axis=0)
    X_test_full, y_test_full = make_sequences(combined_values, combined_targets, lookback)
    X_test = X_test_full[-len(test_data):]
    y_test = y_test_full[-len(test_data):]

    if Sequential is None:
        print("TensorFlow is not installed. Cannot train LSTM.")
        return None

    # Small model because we have limited data
    # lookback=12 means we use one year of monthly data to predict next month
    # 32 units in LSTM is small enough to not overfit
    # dropout 0.2 helps regularize
    # epochs 30 with early stopping usually converges fast on small data
    # batch size 16 is a reasonable default for small datasets
    model = Sequential([
        LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0,
    )

    predictions = model.predict(X_test, verbose=0).ravel()

    return {
        "predictions": predictions,
        "actual": y_test,
        "metrics": {
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": mean_squared_error(y_test, predictions) ** 0.5,
            "mape": mean_absolute_percentage_error(y_test, predictions) * 100,
        },
    }


if __name__ == "__main__":
    from preprocessing import add_features, load_data, split_train_test

    data = load_data()
    features = add_features(data)
    train, test = split_train_test(features)
    sarimax_result = train_sarimax(train, test)
    print("SARIMAX order:", sarimax_result["order"])
    print("SARIMAX metrics")
    for k, v in sarimax_result["metrics"].items():
        print(f"  {k}: {v:.3f}")
