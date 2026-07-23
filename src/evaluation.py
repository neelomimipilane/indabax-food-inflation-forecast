from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from models import forecast_future_sarimax


def _load_hcp_data():
    hcp_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "05_human_capital_project.csv"
    hcp = pd.read_csv(hcp_path, parse_dates=["Date"])
    hcp = hcp[hcp["REF_AREA"] == "BWA"]
    hcp = hcp[hcp["INDICATOR"].isin(["FAO_CP_23012", "FAO_CP_23013"])]
    hcp = hcp[["Date", "INDICATOR", "Value"]]
    hcp = hcp.pivot(index="Date", columns="INDICATOR", values="Value").reset_index()
    hcp["month"] = hcp["Date"].dt.to_period("M")
    hcp = hcp.groupby("month")[["FAO_CP_23012", "FAO_CP_23013"]].mean()
    hcp.index = hcp.index.to_timestamp()
    return hcp


def _align_with_monthly_index(data):
    aligned = data.copy()
    aligned.index = aligned.index.to_period("M").to_timestamp()
    return aligned


def _write_predictions_csv(output_path, forecast_dates, forecasts):
    forecast_values = pd.to_numeric(pd.Series(forecasts), errors="coerce")
    if forecast_values.isna().any():
        raise ValueError("Forecast values contain missing values")

    prediction_frame = pd.DataFrame(
        {
            "year_month": forecast_dates.strftime("%Y-%m"),
            "forecast": forecast_values.astype(float).to_numpy(),
        }
    )
    prediction_frame.to_csv(output_path, index=False)

    written_frame = pd.read_csv(output_path)
    if len(written_frame.columns) != 2:
        raise ValueError(f"Predictions CSV must have exactly 2 columns, found {len(written_frame.columns)}")
    if list(written_frame.columns) != ["year_month", "forecast"]:
        raise ValueError(f"Predictions CSV columns must be ['year_month', 'forecast'], found {list(written_frame.columns)}")
    if len(written_frame) != 12:
        raise ValueError(f"Predictions CSV must have exactly 12 rows, found {len(written_frame)}")
    if written_frame["forecast"].isna().any():
        raise ValueError("Predictions CSV contains missing forecast values")

    return written_frame


def _build_submission_forecast(full_data, horizon=12, start_date="2024-01-01"):
    forecast_values = forecast_future_sarimax(full_data, horizon=horizon)
    forecast_values = pd.to_numeric(forecast_values, errors="coerce")
    if forecast_values.isna().any():
        raise ValueError("Future forecast contains missing values")

    forecast_dates = pd.date_range(start=start_date, periods=horizon, freq="MS")
    return forecast_dates, forecast_values.astype(float).to_numpy()


def evaluate_models(sarimax_result, lstm_result, full_data, test_data, figures_dir, outputs_dir):
    figures_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    sarimax_predictions = pd.Series(sarimax_result["predictions"], index=test_data.index)

    if lstm_result is not None:
        lstm_actual = pd.Series(lstm_result["actual"], index=test_data.index[-len(lstm_result["actual"]):])
        lstm_predictions = pd.Series(lstm_result["predictions"], index=lstm_actual.index)
    else:
        lstm_actual = None
        lstm_predictions = None

    # Export the single submission forecast 
    forecast_dates, future_forecasts = _build_submission_forecast(full_data, horizon=12, start_date="2024-01-01")
    _write_predictions_csv(outputs_dir / "predictions.csv", forecast_dates, future_forecasts)

    plot_predictions(
        test_data["food_price_inflation"], sarimax_predictions, "SARIMAX", figures_dir / "sarimax_predictions.png"
    )
    plot_residuals(
        test_data["food_price_inflation"], sarimax_predictions, "SARIMAX", figures_dir / "sarimax_residuals.png"
    )
    plot_historical_inflation(test_data, figures_dir / "historical_food_inflation.png")
    plot_historical_bdi(test_data, figures_dir / "historical_bdi.png")
    plot_hcp_relationship(test_data, figures_dir / "hcp_relationship.png")
    plot_hcp_projection(test_data, figures_dir / "hcp_projection.png")

    if lstm_actual is not None and lstm_predictions is not None:
        plot_predictions(lstm_actual, lstm_predictions, "LSTM", figures_dir / "lstm_predictions.png")
        plot_residuals(lstm_actual, lstm_predictions, "LSTM", figures_dir / "lstm_residuals.png")

    print("SARIMAX metrics")
    print_metrics(sarimax_result["metrics"])
    if lstm_result is not None:
        print("LSTM metrics")
        print_metrics(lstm_result["metrics"])


def plot_predictions(actual, predictions, name, output_path):
    plt.figure(figsize=(10, 4))
    plt.plot(actual.index, actual.values, label="Actual", color="black")
    plt.plot(predictions.index, predictions.values, label="Predicted", color="tab:red", linestyle="--")
    plt.title(f"{name}: Actual vs Predicted Food Price Inflation")
    plt.xlabel("Month")
    plt.ylabel("Food Price Inflation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residuals(actual, predictions, name, output_path):
    residuals = pd.Series(actual.values - predictions.values, index=actual.index)
    plt.figure(figsize=(10, 4))
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.plot(residuals.index, residuals.values, color="tab:blue")
    plt.title(f"{name} Residual Plot")
    plt.xlabel("Month")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_historical_inflation(data, output_path):
    plt.figure(figsize=(10, 4))
    plt.plot(data.index, data["food_price_inflation"], color="tab:blue")
    plt.title("Historical Food Price Inflation")
    plt.xlabel("Month")
    plt.ylabel("Inflation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_historical_bdi(data, output_path):
    plt.figure(figsize=(10, 4))
    plt.plot(data.index, data["bdi_mean"], color="tab:green")
    plt.title("Historical BDI Mean")
    plt.xlabel("Month")
    plt.ylabel("BDI Mean")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_hcp_relationship(data, output_path):
    hcp = _load_hcp_data()
    merged = _align_with_monthly_index(data).join(hcp, how="left")
    merged = merged.dropna()

    if merged.empty:
        print("Skipping hcp_relationship.png: no overlapping data between model data and HCP indicators")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(merged["FAO_CP_23012"], merged["food_price_inflation"])
    axes[0].set_title("Inflation vs Indicator 23012")
    axes[0].set_xlabel("Indicator 23012")
    axes[0].set_ylabel("Food Inflation")

    axes[1].scatter(merged["FAO_CP_23013"], merged["food_price_inflation"])
    axes[1].set_title("Inflation vs Indicator 23013")
    axes[1].set_xlabel("Indicator 23013")
    axes[1].set_ylabel("Food Inflation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_hcp_projection(data, output_path):
    hcp = _load_hcp_data()
    merged = _align_with_monthly_index(data).join(hcp, how="left").dropna()

    if merged.empty:
        print("Skipping hcp_projection.png: no overlapping data between model data and HCP indicators")
        return

    inflation = merged["food_price_inflation"].to_numpy()
    x = np.column_stack([np.ones(len(inflation)), inflation])
    y1 = merged["FAO_CP_23012"].to_numpy()
    y2 = merged["FAO_CP_23013"].to_numpy()
    model1 = sm.OLS(y1, x).fit()
    model2 = sm.OLS(y2, x).fit()

    future_inflation = np.full(12, merged["food_price_inflation"].iloc[-1])
    future_x = np.column_stack([np.ones(12), future_inflation])
    projected1 = model1.predict(future_x)
    projected2 = model2.predict(future_x)

    plt.figure(figsize=(10, 4))
    plt.plot(pd.date_range("2024-01-01", periods=12, freq="MS"), projected1, label="Indicator 23012", color="tab:blue")
    plt.plot(pd.date_range("2024-01-01", periods=12, freq="MS"), projected2, label="Indicator 23013", color="tab:red")
    plt.title("Projected HCP Indicators")
    plt.xlabel("Month")
    plt.ylabel("Projected Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_metrics(metrics):
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    from preprocessing import add_features, load_data, split_train_test
    from models import train_lstm, train_sarimax

    data = load_data()
    features = add_features(data)
    train, test = split_train_test(features)
    sarimax_result = train_sarimax(train, test)
    root = Path(__file__).resolve().parent.parent
    evaluate_models(sarimax_result, None, features, test, root / "figures", root / "outputs")
