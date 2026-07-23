"""
Main pipeline that runs the full analysis:
1. Loads and prepares data
2. Trains SARIMAX and LSTM models
3. Compares performance
4. Generates submission forecast
5. Produces all outputs and figures
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress convergence warnings - expected with small monthly datasets
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.models import forecast_future_lstm, forecast_future_sarimax, train_lstm, train_sarimax
from src.preprocessing import (
    FEATURE_COLS,
    add_features,
    evaluate_lags,
    get_exog_cols,
    load_data,
    select_best_lags,
    split_train_test,
)

FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")


def load_hcp_data():
    """Load Botswana human capital indicators from the raw data."""
    hcp_path = os.path.join(ROOT_DIR, "data", "raw", "05_human_capital_project.csv")
    hcp = pd.read_csv(hcp_path, parse_dates=["Date"])
    hcp = hcp[hcp["REF_AREA"] == "BWA"]
    hcp = hcp[hcp["INDICATOR"].isin(["FAO_CP_23012", "FAO_CP_23013"])]
    hcp = hcp[["Date", "INDICATOR", "Value"]]
    hcp = hcp.pivot(index="Date", columns="INDICATOR", values="Value").reset_index()
    hcp["month"] = hcp["Date"].dt.to_period("M")
    hcp = hcp.groupby("month")[["FAO_CP_23012", "FAO_CP_23013"]].mean()
    hcp.index = hcp.index.to_timestamp()
    return hcp


def align_monthly(data):
    """Align data to monthly timestamp index."""
    aligned = data.copy()
    aligned.index = aligned.index.to_period("M").to_timestamp()
    return aligned


def save_predictions(output_path, forecast_dates, forecasts):
    """Save and validate the submission predictions CSV."""
    forecast_values = pd.to_numeric(pd.Series(forecasts), errors="coerce")
    if forecast_values.isna().any():
        raise ValueError("Forecast values contain missing values")

    frame = pd.DataFrame(
        {
            "year_month": forecast_dates.strftime("%Y-%m"),
            "forecast": forecast_values.to_numpy(dtype=float),
        }
    )
    frame.to_csv(output_path, index=False)

    # Validate the file matches the submission format exactly
    written = pd.read_csv(output_path)
    if len(written.columns) != 2:
        raise ValueError(f"Predictions CSV must have exactly 2 columns, found {len(written.columns)}")
    if list(written.columns) != ["year_month", "forecast"]:
        raise ValueError(f"Predictions CSV columns must be ['year_month', 'forecast'], found {list(written.columns)}")
    if len(written) != 12:
        raise ValueError(f"Predictions CSV must have exactly 12 rows, found {len(written)}")
    if written["forecast"].isna().any():
        raise ValueError("Predictions CSV contains missing forecast values")

    return frame


def build_submission_forecast(full_data, horizon=12, sarimax_order=None, lstm_result=None, features=None, sarimax_rmse=float("inf"), lstm_rmse=float("inf")):
    """
    Generate the final 12-month forecast using the better-performing model.
    Compares SARIMAX and LSTM RMSE on the test set.
    """
    sarimax_preds = None
    lstm_preds = None

    if sarimax_order is not None:
        sarimax_preds, _ = forecast_future_sarimax(full_data, horizon=horizon, order=sarimax_order)

    if lstm_result is not None:
        lstm_preds = forecast_future_lstm(full_data, horizon=horizon, features=features)

    # Use the model with lower holdout RMSE
    if lstm_rmse < sarimax_rmse and lstm_preds is not None:
        chosen = lstm_preds
        chosen_name = "LSTM"
    elif sarimax_preds is not None:
        chosen = sarimax_preds
        chosen_name = "SARIMAX"
    else:
        raise ValueError("No forecast available from either model")

    chosen = pd.to_numeric(pd.Series(chosen), errors="coerce")
    if chosen.isna().any():
        raise ValueError(f"{chosen_name} forecast contains missing values")

    dates = pd.date_range(start="2024-01-01", periods=horizon, freq="MS")
    print(f"Using {chosen_name} for submission forecast")
    return dates, chosen.to_numpy(dtype=float)


def save_metrics(output_path, sarimax_metrics, lstm_metrics, sarimax_wf=None, lstm_wf=None):
    """Write model comparison metrics to CSV."""
    rows = []

    for name, metrics in [("SARIMAX", sarimax_metrics), ("LSTM", lstm_metrics)]:
        if metrics is None:
            continue
        row = {"model": name}
        row.update({k: f"{v:.4f}" for k, v in metrics.items()})
        row.update({"test_split": "holdout"})
        rows.append(row)

    for name, metrics in [("SARIMAX", sarimax_wf), ("LSTM", lstm_wf)]:
        if metrics is None:
            continue
        row = {"model": name}
        row.update({k: f"{v:.4f}" for k, v in metrics.items()})
        row.update({"test_split": "walk_forward"})
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_feature_importance(output_path, train_data, features):
    """Save feature importance based on correlation with target."""
    target = train_data["food_price_inflation"]
    rows = []

    for feature in features:
        corr = train_data[feature].corr(target)
        importance = abs(corr) if not pd.isna(corr) else 0.0
        rows.append({"feature": feature, "importance": round(float(importance), 4)})

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("importance", ascending=False)
    frame.to_csv(output_path, index=False)


def plot_predictions(actual, predictions, name, output_path):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 4))
    plt.plot(actual.index, actual.values, label="Actual", color="black", linewidth=1.5)
    plt.plot(
        predictions.index,
        predictions.values,
        label="Predicted",
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
    )
    plt.title(f"{name}: Actual vs Predicted Food Price Inflation")
    plt.xlabel("Month")
    plt.ylabel("Food Price Inflation (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residuals(actual, predictions, name, output_path):
    """Plot residuals to check for patterns."""
    residuals = pd.Series(actual.values - predictions.values, index=actual.index)
    plt.figure(figsize=(10, 4))
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.plot(residuals.index, residuals.values, color="tab:blue", linewidth=1.5)
    plt.fill_between(residuals.index, residuals.values, 0, alpha=0.3, color="tab:blue")
    plt.title(f"{name} Residual Plot")
    plt.xlabel("Month")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_forecast(full_data, future_dates, forecasts, output_path):
    """Plot historical data and future forecast together."""
    plt.figure(figsize=(10, 5))
    hist = full_data["food_price_inflation"].iloc[-24:]
    plt.plot(hist.index, hist.values, label="Historical", color="black", linewidth=1.5)
    plt.plot(
        future_dates,
        forecasts,
        label="Forecast",
        color="tab:red",
        linestyle="--",
        marker="o",
        linewidth=1.5,
    )
    plt.axvline(hist.index[-1], color="gray", linestyle=":", linewidth=1)
    plt.title("Food Price Inflation Forecast")
    plt.xlabel("Month")
    plt.ylabel("Inflation (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_historical_inflation(data, output_path):
    plt.figure(figsize=(10, 4))
    plt.plot(data.index, data["food_price_inflation"], color="tab:blue", linewidth=1.5)
    plt.title("Historical Food Price Inflation")
    plt.xlabel("Month")
    plt.ylabel("Inflation (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_historical_bdi(data, output_path):
    plt.figure(figsize=(10, 4))
    plt.plot(data.index, data["bdi_mean"], color="tab:green", linewidth=1.5)
    plt.title("Historical BDI Mean")
    plt.xlabel("Month")
    plt.ylabel("BDI Mean")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_brent_vs_inflation(data, output_path):
    """Double-axis plot to compare Brent prices with food inflation."""
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(data.index, data["food_price_inflation"], color="tab:blue", label="Food Inflation")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Inflation (%)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(data.index, data["brent_price"], color="tab:orange", label="Brent")
    ax2.set_ylabel("Brent (USD/barrel)", color="tab:orange")
    fig.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_policy_rate_vs_inflation(data, output_path):
    """Double-axis plot to compare policy rate with food inflation."""
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(data.index, data["food_price_inflation"], color="tab:blue", label="Food Inflation")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Inflation (%)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(data.index, data["policy_rate"], color="tab:red", label="Policy Rate")
    ax2.set_ylabel("Policy Rate (%)", color="tab:red")
    fig.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_bdi_vs_inflation(data, output_path):
    """Double-axis plot to compare BDI with food inflation."""
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(data.index, data["food_price_inflation"], color="tab:blue", label="Food Inflation")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Inflation (%)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(data.index, data["bdi_mean"], color="tab:green", label="BDI")
    ax2.set_ylabel("BDI Mean", color="tab:green")
    fig.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_hcp_relationship(data, output_path):
    """Scatter plots of HCP indicators vs inflation for the report."""
    hcp = load_hcp_data()
    merged = align_monthly(data).join(hcp, how="left").dropna()
    if merged.empty:
        print("Skipping hcp_relationship.png: no overlapping data between model data and HCP indicators")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(merged["FAO_CP_23012"], merged["food_price_inflation"], alpha=0.7)
    axes[0].set_title("Inflation vs Indicator 23012")
    axes[0].set_xlabel("Indicator 23012")
    axes[0].set_ylabel("Food Inflation")

    axes[1].scatter(merged["FAO_CP_23013"], merged["food_price_inflation"], alpha=0.7)
    axes[1].set_title("Inflation vs Indicator 23013")
    axes[1].set_xlabel("Indicator 23013")
    axes[1].set_ylabel("Food Inflation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_hcp_projection(data, output_path):
    """Plot projected HCP indicators based on forecast inflation."""
    from src.human_capital_analysis import generate_hcp_projections, load_submission_forecast

    try:
        future_inflation = load_submission_forecast()
        projections = generate_hcp_projections(future_inflation, data)
    except Exception as e:
        print(f"Could not generate HCP projection: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in projections.columns:
        if col != "year_month" and col != "forecast_food_inflation":
            ax.plot(
                range(1, len(future_inflation) + 1),
                projections[col],
                marker="o",
                linewidth=1.6,
                label=col,
            )
    ax.set_title("Projected Human-Capital Indicator Trends")
    ax.set_xlabel("Submission month")
    ax.set_ylabel("Projected value")
    ax.legend(loc="best")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_lstm_loss(history, output_path):
    """Plot LSTM training and validation loss curves."""
    if history is None:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Training Loss", color="tab:blue", linewidth=1.5)
    plt.plot(history.history["val_loss"], label="Validation Loss", color="tab:red", linewidth=1.5)
    plt.title("LSTM Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_lag_correlations(results, output_path):
    """Bar charts showing how correlation changes across lags 1-6."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    key_features = ["brent_price", "policy_rate", "bdi_mean"]
    titles = ["Brent vs Inflation", "Policy Rate vs Inflation", "BDI vs Inflation"]

    for ax, feature, title in zip(axes, key_features, titles):
        if feature not in results:
            continue
        lags = list(results[feature].keys())
        corrs = [results[feature][lag]["correlation"] for lag in lags]
        ax.bar([str(l) for l in lags], corrs, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Correlation")
        ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_lag_selection(full_data):
    """Run automatic lag selection for all exogenous features."""
    exog_cols = get_exog_cols()
    main_exog = [c for c in exog_cols if not c.startswith("inflation_")]

    target = full_data["food_price_inflation"]
    exog = full_data[main_exog]

    results = evaluate_lags(target, exog, max_lag=6)
    best_lags = select_best_lags(results, main_exog, max_lag=6)

    return {"lag_results": results, "best_lags": best_lags}


def evaluate_models(full_data, test_data, figures_dir, outputs_dir):
    """
    Run the full evaluation pipeline:
    - Train SARIMAX and LSTM
    - Compare metrics
    - Generate submission forecast
    - Save all outputs and figures
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    features = [c for c in FEATURE_COLS if c in full_data.columns]
    train_data = full_data.iloc[: -len(test_data)]

    print("Training SARIMAX...")
    sarimax_result = train_sarimax(train_data, test_data, features=features)

    print("Training LSTM...")
    lstm_result = train_lstm(train_data, test_data, features=features)
    lstm_wf = None
    if lstm_result is not None:
        lstm_wf = {
            "mae": lstm_result["metrics"]["mae"],
            "rmse": lstm_result["metrics"]["rmse"],
            "mape": lstm_result["metrics"]["mape"],
        }

    sarimax_metrics = sarimax_result["metrics"]
    lstm_metrics = lstm_result["metrics"] if lstm_result else None
    sarimax_wf = sarimax_result.get("walk_forward_metrics")

    print("Building submission forecast...")
    sarimax_rmse = sarimax_result["metrics"]["rmse"]
    lstm_rmse = lstm_result["metrics"]["rmse"] if lstm_result else float("inf")
    forecast_dates, future_forecasts = build_submission_forecast(
        full_data,
        horizon=12,
        sarimax_order=sarimax_result["order"],
        lstm_result=lstm_result,
        features=features,
        sarimax_rmse=sarimax_rmse,
        lstm_rmse=lstm_rmse,
    )
    save_predictions(os.path.join(outputs_dir, "predictions.csv"), forecast_dates, future_forecasts)

    save_metrics(
        os.path.join(outputs_dir, "metrics.csv"),
        sarimax_metrics,
        lstm_metrics,
        sarimax_wf=sarimax_wf,
        lstm_wf=lstm_wf,
    )

    save_feature_importance(
        os.path.join(outputs_dir, "feature_importances.csv"),
        train_data,
        features,
    )

    sarimax_preds = pd.Series(sarimax_result["predictions"], index=test_data.index)

    plot_predictions(
        test_data["food_price_inflation"],
        sarimax_preds,
        "SARIMAX",
        os.path.join(figures_dir, "sarimax_predictions.png"),
    )
    plot_residuals(
        test_data["food_price_inflation"],
        sarimax_preds,
        "SARIMAX",
        os.path.join(figures_dir, "sarimax_residuals.png"),
    )

    if lstm_result is not None and "predictions" in lstm_result:
        lstm_actual = pd.Series(lstm_result["actual"], index=test_data.index[-len(lstm_result["actual"]):])
        lstm_preds = pd.Series(lstm_result["predictions"], index=lstm_actual.index)
        lstm_preds = lstm_preds.dropna()
        if not lstm_preds.empty:
            plot_predictions(lstm_actual, lstm_preds, "LSTM", os.path.join(figures_dir, "lstm_predictions.png"))
            plot_residuals(lstm_actual, lstm_preds, "LSTM", os.path.join(figures_dir, "lstm_residuals.png"))
            plot_lstm_loss(lstm_result.get("history"), os.path.join(figures_dir, "lstm_training_history.png"))

    plot_forecast(full_data, forecast_dates, future_forecasts, os.path.join(figures_dir, "forecast_food_inflation.png"))
    plot_historical_inflation(test_data, os.path.join(figures_dir, "historical_food_inflation.png"))
    plot_historical_bdi(test_data, os.path.join(figures_dir, "historical_bdi.png"))
    plot_brent_vs_inflation(test_data, os.path.join(figures_dir, "brent_vs_inflation.png"))
    plot_policy_rate_vs_inflation(test_data, os.path.join(figures_dir, "policy_rate_vs_inflation.png"))
    plot_bdi_vs_inflation(test_data, os.path.join(figures_dir, "bdi_vs_inflation.png"))
    plot_hcp_relationship(test_data, os.path.join(figures_dir, "hcp_relationship.png"))
    plot_hcp_projection(test_data, os.path.join(figures_dir, "forecast_human_capital_projection.png"))

    lag_results = run_lag_selection(full_data)
    plot_lag_correlations(lag_results["lag_results"], os.path.join(figures_dir, "lag_correlations.png"))

    print("SARIMAX metrics (holdout):")
    for k, v in sarimax_metrics.items():
        print(f"  {k}: {v:.3f}")
    if sarimax_wf:
        print("SARIMAX metrics (walk-forward):")
        for k, v in sarimax_wf.items():
            print(f"  {k}: {v:.3f}")

    if lstm_metrics:
        print("LSTM metrics (holdout):")
        for k, v in lstm_metrics.items():
            print(f"  {k}: {v:.3f}")

    print("Lag selection:")
    for col, lag in lag_results["best_lags"].items():
        print(f"  {col}: lag {lag}")

    return {
        "sarimax": sarimax_result,
        "lstm": lstm_result,
        "lag_selection": lag_results,
        "submission_dates": forecast_dates,
        "submission_forecasts": future_forecasts,
    }


if __name__ == "__main__":
    data = load_data()
    data = add_features(data)
    train, test = split_train_test(data)
    evaluate_models(data, test, FIGURES_DIR, OUTPUTS_DIR)
