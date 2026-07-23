from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
FIGURES_DIR = ROOT_DIR / "figures"
OUTPUTS_DIR = ROOT_DIR / "outputs"
INDICATORS = ["FAO_CP_23012", "FAO_CP_23013"]


def load_human_capital_data() -> pd.DataFrame:
    # Load Botswana HCP indicators from the raw data
    hcp = pd.read_csv(DATA_DIR / "05_human_capital_project.csv", parse_dates=["Date"])
    hcp = hcp[hcp["REF_AREA"] == "BWA"]
    hcp = hcp[hcp["INDICATOR"].isin(INDICATORS)]
    hcp = hcp[["Date", "INDICATOR", "Value"]]
    hcp = hcp.pivot(index="Date", columns="INDICATOR", values="Value").reset_index()
    hcp["month"] = hcp["Date"].dt.to_period("M")
    monthly = hcp.groupby("month")[INDICATORS].mean()
    monthly.index = monthly.index.to_timestamp()
    return monthly


def build_analysis_frame() -> pd.DataFrame:
    # Merge Botswana food inflation with the HCP indicators
    import sys

    sys.path.append(str(ROOT_DIR))
    from src.preprocessing import load_data

    inflation_data = load_data()
    hcp_data = load_human_capital_data()
    merged = inflation_data.join(hcp_data, how="inner")
    return merged.dropna().copy()


def run_regressions(merged: pd.DataFrame) -> dict[str, dict[str, Any]]:
    # Run OLS regression for each indicator
    results: dict[str, dict[str, Any]] = {}
    for indicator in INDICATORS:
        x = sm.add_constant(merged["food_price_inflation"])
        model = sm.OLS(merged[indicator], x).fit()
        results[indicator] = {
            "coefficients": model.params.to_dict(),
            "pvalues": model.pvalues.to_dict(),
            "r_squared": float(model.rsquared),
        }
    return results


def load_submission_forecast() -> pd.Series:
    # Read the official submission forecast
    forecast_path = OUTPUTS_DIR / "predictions.csv"
    if not forecast_path.exists():
        raise FileNotFoundError("outputs/predictions.csv is required for the human-capital projection step")

    forecast_frame = pd.read_csv(forecast_path)
    forecast_values = pd.to_numeric(forecast_frame["forecast"], errors="coerce")
    if forecast_values.isna().any():
        raise ValueError("The submission forecast contains missing values")

    return pd.Series(forecast_values.astype(float).to_numpy(), index=forecast_frame["year_month"])


def generate_plots(merged: pd.DataFrame, results: dict[str, dict[str, Any]]) -> None:
    # Make the plots and save the projection CSV
    FIGURES_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(merged.index, merged["food_price_inflation"], color="tab:blue", linewidth=1.6)
    axes[0].set_title("Historical Food Inflation")
    axes[0].set_ylabel("Inflation (%)")
    axes[1].plot(merged.index, merged["FAO_CP_23012"], color="tab:green", label="FAO_CP_23012", linewidth=1.6)
    axes[1].plot(merged.index, merged["FAO_CP_23013"], color="tab:red", label="FAO_CP_23013", linewidth=1.6)
    axes[1].set_title("Historical Human-Capital Indicator Proxies")
    axes[1].set_ylabel("Indicator value")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "historical_food_inflation_vs_indicator.png", dpi=200)
    plt.close(fig)

    future_inflation = load_submission_forecast()
    projection = pd.DataFrame(
        {
            "year_month": future_inflation.index,
            "forecast_food_inflation": future_inflation.values,
        }
    )
    for indicator in INDICATORS:
        coeff = results[indicator]["coefficients"]
        intercept = coeff.get("const", 0.0)
        slope = coeff.get("food_price_inflation", 0.0)
        projection[indicator] = intercept + slope * projection["forecast_food_inflation"]
    projection.to_csv(OUTPUTS_DIR / "human_capital_projections.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(future_inflation) + 1), future_inflation.values, marker="o", color="tab:blue", linewidth=1.6)
    ax.set_title("Forecast Food Inflation")
    ax.set_xlabel("Submission month")
    ax.set_ylabel("Inflation (%)")
    ax.set_xticks(range(1, len(future_inflation) + 1))
    ax.set_xticklabels(future_inflation.index)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_food_inflation.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for indicator in INDICATORS:
        ax.plot(range(1, len(future_inflation) + 1), projection[indicator], marker="o", linewidth=1.6, label=indicator)
    ax.set_title("Projected Human-Capital Indicator Trends")
    ax.set_xlabel("Submission month")
    ax.set_ylabel("Projected value")
    ax.set_xticks(range(1, len(future_inflation) + 1))
    ax.set_xticklabels(future_inflation.index)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_human_capital_projection.png", dpi=200)
    plt.close(fig)

    corr = merged[["food_price_inflation", *INDICATORS]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.columns)):
        for j in range(len(corr.index)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Correlation Heatmap")
    fig.colorbar(cax, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, indicator in zip(axes, INDICATORS):
        x = sm.add_constant(merged["food_price_inflation"])
        model = sm.OLS(merged[indicator], x).fit()
        yhat = model.predict(x)
        ax.scatter(merged["food_price_inflation"], merged[indicator], color="tab:blue", alpha=0.8)
        ax.plot(merged["food_price_inflation"], yhat, color="tab:red", linewidth=1.6)
        ax.set_title(f"{indicator} vs inflation")
        ax.set_xlabel("Food inflation")
        ax.set_ylabel(indicator)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regression_plot.png", dpi=200)
    plt.close(fig)


def generate_outputs() -> None:
    # Run the whole analysis and save the outputs
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    merged = build_analysis_frame()
    results = run_regressions(merged)
    generate_plots(merged, results)


if __name__ == "__main__":
    generate_outputs()
