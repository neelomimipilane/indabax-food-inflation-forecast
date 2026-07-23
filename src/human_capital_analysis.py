"""
Human capital analysis - links food inflation forecasts to HCP indicators.
Must be run AFTER evaluation.py so predictions.csv exists.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
INDICATORS = ["FAO_CP_23012", "FAO_CP_23013"]


def load_human_capital_data():
    """Load Botswana HCP indicators from the raw data."""
    hcp = pd.read_csv(os.path.join(DATA_DIR, "05_human_capital_project.csv"), parse_dates=["Date"])
    hcp = hcp[hcp["REF_AREA"] == "BWA"]
    hcp = hcp[hcp["INDICATOR"].isin(INDICATORS)]
    hcp = hcp[["Date", "INDICATOR", "Value"]]
    hcp = hcp.pivot(index="Date", columns="INDICATOR", values="Value").reset_index()
    hcp["month"] = hcp["Date"].dt.to_period("M")
    monthly = hcp.groupby("month")[INDICATORS].mean()
    monthly.index = monthly.index.to_timestamp()
    return monthly


def build_analysis_frame():
    """Merge Botswana food inflation with the HCP indicators."""
    from src.preprocessing import load_data

    inflation_data = load_data()
    hcp_data = load_human_capital_data()
    merged = inflation_data.join(hcp_data, how="inner")
    return merged.dropna().copy()


def run_regressions(merged):
    """Run OLS regression for each HCP indicator against food inflation."""
    results = {}
    for indicator in INDICATORS:
        x = sm.add_constant(merged["food_price_inflation"])
        model = sm.OLS(merged[indicator], x).fit()
        results[indicator] = {
            "coefficients": model.params.to_dict(),
            "pvalues": model.pvalues.to_dict(),
            "r_squared": float(model.rsquared),
        }
    return results


def load_submission_forecast():
    """Read the official submission forecast from predictions.csv."""
    forecast_path = os.path.join(OUTPUTS_DIR, "predictions.csv")
    if not os.path.exists(forecast_path):
        raise FileNotFoundError("outputs/predictions.csv is required. Run evaluation.py first.")

    frame = pd.read_csv(forecast_path)
    values = pd.to_numeric(frame["forecast"], errors="coerce")
    if values.isna().any():
        raise ValueError("The submission forecast contains missing values")

    return pd.Series(values.astype(float).to_numpy(), index=frame["year_month"])


def generate_hcp_projections(future_inflation, historical_data):
    """Project HCP indicators using the inflation-forecast relationship."""
    hcp = load_human_capital_data()
    aligned = historical_data.copy()
    aligned.index = aligned.index.to_period("M").to_timestamp()
    merged = aligned.join(hcp, how="left", rsuffix="_hcp").dropna()

    projection = pd.DataFrame(
        {
            "year_month": future_inflation.index,
            "forecast_food_inflation": future_inflation.values,
        }
    )

    for indicator in INDICATORS:
        indicator_col = indicator if indicator in merged.columns else f"{indicator}_hcp"
        if indicator_col not in merged.columns or merged[indicator_col].empty:
            continue
        x = sm.add_constant(merged["food_price_inflation"])
        model = sm.OLS(merged[indicator_col], x).fit()
        coef_const = model.params.get("const", 0.0)
        coef_inf = model.params.get("food_price_inflation", 0.0)
        projection[indicator] = coef_const + coef_inf * projection["forecast_food_inflation"]

    return projection


def generate_plots(merged, projections):
    """Make all the HCP-related plots and save the projection CSV."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

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
    fig.savefig(os.path.join(FIGURES_DIR, "historical_food_inflation_vs_indicator.png"), dpi=200)
    plt.close(fig)

    future_inflation = load_submission_forecast()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        range(1, len(future_inflation) + 1),
        future_inflation.values,
        marker="o",
        color="tab:blue",
        linewidth=1.6,
    )
    ax.set_title("Forecast Food Inflation")
    ax.set_xlabel("Submission month")
    ax.set_ylabel("Inflation (%)")
    ax.set_xticks(range(1, len(future_inflation) + 1))
    ax.set_xticklabels(future_inflation.index, rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "forecast_food_inflation.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for indicator in INDICATORS:
        if indicator in projections.columns:
            ax.plot(
                range(1, len(future_inflation) + 1),
                projections[indicator],
                marker="o",
                linewidth=1.6,
                label=indicator,
            )
    ax.set_title("Projected Human-Capital Indicator Trends")
    ax.set_xlabel("Submission month")
    ax.set_ylabel("Projected value")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "forecast_human_capital_projection.png"), dpi=200)
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
    fig.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, indicator in zip(axes, INDICATORS):
        if indicator not in merged.columns:
            continue
        x = sm.add_constant(merged["food_price_inflation"])
        model = sm.OLS(merged[indicator], x).fit()
        yhat = model.predict(x)
        ax.scatter(merged["food_price_inflation"], merged[indicator], color="tab:blue", alpha=0.8)
        ax.plot(merged["food_price_inflation"], yhat, color="tab:red", linewidth=1.6)
        ax.set_title(f"{indicator} vs inflation")
        ax.set_xlabel("Food inflation")
        ax.set_ylabel(indicator)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "regression_plot.png"), dpi=200)
    plt.close(fig)


def generate_outputs():
    """Run the full HCP analysis and save outputs."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    merged = build_analysis_frame()
    results = run_regressions(merged)
    future_inflation = load_submission_forecast()
    projections = generate_hcp_projections(future_inflation, merged)

    projections.to_csv(os.path.join(OUTPUTS_DIR, "human_capital_projections.csv"), index=False)

    for indicator, res in results.items():
        print(f"{indicator} R-squared: {res['r_squared']:.3f}")

    generate_plots(merged, projections)


if __name__ == "__main__":
    generate_outputs()
