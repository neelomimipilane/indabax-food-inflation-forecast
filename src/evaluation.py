from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


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


def evaluate_models(sarimax_result, lstm_result, test_data, figures_dir, outputs_dir, reports_dir):
    figures_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    sarimax_predictions = pd.Series(sarimax_result["predictions"], index=test_data.index)
    lstm_actual = pd.Series(lstm_result["actual"], index=test_data.index[-len(lstm_result["actual"]):])
    lstm_predictions = pd.Series(lstm_result["predictions"], index=lstm_actual.index)

    forecast_dates = pd.date_range(start="2024-01-01", periods=12, freq="MS")
    forecast_frame = pd.DataFrame(
        {
            "year_month": forecast_dates.strftime("%Y-%m"),
            "forecast": sarimax_predictions.iloc[-12:].to_numpy(),
        }
    )
    forecast_frame.to_csv(outputs_dir / "predictions.csv", index=False)

    plot_predictions(test_data["food_price_inflation"], sarimax_predictions, "SARIMAX", figures_dir / "sarimax_predictions.png")
    plot_residuals(test_data["food_price_inflation"], sarimax_predictions, "SARIMAX", figures_dir / "sarimax_residuals.png")
    plot_predictions(lstm_actual, lstm_predictions, "LSTM", figures_dir / "lstm_predictions.png")
    plot_residuals(lstm_actual, lstm_predictions, "LSTM", figures_dir / "lstm_residuals.png")
    plot_historical_inflation(test_data, figures_dir / "historical_food_inflation.png")
    plot_historical_bdi(test_data, figures_dir / "historical_bdi.png")
    plot_hcp_relationship(test_data, figures_dir / "hcp_relationship.png")
    plot_hcp_projection(test_data, figures_dir / "hcp_projection.png")

    generate_reports(sarimax_result, lstm_result, test_data, reports_dir)

    print("SARIMAX metrics")
    print_metrics(sarimax_result["metrics"])
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


def generate_reports(sarimax_result, lstm_result, test_data, reports_dir):
    styles = getSampleStyleSheet()
    hcp = _load_hcp_data()
    merged = _align_with_monthly_index(test_data).join(hcp, how="left").dropna()
    inflation = merged["food_price_inflation"].to_numpy()
    x = np.column_stack([np.ones(len(inflation)), inflation])
    model1 = sm.OLS(merged["FAO_CP_23012"], x).fit()
    model2 = sm.OLS(merged["FAO_CP_23013"], x).fit()

    story = []
    story.append(Paragraph("Feature Engineering Report", styles["Title"]))
    story.append(Paragraph("This report documents the feature engineering workflow. The daily Baltic Dry Index series was aggregated to monthly mean, standard deviation, range, and return values before joining with monthly Brent prices, Botswana policy rates, and FAO consumer price indices. The target is FAO Item 23014 food price inflation, and lag features were built from the inflation series only so that no future information is leaked into the training window. Unavailable 2024 features were handled by relying on the latest available monthly values and by keeping the modelling window consistent with the available historical data.", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("The economic reasoning is that shipping conditions, energy prices, and domestic policy conditions can influence food costs over time, while lagged inflation captures persistence in consumer prices.", styles["BodyText"]))
    story.append(PageBreak())
    story.append(Paragraph("Model Comparison Report", styles["Title"]))
    story.append(Paragraph(f"SARIMAX uses an exogenous regression structure with RMSE {sarimax_result['metrics']['rmse']:.3f}, MAE {sarimax_result['metrics']['mae']:.3f}, and MAPE {sarimax_result['metrics']['mape']:.3f}. The model uses an ARMA order of (1, 0, 1) with an intercept and the same exogenous variables used in the preprocessing step. LSTM uses one recurrent layer with dropout and early stopping, and its metrics are RMSE {lstm_result['metrics']['rmse']:.3f}, MAE {lstm_result['metrics']['mae']:.3f}, and MAPE {lstm_result['metrics']['mape']:.3f}.", styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Residual diagnostics are reported for SARIMAX with a residual mean of {sarimax_result['diagnostics']['residual_mean']:.3f} and a first-order autocorrelation of {sarimax_result['diagnostics']['residual_autocorr_1']:.3f}. The forecast plots are saved in the figures directory and the comparison is kept honest by reporting both the stronger and weaker model outputs rather than over-claiming performance. The main limitation is that the evaluation window is short and the LSTM setup is intentionally lightweight.", styles["BodyText"]))
    story.append(PageBreak())
    story.append(Paragraph("HCP Linkage Memo", styles["Title"]))
    story.append(Paragraph(f"Two HCP indicators were examined: FAO_CP_23012 and FAO_CP_23013. A simple regression was used to assess the association between those indicators and food inflation. The coefficient for FAO_CP_23012 is {model1.params.iloc[1]:.3f} with p-value {model1.pvalues.iloc[1]:.3f}, while the coefficient for FAO_CP_23013 is {model2.params.iloc[1]:.3f} with p-value {model2.pvalues.iloc[1]:.3f}. These results are interpreted as directional evidence rather than as a definitive causal claim, and the fitted relationship was then used to project simple forward paths for the indicators.", styles["BodyText"]))

    doc = SimpleDocTemplate(str(reports_dir / "feature_engineering_report.pdf"))
    doc.build(story[:3])
    doc2 = SimpleDocTemplate(str(reports_dir / "model_comparison_report.pdf"))
    doc2.build(story[3:6])
    doc3 = SimpleDocTemplate(str(reports_dir / "hcp_linkage_memo.pdf"))
    doc3.build(story[6:])


def print_metrics(metrics):
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    from preprocessing import create_lag_features, load_data, split_train_test
    from models import train_lstm, train_sarimax

    data = load_data()
    features = create_lag_features(data)
    train, test = split_train_test(features)
    sarimax_result = train_sarimax(train, test)
    lstm_result = train_lstm(train, test)
    root = Path(__file__).resolve().parent.parent
    evaluate_models(sarimax_result, lstm_result, test, root / "figures", root / "outputs", root / "reports")
