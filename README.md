# Botswana Food Price Inflation Forecasting

Welcome to the official repository for our IndabaX Botswana AI Hackathon project. This project provides a robust, end-to-end forecasting pipeline that models and predicts Botswana's monthly food price inflation. By combining traditional econometric modeling (SARIMAX) with deep learning (LSTM), and tying these projections to broader human capital outcomes, this tool bridges the gap between macroeconomic data science and actionable policy insights.

---

## Project Overview and Motivation

Food price stability is essential for sustainable national development. In recent years, Botswana's food prices have experienced notable volatility driven by post-pandemic supply chain disruptions and shifting global energy markets. 

To address this challenge, our solution integrates key macroeconomic and supply-chain indicators:
* **Global Shipping Costs:** Captured via the **Baltic Dry Index (BDI)**, reflecting international shipping friction and trade volumes.
* **Energy Markets:** Tracked using **Brent Crude Oil** prices, influencing agricultural input, processing, and transportation costs.
* **Monetary Policy:** Monitored through **Botswana's Policy Rate**, accounting for domestic interest rate adjustments and credit conditions.
* **Target Variable:** **FAO Food Prices / Botswana Food CPI and Inflation**.

Additionally, the pipeline explores downstream socioeconomic impacts by analyzing how forecasted food inflation correlates with key human capital indicators (such as school attendance and health outcomes).

---

## Project Architecture

```text
src/
    preprocessing.py            - Data loading, cleaning, and feature engineering
    models.py                   - SARIMAX and LSTM model architectures
    evaluation.py               - Automated training, metric comparison, and artifact saving
    human_capital_analysis.py   - Linking inflation forecasts to human capital outcomes

data/raw/                       - Raw hackathon datasets (BDI, Brent, Policy Rate, FAO, Human Capital)
outputs/                        - Generated CSVs (predictions, metrics, feature importances)
figures/                        - Visualizations (forecasts, residuals, lag correlations, charts)
requirements.txt                - Python dependency manifest
README.md                       - Project documentation
```

---

## Getting Started: Step-by-Step Installation and Execution Guide

Whether you are a technical evaluator, a domain expert, or exploring this project for the first time, follow these comprehensive instructions to set up and run the system on your machine.

### Step 1: Clone the Repository
Open your terminal (**Command Prompt**, **PowerShell**, or **Terminal** on macOS/Linux) and clone the repository to your local machine:
```bash
git clone https://github.com/your-username/botswana-food-inflation-forecasting.git
cd botswana-food-inflation-forecasting
```

### Step 2: Verify Your Python Installation
This project requires **Python 3.11 or 3.12**. 
> *Note: TensorFlow does not yet support Python 3.13+. Please ensure your active Python version is within the 3.11–3.12 range.*

Check your Python version by running:
```bash
python --version
# or on some systems:
python3 --version
```

### Step 3: Set Up a Virtual Environment (Recommended)
Creating an isolated virtual environment ensures that project dependencies do not conflict with your global Python installation.

* **On Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 4: Install Project Dependencies
With your virtual environment activated, upgrade `pip` and install the required packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
> *Tip: If TensorFlow installation fails or is skipped due to system compatibility (such as Apple Silicon native wheels or specific Windows build tools), the pipeline will handle it gracefully, running the robust SARIMAX forecasting model seamlessly.*

---

## Running the Pipeline

Once your environment is fully configured, execute the scripts in the following order:

### 1. Run the Main Forecasting and Evaluation Pipeline
This script loads the raw data, applies lag selections, trains both the **SARIMAX** and **LSTM** models using walk-forward validation, compares their performance metrics (**RMSE**, **MAE**, **MAPE**), and generates final predictions.
```bash
python src/evaluation.py
```

### 2. Run the Human Capital Analysis
This script takes the generated inflation projections and maps their downstream implications onto human capital indicators.
```bash
python src/human_capital_analysis.py
```

---

## Methodology and Feature Engineering

### Feature Engineering Approach
To extract predictive signal from raw data, we engineered features tailored to each domain:
* **Baltic Dry Index (BDI):** Calculated monthly means, standard deviations, ranges, volatility metrics, 3-month and 6-month momentum, trend indicators, extreme day counts, and rolling averages.
* **Brent Crude Oil:** Incorporated **3 distinct economic lags**, rolling means, and monthly percentage changes.
* **Botswana Policy Rate:** Included **3 lags**, rolling means, monthly changes, and directional increase/decrease flags.
* **Food CPI:** Extracted **lags 1 through 6 months** alongside rolling means and rolling standard deviations.

> *Note: All features undergo rigorous lag selection via correlation analysis, cross-correlation, **AIC**, and **BIC** scoring, and any rows with missing values resulting from lag generation are systematically cleaned.*

### Model Architecture
1. **SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Inputs):** Tuned parameters ($p, d, q$) using grid search optimized by **AIC** alongside seasonal parameters. Evaluated via rigorous walk-forward validation.
2. **LSTM (Long Short-Term Memory Neural Network):** Processes **12-month sequential windows** with feature/target normalization, featuring a dual-layer **LSTM** architecture with dropout regularizations and callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`).

---

## Output Artifacts

Upon successful execution, the following files and directories will be populated:
* `outputs/predictions.csv` - The final **12-month food price inflation forecast**.
* `outputs/metrics.csv` - Comparative performance metrics (**RMSE**, **MAE**, **MAPE**) for both models.
* `outputs/feature_importances.csv` - **SARIMAX-derived** feature importance rankings.
* `outputs/human_capital_projections.csv` - Projected human capital indicators based on inflation trends.
* `figures/` - Comprehensive visual charts including forecast plots, actual vs. predicted overlays, residual diagnostics, lag correlation heatmaps, and human capital impact curves.

---

## Limitations

* **Data Depth:** Limited to approximately **20 years** of monthly historical observations, which restricts the data-hungry nature of deep learning architectures.
* **Exogenous Assumptions:** Future values for exogenous variables (such as oil prices and shipping indices) are held flat at their last observed values for forward-looking simulations.
