"""
Data loading and feature engineering for the food inflation project.
"""

import os
import sys

import pandas as pd
from statsmodels.tsa.stattools import ccf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data():
    """Load all datasets and join them into one monthly table."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    bdi = pd.read_csv(os.path.join(data_dir, "01_baltic_dry_index_daily.csv"), parse_dates=["Date"])
    brent = pd.read_csv(os.path.join(data_dir, "02_brent_crude_monthly.csv"), parse_dates=["Date"])
    policy = pd.read_csv(os.path.join(data_dir, "03_botswana_policy_rate.csv"), parse_dates=["Date"])
    prices = pd.read_csv(os.path.join(data_dir, "04_fao_botswana_prices.csv"), parse_dates=["Date"])

    monthly_bdi = make_monthly_bdi(bdi)

    # Brent is already monthly, just keep the average
    brent = brent.rename(columns={"Brent_USD_per_barrel": "brent_price"})
    brent["month"] = brent["Date"].dt.to_period("M")
    brent = brent.groupby("month")["brent_price"].mean().to_frame()

    # Policy rate - monthly values
    policy = policy.rename(columns={"policy_rate": "policy_rate"})
    policy["month"] = policy["Date"].dt.to_period("M")
    policy = policy.groupby("month")["policy_rate"].mean().to_frame()

    # FAO prices - pivot so each item becomes a column
    prices = prices.pivot_table(index="Date", columns="Item", values="Value", aggfunc="first")
    prices = prices.reset_index()
    prices = prices.rename(
        columns={
            "Consumer Prices, General Indices (2015 = 100)": "general_cpi",
            "Consumer Prices, Food Indices (2015 = 100)": "food_cpi",
            "Food price inflation": "food_price_inflation",
        }
    )
    prices["month"] = prices["Date"].dt.to_period("M")
    prices = prices.groupby("month")[["general_cpi", "food_cpi", "food_price_inflation"]].mean()

    # Join everything together
    data = monthly_bdi.join(brent, how="left")
    data = data.join(policy, how="left")
    data = data.join(prices, how="left")
    data = data.sort_index().ffill()

    # Year-on-year food price inflation using food CPI
    # pct_change(12) because monthly data and 12 months = 1 year
    data["food_price_inflation"] = data["food_cpi"].pct_change(12) * 100
    data = data.dropna().copy()
    data.index = data.index.to_timestamp()
    return data


def make_monthly_bdi(bdi):
    """Convert daily BDI data into monthly features."""
    bdi = bdi.set_index("Date")

    monthly = bdi.resample("ME").agg(
        {
            "BDI_Close": ["mean", "std", "min", "max"],
        }
    )
    monthly.columns = ["bdi_mean", "bdi_std", "bdi_min", "bdi_max"]

    # Range shows how much the index moved within each month
    monthly["bdi_range"] = monthly["bdi_max"] - monthly["bdi_min"]

    # 3-month momentum - captures medium-term trend direction
    monthly["bdi_momentum_3"] = monthly["bdi_mean"].pct_change(3) * 100

    # 6-month momentum - captures longer trend direction
    monthly["bdi_momentum_6"] = monthly["bdi_mean"].pct_change(6) * 100

    # Trend: difference between short and long moving averages
    # Positive means short-term average is above long-term = uptrend
    monthly["bdi_ma3"] = monthly["bdi_mean"].rolling(3).mean()
    monthly["bdi_ma6"] = monthly["bdi_mean"].rolling(6).mean()
    monthly["bdi_trend"] = monthly["bdi_ma3"] - monthly["bdi_ma6"]

    # Volatility: coefficient of variation to normalize across price levels
    monthly["bdi_volatility"] = monthly["bdi_std"] / monthly["bdi_mean"].replace(0, pd.NA)

    # Extreme days: days where close is more than 1 std above the mean
    # High count signals unusual bullish activity
    extreme = bdi["BDI_Close"].groupby(pd.Grouper(freq="ME")).apply(
        lambda x: (x > x.mean() + x.std()).sum() if x.std() > 0 else 0
    )
    monthly["bdi_extreme_days"] = extreme

    # Rolling averages for smoothing
    monthly["bdi_roll3_mean"] = monthly["bdi_mean"].rolling(3).mean()
    monthly["bdi_roll6_mean"] = monthly["bdi_mean"].rolling(6).mean()

    monthly = monthly.drop(columns=["bdi_ma3", "bdi_ma6"])
    monthly.index = monthly.index.to_period("M")
    return monthly


def add_brent_features(data):
    """Add engineered features for Brent crude oil prices."""
    data = data.copy()
    # One-month lag captures delayed fuel price transmission to food costs
    data["brent_lag_1"] = data["brent_price"].shift(1)
    data["brent_lag_2"] = data["brent_price"].shift(2)
    data["brent_lag_3"] = data["brent_price"].shift(3)

    # Rolling means smooth out monthly volatility
    data["brent_roll3_mean"] = data["brent_price"].rolling(3).mean()
    data["brent_roll6_mean"] = data["brent_price"].rolling(6).mean()

    # Month-to-month change captures recent direction
    data["brent_monthly_change"] = data["brent_price"].diff()
    return data


def add_policy_rate_features(data):
    """Add engineered features for Botswana policy rate."""
    data = data.copy()
    # Policy rate changes take time to affect inflation, so lags matter
    data["policy_lag_1"] = data["policy_rate"].shift(1)
    data["policy_lag_2"] = data["policy_rate"].shift(2)
    data["policy_lag_3"] = data["policy_rate"].shift(3)

    # Rolling mean shows the recent stance of monetary policy
    data["policy_roll3_mean"] = data["policy_rate"].rolling(3).mean()

    # Month-to-month change and direction flag
    data["policy_monthly_change"] = data["policy_rate"].diff()
    data["policy_increase"] = (data["policy_monthly_change"] > 0).astype(int)
    return data


def add_inflation_features(data):
    """Add lag and rolling features for the target variable."""
    data = data.copy()
    # Lags 1-6 months test whether past inflation predicts future inflation
    for lag in range(1, 7):
        data[f"inflation_lag_{lag}"] = data["food_price_inflation"].shift(lag)

    # Rolling statistics - shift(1) avoids using the current month's value
    # Without the shift this would be data leakage
    data["inflation_roll3_mean"] = data["food_price_inflation"].shift(1).rolling(3).mean()
    data["inflation_roll6_mean"] = data["food_price_inflation"].shift(1).rolling(6).mean()
    data["inflation_roll3_std"] = data["food_price_inflation"].shift(1).rolling(3).std()
    return data


def add_features(data):
    """Add all engineered features and drop rows with missing values."""
    data = add_brent_features(data)
    data = add_policy_rate_features(data)
    data = add_inflation_features(data)
    return data.dropna().copy()


def evaluate_lags(target, exog, max_lag=6):
    """
    Test lags 1 through max_lag for each exogenous feature.
    Reports correlation, cross-correlation, AIC and BIC.
    """
    results = {}
    target_aligned = target.dropna()
    exog_aligned = exog.loc[target_aligned.index]

    for col in exog_aligned.columns:
        col_results = {}
        for lag in range(1, max_lag + 1):
            if len(target_aligned) <= lag:
                continue

            y = target_aligned.iloc[lag:]
            x = exog_aligned[col].iloc[: len(y)]

            if len(y) == 0 or len(x) == 0:
                continue

            corr = y.corr(x)

            # Cross-correlation at this lag
            ccf_vals = ccf(target_aligned.dropna(), exog_aligned[col].dropna(), unbiased=False)
            ccf_at_lag = ccf_vals[lag - 1] if len(ccf_vals) >= lag else 0.0

            # AIC/BIC from a simple OLS with this lag
            aic_val = float("inf")
            bic_val = float("inf")
            try:
                import statsmodels.api as sm

                combined = pd.concat([y, x], axis=1).dropna()
                if len(combined) >= 10:
                    y_c = combined.iloc[:, 0]
                    x_c = combined.iloc[:, 1]
                    model = sm.OLS(y_c, sm.add_constant(x_c)).fit()
                    aic_val = model.aic
                    bic_val = model.bic
            except Exception:
                pass

            col_results[lag] = {
                "correlation": float(corr) if not pd.isna(corr) else 0.0,
                "ccf": float(ccf_at_lag),
                "aic": aic_val,
                "bic": bic_val,
            }
        results[col] = col_results

    return results


def select_best_lags(lag_results, exog_cols, max_lag=6):
    """Pick the lag with the lowest AIC for each feature."""
    best_lags = {}
    for col in exog_cols:
        col_lags = lag_results.get(col, {})
        if not col_lags:
            best_lags[col] = 1
            continue

        valid_lags = {
            lag: metrics
            for lag, metrics in col_lags.items()
            if metrics["aic"] != float("inf") and not pd.isna(metrics["correlation"])
        }
        if not valid_lags:
            best_lags[col] = 1
            continue

        best_lag = min(valid_lags.keys(), key=lambda lag: valid_lags[lag]["aic"])
        best_lags[col] = best_lag
        print(
            f"Selected lag {best_lag} for {col} "
            f"(corr={valid_lags[best_lag]['correlation']:.3f}, AIC={valid_lags[best_lag]['aic']:.2f})"
        )
    return best_lags


# All features used by the models
FEATURE_COLS = [
    "brent_price",
    "brent_lag_1",
    "brent_lag_2",
    "brent_lag_3",
    "brent_roll3_mean",
    "brent_roll6_mean",
    "brent_monthly_change",
    "policy_rate",
    "policy_lag_1",
    "policy_lag_2",
    "policy_lag_3",
    "policy_roll3_mean",
    "policy_monthly_change",
    "policy_increase",
    "bdi_mean",
    "bdi_std",
    "bdi_range",
    "bdi_momentum_3",
    "bdi_momentum_6",
    "bdi_trend",
    "bdi_volatility",
    "bdi_extreme_days",
    "bdi_roll3_mean",
    "bdi_roll6_mean",
    "inflation_lag_1",
    "inflation_lag_2",
    "inflation_lag_3",
    "inflation_lag_4",
    "inflation_lag_5",
    "inflation_lag_6",
    "inflation_roll3_mean",
    "inflation_roll6_mean",
    "inflation_roll3_std",
]


def get_exog_cols():
    """Return exogenous feature columns (everything except the target)."""
    return [c for c in FEATURE_COLS if c != "food_price_inflation"]


def split_train_test(data, test_months=12):
    """Simple train/test split - last 12 months as holdout."""
    train = data.iloc[:-test_months]
    test = data.iloc[-test_months:]
    return train, test
