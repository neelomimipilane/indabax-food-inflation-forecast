from pathlib import Path

import pandas as pd


def load_data():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "raw"

    bdi = pd.read_csv(data_dir / "01_baltic_dry_index_daily.csv", parse_dates=["Date"])
    brent = pd.read_csv(data_dir / "02_brent_crude_monthly.csv", parse_dates=["Date"])
    prices = pd.read_csv(data_dir / "04_fao_botswana_prices.csv", parse_dates=["Date"])

    # Aggregate daily BDI to monthly features
    monthly_bdi = make_monthly_bdi(bdi)

    # Keep only monthly average for brent
    brent = brent.rename(columns={"Brent_USD_per_barrel": "brent_price"})
    brent["month"] = brent["Date"].dt.to_period("M")
    brent = brent.groupby("month")["brent_price"].mean().to_frame()

    # Pivot FAO prices: want general CPI, food CPI, and food inflation
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

    # Merge everything together
    data = monthly_bdi.join(brent, how="left")
    data = data.join(prices, how="left")
    data = data.sort_index().ffill()

    # Target is year-on-year food price inflation
    data["food_price_inflation"] = data["food_cpi"].pct_change(12) * 100
    data = data.dropna().copy()
    data.index = data.index.to_timestamp()
    return data


def make_monthly_bdi(bdi):
    bdi = bdi.set_index("Date")

    # Basic monthly stats
    monthly = bdi.resample("ME").agg(
        {
            "BDI_Close": ["mean", "std", "min", "max"],
        }
    )
    monthly.columns = ["bdi_mean", "bdi_std", "bdi_min", "bdi_max"]

    # Range shows intra-month volatility
    monthly["bdi_range"] = monthly["bdi_max"] - monthly["bdi_min"]

    # 3-month momentum (percent change of the mean)
    monthly["bdi_momentum_3"] = monthly["bdi_mean"].pct_change(3) * 100

    # 6-month momentum
    monthly["bdi_momentum_6"] = monthly["bdi_mean"].pct_change(6) * 100

    # Trend: difference between 3-month moving average and 6-month moving average
    monthly["bdi_ma3"] = monthly["bdi_mean"].rolling(3).mean()
    monthly["bdi_ma6"] = monthly["bdi_mean"].rolling(6).mean()
    monthly["bdi_trend"] = monthly["bdi_ma3"] - monthly["bdi_ma6"]

    # Extreme day count: days in month where close > mean + 1 std
    extreme = bdi["BDI_Close"].groupby(pd.Grouper(freq="ME")).apply(
        lambda x: (x > x.mean() + x.std()).sum()
    )
    monthly["bdi_extreme_days"] = extreme

    # Drop helper columns we don't need
    keep_cols = [
        "bdi_mean",
        "bdi_std",
        "bdi_range",
        "bdi_momentum_3",
        "bdi_momentum_6",
        "bdi_trend",
        "bdi_extreme_days",
    ]
    monthly = monthly[keep_cols]
    monthly.index = monthly.index.to_period("M")
    return monthly


def add_features(data):
    # Lag features for inflation 
    for lag in [1, 3, 6, 12]:
        data[f"inflation_lag_{lag}"] = data["food_price_inflation"].shift(lag)

    # Lag features for key exogenous variables
    data["brent_lag_1"] = data["brent_price"].shift(1)
    data["bdi_lag_1"] = data["bdi_mean"].shift(1)

    # Rolling statistics for food inflation
    data["inflation_roll3_mean"] = data["food_price_inflation"].rolling(3).mean()
    data["inflation_roll6_mean"] = data["food_price_inflation"].rolling(6).mean()
    data["inflation_roll3_std"] = data["food_price_inflation"].rolling(3).std()

    return data.dropna()


def split_train_test(data):
    # Last 12 months as test set (2023)
    train = data.iloc[:-12]
    test = data.iloc[-12:]
    return train, test


if __name__ == "__main__":
    data = load_data()
    data = add_features(data)
    train, test = split_train_test(data)
    print(data.head().to_string())
    print("total", data.shape)
    print("train", train.shape, "test", test.shape)
