from pathlib import Path

import pandas as pd


def load_data():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "raw"

    bdi = pd.read_csv(data_dir / "01_baltic_dry_index_daily.csv", parse_dates=["Date"])
    brent = pd.read_csv(data_dir / "02_brent_crude_monthly.csv", parse_dates=["Date"])
    policy = pd.read_csv(data_dir / "03_botswana_policy_rate.csv", parse_dates=["Date"])
    prices = pd.read_csv(data_dir / "04_fao_botswana_prices.csv", parse_dates=["Date"])

    monthly_bdi = aggregate_monthly_bdi(bdi)

    brent = brent.rename(columns={"Brent_USD_per_barrel": "brent_price"})
    brent["month"] = brent["Date"].dt.to_period("M")
    brent = brent.groupby("month")["brent_price"].mean().to_frame()

    policy = policy.rename(columns={"policy_rate": "policy_rate"})
    policy["month"] = policy["Date"].dt.to_period("M")
    policy = policy.groupby("month")["policy_rate"].mean().to_frame()

    prices = prices.pivot_table(index="Date", columns="Item", values="Value", aggfunc="first")
    prices = prices.reset_index()
    prices = prices.rename(
        columns={
            "Consumer Prices, General Indices (2015 = 100)": "general_cpi",
            "Consumer Prices, Food Indices (2015 = 100)": "food_cpi",
        }
    )
    prices["month"] = prices["Date"].dt.to_period("M")
    prices = prices.groupby("month")[["general_cpi", "food_cpi"]].mean()

    data = merge_datasets(monthly_bdi, brent, policy, prices)
    data = clean_data(data)
    data["food_price_inflation"] = data["food_cpi"].pct_change(12) * 100
    data = data.dropna().copy()
    data.index = data.index.to_timestamp()
    return data


def clean_data(data):
    return data.dropna().copy()


def merge_datasets(bdi, brent, policy, prices):
    data = bdi.join(brent, how="left")
    data = data.join(policy, how="left")
    data = data.join(prices, how="left")
    return data.sort_index().ffill().bfill()


def aggregate_monthly_bdi(bdi):
    bdi = bdi.set_index("Date")
    monthly = bdi.resample("ME").agg({"BDI_Close": ["mean", "std", "min", "max"]})
    monthly.columns = ["bdi_mean", "bdi_std", "bdi_min", "bdi_max"]
    monthly["bdi_range"] = monthly["bdi_max"] - monthly["bdi_min"]
    monthly["bdi_return"] = monthly["bdi_mean"].pct_change()
    monthly = monthly[["bdi_mean", "bdi_std", "bdi_range", "bdi_return"]]
    monthly.index = monthly.index.to_period("M")
    return monthly


def create_lag_features(data):
    features = data.copy()

    for lag in [1, 3, 6, 12]:
        features[f"food_price_inflation_lag_{lag}"] = features["food_price_inflation"].shift(lag)

    return features.dropna()


def split_train_test(data):
    train = data.iloc[:-12]
    test = data.iloc[-12:]
    return train, test


if __name__ == "__main__":
    data = load_data()
    features = create_lag_features(data)
    train, test = split_train_test(features)
    print(data.head().to_string())
    print("features", features.shape)
    print("train", train.shape, "test", test.shape)
