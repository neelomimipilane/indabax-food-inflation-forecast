# Feature Engineering Report

## Objective

This report documents the feature engineering workflow implemented in the repository for Botswana food-inflation forecasting.

## Data Sources

The forecasting pipeline uses the repository data files:

- data/raw/01_baltic_dry_index_daily.csv
- data/raw/02_brent_crude_monthly.csv
- data/raw/04_fao_botswana_prices.csv

The human-capital analysis also uses data/raw/05_human_capital_project.csv.

## Feature Engineering Steps

1. Daily Baltic Dry Index values were aggregated to monthly statistics: mean, standard deviation, range, minimum, and maximum.
2. A monthly count of extreme BDI days was computed from daily closes above the monthly mean plus one standard deviation.
3. Monthly BDI momentum features were created over 3-month and 6-month horizons.
4. A BDI trend feature was constructed as the difference between 3-month and 6-month moving averages.
5. Brent crude prices were converted to monthly averages and aligned to the monthly index.
6. FAO price indices were pivoted so general CPI, food CPI, and food price inflation were available as separate columns.
7. Inflation lags at 1, 3, 6, and 12 months were created to capture persistence.
8. Rolling statistics for inflation were created using 3-month and 6-month windows.
9. The target is the year-on-year food-price inflation series derived from the FAO food CPI index.

## Implemented Features

The final modeling frame includes the following features:

- brent_price
- general_cpi
- food_cpi
- bdi_mean
- bdi_std
- bdi_range
- bdi_momentum_3
- bdi_momentum_6
- bdi_trend
- bdi_extreme_days
- inflation_lag_1
- inflation_lag_3
- inflation_lag_6
- inflation_lag_12
- brent_lag_1
- bdi_lag_1
- inflation_roll3_mean
- inflation_roll6_mean
- inflation_roll3_std

## Rationale

The feature set combines imported-cost proxies, inflation persistence, and short-run volatility signals. BDI and Brent capture global shipping and energy-cost pressure, while the lag and rolling features capture persistence and short-run momentum in food inflation.

## Modeling Notes

The final submission pipeline uses a single SARIMAX model with the engineered features. The model is fit on the full available engineered frame and used to generate the six-month submission forecast written to outputs/predictions.csv.
