# Botswana Food Price Inflation Forecasting

This is our submission for the IndabaX Botswana AI Hackathon. We forecast Botswana's monthly food price inflation using two models - SARIMAX and LSTM - and compare which works better.

## What we did

Food prices in Botswana have been pretty volatile lately, especially after the post-COVID inflation spike and the war in Ukraine pushing up energy costs. We wanted to see if we could forecast the food price inflation using a mix of global shipping costs (Baltic Dry Index), oil prices (Brent), and Botswana's own policy rate.

We also looked at how food inflation might affect human capital indicators like school attendance or health outcomes, but that part is separate from the actual forecasting.

## Project structure

```
src/
    preprocessing.py      - loads data, cleans it, creates features
    models.py             - SARIMAX and LSTM models
    evaluation.py         - trains models, compares them, saves outputs
    human_capital_analysis.py - links inflation to human capital indicators

data/raw/                - the five datasets given for the hackathon
outputs/                 - predictions.csv, metrics.csv, figures
figures/                 - all the plots
requirements.txt
README.md
```

## Running it

You need Python 3.11 or 3.12 (TensorFlow doesn't support 3.13+ yet).

```bash
py -3 -m pip install -r requirements.txt
py -3 src/evaluation.py
py -3 src/human_capital_analysis.py
```

If TensorFlow is installed, the LSTM will run. If not, it skips gracefully and you still get the SARIMAX results.

## Datasets

We used all five datasets provided:

- **Baltic Dry Index** (daily -> monthly features like mean, volatility, momentum)
- **Brent Crude Oil** (monthly prices with lags and rolling averages)
- **Botswana Policy Rate** (monthly with lags and change indicators)
- **FAO Food Prices** (the target - Botswana food CPI and inflation)
- **Human Capital Data** (only used after forecasting, not as a model input)

## Feature engineering

For each dataset we tried to capture different things:

- **BDI**: monthly mean, standard deviation, range, volatility, 3-month and 6-month momentum, trend, extreme day count, rolling averages
- **Brent**: 3 lags, rolling means, monthly change
- **Policy Rate**: 3 lags, rolling mean, monthly change, increase/decrease flag
- **Food CPI**: lags 1-6 months, rolling mean, rolling std

We dropped any row with missing values after creating lags.

## Lag selection

we actually tested lags 1 through 6 months for each variable using correlation, cross-correlation, AIC and BIC. The best lag for each feature was selected automatically and documented.

## SARIMAX

We tuned `p`, `d`, `q` using grid search with AIC, and also tested seasonal parameters. We used walk-forward validation to get more realistic error estimates. The model outputs RMSE, MAE, MAPE, confidence intervals, and we checked residuals with the Ljung-Box test.

## LSTM

The LSTM uses sequences of 12 months. We normalize features and targets separately, use EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau. The network has two LSTM layers with dropout.

## Model comparison

We compare both models on the same test set using RMSE, MAE, and MAPE. The better one is used for the final `predictions.csv`.

## Outputs

After running, you get:

- `outputs/predictions.csv` - 12-month forecast
- `outputs/human_capital_projections.csv` - projected human capital indicators
- `outputs/metrics.csv` - model comparison
- `outputs/feature_importances.csv` - SARIMAX-based feature importance
- `figures/` - forecast plot, prediction vs actual, residuals, dataset comparisons, lag correlations, human capital charts

## Limitations

- Only about 20 years of monthly data, so deep learning doesn't have much to work with
- Future exogenous values are assumed flat at their last observed value

