# Botswana Food Inflation Forecasting Project

## Overview

This repository implements a reproducible forecasting workflow for Botswana food price inflation and a companion human-capital analysis based only on the datasets supplied in the repository.

## Repository Structure

- src/preprocessing.py: loads the raw data, aligns the monthly series, and creates the engineered features.
- src/models.py: trains the forecasting model used for the submission output.
- src/evaluation.py: fits the model, generates the official forecast, and writes outputs/predictions.csv.
- src/human_capital_analysis.py: estimates regressions between food inflation and the available Botswana human-capital indicators and writes projected impacts.
- outputs/: submission output files and projection tables.
- figures/: generated plots.

## Installation

```bash
py -3 -m pip install -r requirements.txt
```

## Running the Project

Generate the submission forecast:

```bash
py -3 src/evaluation.py
```

Generate the human-capital analysis outputs:

```bash
py -3 src/human_capital_analysis.py
```

## Outputs

The main submission artifact is outputs/predictions.csv, which contains:

- year_month
- forecast

The human-capital workflow writes outputs/human_capital_projections.csv and several plots to figures/.

## Notes

The forecasting model used for the final submission is a single SARIMAX model. The report and projection outputs are generated directly from the repository code and the submission forecast.
