# Botswana Food Inflation Forecasting

## About the Project

This project was done for the IndabaX Botswana AI Hackathon. The aim of the project is to predict Botswana's monthly food price inflation using historical economic data. We also looked at how changes in food inflation may affect some human capital indicators.

During the project we tested both **SARIMAX** and **LSTM** models. After comparing the results, we decided to use the **SARIMAX** model for the final predictions because it performed better on our data.

---

## Project Files

```
src/
    preprocessing.py
    models.py
    evaluation.py
    human_capital_analysis.py

outputs/
figures/
requirements.txt
README.md
```

### preprocessing.py

Reads all the datasets, cleans them, joins them together and creates the features used by the models.

### models.py

Contains the forecasting models used in this project. We experimented with both SARIMAX and LSTM during model development.

### evaluation.py

Runs the forecasting model and creates the final `predictions.csv` file for submission.

### human_capital_analysis.py

Analyses how food inflation is related to human capital indicators and saves the results.

---

## Installing the Project

Install the required packages using:

```bash
py -3 -m pip install -r requirements.txt
```

---

## Running the Forecast

```bash
py -3 src/evaluation.py
```

---

## Running the Human Capital Analysis

```bash
py -3 src/human_capital_analysis.py
```

---

## Output Files

After running the project you should get:

- `outputs/predictions.csv` – the final food inflation forecasts.
- `outputs/human_capital_projections.csv` – projected human capital results.
- Graphs saved inside the `figures` folder.

---

## Datasets Used

The project uses the datasets provided for the hackathon:

- Baltic Dry Index
- Brent Crude Oil Prices
- Botswana Policy Rate
- FAO Food Price Data
- Human Capital Data

---

## Final Model

Although we experimented with both SARIMAX and LSTM, the final submission uses the SARIMAX model because it gave better forecasting results.

