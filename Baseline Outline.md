# Nitrate Forecasting Outline

# Step 1: Build the hourly (or another relevant frequency) dataset

## Goal

Create a single, aligned time-indexed dataset where all variables share the same hourly timestamp structure.

## Description

Data sources such as nitrate sensors, river flow measurements, and weather variables exist at different time frequencies (e.g., minutes, hours, daily). Forecasting models require consistent temporal alignment, so the first step is to resample all variables onto a unified hourly grid.

## Methodology

* Define an hourly master time index covering the full study period (may need to re-consult about this).

* Resample each dataset to hourly frequency:
  * Sensor measurements (hourly mean or median)
  * Precipitation (hourly sum)
  * Temperature and flow (hourly average)
  * Operational events (convert to state indicators or hourly aggregates)

* Merge all sources into one modeling table using the timestamp as the key.

* Address missing data:
  * Short gaps 
  * Longer gaps 
  * Preserve missingness indicators as features where appropriate.

## Output

A clean dataframe structured as:

timestamp | nitrate | temperature | precipitation | flow | reservoir_state | ...


# Step 2: EDA/Cross-Correlation/Lag Discovery

## Goal
* Explore data, make some visuals (we don't need to spend a whole lot of time on this part because the clients mostly know this information already)
* Identify delayed relationships between nitrate levels and potential predictors to inform feature engineering and model structure.

## Description (correlation/lag discovery)

Environmental processes exhibit lagged responses. For example, precipitation may influence nitrate concentrations several hours or days later. Cross-correlation analysis helps determine these temporal dependencies.

## Methodology (correlation/lag discovery)

* Compute autocorrelation (ACF/PACF) for nitrate to understand temporal structure.
* Calculate cross-correlation functions (CCF) between nitrate and candidate predictors.
* Examine positive and negative lags to identify leading indicators.
* Focus on physically interpretable lag ranges (e.g., 0–72 hrs).
* Validate findings visually using lag plots and domain knowledge.

## Output

A set of candidate lagged predictors such as:
* precip_lag_12
* flow_lag_6
* temperature_lag_24

These will guide feature engineering for forecasting models.


# Step 3: Create a baseline model (I was thinking SARIMAX)

## Description

SARIMAX (Seasonal ARIMA with exogenous variables) provides a strong initial model because it accounts for autoregressive behavior while allowing weather and operational variables to influence predictions.

## Methodology

* Use the unified hourly dataset as input.
* Select autoregressive and seasonal orders based on ACF/PACF diagnostics.
* Include lagged external variables identified in Step 2 as exogenous regressors.
* Fit the model on a training window using maximum likelihood estimation.

* Diagnose residuals:
  * Check for remaining autocorrelation.
  * Evaluate normality and variance stability.

* Ensure model interpretability by reviewing coefficient signs and magnitudes.

## Output

A baseline forecasting model capable of producing hourly nitrate predictions and interpretable relationships with external drivers.


# Step 4: Rolling Forecasts

## Goal

Assess model performance using realistic time-series validation rather than random train/test splits.

## Description

Forecast models must be evaluated in a way that mimics real-time deployment. Rolling forecasts simulate how predictions would perform as new data arrives.

## Methodology

* Implement rolling or expanding window validation:
  * Train on historical data up to time *t*.
  * Forecast the next horizon (e.g., 6h, 24h).
  * Move the training window forward and repeat.

* Compare predicted vs. observed nitrate levels across all forecast windows.

* Use performance metrics appropriate for forecasting:
  * RMSE or MAE for continuous prediction.
  * Threshold-based metrics (precision, recall) if evaluating early-warning alerts.
* Visualize forecast accuracy over time to identify systematic bias or drift.

## Output

Quantitative performance metrics and diagnostic plots that evaluate the reliability of the early-warning system.


## Notes:

* This is a simple outline to help us decide what further steps we need to take that align with client needs. After our baseline is complete and we have something interpretable for the clients, then we can explore more complex models as well as the integration of our model into a dashboard. 
