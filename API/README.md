# Iowa River Nitrate Forecasting & Synoptic Weather Data Pipeline

This README covers the two Python scripts provided for (1) forecasting nitrate levels and writing results to PI, and (2) pulling daily weather data from the Synoptic API and writing it to PI.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Script 1: forecast_and_write_nitrates.py](#script-1-forecast_and_write_nitratespy)
3. [Script 2: synoptic_api.py](#script-2-synoptic_apipy)
4. [Toggleable Settings](#toggleable-settings)
5. [PI Authentication](#pi-authentication)
6. [Running the Scripts](#running-the-scripts)

---

## Prerequisites

Install the required Python packages before running either script:

```bash
pip install pandas numpy statsmodels requests
```

You will also need:
- A **Synoptic API token** (see [Script 2 setup](#script-2-synoptic_apipy))
- Your **PI Web API credentials** (username and password)

---

## Script 1: `forecast_and_write_nitrates.py`

### What it does
This script reads a CSV of historical daily nitrate, flow, and precipitation data, fits an ARIMAX model to it, and produces a 4-day nitrate forecast. A minimum of 1–2 years of data is required, though 3–5 years is recommended for best accuracy. The forecast is then written to the PI tag:
```
IowaRiver_IowaCity_NitrateLevel_UIDataScience_Forecast
```

### Input Data

The script reads from a CSV file. Update this line to point to your file:

```python
df_daily = pd.read_csv("daily_nitrate.csv")  # ← Change to your file path
```

The CSV must contain the following columns:

| Column | Description |
|--------|-------------|
| `timestamp` | Date of observation (e.g. `2026-04-01`) |
| `WP_WC_Nitrate_River` | Daily nitrate measurement |
| `IowaRiver_IowaCity_Flow` | Daily river flow |
| `IowaCity_Weather_SynopticDailyPrecipitationAccumulation` | Daily precipitation accumulation |

### Output

The script prints a forecast table and writes it to PI:

| Column | Description |
|--------|-------------|
| `forecast_date` | The forecasted date |
| `nitrate_forecast` | Predicted nitrate level |

---

## Script 2: `synoptic_api.py`

### What it does
This script pulls daily weather data from the [Synoptic Data API](https://docs.synopticdata.com/services/welcome-to-synoptic-data-s-web-services) for a specified station and date range, fills any missing days using interpolation, and writes each weather variable to its corresponding PI tag.

### PI Tags Written

| Variable | PI Tag |
|----------|--------|
| Air Temperature | `IowaCity_Weather_SynopticDailyAirTemp` |
| Snow Depth | `IowaCity_Weather_SynopticDailySnowDepth` |
| Snow Accumulation | `IowaCity_Weather_SynopticDailySnowAccumulation` |
| Precipitation Accumulation | `IowaCity_Weather_SynopticDailyPrecipitationAccumulation` |
| Air Temp High | `IowaCity_Weather_SynopticDailyAirTempHigh` |
| Air Temp Low | `IowaCity_Weather_SynopticDailyAirTempLow` |

### Synoptic API Token Setup

You need a Synoptic API token to pull data. To get one:

1. Sign up at https://customer.synopticdata.com/activate/
2. Once you have your API key, generate a token by visiting this URL in your browser:
   ```
   https://api.synopticdata.com/v2/auth?apikey=YOUR_API_KEY
   ```
3. Copy the token from the response and paste it into the script:
   ```python
   TOKEN = "your_token_here"
   ```

---

## Toggleable Settings

### Script 2 — `synoptic_api.py`

These settings are at the top of the script:

```python
TOKEN = "your_token_here"   # Your Synoptic API token (required).

STATION_ID = "COOPICYI4"    # The Synoptic station ID to pull data from.
                             # COOPICYI4 is the Iowa City COOP station.

START_DATE = "202604081521" # Start of the data pull window.
END_DATE   = "202604301521" # End of the data pull window.
                             # Format: YYYYmmddHHMM in UTC.

# Units for returned data. Options: "english" (°F, inches) or "metric" (°C, mm)
"units": "english"
```

---

## PI Authentication

Both scripts connect to the PI Web API using your university credentials. Update this section in **both scripts** before running:

```python
session = requests.Session()
session.auth = ("your_username", "your_password")  # ← Enter your credentials here
```


---

## Running the Scripts

Run each script from the command line or from within a Jupyter notebook:

```bash
# Pull weather data and write to PI
python synoptic_api.py

# Run nitrate forecast and write to PI
python forecast_and_write_nitrates.py
```

### Recommended order of operations

1. Run `synoptic_api.py` first to ensure the latest precipitation data is in PI, as it is used as a covariate in the nitrate forecast model.
2. Run `forecast_and_write_nitrates.py` after to generate and post the updated 4-day forecast.

---

## Troubleshooting

**`Missing required columns` error in forecast script**
Ensure your input CSV has all four required columns with the exact names listed in the [Input Data](#input-data) section above.

**`Status: 200` / authentication error from Synoptic**
Your token may be expired or invalid. Generate a new one using your API key as described in the [Synoptic API Token Setup](#synoptic-api-token-setup) section.

**PI write returns a non-200 status**
This typically means a permissions issue with the PI Web API. Confirm your credentials are correct and that your account has write access to the relevant PI tags.

**Missing days in weather data**
The script automatically detects and fills gaps of up to 3 consecutive missing days using time-based interpolation. Gaps larger than 3 days will remain as `NaN` and will be skipped when writing to PI.
