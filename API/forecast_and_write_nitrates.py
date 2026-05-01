import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
HORIZON = 4
FIXED_ORDER = (2, 1, 3)
K_BEST = 2
USE_LOG_NITRATE = True
USE_LOG_FLOW = False

# ── Fourier Terms Helper ──────────────────────────────────────────────────────
def make_fourier_terms(dates, k, period=365.25, h=None):
    """
    Generate Fourier sin/cos terms for seasonality.
    If h is provided, generates future terms starting after the last date.
    """
    if h is not None:
        last_date = dates.iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=h)
        t = (future_dates - dates.iloc[0]).days.values + 1
    else:
        t = (dates - dates.iloc[0]).dt.days.values + 1

    cols = {}
    for i in range(1, k + 1):
        cols[f"S{i}"] = np.sin(2 * np.pi * i * t / period)
        cols[f"C{i}"] = np.cos(2 * np.pi * i * t / period)
    return pd.DataFrame(cols)


# ── Main Forecast Function ────────────────────────────────────────────────────
def make_arimax_forecast(df_daily):
    df = df_daily.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Feature Engineering ───────────────────────────────────────────────────
    df["nitrate_response"] = np.log(df["WP_WC_Nitrate_River"]) if USE_LOG_NITRATE else df["WP_WC_Nitrate_River"]

    df["flow_lag4"] = df["IowaRiver_IowaCity_Flow"].shift(4)
    df["log_flow_lag4"] = np.log(df["IowaRiver_IowaCity_Flow"]).shift(4)
    df["flow_pred"]  = df["log_flow_lag4"] if USE_LOG_FLOW else df["flow_lag4"]
    df["flow_trend"] = df["IowaRiver_IowaCity_Flow"] - df["IowaRiver_IowaCity_Flow"].shift(7)

    df["flow_trend_lag4"]  = df["flow_trend"].shift(4)
    df["precip_roll7"]     = (
        df["IowaCity_Weather_SynopticDailyPrecipitationAccumulation"]
        .rolling(7, min_periods=1)
        .sum()
        .shift(4)
    )

    # ── Filter to complete cases ───────────────────────────────────────────────
    df_model = df.dropna(subset=[
        "nitrate_response", "flow_pred", "flow_trend_lag4", "precip_roll7"
    ]).reset_index(drop=True)

    # ── Fourier Terms ─────────────────────────────────────────────────────────
    fourier_train = make_fourier_terms(df_model["timestamp"], k=K_BEST)

    xreg_train = pd.concat([
        fourier_train,
        df_model[["flow_pred"]].rename(columns={"flow_pred": "flow"}),
        df_model[["flow_trend_lag4"]].rename(columns={"flow_trend_lag4": "flow_trend"}),
        df_model[["precip_roll7"]].rename(columns={"precip_roll7": "precip"})
    ], axis=1)

    # ── Fit ARIMAX ────────────────────────────────────────────────────────────
    fit = ARIMA(
        df_model["nitrate_response"].values,
        order=FIXED_ORDER,
        exog=xreg_train.values,
        trend="n"
    ).fit()

    # ── Build Future Regressors ───────────────────────────────────────────────
    n = len(df)
    lag_rows = df.iloc[(n - HORIZON):n].reset_index(drop=True)
    lag_indices = list(range(n - HORIZON, n))

    future_flow = lag_rows["IowaRiver_IowaCity_Flow"].values
    if USE_LOG_FLOW:
        future_flow = np.log(future_flow)

    future_flow_trend = np.array([
        df["IowaRiver_IowaCity_Flow"].iloc[i] - df["IowaRiver_IowaCity_Flow"].iloc[i - 7]
        for i in lag_indices
    ])

    future_precip = np.array([
        df["IowaCity_Weather_SynopticDailyPrecipitationAccumulation"].iloc[max(0, i - 6):i + 1].sum()
        for i in lag_indices
    ])

    fourier_future = make_fourier_terms(df_model["timestamp"], k=K_BEST, h=HORIZON)

    xreg_future = pd.DataFrame({
        **fourier_future.to_dict(orient="list"),
        "flow":       future_flow,
        "flow_trend": future_flow_trend,
        "precip":     future_precip
    })

    # ── Forecast ──────────────────────────────────────────────────────────────
    fc = fit.get_forecast(steps=HORIZON, exog=xreg_future.values)
    mean_pred = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)   # 95% CI
    ci80 = fc.conf_int(alpha=0.20) # 80% CI

    lo80 = ci80[:, 0]
    hi80 = ci80[:, 1]
    lo95 = ci[:, 0]
    hi95 = ci[:, 1]

    # ── Back-transform if log ─────────────────────────────────────────────────
    if USE_LOG_NITRATE:
        mean_pred = np.exp(mean_pred)
        lo80, hi80 = np.exp(lo80), np.exp(hi80)
        lo95, hi95 = np.exp(lo95), np.exp(hi95)

    # ── Build Output DataFrame ────────────────────────────────────────────────
    last_date = df["timestamp"].max()
    fc_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON)

    return pd.DataFrame({
        "forecast_date":     fc_dates,
        "horizon_day":       range(1, HORIZON + 1),
        "nitrate_forecast":  np.round(mean_pred, 3),
        "lo80":              np.round(lo80, 3),
        "hi80":              np.round(hi80, 3),
        "lo95":              np.round(lo95, 3),
        "hi95":              np.round(hi95, 3)
    })


# ── Run ───────────────────────────────────────────────────────────────────────
df_daily = pd.read_csv("daily_nitrate.csv") # Change as needed

# Sanity check
print(df_daily.dtypes)
print(df_daily.describe())

# Required columns check
required_cols = ["timestamp", "WP_WC_Nitrate_River", "IowaRiver_IowaCity_Flow", "IowaCity_Weather_SynopticDailyPrecipitationAccumulation"]
missing_cols = [c for c in required_cols if c not in df_daily.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Run forecast
forecast_4day = make_arimax_forecast(df_daily)
print(forecast_4day)



# ── Write to PI Tags ───────────────────────────────────────────────────────────────────────

from datetime import datetime
import os
import pandas as pd
import requests
import json

# ── Load credentials from config file ────────────────────────────────────────
with open("config.json", "r") as f:
    config = json.load(f)

session = requests.Session()
session.auth = (config["pi_username"], config["pi_password"])

NITRATE_FORECAST_TAG  = "IowaRiver_IowaCity_NitrateLevel_UIDataScience_Forecast"
BASE_URL = "https://pi-vision.facilities.uiowa.edu/piwebapi"
PI_SERVER = "\\piserver.facilities.uiowa.edu"
session.headers.update({
    "Accept": "application/json",
    "Content-Type": "application/json",
    "X-Requested-With": ""
})
def get_point_webid(tag_name):
    """
    Look up the PI WebID for a tag name.
    """
    tag_path = rf"{PI_SERVER}\{tag_name}"
    r = session.get(
        f"{BASE_URL}/points",
        params={"path": tag_path},
        timeout=60
    )
    print("\n" + "=" * 80)
    print("POINT LOOKUP")
    print(f"TAG: {tag_name}")
    print(f"PATH: {tag_path}")
    print(f"STATUS: {r.status_code}")
    print(f"FINAL URL: {r.url}")
    print(f"RESPONSE TEXT: {r.text[:1500]}")
    print("=" * 80)
    r.raise_for_status()
    return r.json()["WebId"]

def build_payload(rows, time_col, value_col):
    payload = []
    for _, row in rows.iterrows():
        ts = pd.to_datetime(row[time_col], utc=True)
        val = row[value_col]
        if pd.isna(val):
            continue
        payload.append({
            "Timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Value": float(val)
        })
    return payload

def write_recorded_values(
    tag_name,
    rows,
    time_col,
    value_col,
    update_option="Replace"
):
    webid = get_point_webid(tag_name)
    payload = build_payload(rows, time_col, value_col)
    if not payload:
        print(f"No valid values to write for {tag_name}")
        return None
    write_url = f"{BASE_URL}/streams/{webid}/recorded"
    params = {
        "updateOption": update_option,
        "bufferOption": "BufferIfPossible"
    }
    print("\n" + "=" * 80)
    print("RECORDED WRITE TEST")
    print(f"TAG: {tag_name}")
    print(f"WEBID: {webid}")
    print(f"WRITE URL: {write_url}")
    print(f"PARAMS: {params}")
    print("PAYLOAD PREVIEW:")
    print(payload[:3])
    print("=" * 80)

    r = session.post(
        write_url,
        params=params,
        json=payload,
        timeout=60
    )
    print("POST STATUS:", r.status_code)
    print("FINAL URL:", r.url)
    print("RESPONSE HEADERS:", dict(r.headers))
    print("RESPONSE TEXT:", r.text[:2000])
    if not r.ok:
        print(
            "\nWrite failed. Since point lookup succeeded but POST failed, "
            "this likely indicates PI write permission, Web API write access, "
            "or point data security issue."
        )
        r.raise_for_status()
    print(f"Wrote {len(payload)} values to {tag_name}")
    return r

forecast_4day = forecast_4day[['forecast_date','nitrate_forecast']]

# Clean timestamps and values
forecast_4day["forecast_date"] = pd.to_datetime(
    forecast_4day["forecast_date"],
    utc=True
    )

forecast_4day["nitrate_forecast"] = pd.to_numeric(
    forecast_4day["nitrate_forecast"],
    errors="coerce"
    )


write_recorded_values(
tag_name=NITRATE_FORECAST_TAG,
rows=forecast_4day,
time_col="forecast_date",
value_col="nitrate_forecast",
update_option="Replace"
)
