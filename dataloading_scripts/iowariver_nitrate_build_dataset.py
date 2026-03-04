from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE = "https://pi-vision.facilities.uiowa.edu/piwebapi"

BAD_STATE_NAMES = {"No Data", "Bad Input", "Configure", "Pt Created", "Shutdown", "I/O Timeout"}

# -----------------------------
# CONFIG
# -----------------------------
TARGET_TAG = "IowaRiver_IowaCity_NitrateLevel"
TARGET_PATH = rf"\\piserver.facilities.uiowa.edu\{TARGET_TAG}"

PREDICTOR_TAGS = [
    "IowaRiver_IowaCity_Flow",
    "IowaRiver_IowaCity_GaugeHeight",
    "IowaRiver_BelowCoralvilleDam_Flow",
    "IowaRiver_BelowCoralvilleDam_GaugeHeight",
    "CoralvilleReservoir_USACE_Outflow_Forecast",
    "CoralvilleReservoir_USACE_Level_Forecast",
    "CoralvilleReservoir_USACE_Level",
    "CoralvilleReservoir_USACE_Outflow",
    "CoralvilleReservoir_USACE_Inflow_Davg",
    "IowaCityAirport_Weather_Temperature",
    "IowaCityAirport_Weather_RelativeHumidity",
    "IowaCityAirport_Weather_Precipitation_Dtot",
    "WCP_00_TT_091",
    "WCP_00_MT_091",
    "WCP_00_LS_091",
]

START_TIME = "*-5y"
END_TIME = "*"

# target nitrate recorded pull (hourly-ish, so this is plenty)
RECORDED_MAXCOUNT = 90000

# predictor interpolated pull
PRED_INTERVAL = "1h"
INTERP_MAXCOUNT = 50000  # 1h * 5y ~ 44k

# alignment behavior: require real predictor points within this distance on both sides
MAX_BRACKET = pd.Timedelta("3D")

OUT_DIR = Path("DATA") / "RAW"
OUT_CSV = OUT_DIR / f"{TARGET_TAG}_hourly_with_covariates_5y.csv"


# -----------------------------
# Auth + parsing
# -----------------------------
def get_auth() -> tuple[str, str]:
    load_dotenv()
    user = os.getenv("PI_USERNAME")
    pw = os.getenv("PI_PASSWORD")
    if not user or not pw:
        raise RuntimeError(
            "Missing PI_USERNAME / PI_PASSWORD. Check that:\n"
            "1) your .env is in the same folder you run the script from\n"
            "2) load_dotenv() is called before os.getenv\n"
            "3) .env lines look like PI_USERNAME=... (no quotes needed)"
        )
    return (user, pw)


def normalize_pi_value(v):
    """Return numeric float or NaN."""
    if isinstance(v, dict):
        name = v.get("Name")
        inner = v.get("Value")

        if isinstance(name, str) and name in BAD_STATE_NAMES:
            return np.nan

        if isinstance(inner, (int, float)):
            return float(inner)

        if isinstance(inner, str):
            num = pd.to_numeric(inner, errors="coerce")
            return float(num) if pd.notna(num) else np.nan

        return np.nan

    if isinstance(v, (int, float)):
        return float(v)

    if isinstance(v, str):
        num = pd.to_numeric(v, errors="coerce")
        return float(num) if pd.notna(num) else np.nan

    return np.nan


# -----------------------------
# PI Web API calls (no paging)
# -----------------------------
def get_point_webid_by_path(session: requests.Session, tag_path: str) -> str:
    r = session.get(f"{BASE}/points", params={"path": tag_path})
    r.raise_for_status()
    j = r.json()
    if "WebId" not in j:
        raise RuntimeError(f"No WebId in /points response for path={tag_path}. Keys: {list(j.keys())}")
    return j["WebId"]



def fetch_recorded_once(
    session: requests.Session,
    webid: str,
    start_time: str,
    end_time: str,
    max_count: int,
) -> pd.DataFrame:
    """Single-shot recorded pull."""
    url = f"{BASE}/streams/{webid}/recorded"
    params = {
        "startTime": start_time,
        "endTime": end_time,
        "maxCount": max_count,
        "selectedFields": "Items.Timestamp;Items.Value",
    }

    r = session.get(url, params=params)
    r.raise_for_status()
    j = r.json()
    items = j.get("Items", []) or []

    rows = []
    for it in items:
        ts = it.get("Timestamp")
        if ts is None:
            continue
        v = it.get("Value")
        rows.append((ts, normalize_pi_value(v)))

    df = pd.DataFrame(rows, columns=["timestamp", "value_num"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_interpolated_numeric_once(
    session: requests.Session,
    webid: str,
    start_time: str,
    end_time: str,
    interval: str,
    max_count: int,
) -> pd.Series:
    """
    Single-shot interpolated pull -> numeric Series indexed by timestamp (UTC).
    """
    url = f"{BASE}/streams/{webid}/interpolated"
    params = {
        "startTime": start_time,
        "endTime": end_time,
        "interval": interval,
        "maxCount": max_count,
        "selectedFields": "Items.Timestamp;Items.Value",
    }

    r = session.get(url, params=params)
    r.raise_for_status()
    j = r.json()
    items = j.get("Items", []) or []

    if not items:
        return pd.Series(dtype="float64")

    rows = []
    for it in items:
        ts = it.get("Timestamp")
        if ts is None:
            continue
        v = it.get("Value")
        rows.append((ts, normalize_pi_value(v)))

    df = pd.DataFrame(rows, columns=["timestamp", "value_num"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return pd.Series(dtype="float64")

    s = df.set_index("timestamp")["value_num"].astype("float64")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


# -----------------------------
# Hourly reduction for target
# -----------------------------
def reduce_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep ~1 value per hour by selecting the observation closest to the top of the hour.
    """
    if df.empty:
        return df

    df = df.copy()
    df["hour"] = df["timestamp"].dt.floor("h")
    df["dist_to_hour_s"] = (df["timestamp"] - df["hour"]).dt.total_seconds().abs()

    df = df.sort_values(["hour", "dist_to_hour_s", "timestamp"])
    chosen = df.groupby("hour", as_index=False).first()

    chosen = chosen.drop(columns=["timestamp", "dist_to_hour_s"], errors="ignore")
    chosen = chosen.rename(columns={"hour": "timestamp"})
    chosen = chosen[["timestamp", "value_num"]].sort_values("timestamp").reset_index(drop=True)
    return chosen


# -----------------------------
# Align predictors to hourly nitrate timestamps
# -----------------------------
def align_predictor_to_targets(
    predictor: pd.Series,
    target_times: pd.DatetimeIndex,
    max_bracket: pd.Timedelta,
) -> pd.Series:
    """
    Reindex predictor onto target_times (time interpolation), no extrapolation.
    Enforce bracketing tolerance so gaps become NaN.
    """
    target_times = pd.DatetimeIndex(pd.to_datetime(target_times, utc=True, errors="coerce")).dropna()
    target_times = pd.DatetimeIndex(target_times.unique()).sort_values()

    if predictor.empty:
        return pd.Series(index=target_times, dtype="float64", name=predictor.name)

    predictor = predictor.sort_index()
    obs_idx = predictor.index

    # interpolate on union index, inside only
    union_index = target_times.union(obs_idx)
    su = predictor.reindex(union_index)
    su_interp = su.interpolate(method="time", limit_area="inside")

    # bracketing check at each target time
    left_pos = obs_idx.searchsorted(target_times, side="right") - 1
    prev_obs = pd.DatetimeIndex([obs_idx[i] if i >= 0 else pd.NaT for i in left_pos])

    right_pos = obs_idx.searchsorted(target_times, side="left")
    next_obs = pd.DatetimeIndex([obs_idx[i] if i < len(obs_idx) else pd.NaT for i in right_pos])

    prev_dist = target_times - prev_obs
    next_dist = next_obs - target_times

    ok = (
        prev_obs.notna()
        & next_obs.notna()
        & (prev_dist <= max_bracket)
        & (next_dist <= max_bracket)
    )

    out = su_interp.reindex(target_times)
    out = out.where(ok, np.nan)
    out.name = predictor.name
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tag_to_path = {t: rf"\\piserver.facilities.uiowa.edu\{t}" for t in [TARGET_TAG] + PREDICTOR_TAGS}

    with requests.Session() as sess:
        sess.auth = get_auth()

        # 1) Pull target nitrate recorded and reduce to hourly
        target_webid = get_point_webid_by_path(sess, TARGET_PATH)
        raw_target = fetch_recorded_once(sess, target_webid, START_TIME, END_TIME, RECORDED_MAXCOUNT)

        if raw_target.empty:
            raise RuntimeError("Target nitrate series is empty.")

        target_hourly = reduce_to_hourly(raw_target)
        target_hourly = target_hourly.rename(columns={"value_num": TARGET_TAG})

        # Output frame indexed by hourly timestamps
        df_out = target_hourly.copy()
        df_out["timestamp"] = pd.to_datetime(df_out["timestamp"], utc=True)
        df_out = df_out.sort_values("timestamp").reset_index(drop=True)
        target_times = pd.DatetimeIndex(df_out["timestamp"].values)

        print("Target raw rows:", len(raw_target))
        print("Target hourly rows:", len(df_out))
        deltas = df_out["timestamp"].diff().dropna()
        if not deltas.empty:
            print("\nTarget most common gaps (top 10):")
            print(deltas.value_counts().head(10))

        # 2) Pull predictors interpolated (1h) and align to target hourly timestamps
        for tag in PREDICTOR_TAGS:
            print(f"\n--- Pulling predictor {tag} via interpolated interval={PRED_INTERVAL} ---")
            webid = get_point_webid_by_path(sess, tag_to_path[tag])

            pred = fetch_interpolated_numeric_once(
                sess,
                webid,
                START_TIME,
                END_TIME,
                interval=PRED_INTERVAL,
                max_count=INTERP_MAXCOUNT,
            )
            pred.name = tag

            aligned = align_predictor_to_targets(pred, target_times=target_times, max_bracket=MAX_BRACKET)
            df_out[tag] = aligned.values

    # 3) Save
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    print("\nPreview:")
    print(df_out.head(15))
    print("\nMissingness (top 15):")
    print(df_out.isna().mean().sort_values(ascending=False).head(15))


if __name__ == "__main__":
    main()