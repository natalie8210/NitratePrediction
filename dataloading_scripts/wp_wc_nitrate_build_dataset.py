from __future__ import annotations

import os
import json
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE = "https://pi-vision.facilities.uiowa.edu/piwebapi"

BAD_STATE_NAMES = {
    "No Data", "Bad Input", "Configure", "Pt Created", "Shutdown", "I/O Timeout"
}

# -----------------------------
# Auth + value parsing
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
    """
    Returns (value_num, value_str, value_raw_json)
    """
    try:
        value_raw = json.dumps(v, default=str)
    except Exception:
        value_raw = str(v)

    if isinstance(v, dict):
        name = v.get("Name")
        inner = v.get("Value")

        if isinstance(name, str) and name in BAD_STATE_NAMES:
            return (np.nan, name, value_raw)

        if isinstance(inner, (int, float)):
            return (float(inner), name if isinstance(name, str) else None, value_raw)

        if isinstance(inner, str):
            num = pd.to_numeric(inner, errors="coerce")
            if pd.notna(num):
                return (float(num), name if isinstance(name, str) else None, value_raw)

        if isinstance(name, str):
            return (np.nan, name, value_raw)

        return (np.nan, None, value_raw)

    if isinstance(v, (int, float)):
        return (float(v), None, value_raw)

    if isinstance(v, str):
        num = pd.to_numeric(v, errors="coerce")
        if pd.notna(num):
            return (float(num), None, value_raw)
        return (np.nan, v, value_raw)

    return (np.nan, str(v), value_raw)

# -----------------------------
# PI Web API basics
# -----------------------------
def get_point_webid_by_path(session: requests.Session, tag_path: str) -> str:
    r = session.get(f"{BASE}/points", params={"path": tag_path})
    r.raise_for_status()
    j = r.json()
    if "WebId" not in j:
        raise RuntimeError(f"No WebId in /points response for path={tag_path}. Keys: {list(j.keys())}")
    return j["WebId"]

def _fetch_paged_items(
    session: requests.Session,
    url: str,
    params: dict | None,
    debug_label: str = "",
) -> list[dict]:
    """
    Generic pager: keeps following Links.Next if present.
    Works for recorded/interpolated endpoints alike.
    """
    all_items: list[dict] = []
    page = 1

    while True:
        r = session.get(url, params=params)
        r.raise_for_status()
        j = r.json()

        items = j.get("Items", []) or []
        next_link = (j.get("Links") or {}).get("Next")

        if debug_label:
            print(f"[{debug_label}] Page {page}: items={len(items)} has_next={bool(next_link)}")

        all_items.extend(items)

        if not next_link:
            break

        url = urljoin(BASE, next_link)
        params = None
        page += 1

    return all_items

def fetch_recorded_numeric(
    session: requests.Session,
    webid: str,
    start_time: str,
    end_time: str,
    page_size: int = 10000,
    debug: bool = False,
) -> pd.Series:
    """
    Recorded values -> numeric Series indexed by timestamp (UTC).
    """
    url = f"{BASE}/streams/{webid}/recorded"
    params = {
        "startTime": start_time,
        "endTime": end_time,
        "maxCount": page_size,
        # Request Next link explicitly (helps many PI setups include Links.Next)
        "selectedFields": "Items.Timestamp;Items.Value;Links.Next",
    }

    items = _fetch_paged_items(session, url, params, debug_label="RECORDED" if debug else "")

    if not items:
        return pd.Series(dtype="float64")

    rows = []
    for it in items:
        ts = it.get("Timestamp")
        if ts is None:
            continue
        v = it.get("Value")
        value_num, _, _ = normalize_pi_value(v)
        rows.append((ts, value_num))

    df = pd.DataFrame(rows, columns=["timestamp", "value_num"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return pd.Series(dtype="float64")

    s = df.set_index("timestamp")["value_num"].astype("float64")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s

def fetch_interpolated_numeric(
    session: requests.Session,
    webid: str,
    start_time: str,
    end_time: str,
    interval: str,
    page_size: int = 100000,
    debug: bool = False,
) -> pd.Series:
    """
    Interpolated endpoint -> numeric Series indexed by timestamp (UTC).

    interval examples: "1h", "30m", "8h"
    """
    url = f"{BASE}/streams/{webid}/interpolated"
    params = {
        "startTime": start_time,
        "endTime": end_time,
        "interval": interval,
        "maxCount": page_size,
        "selectedFields": "Items.Timestamp;Items.Value;Links.Next",
    }

    items = _fetch_paged_items(session, url, params, debug_label="INTERP" if debug else "")

    if not items:
        return pd.Series(dtype="float64")

    rows = []
    for it in items:
        ts = it.get("Timestamp")
        if ts is None:
            continue
        v = it.get("Value")
        value_num, _, _ = normalize_pi_value(v)
        rows.append((ts, value_num))

    df = pd.DataFrame(rows, columns=["timestamp", "value_num"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return pd.Series(dtype="float64")

    s = df.set_index("timestamp")["value_num"].astype("float64")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s

# -----------------------------
# Align predictors to nitrate times
# -----------------------------
def align_predictor_to_nitrate(
    predictor: pd.Series,
    nitrate_index: pd.DatetimeIndex,
    max_bracket: pd.Timedelta = pd.Timedelta("3D"),
) -> pd.Series:
    """
    Interpolate predictor onto nitrate timestamps (time interpolation),
    but only keep values when each nitrate timestamp is bracketed by
    real predictor observations within max_bracket on both sides.

    No extrapolation.
    """
    nitrate_index = pd.DatetimeIndex(pd.to_datetime(nitrate_index, utc=True, errors="coerce")).dropna()
    nitrate_index = pd.DatetimeIndex(nitrate_index.unique()).sort_values()

    if predictor.empty:
        return pd.Series(index=nitrate_index, dtype="float64", name=predictor.name)

    predictor = predictor.sort_index()
    obs_idx = predictor.index

    # union index for time interpolation
    union_index = nitrate_index.union(obs_idx)
    su = predictor.reindex(union_index)

    # interpolate only inside observed domain
    su_interp = su.interpolate(method="time", limit_area="inside")

    # bracketing check
    left_pos = obs_idx.searchsorted(nitrate_index, side="right") - 1
    prev_obs = pd.DatetimeIndex([obs_idx[i] if i >= 0 else pd.NaT for i in left_pos])

    right_pos = obs_idx.searchsorted(nitrate_index, side="left")
    next_obs = pd.DatetimeIndex([obs_idx[i] if i < len(obs_idx) else pd.NaT for i in right_pos])

    prev_dist = nitrate_index - prev_obs
    next_dist = next_obs - nitrate_index

    ok = (
        prev_obs.notna()
        & next_obs.notna()
        & (prev_dist <= max_bracket)
        & (next_dist <= max_bracket)
    )

    out = su_interp.reindex(nitrate_index)
    out = out.where(ok, np.nan)
    out.name = predictor.name
    return out

# -----------------------------
# Main build
# -----------------------------
def main():
    # Output
    OUT_DIR = Path("DATA") / "RAW"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NITRATE_TAG = "WP_WC_Nitrate_River"
    out_csv = OUT_DIR / f"{NITRATE_TAG}_hourly_with_covariates_5y.csv"

    # Time range
    START_TIME = "*-5y"
    END_TIME = "*"

    # Interpolation settings
    PREDICTOR_INTERVAL = "1h"              # PI interpolated interval for predictors
    MAX_BRACKET = pd.Timedelta("3D")       # require predictor points within this distance on both sides

    # Tags
    NITRATE_TAG = "WP_WC_Nitrate_River"

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

    # Build full PI paths
    tag_to_path = {t: rf"\\piserver.facilities.uiowa.edu\{t}" for t in [NITRATE_TAG] + PREDICTOR_TAGS}

    with requests.Session() as sess:
        sess.auth = get_auth()

        # 1) Pull nitrate recorded (irregular, keep as-is)
        nitrate_webid = get_point_webid_by_path(sess, tag_to_path[NITRATE_TAG])
        nitrate = fetch_recorded_numeric(sess, nitrate_webid, START_TIME, END_TIME, page_size=10000, debug=True)
        nitrate.name = NITRATE_TAG

        if nitrate.empty:
            raise RuntimeError("Nitrate series is empty; cannot build dataset.")

        nitrate_index = nitrate.index

        # Start output frame on nitrate timestamps
        #df_out = pd.DataFrame({"timestamp": nitrate_index})
        #df_out = df_out.sort_values("timestamp").reset_index(drop=True)
        #df_out[NITRATE_TAG] = nitrate.reindex(df_out["timestamp"].values).values

        df_out = pd.DataFrame(index=nitrate_index)
        df_out.index.name = "timestamp"

        # align nitrate automatically
        df_out[NITRATE_TAG] = nitrate

        df_out = df_out.reset_index()

        # 2) Pull predictors as interpolated (1h), then align to nitrate timestamps
        for tag in PREDICTOR_TAGS:
            print(f"\n--- Predictor: {tag} (PI interpolated interval={PREDICTOR_INTERVAL}) ---")
            webid = get_point_webid_by_path(sess, tag_to_path[tag])

            pred_hourly = fetch_interpolated_numeric(
                sess,
                webid,
                START_TIME,
                END_TIME,
                interval=PREDICTOR_INTERVAL,
                page_size=200000,  # 1h * 5y ~ 43800, so this is plenty
                debug=False,
            )
            pred_hourly.name = tag

            aligned = align_predictor_to_nitrate(
                pred_hourly,
                nitrate_index=pd.DatetimeIndex(df_out["timestamp"].values),
                max_bracket=MAX_BRACKET,
            )
            df_out[tag] = aligned.values

        # 3) Save CSV
        df_out.to_csv(out_csv, index=False)
        print(f"\nSaved -> {out_csv}")

        # Quick inspection prints
        print("\nHead:")
        print(df_out.head(10))
        print("\nMissingness (top 15):")
        print(df_out.isna().mean().sort_values(ascending=False).head(15))

if __name__ == "__main__":
    main()