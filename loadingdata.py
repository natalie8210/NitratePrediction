from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from dotenv import load_dotenv
import numpy as np
import json

#load_dotenv()

BASE = "https://pi-vision.facilities.uiowa.edu/piwebapi"
#USERNAME = os.getenv("PI_USERNAME")
#PASSWORD = os.getenv("PI_PASSWORD")
#AUTH = (USERNAME, PASSWORD)

#TAG_PATH = r"\\piserver.facilities.uiowa.edu\WP_WC_Nitrate_River"

BAD_STATE_NAMES = {
    "No Data", "Bad Input", "Configure", "Pt Created", "Shutdown", "I/O Timeout"
}

def normalize_pi_value(v):
    """
    Returns (value_num, value_str, value_raw_json)
    - value_num: float or NaN
    - value_str: string label (or None)
    - value_raw_json: raw JSON string for debugging
    """
    # raw json (safe)
    try:
        value_raw = json.dumps(v, default=str)
    except Exception:
        value_raw = str(v)

    # Struct/dict case (digital/system states often)
    if isinstance(v, dict):
        name = v.get("Name")
        inner = v.get("Value")

        # If it's a known bad/system state, treat as missing
        if isinstance(name, str) and name in BAD_STATE_NAMES:
            return (np.nan, name, value_raw)

        # Prefer numeric inner value if it exists
        if isinstance(inner, (int, float)):
            return (float(inner), name if isinstance(name, str) else None, value_raw)

        # Sometimes inner is numeric-looking string
        if isinstance(inner, str):
            num = pd.to_numeric(inner, errors="coerce")
            if pd.notna(num):
                return (float(num), name if isinstance(name, str) else None, value_raw)

        # Fallback: store the name as text
        if isinstance(name, str):
            return (np.nan, name, value_raw)

        return (np.nan, None, value_raw)

    # Plain numeric
    if isinstance(v, (int, float)):
        return (float(v), None, value_raw)

    # String (try numeric coercion, else keep text)
    if isinstance(v, str):
        num = pd.to_numeric(v, errors="coerce")
        if pd.notna(num):
            return (float(num), None, value_raw)
        return (np.nan, v, value_raw)

    # Anything else
    return (np.nan, str(v), value_raw)

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

def get_point_webid_by_path(session: requests.Session, tag_path: str) -> str:
    """Resolve \\piserver...\\TAG to a point WebId."""
    r = session.get(f"{BASE}/points", params={"path": tag_path})
    r.raise_for_status()
    j = r.json()
    if "WebId" not in j:
        raise RuntimeError(f"No WebId in /points response for path={tag_path}. Keys: {list(j.keys())}")
    return j["WebId"]


def fetch_recorded_all(
    session: requests.Session,
    webid: str,
    start_time: str = "*-5y",
    end_time: str = "*",
    page_size: int = 10000,
) -> pd.DataFrame:
    """
    Pull recorded data, following Links.Next until exhausted.
    Returns a DataFrame with columns: timestamp, value
    """
    url = f"{BASE}/streams/{webid}/recorded"
    params = {
        "startTime": start_time,
        "endTime": end_time,
        "maxCount": page_size,
        "selectedFields": "Items.Timestamp;Items.Value;Links.Next",
    }

    rows: list[dict] = []

    while True:
        r = session.get(url, params=params)
        r.raise_for_status()
        j = r.json()

        items = j.get("Items", [])
        for it in items:
            ts = it.get("Timestamp")
            if ts is None:
                continue
            v = it.get("Value")

            value_num, value_str, value_raw = normalize_pi_value(v)

            rows.append({
                "timestamp": ts,
                "value_num": value_num,
                "value_str": value_str,
                "value_raw": value_raw,
            })
            #rows.append({"timestamp": it.get("Timestamp"), "value": it.get("Value")})

        next_link = (j.get("Links") or {}).get("Next")
        if not next_link:
            break

        # Next is usually a full URL; once we use it, params must be empty
        url = next_link
        params = None

    df = pd.DataFrame(rows, columns=["timestamp", "value_num", "value_str", "value_raw"])
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def pull_one_tag_to_parquet(
    tag_name: str,
    tag_path: str,
    raw_dir: Path,
    start_time: str = "*-5y",
    end_time: str = "*",
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as s:
        s.auth = get_auth()

        webid = get_point_webid_by_path(s, tag_path)
        df = fetch_recorded_all(s, webid, start_time=start_time, end_time=end_time)

    outpath = raw_dir / f"{tag_name}.parquet"
    # Store both the tag and path for provenance
    if not df.empty:
        df.insert(0, "tag", tag_name)
        df.insert(1, "path", tag_path)
    df.to_parquet(outpath, index=False)
    return outpath


if __name__ == "__main__":
    RAW_DIR = Path("DATA") / "RAW"

    TAG_NAMES = [
        "IowaRiver_IowaCity_NitrateLevel",
        "WP_WC_Nitrate_River",
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

    # Build tag_name -> full path
    tag_to_path = {t: rf"\\piserver.facilities.uiowa.edu\{t}" for t in TAG_NAMES}

    for tag, path in tag_to_path.items():
        try:
            out = pull_one_tag_to_parquet(tag, path, RAW_DIR, start_time="*-5y", end_time="*")
            print(f"Saved {tag} -> {out}")
        except Exception as e:
            print(f"[FAILED] {tag} ({path}) :: {e}")

#resp = requests.get(
#    f"{BASE}/points",
#    params={"path": TAG_PATH},
#    auth=AUTH
#)
#resp.raise_for_status()

#point_json = resp.json()

#WEBID = point_json["WebId"]  

#params = {
#    "startTime": "*-1y",
#    "endTime": "*",
#    "maxCount": 100000, 
#    "selectedFields": "Items.Timestamp;Items.Value"
#}

#data_resp = requests.get(
#    f"{BASE}/streams/{WEBID}/recorded",
#    params=params,
#    auth=AUTH
#)
#data_resp.raise_for_status()

#data_json = data_resp.json()
#items = data_json["Items"]

#print("Number of observations:", len(items))
#print(items[:3])  