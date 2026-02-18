import requests
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

BASE = "https://pi-vision.facilities.uiowa.edu/piwebapi"
USERNAME = os.getenv("PI_USERNAME")
PASSWORD = os.getenv("PI_PASSWORD")

print (USERNAME)

AUTH = (USERNAME, PASSWORD)

TAG_PATH = r"\\piserver.facilities.uiowa.edu\WP_WC_Nitrate_River"

resp = requests.get(
    f"{BASE}/points",
    params={"path": TAG_PATH},
    auth=AUTH
)
resp.raise_for_status()

point_json = resp.json()

WEBID = point_json["WebId"]  

params = {
    "startTime": "*-1y",
    "endTime": "*",
    "maxCount": 100000, 
    "selectedFields": "Items.Timestamp;Items.Value"
}

data_resp = requests.get(
    f"{BASE}/streams/{WEBID}/recorded",
    params=params,
    auth=AUTH
)
data_resp.raise_for_status()

data_json = data_resp.json()
items = data_json["Items"]

print("Number of observations:", len(items))
print(items[:3])  