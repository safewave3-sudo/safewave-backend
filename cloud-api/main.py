from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
import pytz
from pathlib import Path

# =====================================
# Firebase Initialization
# =====================================
cred = credentials.Certificate("firebase_key.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# =====================================
# FastAPI Setup
# =====================================
app = FastAPI(title="SAFEWAVE ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# Load Trained ML Model
# =====================================
BASE_DIR = Path(__file__).resolve().parent
saved = joblib.load(BASE_DIR / "rf_model.joblib")

rf = saved["model"]
FEATURE_COLS = saved["features"]

# =====================================
# Time Helper
# =====================================
ist = pytz.timezone("Asia/Kolkata")

def now_ist():
    return datetime.now(ist).isoformat()

# =====================================
# Firestore Collections
# =====================================
DEVICE_COLLECTION = "devices"
STATE_COLLECTION = "device_state"
RAW_COLLECTION = "sensor_raw"
READINGS_COLLECTION = "safewave_readings"

OFFLINE_TIMEOUT = 60

# =====================================
# Device State Management
# =====================================
def get_state(device_id):
    doc = db.collection(STATE_COLLECTION).document(device_id).get()
    if doc.exists:
        return doc.to_dict()
    return {"high_count": 0}

def save_state(device_id, high_count, status):
    db.collection(STATE_COLLECTION).document(device_id).set({
        "high_count": high_count,
        "status": status,
        "timestamp": now_ist()
    })

# =====================================
# Device Heartbeat
# =====================================
def update_device_status(device_id, device_name, location_name):
    db.collection(DEVICE_COLLECTION).document(device_id).set({
        "device_id": device_id,
        "device_name": device_name,
        "location_name": location_name,
        "last_seen": now_ist(),
        "status": "ONLINE"
    }, merge=True)

# =====================================
# Input Schema
# =====================================
class SensorData(BaseModel):
    device_id: str
    device_name: str
    location_name: str
    ph: float
    temp: float
    tds: float
    turb: float
    flow: int

# =====================================
# Rolling Feature Builder
# =====================================
def build_features(df_hist, latest_data):
    now_time = datetime.now(ist)

    return {
        "temp_mean": df_hist["temp"].mean(),
        "temp_std": df_hist["temp"].std(),
        "temp_last": latest_data.temp,
        "turb_mean": df_hist["turb"].mean(),
        "turb_std": df_hist["turb"].std(),
        "turb_last": latest_data.turb,
        "ph_mean": df_hist["ph"].mean(),
        "ph_std": df_hist["ph"].std(),
        "ph_last": latest_data.ph,
        "tds_mean": df_hist["tds"].mean(),
        "tds_std": df_hist["tds"].std(),
        "tds_last": latest_data.tds,
        "flow_mean": df_hist["flow"].mean(),
        "flow_std": df_hist["flow"].std(),
        "flow_last": latest_data.flow,
        "hour": now_time.hour,
        "hour_sin": np.sin(2 * np.pi * now_time.hour / 24),
        "hour_cos": np.cos(2 * np.pi * now_time.hour / 24),
        "dayofweek": now_time.weekday()
    }

# =====================================
# Prediction Endpoint
# =====================================
@app.post("/predict")
def predict(data: SensorData):

    update_device_status(data.device_id, data.device_name, data.location_name)

    # Store raw reading
    raw_entry = {
        "device_id": data.device_id,
        "timestamp": now_ist(),
        "ph": data.ph,
        "temp": data.temp,
        "tds": data.tds,
        "turb": data.turb,
        "flow": data.flow
    }
    db.collection(RAW_COLLECTION).add(raw_entry)

    # Fetch last 6 hours
    now_time = datetime.now(ist)
    six_hours_ago = now_time - timedelta(hours=6)

    docs = db.collection(RAW_COLLECTION)\
        .where("device_id", "==", data.device_id)\
        .stream()

    records = []
    for doc in docs:
        d = doc.to_dict()
        ts = datetime.fromisoformat(d["timestamp"])
        if ts >= six_hours_ago:
            records.append(d)

    # ML Probability
    if len(records) < 3:
        prob_high = 0.0
    else:
        df_hist = pd.DataFrame(records)
        feature_dict = build_features(df_hist, data)
        row_df = pd.DataFrame([{col: feature_dict[col] for col in FEATURE_COLS}])
        prob_high = rf.predict_proba(row_df)[0][1]

    # =====================================
    # ECOLOGICAL SUITABILITY MODEL
    # =====================================
    state = get_state(data.device_id)
    high_count = state.get("high_count", 0)

    risk_factors = []

    # Hard biological brake
    if data.temp < 30:
        high_count = max(0, high_count - 3)
        status = "SAFE"
        bio_score = 0
        final_score = 0
        risk_factors.append("Temperature below growth threshold")

    else:

        # Temperature zoning
        if 30 <= data.temp < 34:
            temp_score = 2
            risk_factors.append("Warm water zone")
        elif 34 <= data.temp <= 38:
            temp_score = 4
            risk_factors.append("Optimal growth temperature")
        else:
            temp_score = 5
            risk_factors.append("High metabolic temperature zone")

        turb_score = 2 if data.turb >= 60 else 0
        if turb_score:
            risk_factors.append("Elevated turbidity")

        flow_score = 3 if data.flow == 0 else 0
        if flow_score:
            risk_factors.append("Stagnant water")

        tds_score = 1 if data.tds >= 250 else 0
        if tds_score:
            risk_factors.append("High dissolved solids")

        ph_score = 1 if data.ph >= 7.5 else 0
        if ph_score:
            risk_factors.append("Alkaline pH")

        bio_score = temp_score + turb_score + flow_score + tds_score + ph_score

        ml_influence = prob_high * 3
        final_score = bio_score + ml_influence

        # Escalation logic
        if final_score < 4:
            high_count = max(0, high_count - 1)
        elif 4 <= final_score < 8:
            high_count += 1
        else:
            high_count += 2

        # Persistence thresholds
        if high_count >= 12:
            status = "HIGH_GROWTH_POTENTIAL"
        elif high_count >= 6:
            status = "SUITABLE_CONDITIONS"
        else:
            status = "SAFE"

    risk_percent = min(100, high_count * 8)

    save_state(data.device_id, high_count, status)

    result = {
        "device_id": data.device_id,
        "device_name": data.device_name,
        "location_name": data.location_name,
        "ph": data.ph,
        "temp": data.temp,
        "tds": data.tds,
        "turb": data.turb,
        "flow": data.flow,
        "ml_probability": round(prob_high, 4),
        "bio_score": round(bio_score, 2),
        "final_score": round(final_score, 2),
        "risk_percent": round(risk_percent, 2),
        "high_count": high_count,
        "status": status,
        "risk_factors": risk_factors,
        "timestamp": now_ist()
    }

    db.collection(READINGS_COLLECTION).add(result)

    return result

# =====================================
# Devices Endpoint
# =====================================
@app.get("/devices")
def get_devices():
    devices = []
    now = datetime.now(ist)

    docs = db.collection(DEVICE_COLLECTION).stream()

    for doc in docs:
        d = doc.to_dict()
        if "last_seen" in d:
            last_seen = datetime.fromisoformat(d["last_seen"])
            diff = (now - last_seen).total_seconds()
            d["live_status"] = "OFFLINE" if diff > OFFLINE_TIMEOUT else "ONLINE"
        else:
            d["live_status"] = "UNKNOWN"

        devices.append(d)

    return devices

# =====================================
# Latest Reading Endpoint
# =====================================
@app.get("/latest/{device_id}")
def latest_device(device_id: str):

    docs = (
        db.collection(READINGS_COLLECTION)
        .where("device_id", "==", device_id)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )

    for doc in docs:
        return doc.to_dict()

    return {"error": "No data"}

# =====================================
# Health Endpoint
# =====================================
@app.get("/health")
def health():
    return {"status": "ok"}
