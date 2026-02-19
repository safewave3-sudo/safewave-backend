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
import traceback
import os

# =====================================
# Firebase Initialization
# =====================================
BASE_DIR = Path(__file__).resolve().parent

cred = credentials.Certificate(
    os.path.join(BASE_DIR, "firebase_key.json")
)

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
# Load ML Model
# =====================================
saved = joblib.load(BASE_DIR / "rf_model.joblib")
rf = saved["model"]
FEATURE_COLS = saved["features"]

# =====================================
# Time
# =====================================
ist = pytz.timezone("Asia/Kolkata")

def now_ist():
    return datetime.now(ist)

# =====================================
# Collections
# =====================================
DEVICE_COLLECTION = "devices"
STATE_COLLECTION = "device_state"
RAW_COLLECTION = "sensor_raw"
READINGS_COLLECTION = "safewave_readings"

# =====================================
# Schema
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
# Helpers
# =====================================
def safe_std(series):
    val = series.std()
    return 0.0 if pd.isna(val) else float(val)

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

def update_device_status(device_id, device_name, location_name):
    db.collection(DEVICE_COLLECTION).document(device_id).set({
        "device_id": device_id,
        "device_name": device_name,
        "location_name": location_name,
        "last_seen": now_ist(),
        "status": "ONLINE"
    }, merge=True)

# =====================================
# Feature Builder
# =====================================
def build_features(df_hist, latest):
    now_time = now_ist()

    return {
        "temp_mean": float(df_hist["temp"].mean()),
        "temp_std": safe_std(df_hist["temp"]),
        "temp_last": float(latest.temp),

        "turb_mean": float(df_hist["turb"].mean()),
        "turb_std": safe_std(df_hist["turb"]),
        "turb_last": float(latest.turb),

        "ph_mean": float(df_hist["ph"].mean()),
        "ph_std": safe_std(df_hist["ph"]),
        "ph_last": float(latest.ph),

        "tds_mean": float(df_hist["tds"].mean()),
        "tds_std": safe_std(df_hist["tds"]),
        "tds_last": float(latest.tds),

        "flow_mean": float(df_hist["flow"].mean()),
        "flow_std": safe_std(df_hist["flow"]),
        "flow_last": float(latest.flow),

        "hour": now_time.hour,
        "hour_sin": float(np.sin(2 * np.pi * now_time.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * now_time.hour / 24)),
        "dayofweek": now_time.weekday()
    }

# =====================================
# Prediction Logic (UNCHANGED LOGIC)
# =====================================
def process_prediction(data: SensorData):

    current_time = now_ist()

    update_device_status(
        data.device_id,
        data.device_name,
        data.location_name
    )

    # 🔥 Overwrite per minute (NO unlimited growth)
    raw_doc_id = f"{data.device_id}_{current_time.strftime('%Y%m%d%H%M')}"
    db.collection(RAW_COLLECTION).document(raw_doc_id).set({
        "device_id": data.device_id,
        "timestamp": current_time,
        "ph": data.ph,
        "temp": data.temp,
        "tds": data.tds,
        "turb": data.turb,
        "flow": data.flow
    })

    six_hours_ago = current_time - timedelta(hours=6)

    docs = (
        db.collection(RAW_COLLECTION)
        .where("device_id", "==", data.device_id)
        .where("timestamp", ">=", six_hours_ago)
        .limit(40)   # 🔥 important
        .stream()
    )

    records = [doc.to_dict() for doc in docs]

    if len(records) < 3:
        prob_high = 0.0
    else:
        df_hist = pd.DataFrame(records)
        feature_dict = build_features(df_hist, data)
        row_df = pd.DataFrame(
            [{col: feature_dict.get(col, 0.0) for col in FEATURE_COLS}]
        )
        prob_high = float(rf.predict_proba(row_df)[0][1])

    state = get_state(data.device_id)
    high_count = state.get("high_count", 0)

    bio_score = 0
    if data.temp >= 34:
        bio_score += 4
    if data.turb >= 60:
        bio_score += 2
    if data.flow == 0:
        bio_score += 3
    if data.tds >= 250:
        bio_score += 1
    if data.ph >= 7.5:
        bio_score += 1

    final_score = bio_score + (prob_high * 3)

    if final_score < 4:
        high_count = max(0, high_count - 1)
    elif final_score < 8:
        high_count += 1
    else:
        high_count += 2

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
        "risk_percent": round(risk_percent, 2),
        "status": status,
        "timestamp": current_time
    }

    # 🔥 Only ONE document per device
    db.collection(READINGS_COLLECTION).document(data.device_id).set(result)

    return result

# =====================================
# API ENDPOINTS
# =====================================

@app.post("/predict")
def predict(data: SensorData):
    try:
        return process_prediction(data)
    except Exception:
        traceback.print_exc()
        return {"error": "Internal Server Error"}

@app.get("/latest/{device_id}")
def get_latest(device_id: str):
    doc = db.collection(READINGS_COLLECTION).document(device_id).get()
    if doc.exists:
        return doc.to_dict()
    return {"message": "No data found"}

@app.get("/devices")
def get_devices():
    docs = db.collection(DEVICE_COLLECTION).stream()
    return [doc.to_dict() for doc in docs]

# 🔥 OPTIMIZED ALERTS (single read)
@app.get("/alerts")
def get_all_alerts():
    docs = db.collection(READINGS_COLLECTION).stream()
    return [doc.to_dict() for doc in docs]

@app.get("/health")
def health():
    return {"status": "ok"}
