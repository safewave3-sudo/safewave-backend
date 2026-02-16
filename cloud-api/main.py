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
# Load ML Model
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
    return datetime.now(ist)

# =====================================
# Collections
# =====================================
DEVICE_COLLECTION = "devices"
STATE_COLLECTION = "device_state"
RAW_COLLECTION = "sensor_raw"
READINGS_COLLECTION = "safewave_readings"

OFFLINE_TIMEOUT = 60

# =====================================
# Device State
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
# Safe std (prevent NaN)
# =====================================
def safe_std(series):
    val = series.std()
    if pd.isna(val):
        return 0.0
    return float(val)

# =====================================
# Feature Builder
# =====================================
def build_features(df_hist, latest_data):
    now_time = now_ist()

    return {
        "temp_mean": float(df_hist["temp"].mean()),
        "temp_std": safe_std(df_hist["temp"]),
        "temp_last": float(latest_data.temp),

        "turb_mean": float(df_hist["turb"].mean()),
        "turb_std": safe_std(df_hist["turb"]),
        "turb_last": float(latest_data.turb),

        "ph_mean": float(df_hist["ph"].mean()),
        "ph_std": safe_std(df_hist["ph"]),
        "ph_last": float(latest_data.ph),

        "tds_mean": float(df_hist["tds"].mean()),
        "tds_std": safe_std(df_hist["tds"]),
        "tds_last": float(latest_data.tds),

        "flow_mean": float(df_hist["flow"].mean()),
        "flow_std": safe_std(df_hist["flow"]),
        "flow_last": float(latest_data.flow),

        "hour": now_time.hour,
        "hour_sin": float(np.sin(2*np.pi*now_time.hour/24)),
        "hour_cos": float(np.cos(2*np.pi*now_time.hour/24)),
        "dayofweek": now_time.weekday()
    }

# =====================================
# Main Prediction Logic
# =====================================
def process_prediction(data: SensorData):

    update_device_status(data.device_id, data.device_name, data.location_name)

    current_time = now_ist()

    # Store RAW
    db.collection(RAW_COLLECTION).add({
        "device_id": data.device_id,
        "timestamp": current_time,
        "ph": data.ph,
        "temp": data.temp,
        "tds": data.tds,
        "turb": data.turb,
        "flow": data.flow
    })

    # Fetch last 6 hours
    six_hours_ago = current_time - timedelta(hours=6)

    docs = (
        db.collection(RAW_COLLECTION)
        .where("device_id", "==", data.device_id)
        .where("timestamp", ">=", six_hours_ago)
        .stream()
    )

    records = [doc.to_dict() for doc in docs]

    # ML Probability
    if len(records) < 3:
        prob_high = 0.0
    else:
        df_hist = pd.DataFrame(records)
        feature_dict = build_features(df_hist, data)

        try:
            row_df = pd.DataFrame([{col: feature_dict.get(col, 0.0) for col in FEATURE_COLS}])
            prob_high = float(rf.predict_proba(row_df)[0][1])
        except Exception as e:
            print("ML ERROR:", e)
            prob_high = 0.0

    # =====================================
    # Ecological Model
    # =====================================
    state = get_state(data.device_id)
    high_count = state.get("high_count", 0)
    risk_factors = []

    if data.temp < 30:
        high_count = 0
        status = "SAFE"
        bio_score = 0
        final_score = 0
        risk_factors.append("Temperature below growth threshold")

    else:
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
        "ml_probability": round(prob_high, 4),
        "bio_score": round(bio_score, 2),
        "final_score": round(final_score, 2),
        "risk_percent": round(risk_percent, 2),
        "high_count": high_count,
        "status": status,
        "risk_factors": risk_factors,
        "timestamp": current_time
    }

    db.collection(READINGS_COLLECTION).add(result)
    return result

# =====================================
# API Endpoint with crash protection
# =====================================
@app.post("/predict")
def predict(data: SensorData):
    try:
        return process_prediction(data)
    except Exception:
        traceback.print_exc()
        return {"error": "Internal server error"}

# =====================================
# Health
# =====================================
@app.get("/health")
def health():
    return {"status": "ok"}
