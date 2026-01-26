from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# -------------------------------
# Firebase initialization
# -------------------------------
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI()

# -------------------------------
# Load ML models
# -------------------------------
rf = joblib.load("cloud-api/model.pkl")
le = joblib.load("cloud-api/label.pkl")

# -------------------------------
# Persistence state document
# -------------------------------
STATE_DOC = "risk_state"
STATE_COLLECTION = "system"

class SensorData(BaseModel):
    ph: float
    temp: float
    tds: float
    turb: float
    flow: int

# -------------------------------
# Firestore helpers
# -------------------------------
def get_state():
    doc = db.collection(STATE_COLLECTION).document(STATE_DOC).get()
    if doc.exists:
        return doc.to_dict()
    return {"high_count": 0, "status": "SAFE", "timestamp": None}

def save_state(high_count, status):
    db.collection(STATE_COLLECTION).document(STATE_DOC).set({
        "high_count": high_count,
        "status": status,
        "timestamp": datetime.utcnow().isoformat()
    })

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(data: SensorData):

    # ----- ML Instant Prediction -----
    X = np.array([[data.ph, data.temp, data.tds, data.turb, data.flow]])
    pred = rf.predict(X)[0]
    instant = le.inverse_transform([pred])[0]  # "SAFE" or "HIGH_RISK"

    # ----- Sensor Biological Threshold Logic -----
    warm = data.temp > 28        # warm water
    turbid = data.turb > 10      # high organic load
    stagnant = data.flow < 3     # low/zero flow

    # 3-factor biological trigger
    sensor_trigger = warm and turbid and stagnant

    # ML OR biology triggers escalation
    combined_trigger = (instant == "HIGH_RISK") or sensor_trigger

    # ----- Persistence Loading -----
    state = get_state()
    high_count = state.get("high_count", 0)

    # ----- Persistence Accumulation -----
    if combined_trigger:
        high_count += 1
    else:
        high_count = 0  # reset only if safe conditions

    # ----- Escalation Logic -----
    if high_count < 10:
        status = "SAFE"
    elif 10 <= high_count < 20:
        status = "WARNING"
    else:
        # high_count >= 20 (long-term confirming period)
        if (instant == "HIGH_RISK") and sensor_trigger:
            status = "HIGH_RISK"
        else:
            status = "WARNING"

    # Save back to Firebase
    save_state(high_count, status)

    # ----- Store reading into Firebase -----
    data_to_store = {
        "ph": data.ph,
        "temp": data.temp,
        "tds": data.tds,
        "turb": data.turb,
        "flow": data.flow,
        "instant": instant,
        "sensor_trigger": sensor_trigger,
        "high_count": high_count,
        "status": status,
        "timestamp": datetime.utcnow().isoformat()
    }

    db.collection("safewave_readings").add(data_to_store)
    return data_to_store
