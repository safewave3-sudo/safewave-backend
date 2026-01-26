from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from collections import deque
from datetime import datetime

# Firebase setup
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()

# Load model + encoder
rf = joblib.load("cloud-api/model.pkl")
le = joblib.load("cloud-api/label.pkl")

# Persistence & accumulation state
window = deque(maxlen=10)   # short-term instant persistence window

# Firestore persistence document
STATE_DOC = "risk_state"
STATE_COLLECTION = "system"


class SensorData(BaseModel):
    ph: float
    temp: float
    tds: float
    turb: float
    flow: int


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


@app.post("/predict")
def predict(data: SensorData):
    # ML instant prediction
    X = np.array([[data.ph, data.temp, data.tds, data.turb, data.flow]])
    pred = rf.predict(X)[0]
    instant = le.inverse_transform([pred])[0]  # "SAFE" or "HIGH_RISK"

    # ----- Sensor Threshold Logic -----
    sensor_trigger = False
    if data.temp > 30: sensor_trigger = True
    if data.turb > 10: sensor_trigger = True
    if data.flow < 3: sensor_trigger = True

    # Combined Trigger
    combined_trigger = (instant == "HIGH_RISK") or sensor_trigger

    # ===== Load persistence state from Firestore =====
    state = get_state()
    high_count = state.get("high_count", 0)

    # ===== Persistence accumulation =====
    if combined_trigger:
        high_count += 1
    else:
        high_count = 0  # reset on safe readings

    # ===== Escalation =====
    if 10 <= high_count < 20:
        status = "WARNING"
    elif high_count >= 20:
        status = "HIGH_RISK"
    else:
        status = "SAFE"

    # Save state back to Firestore
    save_state(high_count, status)

    # Store raw reading data into Firestore
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
