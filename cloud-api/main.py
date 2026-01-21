from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from collections import deque

# Firebase setup
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()

# Load model + encoder
rf = joblib.load("model.pkl")
le = joblib.load("label.pkl")

# Persistence & accumulation state
window = deque(maxlen=10)   # 10 readings = 10 minutes if 1 reading/min
score = 0                   # long-term accumulation score

class SensorData(BaseModel):
    ph: float
    temp: float
    tds: float
    turb: float
    flow: int

@app.post("/predict")
def predict(data: SensorData):
    global score

    # Instant ML prediction
    X = np.array([[data.ph, data.temp, data.tds, data.turb, data.flow]])
    pred = rf.predict(X)[0]
    instant = le.inverse_transform([pred])[0]  # "SAFE" or "HIGH_RISK"

    # Persistence Layer (short-term)
    window.append(1 if instant == "HIGH_RISK" else 0)

    if len(window) == window.maxlen and sum(window) >= 0.7 * window.maxlen:
        persistent = True
        score += 1
    else:
        persistent = False
        score = max(0, score - 1)

    # Accumulation Layer (long-term)
    if score > 40:
        final_risk = "HIGH_RISK"
    elif score > 10:
        final_risk = "WARNING"
    else:
        final_risk = "SAFE"

    # Prepare record
    data_to_store = {
        "ph": data.ph,
        "temp": data.temp,
        "tds": data.tds,
        "turb": data.turb,
        "flow": data.flow,
        "instant": instant,
        "persistent": persistent,
        "score": score,
        "final_risk": final_risk,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    # Store in Firestore
    db.collection("safewave_readings").add(data_to_store)

    # Return final response
    return data_to_store
