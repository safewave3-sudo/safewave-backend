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
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI(title="SAFEWAVE ML API")

# -------------------------------
# Load ML models (ADVISORY ONLY)
# -------------------------------
rf = joblib.load("cloud-api/model.pkl")
le = joblib.load("cloud-api/label.pkl")

# -------------------------------
# Persistence state (Firestore)
# -------------------------------
STATE_COLLECTION = "system"
STATE_DOC = "risk_state"

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
# Input schema
# -------------------------------
class SensorData(BaseModel):
    ph: float
    temp: float
    tds: float
    turb: float
    flow: int   # 0 = stagnant, 1 = flowing

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(data: SensorData):

    # 1️⃣ BIOLOGICAL RISK FLAGS
    temp_risk = data.temp >= 30
    turb_risk = data.turb >= 10
    tds_risk  = data.tds >= 400
    flow_risk = data.flow == 0
    ph_risk   = data.ph >= 8.5

    # 2️⃣ ML PREDICTION (LOGGING ONLY)
    X = np.array([[data.ph, data.temp, data.tds, data.turb, data.flow]])
    pred = rf.predict(X)[0]
    instant_ml = le.inverse_transform([pred])[0]

    # 3️⃣ HARD SAFE OVERRIDE
    if not temp_risk and not turb_risk:
        save_state(0, "SAFE")

        result = {
            "ph": data.ph,
            "temp": data.temp,
            "tds": data.tds,
            "turb": data.turb,
            "flow": data.flow,
            "instant": instant_ml,
            "sensor_trigger": False,
            "high_count": 0,
            "status": "SAFE",
            "timestamp": datetime.utcnow().isoformat()
        }

        db.collection("safewave_readings").add(result)
        return result

    # 4️⃣ BIOLOGICAL RISK SCORE
    bio_score = 0
    if temp_risk: bio_score += 2
    if turb_risk: bio_score += 2
    if tds_risk:  bio_score += 1
    if flow_risk: bio_score += 1
    if ph_risk:   bio_score += 0.5

    # 5️⃣ PERSISTENCE
    state = get_state()
    high_count = state.get("high_count", 0)

    if bio_score >= 3:
        high_count += 1
    else:
        high_count = max(0, high_count - 2)

    # 6️⃣ FINAL DECISION
    if bio_score >= 5 and high_count >= 10:
        status = "HIGH_RISK"
    elif bio_score >= 3:
        status = "WARNING"
    else:
        status = "SAFE"

    save_state(high_count, status)

    # 7️⃣ STORE + RETURN
    result = {
        "ph": data.ph,
        "temp": data.temp,
        "tds": data.tds,
        "turb": data.turb,
        "flow": data.flow,
        "instant": instant_ml,
        "sensor_trigger": bio_score >= 3,
        "high_count": high_count,
        "status": status,
        "timestamp": datetime.utcnow().isoformat()
    }

    db.collection("safewave_readings").add(result)
    return result

# -------------------------------
# Latest reading for Mobile App
# -------------------------------
@app.get("/latest")
def latest():
    try:
        docs = (
            db.collection("safewave_readings")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )

        for doc in docs:
            return doc.to_dict()

        return {"error": "No data available"}

    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Health endpoint
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
