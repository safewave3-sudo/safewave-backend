from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load ML models (Advisory only)
# -------------------------------
rf = joblib.load("cloud-api/model.pkl")
le = joblib.load("cloud-api/label.pkl")

# -------------------------------
# Persistence state
# -------------------------------
STATE_COLLECTION = "system"
STATE_DOC = "risk_state"

def get_state():
    doc = db.collection(STATE_COLLECTION).document(STATE_DOC).get()
    if doc.exists:
        return doc.to_dict()
    return {"high_count": 0, "status": "SAFE"}

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

    # =====================================================
    # 1️⃣ BIOLOGICAL CONDITIONS — Naegleria fowleri science
    # =====================================================

    # Temperature zones (main driver)
    cool_temp = data.temp < 25
    moderate_temp = 25 <= data.temp < 34
    high_temp = data.temp >= 34   # strong growth zone

    # Environmental risks
    turb_risk = data.turb >= 50        # sediment / biofilm
    tds_risk  = data.tds >= 250        # nutrients
    flow_risk = data.flow == 0         # stagnant water
    ph_risk   = data.ph >= 7.5         # slightly alkaline

    # =====================================================
    # 2️⃣ ML Prediction (Advisory only — logging)
    # =====================================================
    X = np.array([[data.ph, data.temp, data.tds, data.turb, data.flow]])
    pred = rf.predict(X)[0]
    instant_ml = le.inverse_transform([pred])[0]

    # =====================================================
    # 3️⃣ BIOLOGICAL SCORE
    # =====================================================
    bio_score = 0

    if high_temp:
        bio_score += 3
    elif moderate_temp:
        bio_score += 1

    if turb_risk:
        bio_score += 1.5

    if tds_risk:
        bio_score += 1

    if flow_risk:
        bio_score += 1

    if ph_risk:
        bio_score += 0.5

    # Summer seasonal boost (biological realism)
    month = datetime.utcnow().month
    if month in [4,5,6,7,8,9] and data.temp >= 30:
        bio_score += 0.5

    # Convert to percentage for dashboard graph
    risk_percent = min(100, int((bio_score / 7) * 100))

    # =====================================================
    # 4️⃣ PERSISTENCE (avoid false spikes)
    # =====================================================
    state = get_state()
    high_count = state.get("high_count", 0)

    if bio_score >= 4:
        high_count += 1
    else:
        high_count = max(0, high_count - 2)

    # =====================================================
    # 5️⃣ FINAL DECISION — Temperature Dominant Logic
    # =====================================================

    other_score = 0
    if turb_risk: other_score += 1
    if tds_risk:  other_score += 1
    if flow_risk: other_score += 1
    if ph_risk:   other_score += 0.5

    # Cool water → organism dormant
    if cool_temp:
        status = "SAFE"

    # Moderate temperature behaviour
    elif moderate_temp:
        if other_score < 1.5:
            status = "SAFE"
        else:
            status = "WARNING"

    # High favourable temperature
    else:
        if other_score < 1.5:
            status = "WARNING"
        else:
            if high_count >= 6:
                status = "HIGH_RISK"
            else:
                status = "WARNING"

    save_state(high_count, status)

    # =====================================================
    # 6️⃣ STORE RESULT
    # =====================================================
    result = {
        "ph": data.ph,
        "temp": data.temp,
        "tds": data.tds,
        "turb": data.turb,
        "flow": data.flow,
        "instant": instant_ml,
        "bio_score": round(bio_score, 2),
        "risk_percent": risk_percent,
        "high_count": high_count,
        "status": status,
        "timestamp": datetime.utcnow().isoformat()
    }

    db.collection("safewave_readings").add(result)
    return result


# -------------------------------
# Latest reading
# -------------------------------
@app.get("/latest")
def latest():
    docs = (
        db.collection("safewave_readings")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )
    for doc in docs:
        return doc.to_dict()
    return {"error": "No data available"}


# -------------------------------
# Health check
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
