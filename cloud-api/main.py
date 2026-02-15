from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import pytz

# ================================
# Firebase initialization
# ================================
cred = credentials.Certificate("firebase_key.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ================================
# FastAPI setup
# ================================
app = FastAPI(title="SAFEWAVE ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# Load ML model (Advisory only)
# ================================
rf = joblib.load("cloud-api/model.pkl")
le = joblib.load("cloud-api/label.pkl")

# ================================
# Time helper (IST)
# ================================
ist = pytz.timezone("Asia/Kolkata")

def now_ist():
    return datetime.now(ist).isoformat()

# ================================
# Collections
# ================================
STATE_COLLECTION = "system"
STATE_DOC = "risk_state"

DEVICE_COLLECTION = "devices"
OFFLINE_TIMEOUT = 60  # seconds

# ================================
# Persistence state
# ================================
def get_state():
    doc = db.collection(STATE_COLLECTION).document(STATE_DOC).get()
    if doc.exists:
        return doc.to_dict()
    return {"high_count": 0, "status": "SAFE"}

def save_state(high_count, status):
    db.collection(STATE_COLLECTION).document(STATE_DOC).set({
        "high_count": high_count,
        "status": status,
        "timestamp": now_ist()
    })

# ================================
# Device heartbeat
# ================================
def update_device_status(device_id, device_name, location_name):
    doc_ref = db.collection(DEVICE_COLLECTION).document(device_id)
    doc = doc_ref.get()

    data = {
        "device_id": device_id,
        "device_name": device_name,
        "location_name": location_name,
        "last_seen": now_ist(),
        "status": "ONLINE"
    }

    if not doc.exists:
        doc_ref.set(data)
    else:
        doc_ref.update(data)

# ================================
# Input schema
# ================================
class SensorData(BaseModel):
    device_id: str
    device_name: str
    location_name: str
    ph: float
    temp: float
    tds: float
    turb: float
    flow: int  # 0 stagnant, 1 flowing

# ================================
# Prediction endpoint
# ================================
@app.post("/predict")
def predict(data: SensorData):

    # Update device heartbeat
    update_device_status(
        data.device_id,
        data.device_name,
        data.location_name
    )

    # ================= ML Prediction (Advisory) =================
    X = np.array([[data.ph, data.temp, data.tds, data.turb, data.flow]])
    pred = rf.predict(X)[0]
    instant_ml = le.inverse_transform([pred])[0]

    # ================= STRICT BIOLOGICAL MODEL =================
    # Hard gate: No growth below 30°C
    if data.temp < 30:
        bio_score = 0
        high_count = 0
        status = "SAFE"
    else:

        strong_temp = data.temp >= 34
        turb_risk = data.turb >= 60
        tds_risk = data.tds >= 250
        flow_risk = data.flow == 0
        ph_risk = data.ph >= 7.5

        # Strong biological growth condition
        strong_growth = (
            strong_temp and
            turb_risk and
            flow_risk and
            (tds_risk or ph_risk)
        )

        state = get_state()
        high_count = state.get("high_count", 0)

        # Persistence accumulation
        if strong_growth:
            high_count += 1
        else:
            high_count = max(0, high_count - 1)

        # Final decision
        if high_count >= 10:
            status = "HIGH_RISK"
        else:
            status = "SAFE"

        # Biological score only for dashboard
        bio_score = (
            (3 if strong_temp else 1) +
            (1.5 if turb_risk else 0) +
            (1 if tds_risk else 0) +
            (1 if flow_risk else 0) +
            (0.5 if ph_risk else 0)
        )

    # Risk % for visualization only
    risk_percent = min(100, high_count * 10)

    save_state(high_count, status)

    # ================= Store result =================
    result = {
        "device_id": data.device_id,
        "device_name": data.device_name,
        "location_name": data.location_name,
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
        "timestamp": now_ist()
    }

    db.collection("safewave_readings").add(result)
    return result

# ================================
# Device list with ONLINE/OFFLINE
# ================================
@app.get("/devices")
def get_devices():
    devices = []
    now = datetime.now(ist)

    docs = db.collection(DEVICE_COLLECTION).stream()
    for doc in docs:
        d = doc.to_dict()

        last_seen = datetime.fromisoformat(d["last_seen"])
        diff = (now - last_seen).total_seconds()

        d["live_status"] = "OFFLINE" if diff > OFFLINE_TIMEOUT else "ONLINE"
        devices.append(d)

    return devices

# ================================
# Latest reading per device
# ================================
@app.get("/latest/{device_id}")
def latest_device(device_id: str):
    docs = (
        db.collection("safewave_readings")
        .where("device_id", "==", device_id)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )
    for doc in docs:
        return doc.to_dict()
    return {"error": "No data"}

# ================================
# Health check
# ================================
@app.get("/health")
def health():
    return {"status": "ok"}
