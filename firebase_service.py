from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore


# --------------------------------------------------
# Firebase setup
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CREDENTIALS_FILE = (
    BASE_DIR / "firebase-service-account.json"
)

cred = credentials.Certificate(
    str(CREDENTIALS_FILE)
)

firebase_admin.initialize_app(cred)

db = firestore.client()


# --------------------------------------------------
# Write ML telemetry to Firestore
# --------------------------------------------------

def write_telemetry(
    patient_id,
    device_id,
    timestamp,
    accel_x,
    accel_y,
    accel_z,
    gyro_x,
    gyro_y,
    gyro_z,
    ml_result
):

    telemetry_data = {

        "patientId": patient_id,

        "deviceId": device_id,

        "timestamp": firestore.SERVER_TIMESTAMP,

        "accelerometer": {
            "x": float(accel_x),
            "y": float(accel_y),
            "z": float(accel_z)
        },

        "gyroscope": {
            "x": float(gyro_x),
            "y": float(gyro_y),
            "z": float(gyro_z)
        },

        "fatigue": {
            "score": float(
                ml_result["fatigue_score"]
            ),

            "level": (
                "HIGH"
                if ml_result["fatigue_score"] >= 66
                else "MEDIUM"
                if ml_result["fatigue_score"] >= 33
                else "LOW"
            )
        },

        "stability": {
            "score": float(
                ml_result["stability_score"]
            ),

            "level": ml_result["stability_level"]
        },

        "assistance": {
            "score": float(
                ml_result["assistance_score"]
            ),

            "level": ml_result["assistance_level"],

            "controllerCommand": float(
                ml_result["controller_command"]
            )
        }
    }

    doc_ref = (
        db
        .collection("telemetry")
        .document()
    )

    doc_ref.set(telemetry_data)

    return doc_ref.id