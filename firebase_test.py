import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone


# ------------------------------------------------------------
# FIREBASE INITIALIZATION
# ------------------------------------------------------------

cred = credentials.Certificate("firebase-service-account.json")

firebase_admin.initialize_app(cred)

db = firestore.client()


# ------------------------------------------------------------
# TEST FIRESTORE WRITE
# ------------------------------------------------------------

test_data = {
    "message": "Exoskelly Firebase connection test",
    "timestamp": datetime.now(timezone.utc),
    "source": "python_ml"
}


doc_ref = db.collection("test").document("connection_test")

doc_ref.set(test_data)


print("Firebase connection successful!")
print("Firestore test document written successfully.")
print(f"Document path: {doc_ref.path}")