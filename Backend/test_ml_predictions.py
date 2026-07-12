import sys
import os
sys.path.append(os.path.join(os.getcwd()))

from app.services.ml_model import fraud_model
from app.database.db import SessionLocal
from app.database import models

db = SessionLocal()
account = db.query(models.Account).first()
if account:
    print(f"Testing for Account: {account.holder_name} (Baseline Device: {account.baseline_primary_device}, Location: {account.baseline_primary_location})")
    
    # 1. Normal Event
    normal_event = {
        "event_type": "login_success",
        "device": account.baseline_primary_device,
        "location": account.baseline_primary_location,
        "transaction_amount": 0,
        "login_attempts": 1,
        "timestamp": "2026-07-12T10:00:00Z"
    }
    pred_normal = fraud_model.predict(normal_event, db=db, account=account)
    print("\n[NORMAL EVENT PREDICTION]")
    print(pred_normal)

    # 2. Suspicious Event (new device, new location, transaction amount)
    suspicious_event = {
        "event_type": "transaction",
        "device": "unknown_device_hacker",
        "location": "Russia",
        "transaction_amount": 5000,
        "login_attempts": 1,
        "timestamp": "2026-07-12T10:00:00Z"
    }
    pred_susp = fraud_model.predict(suspicious_event, db=db, account=account)
    print("\n[SUSPICIOUS EVENT PREDICTION]")
    print(pred_susp)

    # 3. Another event to test coordinates
    travel_event = {
        "event_type": "transaction",
        "device": account.baseline_primary_device,
        "location": "New York",
        "transaction_amount": 100,
        "login_attempts": 1,
        "timestamp": "2026-07-12T10:05:00Z"
    }
    pred_travel = fraud_model.predict(travel_event, db=db, account=account)
    print("\n[TRAVEL EVENT PREDICTION]")
    print(pred_travel)

db.close()
