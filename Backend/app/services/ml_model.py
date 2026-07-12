import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sqlalchemy.orm import Session
from sqlalchemy import func

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
PCA_PATH = os.path.join(MODEL_DIR, "pca_model.joblib")

# Coordinates for major US cities (demo baseline)
CITY_COORDS = {
    "San Diego": (32.7157, -117.1611),
    "Houston": (29.7604, -95.3698),
    "Austin": (30.2672, -97.7431),
    "Columbus": (39.9612, -82.9988),
    "Seattle": (47.6062, -122.3321),
    "Indianapolis": (39.7684, -86.1581),
    "Louisville": (38.2527, -85.7585),
    "Denver": (39.7392, -104.9903),
    "Dallas": (32.7767, -96.7970),
    "Phoenix": (33.4484, -112.0740),
    "Portland": (45.5152, -122.6784),
    "Chicago": (41.8781, -87.6298),
    "New York": (40.7128, -74.0060),
    "India": (20.5937, 78.9629),
    "Unknown": (0, 0)
}

class FraudDetectionModel:
    """IsolationForest-based anomaly detection for banking fraud (Hardcore Mode)."""

    def __init__(self):
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.is_trained = False
        self.training_samples = 0
        self.feature_names = [
            "hour_of_day",
            "day_of_week",
            "is_known_device",
            "is_known_location",
            "transaction_amount",
            "amount_last_1h",
            "failed_login_1h",
            "event_freq_5m",
            "login_attempts_count",
            "travel_velocity_kmh",
            "amount_z_score",
        ]
        self._load_model()

    def _load_model(self):
        """Load a previously saved model and PCA transformation from disk."""
        if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, PCA_PATH]):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.pca = joblib.load(PCA_PATH)
                self.is_trained = True
                print("[ML] Loaded pre-trained Hardcore fraud detection model.")
            except Exception as e:
                print(f"[ML] Failed to load model: {e}")
                self.is_trained = False

    def _save_model(self):
        """Persist model, scaler, and PCA transformation to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.pca, PCA_PATH)
        print(f"[ML] Models saved to {MODEL_DIR}")

    def _calculate_velocity(self, loc1: str, time1: datetime, loc2: str, time2: datetime) -> float:
        """Calculate km/h between two locations using Haversine formula."""
        if loc1 == loc2: return 0.0
        c1 = CITY_COORDS.get(loc1, CITY_COORDS["Unknown"])
        c2 = CITY_COORDS.get(loc2, CITY_COORDS["Unknown"])
        if c1 == (0, 0) or c2 == (0, 0): return 0.0

        # Haversine distance
        R = 6371  # Earth radius in km
        lat1, lon1 = np.radians(c1)
        lat2, lon2 = np.radians(c2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c

        # Time difference
        time_diff = abs((time2 - time1).total_seconds()) / 3600  # hours
        if time_diff < 0.0001: return 5000.0  # Instant travel (impossible)
        
        return distance / time_diff

    def _extract_features(self, event_data: dict, db: Session = None, account=None) -> np.ndarray:
        """Extract 'Hardcore' feature vector from a single event."""
        # 1. Temporal & Static
        timestamp = event_data.get("timestamp")
        if isinstance(timestamp, str):
            try: 
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
            except: 
                timestamp = datetime.utcnow()
        elif timestamp is None: 
            timestamp = datetime.utcnow()
        elif hasattr(timestamp, "tzinfo") and timestamp.tzinfo is not None:
             timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        
        hour = float(timestamp.hour if hasattr(timestamp, "hour") else 12)
        day_of_week = float(timestamp.weekday() if hasattr(timestamp, "weekday") else 0)

        # 2. Familiarity
        is_known_device = 0.0
        is_known_location = 0.0
        if account:
            if account.baseline_primary_device and event_data.get("device") == account.baseline_primary_device:
                is_known_device = 1.0
            if account.baseline_primary_location and event_data.get("location") == account.baseline_primary_location:
                is_known_location = 1.0

        # 3. Features
        tx_amount = float(event_data.get("transaction_amount", 0) or 0)
        login_attempts_count = float(event_data.get("login_attempts", 1) or 1)
        
        # 4. Behavioral Deltas (Hardcore)
        amount_last_1h = 0.0
        failed_login_1h = 0.0
        event_freq_5m = 0.0
        velocity_kmh = 0.0
        amount_z_score = 0.0
        
        if db and account:
            from ..database.models import Event
            now = datetime.utcnow()
            
            # Historical Stats for Z-Score
            # Historical Stats for Z-Score (SQLite compatible)
            history = db.query(Event.transaction_amount).filter(
                Event.account_id == account.id, Event.event_type == "transaction"
            ).all()
            
            amounts = [h[0] for h in history if h[0] is not None]
            if len(amounts) > 2:
                avg = np.mean(amounts)
                std = np.std(amounts)
                amount_z_score = (tx_amount - avg) / std if std > 0.001 else 0.0

            # Velocity Check (Impossible Travel)
            last_event = db.query(Event).filter(Event.account_id == account.id).order_by(Event.timestamp.desc()).first()
            if last_event and last_event.location:
                velocity_kmh = self._calculate_velocity(
                    last_event.location, last_event.timestamp,
                    event_data.get("location", "Unknown"), timestamp
                )

            # Rolling Windows
            one_hour_ago = now - timedelta(hours=1)
            five_mins_ago = now - timedelta(minutes=5)
            
            event_freq_5m = float(db.query(Event).filter(Event.account_id == account.id, Event.timestamp >= five_mins_ago).count())
            failed_login_1h = float(db.query(Event).filter(Event.account_id == account.id, Event.event_type == "login_failed", Event.timestamp >= one_hour_ago).count())
            amount_last_1h = float(db.query(func.sum(Event.transaction_amount)).filter(
                Event.account_id == account.id, Event.event_type == "transaction", Event.timestamp >= one_hour_ago
            ).scalar() or 0)

        # Assemble (11 features)
        features = np.array([
            hour, day_of_week, is_known_device, is_known_location, tx_amount,
            amount_last_1h, failed_login_1h, event_freq_5m, login_attempts_count,
            velocity_kmh, amount_z_score
        ]).reshape(1, -1)

        return features

    def train_from_csv(self, csv_path: str):
        """Train model and PCA on external CSV data with realistic feature variance."""
        if not os.path.exists(csv_path): return False
        try:
            print(f"[ML] Training Hardcore model on: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Subsample for faster training and balanced representation
            df_sampled = df.sample(n=min(10000, len(df)), random_state=42)
            
            rows = []
            
            # 1. Generate Normal Transactions
            for _, row in df_sampled.iterrows():
                try: dt = pd.to_datetime(row['TransactionDate'])
                except: dt = datetime.now()
                
                rows.append([
                    float(dt.hour), float(dt.weekday()), 
                    1.0, 1.0, # Known device/location
                    float(row['TransactionAmount']),
                    float(row['TransactionAmount']), # amount last 1h
                    0.0, 2.0, # failed login = 0, freq = 2
                    float(row.get('LoginAttempts', 1)),
                    0.0, # travel velocity = 0
                    0.0  # z-score normal
                ])
                
            # 2. Generate Normal Logins (amount = 0)
            for _, row in df_sampled.iterrows():
                try: dt = pd.to_datetime(row['TransactionDate'])
                except: dt = datetime.now()
                
                rows.append([
                    float(dt.hour), float(dt.weekday()), 
                    1.0, 1.0, # Known device/location
                    0.0, # amount = 0
                    0.0, # amount last 1h = 0
                    0.0, 1.0, # failed login = 0, freq = 1
                    1.0, # login attempts = 1
                    0.0, # travel velocity = 0
                    0.0  # z-score normal
                ])
                
            # 3. Inject Simulated Anomalies (to introduce variance for StandardScaler and PCA)
            np.random.seed(42)
            num_anomalies = 600
            
            for i in range(num_anomalies):
                # Random time
                hour = float(np.random.randint(0, 24))
                day = float(np.random.randint(0, 7))
                
                anomaly_type = i % 4
                if anomaly_type == 0:
                    # Impossible Travel
                    rows.append([
                        hour, day, 
                        1.0, 0.0, # Known device, Unknown location
                        0.0, 0.0,
                        0.0, 1.0,
                        1.0,
                        float(np.random.uniform(600.0, 2500.0)), # High velocity
                        0.0
                    ])
                elif anomaly_type == 1:
                    # Brute Force
                    rows.append([
                        hour, day, 
                        0.0, 1.0, # Unknown device, Known location
                        0.0, 0.0,
                        float(np.random.randint(4, 9)), # Multiple failed logins
                        float(np.random.randint(5, 10)), # High event frequency
                        float(np.random.randint(4, 9)), # High login attempts count
                        0.0,
                        0.0
                    ])
                elif anomaly_type == 2:
                    # Transaction Anomaly
                    amount = float(np.random.uniform(5000.0, 15000.0))
                    rows.append([
                        hour, day, 
                        1.0, 1.0,
                        amount,
                        amount,
                        0.0, 2.0,
                        1.0,
                        0.0,
                        float(np.random.uniform(3.5, 8.0)) # High Z-score
                    ])
                elif anomaly_type == 3:
                    # Fingerprint Mismatch / Unknown Device
                    rows.append([
                        hour, day, 
                        0.0, 0.0, # Unknown device & location
                        0.0, 0.0,
                        0.0, 1.0,
                        1.0,
                        0.0,
                        0.0
                    ])
                    
            X = np.array(rows)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Contamination represents the proportion of outliers in the dataset
            contamination_rate = num_anomalies / len(rows)
            
            # 1. Isolation Forest
            self.model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
            self.model.fit(X_scaled)
            
            # 2. PCA (For visualization)
            self.pca = PCA(n_components=2)
            self.pca.fit(X_scaled)
            
            self.is_trained = True
            self.training_samples = len(rows)
            self._save_model()
            print(f"[ML] Hardcore training complete ({len(rows)} samples, contamination={contamination_rate:.4f}).")
            return True
        except Exception as e:
            print(f"[ML] Training failed: {e}")
            return False

    def train(self, db: Session, force_csv: str = None):
        """Standard train hook."""
        if force_csv: return self.train_from_csv(force_csv)
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "bank_transactions_data_2_augmented_clean_2.csv")
        if os.path.exists(dataset_path):
            return self.train_from_csv(dataset_path)
        print("[ML] No dataset found for Hardcore training.")

    def predict(self, event_data: dict, db: Session = None, account=None) -> dict:
        """Predict with auto-visualization coordinates."""
        if not self.is_trained or self.model is None or self.scaler is None:
            return {"fraud_probability": 0.0, "is_anomaly": False, "explanation": [], "pca": [0, 0]}

        features = self._extract_features(event_data, db, account)
        features_scaled = self.scaler.transform(features)

        # 1. Fraud Logic
        raw_score = self.model.decision_function(features_scaled)[0]
        prediction = self.model.predict(features_scaled)[0]
        fraud_probability = max(0.0, min(1.0, 0.5 - raw_score))
        is_anomaly = prediction == -1

        # 2. PCA Visualization (X, Y)
        pca_coords = [float(x) for x in self.pca.transform(features_scaled)[0]]

        # 3. Explanations
        explanation = []
        if is_anomaly:
            mean_vals = self.scaler.mean_
            feat_vals = features[0]
            # Heuristic for top anomalies
            if feat_vals[9] > 900: explanation.append("Impossible travel detected")
            if abs(feat_vals[10]) > 3: explanation.append("Extreme amount deviation")
            if feat_vals[8] > 3: explanation.append("Suspicious login frequency")
            
            if not explanation:
                for i, name in enumerate(self.feature_names):
                    diff = abs((feat_vals[i] - mean_vals[i]) / (mean_vals[i] + 0.1))
                    if diff > 1.5: explanation.append(f"Unusual {name.replace('_', ' ')}")

        return {
            "fraud_probability": float(round(fraud_probability, 4)),
            "is_anomaly": bool(is_anomaly),
            "explanation": explanation[:3],
            "pca": pca_coords
        }

    def get_status(self) -> dict:
        return {
            "is_trained": self.is_trained,
            "training_samples": self.training_samples,
            "feature_names": self.feature_names,
            "model_type": "IsolationForest + PCA"
        }

fraud_model = FraudDetectionModel()
