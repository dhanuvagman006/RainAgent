import os
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

MODELS_DIR = "models"
MODEL_NAMES = ["LSTM", "GRU", "Bi-LSTM", "1D-CNN", "CNN-LSTM", "Transformer"]

# Default lookback — overridden by scaler_meta.json if present
DEFAULT_X_DAYS = 60

# ── Custom objects needed to load models compiled with composite_loss ────────
def _nse_loss(y_true, y_pred):
    residuals   = tf.reduce_sum(tf.square(y_true - y_pred))
    mean_obs    = tf.reduce_mean(y_true)
    denominator = tf.reduce_sum(tf.square(y_true - mean_obs)) + 1e-9
    return residuals / denominator

def _composite_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return 0.7 * _nse_loss(y_true, y_pred) + 0.3 * mse

_CUSTOM_OBJECTS = {'nse_loss': _nse_loss, 'composite_loss': _composite_loss}

class InferenceService:
    def __init__(self):
        self.models = {}
        self.isotonic = {}          # per-model isotonic calibrators
        self.feature_scaler = None
        self.target_scaler = None
        self.target_transform = 'none'  # 'log1p' or 'none'
        self.x_days = DEFAULT_X_DAYS
        self.load_artifacts()

    def load_artifacts(self):
        try:
            feature_scaler_path = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
            target_scaler_path  = os.path.join(MODELS_DIR, 'target_scaler.pkl')
            meta_path           = os.path.join(MODELS_DIR, 'scaler_meta.json')

            if os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):
                self.feature_scaler = joblib.load(feature_scaler_path)
                self.target_scaler  = joblib.load(target_scaler_path)

            # Read metadata written by train.py (target transform + x_days)
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self.target_transform = meta.get('target_transform', 'none')
                self.x_days = int(meta.get('x_days', DEFAULT_X_DAYS))

            for name in MODEL_NAMES:
                model_path = os.path.join(MODELS_DIR, f"{name}.keras")
                if os.path.exists(model_path):
                    self.models[name] = load_model(
                        model_path,
                        compile=False,
                        custom_objects=_CUSTOM_OBJECTS,
                        safe_mode=False
                    )
                else:
                    print(f"Warning: {model_path} not found.")

                # Load isotonic calibrator if available
                iso_path = os.path.join(MODELS_DIR, f"{name}_isotonic.pkl")
                if os.path.exists(iso_path):
                    self.isotonic[name] = joblib.load(iso_path)

            print("ML Artifacts loaded successfully.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")

    def _inverse_transform(self, scaled_pred):
        """Undo MinMax scaling, then undo optional log1p."""
        raw = self.target_scaler.inverse_transform(scaled_pred)
        if self.target_transform == 'log1p':
            raw = np.expm1(raw)
        return np.maximum(raw, 0.0)

    def predict(self, model_name, synthetic_features, horizon):
        """
        synthetic_features: numpy array shape (total_days, num_features).
                            total_days = x_days + horizon - 1
        """
        if self.feature_scaler is None or self.target_scaler is None:
            raise RuntimeError("Scalers are not loaded. Ensure Module 1 is trained.")

        # Scale features
        scaled_features = self.feature_scaler.transform(synthetic_features)

        x_days = self.x_days
        num_windows = len(scaled_features) - x_days + 1

        if num_windows != horizon:
            raise ValueError(
                f"Generated windows {num_windows} != horizon {horizon}. "
                f"Ensure simulator uses lookback_days={x_days}."
            )

        X = []
        for i in range(num_windows):
            X.append(scaled_features[i:i + x_days])
        X = np.array(X)  # shape (horizon, x_days, num_features)
        
        # Determine which models to run
        if model_name == "Ensemble":
            top_models = ["Transformer", "Bi-LSTM", "CNN-LSTM"]
            preds = []
            for name in top_models:
                if name in self.models:
                    p = self.models[name].predict(X)
                    preds.append(p)
            if not preds:
                raise ValueError("None of the ensemble models are available.")
            final_pred_scaled = np.mean(preds, axis=0)
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found.")
            final_pred_scaled = self.models[model_name].predict(X)

        # Inverse transform (handles log1p automatically)
        final_pred = self._inverse_transform(final_pred_scaled).flatten()
        final_pred = np.clip(final_pred, 0.0, None)

        # Apply isotonic calibration when available (reduces PBIAS, boosts NSE)
        if model_name == "Ensemble":
            # Average calibrated predictions per sub-model
            cal_preds = []
            for sub in ["Transformer", "Bi-LSTM", "CNN-LSTM"]:
                if sub in self.isotonic:
                    cal_preds.append(self.isotonic[sub].predict(final_pred))
            if cal_preds:
                final_pred = np.mean(cal_preds, axis=0)
        elif model_name in self.isotonic:
            final_pred = self.isotonic[model_name].predict(final_pred)

        final_pred = np.clip(final_pred, 0.0, None)
        return final_pred.tolist()

# Singleton instance
inference_service = InferenceService()
