import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

MODELS_DIR = "models"
MODEL_NAMES = ["LSTM", "GRU", "Bi-LSTM", "1D-CNN", "CNN-LSTM", "Transformer"]

class InferenceService:
    def __init__(self):
        self.models = {}
        self.feature_scaler = None
        self.target_scaler = None
        self.load_artifacts()

    def load_artifacts(self):
        try:
            # We assume the models directory exists and contains these from Module 1
            feature_scaler_path = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
            target_scaler_path = os.path.join(MODELS_DIR, 'target_scaler.pkl')
            
            if os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):
                self.feature_scaler = joblib.load(feature_scaler_path)
                self.target_scaler = joblib.load(target_scaler_path)
            
            # Load models without compile=False to avoid custom_objects error for NSE
            for name in MODEL_NAMES:
                model_path = os.path.join(MODELS_DIR, f"{name}.keras")
                if os.path.exists(model_path):
                    self.models[name] = load_model(model_path, compile=False)
                else:
                    print(f"Warning: {model_path} not found.")
            print("ML Artifacts loaded successfully.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")

    def predict(self, model_name, synthetic_features, horizon):
        """
        synthetic_features: numpy array shape (total_days, num_features).
                            total_days = 30 + horizon - 1
        """
        if self.feature_scaler is None or self.target_scaler is None:
            raise RuntimeError("Scalers are not loaded. Ensure Module 1 is trained.")

        # Scale features
        scaled_features = self.feature_scaler.transform(synthetic_features)
        
        # Create sliding windows of size 30
        x_days = 30
        num_windows = len(scaled_features) - x_days + 1
        
        if num_windows != horizon:
            raise ValueError(f"Generated windows {num_windows} does not match horizon {horizon}.")
            
        X = []
        for i in range(num_windows):
            X.append(scaled_features[i:i+x_days])
        X = np.array(X) # shape (horizon, 30, num_features)
        
        # Determine which models to run
        if model_name == "Ensemble":
            # Ensemble of top 3 typical best performers for sequential tasks
            top_models = ["Transformer", "Bi-LSTM", "CNN-LSTM"]
            preds = []
            for name in top_models:
                if name in self.models:
                    model = self.models[name]
                    p = model.predict(X) # shape (horizon, 1)
                    preds.append(p)
            
            if not preds:
                raise ValueError("None of the ensemble models are available.")
                
            avg_pred = np.mean(preds, axis=0) # shape (horizon, 1)
            final_pred_scaled = avg_pred
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found.")
            model = self.models[model_name]
            final_pred_scaled = model.predict(X) # shape (horizon, 1)
            
        # Inverse transform to get actual mm
        final_pred = self.target_scaler.inverse_transform(final_pred_scaled)
        
        # Ensure no negative rainfall
        final_pred = np.maximum(final_pred, 0)
        
        return final_pred.flatten().tolist()

# Singleton instance
inference_service = InferenceService()
