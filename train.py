import os
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_preprocessing import load_and_clean_data, scale_data, create_sequences
from models import get_all_models
from visualizer import generate_all_plots

# Global NSE calculation on un-scaled (mm) predictions
def calculate_global_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (numerator / (denominator + 1e-7))

def train_models():
    data_file = "dakshina_kannada_rainfall_daily_2000_2024.csv"
    models_dir = "models"
    metrics_file = "training_metrics.json"
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Parameters
    x_days = 30
    y_days = 1
    epochs = 50
    batch_size = 64
    
    # 1. Prepare data
    df = load_and_clean_data(data_file)
    f_scaled, t_scaled, _ = scale_data(df, save_dir=models_dir)
    X, y = create_sequences(f_scaled, t_scaled, x_days=x_days, y_days=y_days)
    
    # Load the target scaler for inverse transforming during evaluation
    target_scaler = joblib.load(os.path.join(models_dir, 'target_scaler.pkl'))
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    input_shape = (x_days, X_train.shape[2])
    
    # 2. Get models
    models = get_all_models(input_shape, y_days)
    
    all_metrics = {}
    
    # 3. Train each model sequentially
    for model in models:
        model_name = model.name
        print(f"\n--- Training {model_name} ---")
        
        # Compile model with Huber Loss and Gradient Clipping
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae']
        )
        
        # Callbacks
        model_path = os.path.join(models_dir, f"{model_name}.keras")
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        # Predictions
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # Un-scale to calculate true physical metrics in mm
        y_test_mm = target_scaler.inverse_transform(y_test)
        y_pred_mm = target_scaler.inverse_transform(y_pred_scaled)
        
        y_test_mm_flat = y_test_mm.flatten()
        y_pred_mm_flat = y_pred_mm.flatten()
        
        # Compute true Un-Scaled metrics
        unscaled_rmse = float(np.sqrt(mean_squared_error(y_test_mm_flat, y_pred_mm_flat)))
        unscaled_mae = float(mean_absolute_error(y_test_mm_flat, y_pred_mm_flat))
        unscaled_nse = float(calculate_global_nse(y_test_mm_flat, y_pred_mm_flat))
        
        # Save metrics for JSON
        hist_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
        
        final_eval = {
            "loss": float(history.history['loss'][-1]),
            "rmse": unscaled_rmse,
            "mae": unscaled_mae,
            "nse": unscaled_nse
        }
        
        all_metrics[model_name] = {
            "history": hist_dict,
            "final_metrics": final_eval
        }
        
        print(f"Finished training {model_name}. Unscaled metrics -> NSE: {unscaled_nse:.4f}, RMSE: {unscaled_rmse:.2f}mm, MAE: {unscaled_mae:.2f}mm")
        
        # 4. Generate automated plots for this model
        generate_all_plots(y_test_mm_flat, y_pred_mm_flat, hist_dict, model_name)
        
    # 5. Save training metrics
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"\nAll models trained successfully. Metrics saved to {metrics_file} and plots generated in frontend/public/plots/.")

if __name__ == "__main__":
    # Hide verbose TF logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    train_models()
