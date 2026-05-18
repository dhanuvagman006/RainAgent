import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(file_path):
    """
    Loads the dataset and cleans it by handling missing values.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Forward fill to handle NaNs, then interpolate any remaining
    df = df.ffill().interpolate(method='linear', limit_direction='both')
    print(f"Data loaded and cleaned. Shape: {df.shape}")
    return df

def scale_data(df, target_col='prectotcorr', save_dir='models'):
    """
    Scales features and target independently using MinMaxScaler.
    Saves the scalers to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Identify features
    feature_cols = [col for col in df.columns if col != target_col]
    
    print(f"Features used: {feature_cols}")
    print(f"Target used: {target_col}")
    
    features = df[feature_cols].values
    target = df[[target_col]].values # Keep as 2D for scaler
    
    # Initialize scalers
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    # Fit and transform
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)
    
    # Save scalers
    joblib.dump(feature_scaler, os.path.join(save_dir, 'feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(save_dir, 'target_scaler.pkl'))
    print("Scalers saved successfully.")
    
    return scaled_features, scaled_target, feature_cols

def create_sequences(features, target, x_days=30, y_days=1):
    """
    Creates sliding window sequences for time-series forecasting.
    
    Args:
        features (np.array): Scaled features array
        target (np.array): Scaled target array
        x_days (int): Number of past days to use as input
        y_days (int): Number of future days to predict
        
    Returns:
        X (np.array): 3D array of shape (samples, x_days, num_features)
        y (np.array): 2D array of shape (samples, y_days)
    """
    X, y = [], []
    num_samples = len(features) - x_days - y_days + 1
    
    for i in range(num_samples):
        # Input sequence: past x_days
        X.append(features[i:i+x_days])
        # Target sequence: next y_days
        y.append(target[i+x_days:i+x_days+y_days].flatten())
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Sequence generated. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

if __name__ == "__main__":
    # Test the module locally
    df = load_and_clean_data("dakshina_kannada_rainfall_daily_2000_2024.csv")
    f_scaled, t_scaled, f_cols = scale_data(df)
    X, y = create_sequences(f_scaled, t_scaled, x_days=30, y_days=1)
