"""
train.py — High-Accuracy Rainfall Forecasting Training Pipeline
===============================================================
Key improvements for NSE ≥ 0.88:
  • Log1p target transform  → handles skewed zero-inflated rainfall distribution
  • Longer lookback (60 days) → captures seasonal patterns
  • Cosine-decay + warmup LR schedule
  • Batch size 128 + mixed float16 precision → ~2× GPU throughput
  • tf.data pipeline with cache + prefetch → eliminates data starvation
  • Gradient clipping (clipnorm=1.0)
  • EarlyStopping with patience=12 (tight but fair)
  • MSE loss (unbiased for NSE optimisation)
  • Per-model best-weight restoration
"""
import os
import json
import joblib
import numpy as np
import tensorflow as tf

# ── Mixed precision: ~2× faster on any Ampere/Turing GPU, no-op on CPU ──
tf.keras.mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, LambdaCallback
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from data_preprocessing import load_and_clean_data, create_sequences
from models import get_all_models
from visualizer import generate_all_plots

# ─────────────────────────────────────────────
# Hydrological Metrics
# ─────────────────────────────────────────────

def nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency (1 = perfect, 0 = mean baseline, <0 = worse than mean)."""
    numerator   = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-9
    return float(1.0 - numerator / denominator)


def kge(y_true, y_pred):
    """Kling-Gupta Efficiency — complementary metric to NSE."""
    r  = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / (np.std(y_true) + 1e-9)
    beta  = np.mean(y_pred) / (np.mean(y_true) + 1e-9)
    return float(1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))


def pbias(y_true, y_pred):
    """Percent Bias (smaller |value| is better)."""
    return float(100.0 * np.sum(y_true - y_pred) / (np.sum(y_true) + 1e-9))


# ─────────────────────────────────────────────
# Custom NSE Loss  (minimise 1 - NSE)
# ─────────────────────────────────────────────

def nse_loss(y_true, y_pred):
    """TF NSE loss — maximises NSE by minimising 1-NSE."""
    residuals   = tf.reduce_sum(tf.square(y_true - y_pred))
    mean_obs    = tf.reduce_mean(y_true)
    denominator = tf.reduce_sum(tf.square(y_true - mean_obs)) + 1e-9
    return residuals / denominator  # 1-NSE (but constant 1 doesn't affect gradient)


# ─────────────────────────────────────────────
# Scale Data  (log1p target)
# ─────────────────────────────────────────────

def scale_data_log(df, target_col='prectotcorr', save_dir='models'):
    """
    Scale features with MinMaxScaler.
    Apply log1p to the rainfall target BEFORE MinMax scaling.
    This dramatically helps with zero-inflated, heavy-tailed rainfall distributions.
    """
    os.makedirs(save_dir, exist_ok=True)

    feature_cols = [col for col in df.columns if col != target_col]
    print(f"Features : {feature_cols}")
    print(f"Target   : {target_col}  (log1p + MinMax scaled)")

    features = df[feature_cols].values
    target   = df[[target_col]].values

    # --- Feature scaler ---
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)

    # --- Log1p then MinMax for target ---
    target_log = np.log1p(target)          # log1p(0) = 0, handles zeros safely
    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(target_log)

    # Save both scalers + log1p flag
    joblib.dump(feature_scaler, os.path.join(save_dir, 'feature_scaler.pkl'))
    joblib.dump(target_scaler,  os.path.join(save_dir, 'target_scaler.pkl'))

    # Save a small metadata file so inference service knows to inverse log1p
    meta = {'target_transform': 'log1p', 'target_col': target_col,
            'feature_cols': feature_cols,
            'x_days': 60}
    with open(os.path.join(save_dir, 'scaler_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("Scalers + metadata saved.")
    return scaled_features, scaled_target, feature_cols


def inverse_transform_target(scaled_pred, target_scaler):
    """Undo MinMax, then undo log1p  →  original mm scale."""
    log_pred = target_scaler.inverse_transform(scaled_pred)
    return np.expm1(log_pred)          # expm1 is the inverse of log1p


# ─────────────────────────────────────────────
# Cosine Warmup LR Schedule
# ─────────────────────────────────────────────

class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup for `warmup_steps`, then cosine decay to `min_lr`."""
    def __init__(self, base_lr=5e-4, min_lr=1e-5, warmup_steps=200, decay_steps=5000):
        super().__init__()
        self.base_lr     = base_lr
        self.min_lr      = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps  = decay_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.base_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        cos_decay = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1 + tf.cos(np.pi * (step - self.warmup_steps) /
                        tf.cast(self.decay_steps, tf.float32))
        )
        return tf.where(step < self.warmup_steps, warmup, cos_decay)

    def get_config(self):
        return {
            'base_lr': self.base_lr, 'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps, 'decay_steps': self.decay_steps
        }


# ─────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────

def train_models():
    data_file   = "dakshina_kannada_rainfall_daily_2000_2024.csv"
    models_dir  = "models"
    metrics_file = "training_metrics.json"
    os.makedirs(models_dir, exist_ok=True)

    # ── Hyper-parameters ──
    X_DAYS     = 60      # captures seasonal monsoon patterns
    Y_DAYS     = 1
    EPOCHS     = 150     # generous budget; EarlyStopping will cut it short
    BATCH_SIZE = 128     # larger batch → more GPU utilisation; cosine LR compensates
    PATIENCE   = 12      # EarlyStopping patience (tighter = faster dead-run exit)
    LR_BASE    = 5e-4
    LR_MIN     = 5e-6

    # ── Data preparation ──
    df = load_and_clean_data(data_file)
    f_scaled, t_scaled, feature_cols = scale_data_log(df, save_dir=models_dir)
    X, y = create_sequences(f_scaled, t_scaled, x_days=X_DAYS, y_days=Y_DAYS)

    target_scaler = joblib.load(os.path.join(models_dir, 'target_scaler.pkl'))

    # Temporal split — no shuffle to preserve time ordering
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=False
    )
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}\n")

    input_shape = (X_DAYS, X_train.shape[2])
    models = get_all_models(input_shape, Y_DAYS)

    # ── tf.data pipelines — cache in RAM, prefetch overlaps GPU+CPU ──
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .cache()
        .shuffle(buffer_size=len(X_train), seed=42, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    # ── Pre-compute test labels in mm (for metric reporting) ──
    y_test_mm = inverse_transform_target(y_test, target_scaler).flatten()

    all_metrics = {}

    for model in models:
        name = model.name
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        # ── LR Schedule ──
        steps_per_epoch = len(X_train) // BATCH_SIZE
        total_steps     = steps_per_epoch * EPOCHS
        warmup_steps    = steps_per_epoch * 5   # 5-epoch warm-up

        lr_schedule = CosineDecayWithWarmup(
            base_lr=LR_BASE, min_lr=LR_MIN,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps
        )
        optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)

        model.compile(
            optimizer=optimizer,
            loss='mse',                 # MSE is directly related to NSE
            metrics=['mae']
        )
        model.summary(print_fn=lambda s: None)   # suppress verbose summary

        model_path = os.path.join(models_dir, f"{name}.keras")

        callbacks = [
            EarlyStopping(
                monitor='val_loss', patience=PATIENCE,
                restore_best_weights=True, verbose=1
            ),
            ModelCheckpoint(
                model_path, monitor='val_loss',
                save_best_only=True, verbose=0
            ),
            # NOTE: ReduceLROnPlateau is intentionally removed — it conflicts
            # with LearningRateSchedule-based optimizers (cannot set .learning_rate
            # on a schedule). CosineDecayWithWarmup already handles LR decay.
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=2   # one clean line per epoch; no per-batch progress bar I/O
        )

        # ── Evaluation in original mm scale ──
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred_mm     = inverse_transform_target(y_pred_scaled, target_scaler).flatten()

        # Clip negatives (physical impossibility)
        y_pred_mm = np.clip(y_pred_mm, 0.0, None)

        rmse     = float(np.sqrt(mean_squared_error(y_test_mm, y_pred_mm)))
        mae      = float(mean_absolute_error(y_test_mm, y_pred_mm))
        nse_val  = nse(y_test_mm, y_pred_mm)
        kge_val  = kge(y_test_mm, y_pred_mm)
        pb       = pbias(y_test_mm, y_pred_mm)

        print(f"\n  ✔ {name} Results:")
        print(f"    NSE   : {nse_val:+.4f}   (target ≥ 0.880)")
        print(f"    KGE   : {kge_val:+.4f}")
        print(f"    RMSE  : {rmse:.2f} mm")
        print(f"    MAE   : {mae:.2f} mm")
        print(f"    PBIAS : {pb:+.2f}%")

        # ── Save metrics JSON ──
        hist_dict = {k: [float(v) for v in vals]
                     for k, vals in history.history.items()}
        all_metrics[name] = {
            "history": hist_dict,
            "final_metrics": {
                "loss": float(history.history['val_loss'][
                    int(np.argmin(history.history['val_loss']))
                ]),
                "rmse":  rmse,
                "mae":   mae,
                "nse":   nse_val,
                "kge":   kge_val,
                "pbias": pb,
            }
        }

        # ── Generate plots ──
        generate_all_plots(y_test_mm, y_pred_mm, hist_dict, name)

    # ── Persist metrics ──
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n{'='*60}")
    print(f"All models trained. Metrics → {metrics_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.random.set_seed(42)
    np.random.seed(42)
    train_models()
