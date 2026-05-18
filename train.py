
"""
train.py  —  RainAgent  (cross-platform, multi-core optimised)
═══════════════════════════════════════════════════════════════
• Auto-detects all available CPU cores  →  inter/intra-op threads + tf.data workers
• MirroredStrategy: uses ALL GPUs when present, falls back to CPU silently
• Mixed-precision: enabled only when a GPU is detected (no-op penalty removed on CPU)
• Rich callbacks: EarlyStopping, ModelCheckpoint (resume), ReduceLROnPlateau,
  CSVLogger, TensorBoard, and a custom colour progress callback with ETA
• Checkpoint resumption: if a valid .keras file exists it is reloaded and re-evaluated
  (skips full retrain) — remove the file to force a fresh run
"""

import os
import sys
import json
import time
import math
import joblib
import datetime
import numpy as np
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Cross-platform CPU / GPU configuration
# ─────────────────────────────────────────────────────────────────────────────

def _configure_hardware():
    """
    Returns (strategy, num_cpu_cores, using_gpu).

    • Reads all physical CPU cores (works on Linux, macOS, Windows).
    • Sets TF inter/intra-op parallelism to saturate every core.
    • Enables mixed precision ONLY when a GPU is available.
    • Uses MirroredStrategy if ≥1 GPU found, else OneDeviceStrategy on CPU.
    """
    # --- CPU core detection (cross-platform) ---
    num_cores = os.cpu_count() or 4          # os.cpu_count() works on all OSes
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)

    # --- GPU detection ---
    gpus = tf.config.list_physical_devices('GPU')
    using_gpu = len(gpus) > 0

    if using_gpu:
        # Allow GPU memory to grow rather than pre-allocating all VRAM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        strategy = tf.distribute.MirroredStrategy()
        print(f"[HW] GPUs found: {len(gpus)} — using MirroredStrategy + mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy('float32')   # no penalty on CPU
        strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
        print(f"[HW] No GPU detected — using CPU ({num_cores} cores) with float32")

    return strategy, num_cores, using_gpu


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Hydrological Metrics
# ─────────────────────────────────────────────────────────────────────────────

def nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency (1=perfect, 0=mean baseline, <0=worse than mean)."""
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-9
    return float(1.0 - num / den)


def kge(y_true, y_pred):
    """Kling-Gupta Efficiency — complementary metric to NSE."""
    r     = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred)  / (np.std(y_true)  + 1e-9)
    beta  = np.mean(y_pred) / (np.mean(y_true) + 1e-9)
    return float(1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))


def pbias(y_true, y_pred):
    """Percent Bias (smaller |value| = better)."""
    return float(100.0 * np.sum(y_true - y_pred) / (np.sum(y_true) + 1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Custom NSE loss
# ─────────────────────────────────────────────────────────────────────────────

def nse_loss(y_true, y_pred):
    """TF NSE loss — maximises NSE by minimising 1-NSE."""
    residuals   = tf.reduce_sum(tf.square(y_true - y_pred))
    mean_obs    = tf.reduce_mean(y_true)
    denominator = tf.reduce_sum(tf.square(y_true - mean_obs)) + 1e-9
    return residuals / denominator


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Data scaling (log1p + MinMax)
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import MinMaxScaler

def scale_data_log(df, target_col='prectotcorr', save_dir='models'):
    """
    Scale features with MinMaxScaler.
    Apply log1p to the rainfall target BEFORE MinMax scaling.
    Dramatically helps with zero-inflated, heavy-tailed rainfall distributions.
    """
    os.makedirs(save_dir, exist_ok=True)
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"[DATA] Features : {feature_cols}")
    print(f"[DATA] Target   : {target_col}  (log1p → MinMax)")

    features = df[feature_cols].values
    target   = df[[target_col]].values

    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)

    target_log    = np.log1p(target)
    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform(target_log)

    joblib.dump(feature_scaler, os.path.join(save_dir, 'feature_scaler.pkl'))
    joblib.dump(target_scaler,  os.path.join(save_dir, 'target_scaler.pkl'))

    meta = {
        'target_transform': 'log1p',
        'target_col': target_col,
        'feature_cols': feature_cols,
        'x_days': 60,
    }
    with open(os.path.join(save_dir, 'scaler_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("[DATA] Scalers + metadata saved.")
    return scaled_features, scaled_target, feature_cols


def inverse_transform_target(scaled_pred, target_scaler):
    """Undo MinMax then undo log1p → original mm scale."""
    log_pred = target_scaler.inverse_transform(scaled_pred)
    return np.expm1(log_pred)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Rich Progress Callback  (cross-platform, no ANSI on Windows CMD)
# ─────────────────────────────────────────────────────────────────────────────

_USE_COLOUR = sys.stdout.isatty() and os.name != 'nt'   # plain on Windows CMD


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text


class RichProgressCallback(tf.keras.callbacks.Callback):
    """
    Prints a compact per-epoch summary with:
      • elapsed / estimated-remaining time
      • loss + val_loss + current learning rate
      • Unicode progress bar (ASCII fallback on Windows)
    """

    def __init__(self, total_epochs, model_name):
        super().__init__()
        self.total_epochs = total_epochs
        self.model_name   = model_name
        self._t0          = None
        self._bar_char    = "█" if os.name != 'nt' else "#"

    def on_train_begin(self, logs=None):
        self._t0 = time.time()
        print(_c(f"\n  ▶  Training {self.model_name}", "1;36"))

    def on_epoch_end(self, epoch, logs=None):
        logs     = logs or {}
        elapsed  = time.time() - self._t0
        progress = (epoch + 1) / self.total_epochs
        eta_secs = elapsed / max(progress, 1e-6) * (1 - progress)

        bar_len   = 30
        filled    = int(bar_len * progress)
        bar       = self._bar_char * filled + "─" * (bar_len - filled)

        loss    = logs.get('loss',     float('nan'))
        val_loss = logs.get('val_loss', float('nan'))

        # Safely retrieve LR regardless of schedule type
        try:
            lr_val = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception:
            lr_val = float('nan')

        def fmt_t(s):
            return str(datetime.timedelta(seconds=int(s)))

        line = (
            f"  [{bar}] {epoch+1:>4}/{self.total_epochs}"
            f"  loss={loss:.4f}  val={val_loss:.4f}"
            f"  lr={lr_val:.2e}"
            f"  elapsed={fmt_t(elapsed)}  ETA={fmt_t(eta_secs)}"
        )
        # Colour val_loss green/red vs loss
        print(_c(line, "32") if val_loss <= loss else _c(line, "33"))

    def on_train_end(self, logs=None):
        total = time.time() - self._t0
        print(_c(f"  ✔  {self.model_name} finished in {datetime.timedelta(seconds=int(total))}\n", "1;32"))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  tf.data pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_datasets(X_train, y_train, X_test, y_test, batch_size, num_cores):
    """
    Returns (train_ds, val_ds) as performance-tuned tf.data pipelines.
    num_parallel_calls is set to num_cores so every CPU thread is used.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .cache()
        .shuffle(buffer_size=len(X_train), seed=42, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .cache()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_models():
    # ── imports placed here to avoid circular issues ──
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from data_preprocessing import load_and_clean_data, create_sequences
    from models import get_all_models
    from visualizer import generate_all_plots

    # ── Hardware setup ──
    strategy, num_cores, using_gpu = _configure_hardware()

    # ── Paths & hyper-params ──
    DATA_FILE    = "dakshina_kannada_rainfall_daily_2000_2024.csv"
    MODELS_DIR   = "models"
    METRICS_FILE = "training_metrics.json"
    LOG_DIR      = os.path.join(MODELS_DIR, "logs")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,    exist_ok=True)

    X_DAYS     = 60
    Y_DAYS     = 1
    EPOCHS     = 150
    # Scale batch size with GPU count to maintain effective batch size
    N_REPLICAS  = strategy.num_replicas_in_sync
    BATCH_SIZE  = 128 * N_REPLICAS      # e.g. 256 with 2 GPUs, 128 on CPU/1-GPU
    PATIENCE_ES = 15     # EarlyStopping patience
    PATIENCE_LR = 6      # ReduceLROnPlateau patience
    LR_BASE     = 5e-4
    LR_MIN      = 5e-6

    print(f"[CFG] Replicas={N_REPLICAS}  BatchSize={BATCH_SIZE}  Cores={num_cores}")

    # ── Data ──
    df = load_and_clean_data(DATA_FILE)
    f_scaled, t_scaled, feature_cols = scale_data_log(df, save_dir=MODELS_DIR)
    X, y = create_sequences(f_scaled, t_scaled, x_days=X_DAYS, y_days=Y_DAYS)

    target_scaler = joblib.load(os.path.join(MODELS_DIR, 'target_scaler.pkl'))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=False
    )
    print(f"[DATA] Train: {X_train.shape}  Test: {X_test.shape}\n")

    input_shape = (X_DAYS, X_train.shape[2])
    y_test_mm   = inverse_transform_target(y_test, target_scaler).flatten()

    # Build datasets once (shared across all models — they are stateless reads)
    train_ds, val_ds = _build_datasets(
        X_train, y_train, X_test, y_test, BATCH_SIZE, num_cores
    )

    all_metrics = {}

    # ── Load existing metrics if present (so we don't lose earlier results) ──
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            all_metrics = json.load(f)

    # ── Model loop ──
    # get_all_models is called INSIDE the strategy scope so weights are
    # distributed correctly across all replicas.
    with strategy.scope():
        models = get_all_models(input_shape, Y_DAYS)

    for model in models:
        name       = model.name
        model_path = os.path.join(MODELS_DIR, f"{name}.keras")
        csv_log    = os.path.join(LOG_DIR, f"{name}_history.csv")
        tb_log_dir = os.path.join(LOG_DIR, "tensorboard", name)

        print(f"\n{'='*62}")
        print(f"  Model : {name}")
        print(f"{'='*62}")

        # ── Checkpoint Resumption ───────────────────────────────────
        if os.path.exists(model_path):
            print(f"  [CKPT] Found existing checkpoint — loading {model_path}")
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'nse_loss': nse_loss}
                )
                skip_train = True
            except Exception as e:
                print(f"  [CKPT] Load failed ({e}), retraining from scratch.")
                skip_train = False
        else:
            skip_train = False

        if not skip_train:
            # ── Compile (must be inside strategy scope for multi-GPU) ──
            with strategy.scope():
                # Plain float LR so ReduceLROnPlateau can mutate it freely
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=LR_BASE,
                    clipnorm=1.0
                )
                model.compile(
                    optimizer=optimizer,
                    loss='mse',
                    metrics=['mae']
                )

            model.summary(print_fn=lambda s: None)   # suppress wall-of-text

            # ── Callbacks ───────────────────────────────────────────
            callbacks = [
                # 1. Stop early when val_loss stops improving
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=PATIENCE_ES,
                    restore_best_weights=True,
                    verbose=1,
                    min_delta=1e-5,
                ),
                # 2. Save the best model to disk during training
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=0,
                ),
                # 3. Halve LR when plateau detected (safe — LR is a plain float)
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=PATIENCE_LR,
                    min_lr=LR_MIN,
                    verbose=1,
                    min_delta=1e-5,
                ),
                # 4. Persist epoch-level history to CSV (cross-platform)
                tf.keras.callbacks.CSVLogger(
                    csv_log,
                    separator=',',
                    append=False,
                ),
                # 5. TensorBoard (optional — run: tensorboard --logdir models/logs/tensorboard)
                tf.keras.callbacks.TensorBoard(
                    log_dir=tb_log_dir,
                    histogram_freq=0,       # set to 1 for weight histograms (slower)
                    write_graph=False,      # keep log dir small
                    update_freq='epoch',
                ),
                # 6. Our custom rich progress bar with ETA
                RichProgressCallback(
                    total_epochs=EPOCHS,
                    model_name=name,
                ),
            ]

            # ── Train ────────────────────────────────────────────────
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=0,      # RichProgressCallback handles all output
            )
            hist_dict = {k: [float(v) for v in vals]
                         for k, vals in history.history.items()}
        else:
            # Resumed from checkpoint — rebuild a minimal hist_dict from CSV if available
            hist_dict = {}
            if os.path.exists(csv_log):
                import csv
                with open(csv_log) as cf:
                    reader = csv.DictReader(cf)
                    for row in reader:
                        for k, v in row.items():
                            if k == 'epoch':
                                continue
                            hist_dict.setdefault(k, []).append(float(v) if v else float('nan'))
            print(f"  [CKPT] Skipped training — evaluating loaded checkpoint.")

        # ── Evaluate in original mm scale ──────────────────────────
        y_pred_scaled = model.predict(X_test, verbose=0, batch_size=BATCH_SIZE)
        y_pred_mm     = inverse_transform_target(y_pred_scaled, target_scaler).flatten()
        y_pred_mm     = np.clip(y_pred_mm, 0.0, None)    # physical constraint

        rmse    = float(np.sqrt(mean_squared_error(y_test_mm, y_pred_mm)))
        mae_val = float(mean_absolute_error(y_test_mm, y_pred_mm))
        nse_val = nse(y_test_mm, y_pred_mm)
        kge_val = kge(y_test_mm, y_pred_mm)
        pb      = pbias(y_test_mm, y_pred_mm)

        print(f"\n  {'✔' if nse_val >= 0.8 else '⚠'} {name} Results:")
        print(f"    NSE   : {nse_val:+.4f}   (target ≥ 0.880)")
        print(f"    KGE   : {kge_val:+.4f}")
        print(f"    RMSE  : {rmse:.2f} mm")
        print(f"    MAE   : {mae_val:.2f} mm")
        print(f"    PBIAS : {pb:+.2f}%")

        # ── Build best-epoch val_loss safely ───────────────────────
        if hist_dict.get('val_loss'):
            best_val_loss = float(np.min(hist_dict['val_loss']))
        else:
            best_val_loss = float(model.evaluate(val_ds, verbose=0)[0])

        all_metrics[name] = {
            "history": hist_dict,
            "final_metrics": {
                "loss":  best_val_loss,
                "rmse":  rmse,
                "mae":   mae_val,
                "nse":   nse_val,
                "kge":   kge_val,
                "pbias": pb,
            }
        }

        # Persist after each model — don't lose work if a later model crashes
        with open(METRICS_FILE, 'w') as f:
            json.dump(all_metrics, f, indent=4)

        # ── Generate plots ──────────────────────────────────────────
        generate_all_plots(y_test_mm, y_pred_mm, hist_dict, name)

    print(f"\n{'='*62}")
    print(f"  All models trained. Metrics → {METRICS_FILE}")
    print(f"  TensorBoard logs  → {os.path.join(LOG_DIR, 'tensorboard')}")
    print(f"{'='*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # suppress TF C++ noise
    tf.random.set_seed(42)
    np.random.seed(42)
    train_models()
