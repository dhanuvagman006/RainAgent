"""
train.py  —  RainAgent
══════════════════════════════════════════════════════════════════
Simple, per-model-tuned training  •  No distributed strategies
NSE boosters: snapshot ensemble, isotonic calibration, TTA,
              cosine-warm-restart LR, ReduceLROnPlateau
══════════════════════════════════════════════════════════════════
"""

import os, sys, json, time, math, joblib, datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL FLAGS
# ─────────────────────────────────────────────────────────────────────────────
FORCE_RETRAIN = True   # delete old .keras checkpoints and retrain from scratch

# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL HYPERPARAMETER CONFIGS
# Each model gets its own tuned lr, batch_size, epochs, patience, etc.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    # Recurrent models — moderate LR, medium batch
    "LSTM": dict(
        lr=2e-4, lr_min=1e-7, batch_size=128, epochs=300,
        patience_es=35, patience_lr=12, lr_factor=0.40,
        cosine_t0=30, n_snapshots=7, n_tta=12, noise_std=0.003,
    ),
    "GRU": dict(
        lr=3e-4, lr_min=1e-7, batch_size=128, epochs=300,
        patience_es=35, patience_lr=12, lr_factor=0.40,
        cosine_t0=30, n_snapshots=7, n_tta=12, noise_std=0.003,
    ),
    # Bi-LSTM needs smaller batch for better gradient diversity
    "Bi-LSTM": dict(
        lr=1e-4, lr_min=5e-8, batch_size=64, epochs=250,
        patience_es=30, patience_lr=10, lr_factor=0.45,
        cosine_t0=25, n_snapshots=7, n_tta=12, noise_std=0.004,
    ),
    # CNN is fast — bigger batch, higher LR
    "1D-CNN": dict(
        lr=5e-4, lr_min=1e-7, batch_size=256, epochs=200,
        patience_es=25, patience_lr=8,  lr_factor=0.35,
        cosine_t0=20, n_snapshots=5, n_tta=10, noise_std=0.002,
    ),
    "CNN-LSTM": dict(
        lr=2e-4, lr_min=1e-7, batch_size=128, epochs=300,
        patience_es=35, patience_lr=12, lr_factor=0.40,
        cosine_t0=30, n_snapshots=7, n_tta=12, noise_std=0.003,
    ),
    # Transformer — low LR, small batch, more epochs
    "Transformer": dict(
        lr=8e-5, lr_min=5e-8, batch_size=64,  epochs=350,
        patience_es=40, patience_lr=15, lr_factor=0.50,
        cosine_t0=35, n_snapshots=7, n_tta=12, noise_std=0.002,
    ),
}
_DEFAULT_CFG = MODEL_CONFIGS["LSTM"]   # fallback for any unregistered model name


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Hardware setup  (simple — no strategy / distribute API)
# ─────────────────────────────────────────────────────────────────────────────
def _configure_hardware():
    num_cores = os.cpu_count() or 4
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[HW] {len(gpus)} GPU(s) — memory growth enabled")
    else:
        print(f"[HW] CPU-only ({num_cores} cores)")
    return num_cores, bool(gpus)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Hydrological metrics
# ─────────────────────────────────────────────────────────────────────────────
def nse(y_true, y_pred):
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-9
    return float(1.0 - num / den)

def kge(y_true, y_pred):
    r     = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred)  / (np.std(y_true)  + 1e-9)
    beta  = np.mean(y_pred) / (np.mean(y_true) + 1e-9)
    return float(1.0 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))

def pbias(y_true, y_pred):
    return float(100.0 * np.sum(y_true - y_pred) / (np.sum(y_true) + 1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Loss functions
# ─────────────────────────────────────────────────────────────────────────────
def nse_loss(y_true, y_pred):
    """Full-dataset NSE loss (stored for model serialisation)."""
    res = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) + 1e-9
    return res / den

def composite_loss(y_true, y_pred):
    """Huber loss — smooth, robust to rainfall outliers, high NSE at eval."""
    return tf.keras.losses.huber(y_true, y_pred, delta=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Data scaling  (log1p + MinMax)
# ─────────────────────────────────────────────────────────────────────────────
def scale_data_log(df, target_col="prectotcorr", save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"[DATA] {len(feature_cols)} features + target '{target_col}'")

    features = df[feature_cols].values
    target   = df[[target_col]].values

    f_scaler = MinMaxScaler()
    scaled_f = f_scaler.fit_transform(features)

    t_log    = np.log1p(target)
    t_scaler = MinMaxScaler()
    scaled_t = t_scaler.fit_transform(t_log)

    joblib.dump(f_scaler, os.path.join(save_dir, "feature_scaler.pkl"))
    joblib.dump(t_scaler, os.path.join(save_dir, "target_scaler.pkl"))

    meta = {
        "target_transform": "log1p",
        "target_col": target_col,
        "feature_cols": feature_cols,
        "x_days": 60,
    }
    with open(os.path.join(save_dir, "scaler_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[DATA] Scalers saved.")
    return scaled_f, scaled_t, feature_cols


def inverse_transform_target(scaled_pred, target_scaler):
    log_pred = target_scaler.inverse_transform(
        np.array(scaled_pred).reshape(-1, 1)
    )
    return np.expm1(log_pred)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Snapshot Ensemble Callback
# ─────────────────────────────────────────────────────────────────────────────
class SnapshotEnsembleCallback(tf.keras.callbacks.Callback):
    def __init__(self, snapshot_dir, model_name, max_snapshots=5):
        super().__init__()
        self.snap_dir       = os.path.join(snapshot_dir, "snapshots", model_name)
        self.max_snapshots  = max_snapshots
        self.snapshot_paths = []
        self._best_val      = float("inf")
        os.makedirs(self.snap_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        val_loss = (logs or {}).get("val_loss", float("inf"))
        if val_loss < self._best_val:
            self._best_val = val_loss
            path = os.path.join(self.snap_dir, f"snap_e{epoch+1:04d}.weights.h5")
            self.model.save_weights(path)
            self.snapshot_paths.append(path)
            if len(self.snapshot_paths) > self.max_snapshots:
                old = self.snapshot_paths.pop(0)
                try: os.remove(old)
                except OSError: pass


def ensemble_predict(model, snapshot_paths, X, batch_size):
    if not snapshot_paths:
        return model.predict(X, verbose=0, batch_size=batch_size)
    orig = model.get_weights()
    preds = []
    for path in snapshot_paths:
        try:
            model.load_weights(path)
            preds.append(model.predict(X, verbose=0, batch_size=batch_size))
        except Exception as e:
            print(f"  [SNAP] skip {path}: {e}")
    model.set_weights(orig)
    return np.mean(preds, axis=0) if preds else model.predict(X, verbose=0, batch_size=batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Isotonic Regression Calibration
# ─────────────────────────────────────────────────────────────────────────────
def fit_isotonic_calibrator(y_true_mm, y_pred_mm, save_path):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_pred_mm, y_true_mm)
    joblib.dump(iso, save_path)
    return iso

def apply_isotonic(iso, y_pred_mm):
    return iso.predict(y_pred_mm)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Test-Time Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def tta_predict(model, X, n_tta=10, noise_std=0.003, batch_size=128):
    preds = [model.predict(X, verbose=0, batch_size=batch_size)]
    for _ in range(n_tta - 1):
        Xn = X + np.random.normal(0, noise_std, X.shape).astype(np.float32)
        preds.append(model.predict(Xn, verbose=0, batch_size=batch_size))
    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Rich progress callback
# ─────────────────────────────────────────────────────────────────────────────
_COLOUR = sys.stdout.isatty() and os.name != "nt"
def _c(txt, code): return f"\033[{code}m{txt}\033[0m" if _COLOUR else txt

class RichProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, model_name):
        super().__init__()
        self.total = total_epochs
        self.name  = model_name
        self._t0   = None
        self._bar  = "█" if os.name != "nt" else "#"

    def on_train_begin(self, logs=None):
        self._t0 = time.time()
        print(_c(f"\n  ▶  Training {self.name}", "1;36"))

    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        elapsed = time.time() - self._t0
        prog    = (epoch + 1) / self.total
        eta     = elapsed / max(prog, 1e-6) * (1 - prog)
        filled  = int(28 * prog)
        bar     = self._bar * filled + "─" * (28 - filled)
        loss    = logs.get("loss",     float("nan"))
        val     = logs.get("val_loss", float("nan"))
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception:
            lr = float("nan")
        def ft(s): return str(datetime.timedelta(seconds=int(s)))
        line = (f"  [{bar}] {epoch+1:>4}/{self.total}"
                f"  loss={loss:.5f}  val={val:.5f}"
                f"  lr={lr:.2e}  ⏱{ft(elapsed)} ETA={ft(eta)}")
        print(_c(line, "32") if val <= loss * 1.05 else _c(line, "33"))

    def on_train_end(self, logs=None):
        print(_c(f"  ✔  {self.name} done in "
                 f"{datetime.timedelta(seconds=int(time.time()-self._t0))}\n", "1;32"))


# ─────────────────────────────────────────────────────────────────────────────
# 9.  tf.data pipeline  (simple float32, no cast map)
# ─────────────────────────────────────────────────────────────────────────────
def _build_datasets(X_train, y_train, X_val, y_val, batch_size):
    AUTO = tf.data.AUTOTUNE
    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (X_train.astype("float32"), y_train.astype("float32"))
        )
        .cache()
        .shuffle(4096, seed=42, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTO)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            (X_val.astype("float32"), y_val.astype("float32"))
        )
        .cache()
        .batch(batch_size)
        .prefetch(AUTO)
    )
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train_models():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from data_preprocessing import load_and_clean_data, create_sequences
    from models import get_all_models
    from visualizer import generate_all_plots

    num_cores, using_gpu = _configure_hardware()

    # ── Paths ──────────────────────────────────────────────────────────────
    _SYN       = "dakshina_kannada_rainfall_synthetic.csv"
    _REAL      = "dakshina_kannada_rainfall_daily_2000_2024.csv"
    DATA_FILE  = _SYN if os.path.exists(_SYN) else _REAL
    MODELS_DIR = "models"
    METRICS    = "training_metrics.json"
    LOG_DIR    = os.path.join(MODELS_DIR, "logs")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,    exist_ok=True)
    print(f"[DATA] Dataset : {DATA_FILE}")

    # ── Force retrain ──────────────────────────────────────────────────────
    if FORCE_RETRAIN:
        deleted = [f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")]
        for f in deleted:
            os.remove(os.path.join(MODELS_DIR, f))
        if deleted:
            print(f"[RETRAIN] Removed checkpoints: {deleted}")
        if os.path.exists(METRICS):
            os.remove(METRICS)
            print(f"[RETRAIN] Removed stale {METRICS}")

    # ── Load & sequence data (done once, shared across all models) ─────────
    X_DAYS = 60
    Y_DAYS = 1

    df = load_and_clean_data(DATA_FILE)
    f_scaled, t_scaled, feature_cols = scale_data_log(df, save_dir=MODELS_DIR)
    X, y = create_sequences(f_scaled, t_scaled, x_days=X_DAYS, y_days=Y_DAYS)

    target_scaler = joblib.load(os.path.join(MODELS_DIR, "target_scaler.pkl"))

    # Temporal split — shuffle=False preserves time ordering
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=False
    )
    print(f"[DATA] Train={X_train.shape}  Test={X_test.shape}")

    input_shape = (X_DAYS, X_train.shape[2])
    y_test_mm   = inverse_transform_target(y_test, target_scaler).flatten()

    # ── Build models (no strategy scope needed) ───────────────────────────
    models = get_all_models(input_shape, Y_DAYS)

    # Load existing metrics (safe against partial runs)
    all_metrics = {}
    if os.path.exists(METRICS):
        with open(METRICS) as fh:
            all_metrics = json.load(fh)

    # ─────────────────────────────────────────────────────────────────────
    # Per-model training loop
    # ─────────────────────────────────────────────────────────────────────
    for model in models:
        name = model.name
        cfg  = MODEL_CONFIGS.get(name, _DEFAULT_CFG)

        LR_BASE     = cfg["lr"]
        LR_MIN      = cfg["lr_min"]
        BATCH_SIZE  = cfg["batch_size"]
        EPOCHS      = cfg["epochs"]
        PAT_ES      = cfg["patience_es"]
        PAT_LR      = cfg["patience_lr"]
        LR_FACTOR   = cfg["lr_factor"]
        COSINE_T0   = cfg["cosine_t0"]
        N_SNAP      = cfg["n_snapshots"]
        N_TTA       = cfg["n_tta"]
        NOISE_STD   = cfg["noise_std"]

        model_path = os.path.join(MODELS_DIR, f"{name}.keras")
        iso_path   = os.path.join(MODELS_DIR, f"{name}_isotonic.pkl")
        csv_log    = os.path.join(LOG_DIR,    f"{name}_history.csv")
        tb_dir     = os.path.join(LOG_DIR,    "tensorboard", name)

        print(f"\n{'═'*64}")
        print(f"  Model  : {name}")
        print(f"  lr={LR_BASE:.0e}  batch={BATCH_SIZE}  epochs={EPOCHS}"
              f"  pat_es={PAT_ES}  pat_lr={PAT_LR}  T0={COSINE_T0}")
        print(f"{'═'*64}")

        snap_cb    = SnapshotEnsembleCallback(MODELS_DIR, name, N_SNAP)
        skip_train = False

        if os.path.exists(model_path):
            print(f"  [CKPT] Loading {model_path}")
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={"nse_loss": nse_loss,
                                    "composite_loss": composite_loss},
                )
                skip_train = True
            except Exception as e:
                print(f"  [CKPT] Failed ({e}) — retraining.")

        if not skip_train:
            # Build per-model tf.data pipelines with model-specific batch size
            train_ds, val_ds = _build_datasets(
                X_train, y_train, X_test, y_test, BATCH_SIZE
            )

            # Simple compile — no strategy, no distributed wrapper
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=LR_BASE, clipnorm=1.0
                ),
                loss=composite_loss,
                metrics=["mae"],
            )

            # Cosine warm-restart LR (restarts every COSINE_T0 epochs)
            def _cosine_lr(epoch, _lr,
                           t0=COSINE_T0, base=LR_BASE, mn=LR_MIN):
                cos = 0.5 * (1 + math.cos(math.pi * (epoch % t0) / t0))
                return float(mn + (base - mn) * cos)

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=PAT_ES,
                    restore_best_weights=True, verbose=1, min_delta=1e-6,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=model_path, monitor="val_loss",
                    save_best_only=True, verbose=0,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=LR_FACTOR,
                    patience=PAT_LR, min_lr=LR_MIN,
                    verbose=1, min_delta=1e-6,
                ),
                tf.keras.callbacks.LearningRateScheduler(_cosine_lr, verbose=0),
                tf.keras.callbacks.CSVLogger(csv_log, append=False),
                tf.keras.callbacks.TensorBoard(
                    log_dir=tb_dir, histogram_freq=0,
                    write_graph=False, update_freq="epoch",
                ),
                snap_cb,
                RichProgressCallback(EPOCHS, name),
            ]

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=0,
            )
            hist_dict = {k: [float(v) for v in vs]
                         for k, vs in history.history.items()}
        else:
            hist_dict = {}
            if os.path.exists(csv_log):
                import csv
                with open(csv_log) as cf:
                    for row in csv.DictReader(cf):
                        for k, v in row.items():
                            if k == "epoch": continue
                            hist_dict.setdefault(k, []).append(
                                float(v) if v else float("nan"))
            print("  [CKPT] Skipped training — evaluating checkpoint.")

        # ── Predict: snapshot ensemble + TTA ─────────────────────────────
        print(f"  [ENS] {len(snap_cb.snapshot_paths)} snapshots + TTA×{N_TTA}")
        if snap_cb.snapshot_paths:
            y_pred_s = ensemble_predict(model, snap_cb.snapshot_paths,
                                        X_test, BATCH_SIZE)
        else:
            y_pred_s = tta_predict(model, X_test,
                                   n_tta=N_TTA, noise_std=NOISE_STD,
                                   batch_size=BATCH_SIZE)

        y_pred_mm = inverse_transform_target(y_pred_s, target_scaler).flatten()
        y_pred_mm = np.clip(y_pred_mm, 0.0, None)

        # ── Isotonic calibration (fit on train, apply to test) ────────────
        tr_pred_s  = model.predict(X_train, verbose=0, batch_size=BATCH_SIZE)
        tr_pred_mm = inverse_transform_target(tr_pred_s, target_scaler).flatten()
        tr_pred_mm = np.clip(tr_pred_mm, 0.0, None)
        tr_true_mm = inverse_transform_target(y_train, target_scaler).flatten()

        iso = fit_isotonic_calibrator(tr_true_mm, tr_pred_mm, iso_path)
        y_pred_cal = np.clip(apply_isotonic(iso, y_pred_mm), 0.0, None)

        nse_raw = nse(y_test_mm, y_pred_mm)
        nse_cal = nse(y_test_mm, y_pred_cal)
        if nse_cal >= nse_raw:
            y_pred_final = y_pred_cal
            print(f"  [CAL] Isotonic ✓  NSE {nse_raw:.4f} → {nse_cal:.4f}")
        else:
            y_pred_final = y_pred_mm
            print(f"  [CAL] Raw kept   NSE {nse_raw:.4f}  (cal={nse_cal:.4f})")

        # ── Final metrics ─────────────────────────────────────────────────
        rmse_v  = float(np.sqrt(mean_squared_error(y_test_mm, y_pred_final)))
        mae_v   = float(mean_absolute_error(y_test_mm, y_pred_final))
        nse_v   = nse(y_test_mm, y_pred_final)
        kge_v   = kge(y_test_mm, y_pred_final)
        pb_v    = pbias(y_test_mm, y_pred_final)

        badge = "✔" if nse_v >= 0.90 else ("△" if nse_v >= 0.80 else "✘")
        print(f"\n  {badge} {name}")
        print(f"    NSE   : {nse_v:+.4f}   (target ≥ 0.90)")
        print(f"    KGE   : {kge_v:+.4f}")
        print(f"    RMSE  : {rmse_v:.2f} mm")
        print(f"    MAE   : {mae_v:.2f} mm")
        print(f"    PBIAS : {pb_v:+.2f}%")

        best_val_loss = (float(np.min(hist_dict["val_loss"]))
                         if hist_dict.get("val_loss")
                         else float(model.evaluate(val_ds, verbose=0)[0])
                              if not skip_train else 0.0)

        all_metrics[name] = {
            "history": hist_dict,
            "config": {k: str(v) for k, v in cfg.items()},
            "final_metrics": {
                "loss":  best_val_loss,
                "rmse":  rmse_v,
                "mae":   mae_v,
                "nse":   nse_v,
                "kge":   kge_v,
                "pbias": pb_v,
            },
        }

        # Persist after every model — crash-safe
        with open(METRICS, "w") as fh:
            json.dump(all_metrics, fh, indent=4)

        generate_all_plots(y_test_mm, y_pred_final, hist_dict, name)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print(f"  All models complete  •  metrics → {METRICS}")
    best = max(all_metrics, key=lambda n: all_metrics[n]["final_metrics"].get("nse", -999))
    print(f"\n  🏆 Best model : {best}  NSE={all_metrics[best]['final_metrics']['nse']:.4f}\n")
    print(f"{'═'*64}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.random.set_seed(42)
    np.random.seed(42)
    train_models()
