import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_visuals(save_dir='frontend/public/plots/'):
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="darkgrid", palette="deep")
    return save_dir

def plot_loss_curve(history, model_name, save_dir):
    if not history or 'loss' not in history or 'val_loss' not in history:
        print(f"  [VIS] Skipping loss curve for {model_name} (missing history data)")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='green', linewidth=2)
    plt.title(f"{model_name} - Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_loss_curve.png"))
    plt.close()

def plot_pred_vs_actual_line(y_true, y_pred, model_name, save_dir, subset_days=60):
    plt.figure(figsize=(12, 5))
    subset_true = y_true[:subset_days]
    subset_pred = y_pred[:subset_days]
    
    plt.plot(subset_true, label='Actual Rainfall', color='blue', linewidth=2)
    plt.plot(subset_pred, label='Predicted Rainfall', color='orange', linestyle='--', linewidth=2)
    plt.title(f"{model_name} - Pred vs Actual Hydrograph ({subset_days} days)")
    plt.xlabel("Days")
    plt.ylabel("Rainfall (mm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_pred_vs_actual_line.png"))
    plt.close()

def plot_pred_vs_actual_scatter(y_true, y_pred, model_name, save_dir):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color='teal')
    
    # y=x line
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
    
    plt.title(f"{model_name} - Fitness Scatter Plot")
    plt.xlabel("Actual Rainfall (mm)")
    plt.ylabel("Predicted Rainfall (mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_pred_vs_actual_scatter.png"))
    plt.close()

def plot_residuals_hist(y_true, y_pred, model_name, save_dir):
    plt.figure(figsize=(10, 6))
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, color='purple', bins=50)
    plt.title(f"{model_name} - Error Residuals Distribution")
    plt.xlabel("Error (Actual - Predicted) (mm)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_residuals_hist.png"))
    plt.close()

def plot_cumulative_rainfall(y_true, y_pred, model_name, save_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(y_true), label='Actual Cumulative', color='blue', linewidth=2)
    plt.plot(np.cumsum(y_pred), label='Predicted Cumulative', color='orange', linestyle='--', linewidth=2)
    plt.title(f"{model_name} - Cumulative Rainfall Mass Curve")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Rainfall (mm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_cumulative_rainfall.png"))
    plt.close()

def generate_all_plots(y_true, y_pred, history, model_name):
    save_dir = setup_visuals()
    plot_loss_curve(history, model_name, save_dir)
    plot_pred_vs_actual_line(y_true, y_pred, model_name, save_dir)
    plot_pred_vs_actual_scatter(y_true, y_pred, model_name, save_dir)
    plot_residuals_hist(y_true, y_pred, model_name, save_dir)
    plot_cumulative_rainfall(y_true, y_pred, model_name, save_dir)
