import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.calibration import calibration_curve
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras import layers

# --- CONFIGURATION ---
DATA_DIR = "data/raw/images"
PLOT_DIR = "data/plots"
MODEL_PATH = "models/satellite_mobilenet.h5"
CLASSES_FILE = "models/classes.txt"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Consistent colors
PALETTE = {
    "snow": "#d1d5db", "water": "#3b82f6", "forest": "#16a34a", 
    "urban": "#ef4444", "desert": "#f97316"
}

# --- CUSTOM LAYERS (Pass-through for loading) ---
@tf.keras.utils.register_keras_serializable()
class GentleCloudAugmentation(layers.Layer):
    def call(self, inputs, training=None): return inputs

@tf.keras.utils.register_keras_serializable()
class SensorThermalNoise(layers.Layer):
    def call(self, inputs, training=None): return inputs

@tf.keras.utils.register_keras_serializable()
class AtmosphericScattering(layers.Layer):
    def call(self, inputs, training=None): return inputs

def load_resources():
    """Loads model and ensures class names align with training."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model missing: {MODEL_PATH}")
        return None, None

    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
            'GentleCloudAugmentation': GentleCloudAugmentation,
            'SensorThermalNoise': SensorThermalNoise,
            'AtmosphericScattering': AtmosphericScattering
        })
        
        # Load the EXACT classes used during training
        if os.path.exists(CLASSES_FILE):
            with open(CLASSES_FILE, 'r') as f:
                model_classes = f.read().splitlines()
            print(f"✅ Loaded {len(model_classes)} classes from metadata.")
        else:
            print("⚠️ Classes file missing. Inferring from folder structure (risk of mismatch).")
            model_classes = None
            
        return model, model_classes
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def load_validation_data(expected_classes=None):
    """Loads validation data and verifies class alignment."""
    print("⏳ Loading Validation Data...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    
    ds_classes = val_ds.class_names
    
    # Critical Check: Ensure Dataset classes match Model classes
    if expected_classes and ds_classes != expected_classes:
        print(f"⚠️ CRITICAL WARNING: Class Mismatch!")
        print(f"   Model expects: {expected_classes}")
        print(f"   Dataset found: {ds_classes}")
        print("   -> Predictions will be wrong. Check folder names or sorting.")
    
    images, labels = [], []
    for imgs, lbls in val_ds:
        images.append(imgs.numpy())
        labels.append(lbls.numpy())
    
    return np.concatenate(images), np.concatenate(labels), ds_classes

def run_analytics():
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    
    # 1. Load Model & Metadata
    model, model_classes = load_resources()
    if not model: return

    # 2. Load Data
    # Use model_classes for validation if available
    images, labels, ds_classes = load_validation_data(model_classes)
    class_names = model_classes if model_classes else ds_classes

    # 3. Inference
    print("🧠 Running Inference...")
    probs = model.predict(images, verbose=1)
    preds = np.argmax(probs, axis=1).flatten() # Flatten to ensure shape (N,)
    labels = labels.flatten() # Flatten to ensure shape (N,)
    confidences = np.max(probs, axis=1)

    # ---------------------------------------------------------
    # PLOT: WORST FAILURES (Fixed Logic)
    # ---------------------------------------------------------
    print("❌ Generating Figure 6.2: Top-5 Confident Failures...")
    
    # Identify errors
    incorrect_mask = (preds != labels)
    incorrect_indices = np.where(incorrect_mask)[0]
    
    if len(incorrect_indices) > 0:
        # Get confidences ONLY for the errors
        error_confs = confidences[incorrect_indices]
        
        # Sort errors by confidence (High -> Low)
        # We want the errors where the model was MOST confident (but wrong)
        sorted_indices_local = np.argsort(error_confs)[::-1][:5]
        top_failure_indices = incorrect_indices[sorted_indices_local]
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        if len(top_failure_indices) < 5: 
            # Handle case with <5 errors (rare but possible)
            axes = axes[:len(top_failure_indices)]
            
        for i, idx in enumerate(top_failure_indices):
            # Safe access to image
            img = images[idx].astype("uint8")
            
            # Get integer IDs
            true_id = labels[idx]
            pred_id = preds[idx]
            
            # Lookup string names safely
            true_lbl = class_names[true_id] if true_id < len(class_names) else "Unknown"
            pred_lbl = class_names[pred_id] if pred_id < len(class_names) else "Unknown"
            
            conf = probs[idx, pred_id] * 100
            
            # Plot
            ax = axes[i] if len(top_failure_indices) > 1 else axes
            ax.imshow(img)
            ax.set_title(f"Pred: {pred_lbl} ({conf:.1f}%)\nTrue: {true_lbl}", 
                         color="#d62728", fontweight="bold", fontsize=10)
            ax.axis("off")
        
        plt.suptitle("Figure 6.2: Top-5 'Most Confident' Failures", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/worst_failures.png")
        plt.close()
    else:
        print("   ✨ Amazing! 0 errors found in validation set.")

    # ---------------------------------------------------------
    # PLOT: CALIBRATION CURVE
    # ---------------------------------------------------------
    print("📉 Generating Figure 6.1: Reliability...")
    is_correct = (preds == labels).astype(int)
    prob_true, prob_pred = calibration_curve(is_correct, confidences, n_bins=10, strategy='uniform')
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(prob_pred, prob_true, "s-", color="#2ecc71", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_DIR}/calibration_curve.png")
    plt.close()

    # ---------------------------------------------------------
    # PLOT: 3D COLOR SPACE
    # ---------------------------------------------------------
    print("🌈 Generating Figure 6.4: 3D Scatter...")
    subset_idx = np.random.choice(len(images), min(500, len(images)), replace=False)
    sub_imgs = images[subset_idx]
    sub_lbls = labels[subset_idx]
    
    r = sub_imgs[:, :, :, 0].mean(axis=(1, 2))
    g = sub_imgs[:, :, :, 1].mean(axis=(1, 2))
    b = sub_imgs[:, :, :, 2].mean(axis=(1, 2))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, name in enumerate(class_names):
        if name not in PALETTE: continue
        mask = (sub_lbls == i)
        ax.scatter(r[mask], g[mask], b[mask], c=PALETTE[name], label=name, alpha=0.6)
    
    ax.set_xlabel('Red'); ax.set_ylabel('Green'); ax.set_zlabel('Blue')
    plt.legend()
    plt.savefig(f"{PLOT_DIR}/3d_color_space.png")
    plt.close()

    print("✅ Advanced Analytics Complete.")

if __name__ == "__main__":
    run_analytics()