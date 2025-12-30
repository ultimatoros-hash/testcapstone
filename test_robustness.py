import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras import layers

# --- CONFIG ---
MODEL_PATH = "models/satellite_mobilenet.h5"
DATA_DIR = "data/raw/images"
PLOT_DIR = "data/plots"
IMG_SIZE = (128, 128)

# --- CUSTOM LAYERS (REQUIRED FOR LOADING) ---
@tf.keras.utils.register_keras_serializable()
class ColorJitter(layers.Layer):
    def call(self, inputs, training=None): return inputs

@tf.keras.utils.register_keras_serializable()
class GentleCloudAugmentation(layers.Layer):
    def call(self, inputs, training=None): return inputs

@tf.keras.utils.register_keras_serializable()
class SensorThermalNoise(layers.Layer):
    def call(self, inputs, training=None): return inputs

@tf.keras.utils.register_keras_serializable()
class AtmosphericScattering(layers.Layer):
    def call(self, inputs, training=None): return inputs

def add_noise(image_tensor, severity):
    """Adds Gaussian Noise (simulating sensor thermal noise)"""
    noise = tf.random.normal(shape=tf.shape(image_tensor), mean=0.0, stddev=severity)
    # image_tensor is [0, 255] coming from dataset
    return tf.clip_by_value(image_tensor + noise * 255.0, 0.0, 255.0)

def add_blur(image_tensor, severity):
    """Adds Gaussian Blur (simulating atmospheric scattering)"""
    if severity == 0: return image_tensor
    
    imgs_np = image_tensor.numpy().astype('uint8')
    blurred_batch = []
    # Kernel size must be odd. severity 0.1 -> 3, 0.5 -> 5+
    ksize = int(severity * 10) | 1 
    
    for img in imgs_np:
        b = cv2.GaussianBlur(img, (ksize, ksize), 0)
        blurred_batch.append(b)
        
    return tf.convert_to_tensor(np.array(blurred_batch), dtype=tf.float32)

def run_stress_test():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return
    
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)

    print("🌪️ Running Protocol D: Robustness Stress Test...")
    
    # Load model with custom scope including ColorJitter
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
        'ColorJitter': ColorJitter,
        'GentleCloudAugmentation': GentleCloudAugmentation,
        'SensorThermalNoise': SensorThermalNoise,
        'AtmosphericScattering': AtmosphericScattering
    })
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=32, shuffle=False
    )
    
    levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    acc_noise = []
    acc_blur = []
    
    # --- TEST 1: NOISE ---
    print("   Testing Sensor Noise Resilience...")
    for lvl in levels:
        correct, total = 0, 0
        for imgs, lbls in val_ds:
            noisy = add_noise(imgs, lvl)
            preds = model.predict(noisy, verbose=0)
            correct += np.sum(np.argmax(preds, axis=1) == lbls.numpy())
            total += len(lbls)
        acc = correct / total
        acc_noise.append(acc)
        print(f"     Sigma {lvl}: {acc:.2%}")

    # --- TEST 2: BLUR ---
    print("   Testing Atmospheric Blur Resilience...")
    for lvl in levels:
        correct, total = 0, 0
        for imgs, lbls in val_ds:
            blurred = add_blur(imgs, lvl)
            preds = model.predict(blurred, verbose=0)
            correct += np.sum(np.argmax(preds, axis=1) == lbls.numpy())
            total += len(lbls)
        acc = correct / total
        acc_blur.append(acc)
        print(f"     Factor {lvl}: {acc:.2%}")

    # --- PLOTTING --
    plt.figure(figsize=(10, 6))
    plt.plot(levels, acc_noise, 'o-', linewidth=2, color='#e74c3c', label='Sensor Noise')
    plt.plot(levels, acc_blur, 's-', linewidth=2, color='#3498db', label='Atmospheric Blur')
    
    plt.axhline(0.8, color='gray', linestyle='--', label='Robustness Threshold (80%)')
    plt.title("Figure 5.3: Model Robustness under Degradation")
    plt.xlabel("Severity (Sigma / Blur Factor)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_DIR}/robustness_curve.png")
    plt.close()
    print("✅ Robustness Plot Saved.")

if __name__ == "__main__":
    run_stress_test()