import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.decomposition import PCA
import umap
from scipy import fftpack
from skimage.measure import shannon_entropy
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras import layers

# --- CONFIG --
DATA_DIR = "data/raw/images"
PLOT_DIR = "data/plots"
IMG_SIZE = (128, 128)
MODEL_PATH = "models/satellite_mobilenet.h5"

PALETTE = {
    "snow": "#d1d5db", "water": "#3b82f6", 
    "forest": "#16a34a", "urban": "#ef4444", "desert": "#f97316"
}

# --- CUSTOM LAYERS ---
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

def load_sample_data(num_samples=300):
    print("Loading sample data for biophysics...")
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=IMG_SIZE, batch_size=32, shuffle=True, seed=42
    )
    images, labels = [], []
    class_names = ds.class_names
    for batch_img, batch_lbl in ds.take(num_samples // 32 + 1):
        images.extend(batch_img.numpy())
        labels.extend(batch_lbl.numpy())
    return np.array(images[:num_samples]), np.array(labels[:num_samples]), class_names

def calculate_vari(image):
    """
    Visible Atmospherically Resistant Index (VARI)
    
    """
    img = image.astype('float32')
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    
    numerator = G - R
    denominator = G + R - B
    vari_map = numerator / (denominator + 1e-6)
    return np.median(np.clip(vari_map, -1.0, 1.0))

def run_analytics():
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    
    imgs, lbls, names = load_sample_data(300)
    
    # --- PART 1: BIOPHYSICAL METRICS ---
    print("📊 Generating Figure 4.1: Biophysical Metrics...")
    metrics = {'entropy': [], 'vari': [], 'label': []}
    
    for img, lbl_idx in zip(imgs, lbls):
        name = names[lbl_idx]
        if name not in PALETTE: continue
        
        gray = tf.image.rgb_to_grayscale(img).numpy().squeeze().astype('uint8')
        metrics['entropy'].append(shannon_entropy(gray))
        
        metrics['vari'].append(calculate_vari(img))
        metrics['label'].append(name)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(x='label', y='entropy', data=metrics, ax=ax1, palette=PALETTE)
    ax1.set_title("Figure 4.1a: Spatial Entropy (Texture Complexity)")
    
    sns.boxplot(x='label', y='vari', data=metrics, ax=ax2, palette=PALETTE)
    ax2.set_title("Figure 4.1b: VARI Index (Vegetation Health)")
    ax2.axhline(0.2, color='green', linestyle='--', alpha=0.5, label='Veg Threshold')
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/physical_metrics.png")
    
    # --- PART 2: FFT ANALYSIS ---
    # 
    print("🌊 Generating Figure 4.2: FFT Frequency Analysis...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, target in enumerate(PALETTE.keys()):
        if target not in names: continue
        idx = np.where(lbls == names.index(target))[0]
        if len(idx) > 0:
            img_gray = np.mean(imgs[idx[0]], axis=2)
            f = fftpack.fft2(img_gray)
            fshift = fftpack.fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 1e-6)
            axes[i].imshow(magnitude, cmap='inferno')
            axes[i].set_title(target.title())
            axes[i].axis('off')
    plt.savefig(f"{PLOT_DIR}/fft_spectrum_analysis.png")
    
    # --- PART 3: LIME EXPLANATION ---
    # 
    print("🍋 Generating Figure 4.4: LIME Explanation...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
            'ColorJitter': ColorJitter,
            'GentleCloudAugmentation': GentleCloudAugmentation, 
            'SensorThermalNoise': SensorThermalNoise, 
            'AtmosphericScattering': AtmosphericScattering
        })
        
        explainer = lime_image.LimeImageExplainer()
        exp = explainer.explain_instance(imgs[0].astype('double'), model.predict, top_labels=1, hide_color=0, num_samples=100)
        temp, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(mark_boundaries(temp/255, mask))
        plt.title(f"LIME XAI: {names[lbls[0]]}")
        plt.axis('off')
        plt.savefig(f"{PLOT_DIR}/lime_explanation.png")
    except Exception as e: 
        print(f"⚠️ LIME Skipped: {e}")

if __name__ == "__main__":
    run_analytics()