import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras import layers

# --- CONFIGURATION ---
DATA_DIR = "data/raw/images"
# IMPORTANT: Evaluates your BEST model (Transfer Learning)
MODEL_PATH = "models/satellite_mobilenet.h5" 
PLOT_DIR = "data/plots"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Official Color Palette
PALETTE = {
    "snow": "#d1d5db",   # Grey/White
    "water": "#3b82f6",  # Blue
    "forest": "#16a34a", # Green
    "urban": "#ef4444",  # Red
    "desert": "#f97316"  # Orange
}

# --- CUSTOM LAYERS (REQUIRED FOR LOADING) ---
class GentleCloudAugmentation(layers.Layer):
    def call(self, inputs, training=None): return inputs

class SensorThermalNoise(layers.Layer):
    def call(self, inputs, training=None): return inputs

class AtmosphericScattering(layers.Layer):
    def call(self, inputs, training=None): return inputs

def load_data_and_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return None, None, None

    print(f"Loading {MODEL_PATH}...")
    try:
        # We must provide the custom layers to load the model, even if they are inactive
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
            'GentleCloudAugmentation': GentleCloudAugmentation,
            'SensorThermalNoise': SensorThermalNoise,
            'AtmosphericScattering': AtmosphericScattering
        })
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    return model, val_ds, val_ds.class_names

def run_evaluation(model, dataset, class_names):
    print("🚀 Generating Confusion Matrix...")
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    
    y_true = []
    y_pred = []
    
    # Run Inference
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0)
    
    print(f"🏆 GLOBAL ACCURACY: {acc*100:.2f}%")
    
    # Save Report
    with open(os.path.join(PLOT_DIR, "metrics_report.txt"), "w") as f:
        f.write(f"GLOBAL_ACCURACY: {acc:.5f}\n") 
        f.write("\n" + report)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='YlGnBu', 
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='gray')
    
    plt.title('Figure 5.1: Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix.png'))
    plt.close()

def plot_tsne(model, dataset, class_names):
    print("🚀 Generating t-SNE Projection (Figure 5.2)...")
    
    # Create Feature Extractor (Remove the last classification layer)
    # We look for the GlobalAveragePooling layer or the Dropout layer
    feature_layer = model.layers[-2].output 
    feature_model = tf.keras.Model(inputs=model.inputs, outputs=feature_layer)
    
    features = []
    labels = []
    
    # Extract features from a subset of validation data
    for img_batch, label_batch in dataset.take(20): # Take 20 batches (~640 images)
        f = feature_model.predict(img_batch, verbose=0)
        features.extend(f)
        labels.extend(label_batch.numpy())
        
    features = np.array(features)
    label_names = [class_names[i] for i in labels]
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    result = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=result[:,0], y=result[:,1], 
        hue=label_names, palette=PALETTE, s=70, alpha=0.8, edgecolor="k"
    )
    plt.title("Figure 5.2: t-SNE Feature Projection")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, 'tsne_clusters.png'))
    plt.close()

if __name__ == "__main__":
    model, val_ds, classes = load_data_and_model()
    if model:
        run_evaluation(model, val_ds, classes)
        plot_tsne(model, val_ds, classes)