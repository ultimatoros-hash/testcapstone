import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import math
import os
import json
import pandas as pd
from PIL import Image
from io import BytesIO
from streamlit_folium import st_folium
import folium
import cv2
from tensorflow.keras import layers

# =============================================================================
# 1. CONFIGURATION
# ============================================================================
IMG_SIZE = (128, 128)
PLOT_DIR = "data/plots"
MODEL_PATH = "models/satellite_mobilenet.h5" 
BACKUP_PATH = "models/satellite_custom_cnn.h5"

st.set_page_config(
    page_title="Eco-Vision: Satellite Analysis", 
    layout="wide", 
    page_icon="🛰️",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. CUSTOM LAYERS (CRITICAL FIX FOR LOADING)
# =============================================================================

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

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

@st.cache_resource
def load_resources():
    """Loads model and class names safely with custom object mapping."""
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else BACKUP_PATH
    if not os.path.exists(path): return None, None, None
    
    try:
        model = tf.keras.models.load_model(path, custom_objects={
            'ColorJitter': ColorJitter,
            'GentleCloudAugmentation': GentleCloudAugmentation,
            'SensorThermalNoise': SensorThermalNoise,
            'AtmosphericScattering': AtmosphericScattering
        })
        
        if os.path.exists('models/classes.txt'):
            with open('models/classes.txt', 'r') as f:
                classes = f.read().splitlines()
        else:
            classes = ['desert', 'forest', 'snow', 'urban', 'water']
            
        return model, classes, path
    except Exception as e:
        st.error(f"❌ Critical Error loading model: {e}")
        return None, None, None

model, class_names, loaded_path = load_resources()

def get_gradcam(img_array, model, last_conv_layer_name=None):
    """Generates Grad-CAM heatmap to visualize model attention."""
    try:
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break
        
        if not last_conv_layer_name: return None

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except:
        return None

def predict_with_analytics(image):
    """Runs prediction pipeline with uncertainty and Grad-CAM."""
    if model is None: return "Error", 0, 0, np.array(image)
    
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) 
    img_batch = np.expand_dims(img_array, 0)
    
    preds = model.predict(img_batch, verbose=0)
    idx = np.argmax(preds)
    
    if idx >= len(class_names):
        return "Unknown", 0, 0, img_array
        
    label = class_names[idx]
    conf = preds[0][idx] * 100
    
    probs = preds[0]
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    uncertainty = min(100, entropy * 20) 
    
    heatmap = get_gradcam(img_batch, model)
    overlay = img_array
    if heatmap is not None:
        try:
            heatmap = np.uint8(255 * heatmap)
            jet = cv2.applyColorMap(cv2.resize(heatmap, (128, 128)), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(np.array(img_resized), 0.6, jet, 0.4, 0)
        except: pass

    return label, conf, uncertainty, overlay

def lat_lon_to_tile(lat, lon, zoom):
    """Converts GPS coordinates to Slippy Map tile indices."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

# =============================================================================
# 4. MAIN APPLICATION
# =============================================================================

st.sidebar.title("🛰️ Eco-Vision")
st.sidebar.caption(f"Model: {os.path.basename(loaded_path) if loaded_path else 'None'}")

page = st.sidebar.radio("Navigation", [
    "1. Live Map Scanner",
    "2. Overview",
    "3. Data & Bias",
    "4. Scientific Validation",
    "5. Live Inference (Upload)",
    "6. Time Machine"
])

# --- PAGE 1: LIVE MAP SCANNER ---
if page == "1. Live Map Scanner":
    st.title("🗺️ Satellite Map Scanner")
    st.markdown("Click anywhere on the world map to classify the environment in real-time.")

    m = folium.Map(location=[20, 0], zoom_start=2)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Esri Satellite'
    ).add_to(m)
    
    out = st_folium(m, height=500, width="100%")

    if out['last_clicked']:
        lat = out['last_clicked']['lat']
        lon = out['last_clicked']['lng']
        
        zoom = 15
        x, y = lat_lon_to_tile(lat, lon, zoom)
        url = f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers, timeout=5)
            
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content)).convert("RGB")
                label, conf, unc, heatmap = predict_with_analytics(img)
                
                st.divider()
                c1, c2, c3 = st.columns([1, 1, 2])
                
                with c1:
                    st.write("**Satellite Tile**")
                    st.image(img, use_container_width=True)
                    st.caption(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
                
                with c2:
                    st.write("**Attention Map**")
                    st.image(heatmap, use_container_width=True, caption="Grad-CAM Heatmap")
                
                with c3:
                    st.write("**Classification**")
                    colors = {"urban":"#ef4444", "forest":"#16a34a", "water":"#3b82f6", "desert":"#f97316", "snow":"#d1d5db"}
                    color = colors.get(label, "#333")
                    
                    st.markdown(f"<h1 style='color:{color}'>{label.upper()}</h1>", unsafe_allow_html=True)
                    st.metric("Confidence", f"{conf:.1f}%")
                    st.metric("Uncertainty", f"{unc:.2f}%")
            else:
                st.warning("⚠️ High-resolution imagery restricted or unavailable for this ocean/remote region.")
        except Exception as e:
            st.error(f"Network/Inference Error: {e}")

# --- PAGE 2: OVERVIEW ---
elif page == "2. Overview":
    st.title("🌍 Eco-Vision Project Overview")
    st.info("Robust Land Cover Classification using Physics-Informed MobileNetV2")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("This research resolves 'Spectral Ambiguity' between classes like Urban and Desert by forcing the model to learn spatial textures.")
        st.markdown("* **Desert** 🏜️ - Low entropy, granular texture\n* **Forest** 🌲 - High VARI index, chlorophyll detection\n* **Water** 🌊 - Low reflectance, smooth surface\n* **Urban** 🏙️ - High spatial frequency, structural edges\n* **Snow** 🏔️ - High albedo, cryospheric texture")
    with c2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/FullMoon2010.jpg/1024px-FullMoon2010.jpg", caption="Planetary Observation")

# --- PAGE 3: DATA & BIAS ---
elif page == "3. Data & Bias":
    st.title("📊 Data Engineering & Audit")
    t1, t2 = st.tabs(["Class Balance", "Bias Audit"])
    with t1:
        if os.path.exists(f"{PLOT_DIR}/class_balance.png"):
            st.image(f"{PLOT_DIR}/class_balance.png", caption="Training Dataset Distribution")
    with t2:
        if os.path.exists(f"{PLOT_DIR}/bias_report.txt"):
            with open(f"{PLOT_DIR}/bias_report.txt") as f: st.code(f.read())
        if os.path.exists(f"{PLOT_DIR}/bias_geographic_density.png"):
            st.image(f"{PLOT_DIR}/bias_geographic_density.png", caption="Latitudinal Sampling Density")

# --- PAGE 4: SCIENTIFIC VALIDATION ---
elif page == "4. Scientific Validation":
    st.title("🔬 Scientific Validation Suite")
    t1, t2, t3 = st.tabs(["Performance", "Robustness", "Physics"])
    
    with t1:
        c1, c2 = st.columns(2)
        if os.path.exists(f"{PLOT_DIR}/confusion_matrix.png"):
            c1.image(f"{PLOT_DIR}/confusion_matrix.png", caption="Normalized Confusion Matrix")
        if os.path.exists(f"{PLOT_DIR}/training_curves_transfer.png"):
            c2.image(f"{PLOT_DIR}/training_curves_transfer.png", caption="Model Convergence Phase 1 & 2")

    with t2:
        if os.path.exists(f"{PLOT_DIR}/robustness_curve.png"):
            st.image(f"{PLOT_DIR}/robustness_curve.png", caption="Protocol D: Robustness to Noise/Blur")

    with t3:
        if os.path.exists(f"{PLOT_DIR}/physical_metrics.png"):
            st.image(f"{PLOT_DIR}/physical_metrics.png", caption="Protocol F: Biophysical Consistency (Entropy/VARI)")
        if os.path.exists(f"{PLOT_DIR}/lime_explanation.png"):
            st.image(f"{PLOT_DIR}/lime_explanation.png", caption="XAI: LIME Super-pixel Attribution")

# --- PAGE 5: UPLOAD INFERENCE ---
elif page == "5. Live Inference (Upload)":
    st.title("🧠 Direct Inference Engine")
    f = st.file_uploader("Upload a Satellite Image", type=["jpg", "png", "jpeg"])
    if f:
        img = Image.open(f).convert("RGB")
        label, conf, unc, heatmap = predict_with_analytics(img)
        
        c1, c2 = st.columns(2)
        c1.image(img, caption="Source Input", use_container_width=True)
        c2.image(heatmap, caption="AI Feature Attention", use_container_width=True)
        
        st.divider()
        st.success(f"**Classification:** {label.upper()} ({conf:.1f}% Confidence)")
        st.caption(f"Entropy-based Uncertainty: {unc:.2f}%")

# --- PAGE 6: TIME MACHINE ---
elif page == "6. Time Machine":
    st.title("⏳ Temporal Change Detection")
    st.markdown("Upload two images of the same location at different times to detect land-cover transitions.")
    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Time T1 (Past)", key="t1")
    f2 = c2.file_uploader("Time T2 (Present)", key="t2")
    
    if f1 and f2:
        img1 = Image.open(f1).convert("RGB")
        img2 = Image.open(f2).convert("RGB")
        l1 = predict_with_analytics(img1)[0]
        l2 = predict_with_analytics(img2)[0]
        
        col1, mid, col2 = st.columns([1, 0.2, 1])
        with col1: st.image(img1, caption=f"T1: {l1.upper()}", use_container_width=True)
        with mid: st.markdown("## ➝")
        with col2: st.image(img2, caption=f"T2: {l2.upper()}", use_container_width=True)
        
        if l1 != l2:
            st.error(f"🚨 TRANSITION DETECTED: {l1.upper()} ➝ {l2.upper()}")
        else:
            st.success("✅ STABLE: No Land-Cover Transition Detected")