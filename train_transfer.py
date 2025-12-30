import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, applications, regularizers
import numpy as np
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# --- CONFIG ---
DATA_DIR = "data/raw/images"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS_HEAD = 12
EPOCHS_FINE = 20
PLOT_DIR = "data/plots"
MODEL_SAVE_PATH = "models/satellite_mobilenet.h5"

# =============================================================================
# 1. SCIENTIFIC LAYERS (The "Correction" Layers)
# =============================================================================

class GentleCloudAugmentation(layers.Layer):
    def call(self, inputs, training=None):
        if training is None: training = False
        def apply():
            noise = tf.random.uniform(shape=tf.shape(inputs), minval=0.0, maxval=1.0)
            return tf.clip_by_value(inputs + (tf.cast(noise>0.9, tf.float32)*0.05), 0.0, 1.0)
        return tf.cond(tf.cast(training, tf.bool), apply, lambda: inputs)

class SensorThermalNoise(layers.Layer):
    def call(self, inputs, training=None):
        if training is None: training = False
        def apply(): 
            return tf.clip_by_value(inputs + tf.random.normal(tf.shape(inputs), 0, 0.02), 0, 1)
        return tf.cond(tf.cast(training, tf.bool), apply, lambda: inputs)

class AtmosphericScattering(layers.Layer):
    def call(self, inputs, training=None):
        if training is None: training = False
        def apply():
            should_blur = tf.random.uniform([]) > 0.95 
            def do_blur():
                s = tf.shape(inputs)
                h_new = tf.cast(tf.cast(s[1], tf.float32) * 0.8, tf.int32)
                w_new = tf.cast(tf.cast(s[2], tf.float32) * 0.8, tf.int32)
                return tf.image.resize(tf.image.resize(inputs, [h_new, w_new]), [s[1], s[2]])
            return tf.cond(should_blur, do_blur, lambda: inputs)
        return tf.cond(tf.cast(training, tf.bool), apply, lambda: inputs)

class ColorJitter(layers.Layer):
    """Custom Layer to fix Desert/Forest and Urban/Water confusion"""
    def call(self, inputs, training=None):
        if training is None: training = False
        def apply():
            x = inputs
            x = tf.image.random_saturation(x, 0.7, 1.3)
            x = tf.image.random_contrast(x, 0.8, 1.2)
            x = tf.image.random_brightness(x, 0.1)
            return tf.clip_by_value(x, 0.0, 1.0)
        return tf.cond(tf.cast(training, tf.bool), apply, lambda: inputs)

# =============================================================================
# 2. MODEL ARCHITECTURE
# =============================================================================

def build_transfer_model(num_classes):
    # Explicit input shape + dtype
    inputs = layers.Input(shape=IMG_SIZE + (3,), dtype=tf.float32)
    
    # 1. Standardize
    x = layers.Rescaling(1./255)(inputs)
    
    # 2. Geometric Augmentation
    x = layers.RandomFlip("horizontal_and_vertical")(x)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # 3. Color/Physics Augmentation
    x = ColorJitter()(x)
    x = AtmosphericScattering()(x)
    x = SensorThermalNoise()(x)
    x = GentleCloudAugmentation()(x)

    # 4. Backbone (MobileNetV2)
    x = layers.Rescaling(2.0, offset=-1.0)(x) 
    base_model = applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False 
    x = base_model(x, training=False)
    
    # 5. Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

def plot_history(h1, h2):
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.axvline(x=len(h1.history['accuracy']), color='green', linestyle='--', label='Fine Tune')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.savefig(os.path.join(PLOT_DIR, "training_curves_transfer.png"))
    plt.close()

def process_data(img, label):
    """Explicitly cast types to prevent 'dtype=string' errors."""
    return tf.cast(img, tf.float32), tf.cast(label, tf.int32)

def train():
    print("🚀 ECO-VISION: TRANSFER LEARNING (FIXED)...")
    
    # 1. Load Data with explicit integer labels
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123, 
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123, 
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
    )
    
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    # 2. Compute Weights
    y_train = []
    for _, labels in train_ds.unbatch():
        y_train.append(labels.numpy())
    
    y_train = np.array(y_train)
    weights = dict(enumerate(class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )))

    # 3. Apply Type Casting & Optimization
    train_ds = train_ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    # --- PHASE 1 ---
    print("\n📦 Phase 1: Head Training...")
    model = build_transfer_model(len(class_names))
    model.compile(optimizer=optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, class_weight=weights)

    # --- PHASE 2 ---
    print("\n🔓 Phase 2: Deep Fine-Tuning...")
    base_model = None
    for layer in model.layers:
        if "mobilenet" in layer.name:
            base_model = layer
            break
            
    if base_model:
        base_model.trainable = True
        # Unfreeze last 50 layers for complex geometry learning
        for layer in base_model.layers[:-50]: 
            layer.trainable = False
        print(f"✅ Unfrozen last 50 layers of {base_model.name}")

    model.compile(optimizer=optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, class_weight=weights, callbacks=[
        callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy'),
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ])
    
    with open('models/classes.txt', 'w') as f: f.write('\n'.join(class_names))
    plot_history(h1, h2)
    print(f"✅ Model Saved: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()