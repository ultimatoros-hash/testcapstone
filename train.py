import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
from sklearn.utils import class_weight

DATA_DIR = "data/raw/images"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25

# --- LAYERS ---
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
        def apply(): return tf.clip_by_value(inputs + tf.random.normal(tf.shape(inputs), 0, 0.02), 0, 1)
        return tf.cond(tf.cast(training, tf.bool), apply, lambda: inputs)

class AtmosphericScattering(layers.Layer):
    def call(self, inputs, training=None):
        if training is None: training = False
        def apply():
            s = tf.shape(inputs)
            h_new = tf.cast(tf.cast(s[1], tf.float32) * 0.8, tf.int32)
            w_new = tf.cast(tf.cast(s[2], tf.float32) * 0.8, tf.int32)
            return tf.image.resize(tf.image.resize(inputs, [h_new, w_new]), [s[1], s[2]])
        return tf.cond(tf.cast(training, tf.bool), apply, lambda: inputs)

class ColorJitter(layers.Layer):
    def call(self, inputs, training=None):
        if training is None: training = False
        def apply():
            x = inputs
            x = tf.image.random_saturation(x, 0.7, 1.3)
            x = tf.image.random_contrast(x, 0.8, 1.2)
            x = tf.image.random_brightness(x, 0.1)
            return tf.clip_by_value(x, 0.0, 1.0)
        return tf.cond(tf.cast(training, tf.bool), apply, lambda: inputs)

def build_model(num_classes):
    # Explicitly defining input shape helps Keras understand the dtyp
    inputs = layers.Input(shape=IMG_SIZE + (3,), dtype=tf.float32)
    x = layers.Rescaling(1./255)(inputs)
    
    x = layers.RandomFlip("horizontal_and_vertical")(x)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.1)(x)
    
    x = ColorJitter()(x)
    x = AtmosphericScattering()(x)
    x = SensorThermalNoise()(x)
    x = GentleCloudAugmentation()(x)
    
    for f in [32, 64, 128, 256]:
        x = layers.Conv2D(f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.MaxPooling2D()(x)
        
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256)(x); x = layers.LeakyReLU(0.1)(x); x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

def process_data(img, label):
    """Explicitly cast types to prevent string errors."""
    return tf.cast(img, tf.float32), tf.cast(label, tf.int32)

def train():
    print("🚀 BASELINE: CUSTOM CNN START...")
    
    # Load Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123, 
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123, 
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
    )
    
    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    # Compute weights before optimization
    y_train = []
    for _, labels in train_ds.unbatch():
        y_train.append(labels.numpy())
    
    y_train = np.array(y_train)
    weights = dict(enumerate(class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )))
    
    # Apply type enforcement
    train_ds = train_ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)

    # Optimization
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    model = build_model(len(class_names))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=weights, callbacks=[
        callbacks.ModelCheckpoint('models/satellite_custom_cnn.h5', save_best_only=True, monitor='val_accuracy'),
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ])
    
    with open('models/classes.txt', 'w') as f: f.write('\n'.join(class_names))
    print("✅ Baseline Saved.")

if __name__ == "__main__":
    train()