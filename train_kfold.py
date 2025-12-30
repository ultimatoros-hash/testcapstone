import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pathlib
from sklearn.model_selection import KFold
import os

# --- CONFIG ---
DATA_DIR = pathlib.Path("data/raw/images")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
N_FOLDS = 5

def get_model(num_classes):
    """
    Standard ResNet for Stability Verification.
    We use a clean architecture to test DATA stability, independent of transfer learning.
    """
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.Rescaling(1./255)(inputs)
    
    # Feature Extraction
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_kfold():
    if not os.path.exists(DATA_DIR):
        print("❌ Data not found.")
        return

    print(f"🚀 Running Protocol A: {N_FOLDS}-Fold Cross-Validation...")
    
    # 1. Get Paths & Labels
    image_paths = list(DATA_DIR.glob('*/*.jpg'))
    image_paths = [str(path) for path in image_paths]
    np.random.seed(42)
    np.random.shuffle(image_paths)
    
    class_names = sorted([item.name for item in DATA_DIR.glob('*') if item.is_dir()])
    class_indices = {name: i for i, name in enumerate(class_names)}
    labels = [class_indices[pathlib.Path(path).parent.name] for path in image_paths]
    
    X = np.array(image_paths)
    y = np.array(labels)
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    accuracies = []
    fold_no = 1

    for train_idx, val_idx in kfold.split(X):
        print(f"\n🔄 Fold {fold_no}/{N_FOLDS}")
        
        # Prepare Datasets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        def process(path, label):
            img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
            return tf.image.resize(img, IMG_SIZE), label

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(process).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(process).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        # Train & Evaluate
        model = get_model(len(class_names))
        model.fit(train_ds, epochs=EPOCHS, verbose=0) # Silent trainin
        
        _, acc = model.evaluate(val_ds, verbose=0)
        print(f"   ✅ Score: {acc*100:.2f}%")
        accuracies.append(acc)
        fold_no += 1

    # Report
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    
    print(f"\n{'='*40}")
    print(f"🏆 PROTOCOL A RESULTS")
    print(f"   Mean Accuracy: {mean_acc:.2f}%")
    print(f"   Stability (Std): ± {std_acc:.2f}%")
    print(f"   Conclusion: {'STABLE' if std_acc < 5.0 else 'UNSTABLE'}")
    print(f"{'='*40}")

if __name__ == "__main__":
    run_kfold()