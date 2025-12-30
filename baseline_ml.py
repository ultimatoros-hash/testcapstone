import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# --- CONFIG --
DATA_DIR = "data/raw/images"
PLOT_DIR = "data/plots"
IMG_SIZE = (128, 128)

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
        [0, 180, 0, 256, 0, 256])
    if hist.sum() > 0:
        cv2.normalize(hist, hist)
    return hist.flatten()

def load_data():
    print("⏳ Loading data for Random Forest Baseline...")
    data = []
    labels = []
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    for label in class_names:
        path = os.path.join(DATA_DIR, label)
        files = os.listdir(path)[:500] 
        for f in files:
            img_path = os.path.join(path, f)
            try:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMG_SIZE)
                hist = extract_color_histogram(image)
                
                data.append(hist)
                labels.append(label)
            except:
                continue
    
    return np.array(data), np.array(labels), class_names

def run_baseline():
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    
    X, y, classes = load_data()
    print(f"   Loaded {len(X)} images. Features shape: {X.shape}")

    # Split
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train Random Forest
    print("🌲 Training Random Forest (this establishes the baseline)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"🏆 RANDOM FOREST BASELINE ACCURACY: {acc*100:.2f}%")
    
    with open(f"{PLOT_DIR}/baseline_ml_report.txt", "w") as f:
        f.write(f"RF_ACCURACY: {acc:.5f}\n")
        f.write(classification_report(y_test, preds, zero_division=0))
    
    cm = confusion_matrix(y_test, preds, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Greys', xticklabels=classes, yticklabels=classes)
    plt.title(f"Baseline (Random Forest): {acc*100:.1f}%")
    plt.savefig(f"{PLOT_DIR}/baseline_confusion_matrix.png")
    
    joblib.dump(model, "models/baseline_rf.pkl")
    print("✅ Baseline Complete.")

if __name__ == "__main__":
    run_baseline()