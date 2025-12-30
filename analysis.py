import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np

# --- CONFIGURATION ---
CSV_PATH = "data/dataset.csv"
IMG_DIR = "data/raw/images"
PLOT_DIR = "data/plots"

# OFFICIAL PALETTE (Matches Report Figures)
PALETTE = {
    "snow": "#d1d5db",   # Light Gray
    "water": "#3b82f6",  # Blue
    "forest": "#16a34a", # Green
    "urban": "#ef4444",  # Red
    "desert": "#f97316"  # Orang
}

def run_analysis():
    if not os.path.exists(CSV_PATH):
        print("❌ Dataset CSV not found. Run crawler.py first.")
        return
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded dataset: {len(df)} records")
    
    # Filter to ensure we only plot known classes
    df = df[df['label'].isin(PALETTE.keys())]

    # --- PLOT 1: CLASS DISTRIBUTION (Balanced?) ---
    print("Generating Figure 3.1: Class Balance...")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='label', data=df, palette=PALETTE)
    plt.title("Figure 3.1: Class Distribution (Target: Balanced)")
    plt.xlabel("Terrain Class")
    plt.ylabel("Sample Count")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "class_balance.png"))
    plt.close()

    # --- PLOT 2: GEOGRAPHIC MAP ---
    print("Generating Figure 3.2: Geographic Sources...")
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='longitude', y='latitude', hue='label', data=df, s=15, palette=PALETTE, alpha=0.7)
    plt.title("Figure 3.2: Global Data Acquisition Sources")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "geo_map.png"))
    plt.close()

    # --- PLOT 3: SPECTRAL ANALYSIS (The Paradox) ---
    print("Generating Figure 3.3: Spectral Density Analysis...")
    plt.figure(figsize=(10, 6))
    
    # We analyze brightness (grayscale intensity)
    for label in PALETTE.keys():
        subset = df[df['label'] == label]
        if subset.empty: continue
        
        # Sample 100 images for speed
        sample_files = subset['filename'].sample(n=min(100, len(subset)), random_state=42)
        intensities = []
        
        for fname in sample_files:
            img_path = os.path.join(IMG_DIR, label, fname)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    intensities.extend(gray.flatten())
        
        # Plot KDE
        if intensities:
            sns.kdeplot(intensities, label=label, color=PALETTE[label], fill=True, alpha=0.2)
            
    plt.title("Figure 3.3: The Spectral Paradox\n(Note the overlap between Urban and Desert intensities)")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Probability Density")
    plt.xlim(0, 255)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "spectral_analysis.png"))
    plt.close()

    print(f"✅ EDA Plots saved to {PLOT_DIR}/")

if __name__ == "__main__":
    run_analysis()