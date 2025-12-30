import os
import shutil
import numpy as np
from PIL import Image
import tqdm

# --- CONFIGURATION ---
BASE_DIR = "data/raw/images"
CLASSES = ["urban", "forest", "water", "desert", "snow"]

# Thresholds (Tuned for Satellite RGB)
VARI_THRESHOLD = 0.15  # Above this is likely Forest/Greenery
SNOW_THRESHOLD = 180   # Average pixel intensity to be considered "White/Snow"

def calculate_metrics(img_path):
    """Calculates VARI (Vegetation) and Albedo (Brightness)."""
    with Image.open(img_path).convert('RGB') as img:
        arr = np.array(img).astype(float)
        
    # Extract channels
    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]
    
    # VARI = (Green - Red) / (Green + Red - Blue)
    # Adding small epsilon to avoid division by zero
    vari = (G - R) / (G + R - B + 1e-6)
    avg_vari = np.mean(vari)
    
    # Simple Albedo (Brightness)
    avg_brightness = np.mean(arr)
    
    return avg_vari, avg_brightness

def sort_and_clean():
    print("🧹 Starting Spectral Data Cleaning...")
    
    moved_count = 0
    deleted_count = 0

    for cls in CLASSES:
        cls_path = os.path.join(BASE_DIR, cls)
        if not os.path.exists(cls_path):
            continue
            
        files = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Checking {cls} folder ({len(files)} images)...")

        for f in tqdm.tqdm(files):
            file_path = os.path.join(cls_path, f)
            
            try:
                # 1. Check file integrity
                if os.path.getsize(file_path) < 2000:
                    os.remove(file_path)
                    deleted_count += 1
                    continue

                vari, brightness = calculate_metrics(file_path)

                # --- REDISTRIBUTION LOGIC ---
                
                # Logic: If it's in Snow but it's very green -> Move to Forest
                if cls == "snow" and vari > VARI_THRESHOLD:
                    target_dir = os.path.join(BASE_DIR, "forest")
                    shutil.move(file_path, os.path.join(target_dir, f))
                    moved_count += 1
                
                # Logic: If it's in Forest but it's very white/bright -> Move to Sno
                elif cls == "forest" and brightness > SNOW_THRESHOLD and vari < 0.05:
                    target_dir = os.path.join(BASE_DIR, "snow")
                    shutil.move(file_path, os.path.join(target_dir, f))
                    moved_count += 1
                
                # Logic: If it's in Water but it's very bright -> likely Cloud/Snow
                elif cls == "water" and brightness > 150:
                    # Water should be dark. If it's bright, it's a bad crawl.
                    os.remove(file_path)
                    deleted_count += 1

            except Exception as e:
                print(f"Error processing {f}: {e}")

    print(f"\n✨ Cleaning Complete!")
    print(f"📦 Images moved to correct folders: {moved_count}")
    print(f"🗑️ Corrupt or invalid images deleted: {deleted_count}")

if __name__ == "__main__":
    sort_and_clean()