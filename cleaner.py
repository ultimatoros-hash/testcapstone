import os
import time
from PIL import Image

DATA_DIR = "data/raw/images"

def is_valid_image(path):
    try:
        # 1. Check if empty
        if os.path.getsize(path) < 100: # Less than 100 bytes = probably empty or text
            print(f"❌ {os.path.basename(path)}: File too small (likely empty/text)")
            return False

        # 2. Check if valid image
        with Image.open(path) as img:
            img.verify() 
        return True

    except Exception as e:
        print(f"❌ {os.path.basename(path)}: Corrupt - {e}")
        return False

def clean_dataset():
    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    print(f"🔍 Scanning {DATA_DIR} for corrupt files...")
    deleted_count = 0
    total_files = 0
    
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            
            if not is_valid_image(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except PermissionError:
                    print(f"   ⚠️ Locked file. Retrying delete...")
                    time.sleep(0.1) 
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except:
                        print(f"   💀 Could not delete {file}. Delete manually.")

    print(f"\n{'='*30}")
    print(f"✅ Scan Complete.")
    print(f"📂 Total Files Scanned: {total_files}")
    print(f"🗑️  Corrupt Files Removed: {deleted_count}")
    print(f"{'='*30}")

if __name__ == "__main__":
    clean_dataset()