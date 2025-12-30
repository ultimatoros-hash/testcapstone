import os
import math
import csv
import requests
import concurrent.futures
from PIL import Image
from io import BytesIO
import random
import time

# --- CONFIGURATION ---
OUTPUT_DIR = "data/raw/images"
CSV_FILE = "data/dataset.csv"
ZOOM_LEVEL = 15   
MAX_WORKERS = 12
STEP_SIZE = 0.0045 # Tuned for ~20k total images

# --- HEADERS (Anti-Blocking) ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Referer': 'https://www.google.com/'
}

# --- PROVIDERS ---
PROVIDERS = {
    "esri": "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "usgs": "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
    "nasa": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/2022-12-01/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg"
}

# --- REGIONS OF INTEREST (5 CLASSES) ---
TARGETS = [
    # --- 1. URBAN ---
    ("urban", 40.7000, -74.0200, 40.8000, -73.9500, "esri"),   # NYC
    ("urban", 35.6800, 139.7000, 35.7500, 139.8000, "esri"),   # Tokyo
    ("urban", 48.8300, 2.2500, 48.9000, 2.4000, "esri"),       # Paris
    ("urban", 31.2000, 121.4000, 31.3000, 121.5000, "esri"),   # Shanghai

    # --- 2. FOREST ---
    ("forest", -3.4653, -62.2159, -3.4000, -62.1000, "esri"),  # Amazon
    ("forest", 48.4647, 7.9552, 48.5500, 8.0500, "esri"),      # Black Forest
    ("forest", 0.3000, 20.0000, 0.4000, 20.1000, "esri"),      # Congo
    ("forest", 61.0000, 99.0000, 61.1000, 99.1000, "esri"),    # Taiga

    # --- 3. WATER ---
    ("water", 25.0343, -77.3963, 25.1000, -77.3000, "esri"),   # Bahamas
    ("water", 34.0000, -119.0000, 34.1000, -118.9000, "nasa"), # Pacific
    ("water", -20.0000, 57.5000, -19.9000, 57.6000, "esri"),   # Mauritius
    ("water", 43.0000, 6.0000, 43.1000, 6.1000, "nasa"),       # Med Sea

    # --- 4. DESERT ---
    ("desert", 24.6857, 46.7023, 24.8000, 46.8000, "esri"),    # Saudi
    ("desert", 36.1699, -115.1398, 36.3000, -115.0000, "usgs"), # Nevada
    ("desert", -25.3000, 131.0000, -25.2000, 131.1000, "esri"), # Outback
    ("desert", 21.0000, 10.0000, 21.1000, 10.1000, "nasa"),    # Sahara

    # --- 5. SNOW (ADDED TO FIX MISSING CLASS) ---
    ("snow", 64.2008, -149.4937, 64.3000, -149.4000, "usgs"),  # Alaska
    ("snow", 72.0000, -40.0000, 72.1000, -39.9000, "esri"),    # Greenland
    ("snow", -80.0000, 0.0000, -79.9000, 0.1000, "nasa"),      # Antarctica
    ("snow", 46.5000, 8.0000, 46.6000, 8.1000, "esri"),        # Swiss Alps
]

def lat_lon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def generate_grid(lat_min, lon_min, lat_max, lon_max, step):
    points = []
    lat = lat_min
    while lat < lat_max:
        lon = lon_min
        while lon < lon_max:
            points.append((lat, lon))
            lon += step
        lat += step
    return points

def download_task(args):
    lat, lon, label, provider_name = args
    xtile, ytile = lat_lon_to_tile(lat, lon, ZOOM_LEVEL)
    
    url_template = PROVIDERS.get(provider_name, PROVIDERS['esri'])
    url = url_template.format(z=ZOOM_LEVEL, y=ytile, x=xtile)
    
    filename = f"{label}_{provider_name}_{xtile}_{ytile}.jpg"
    save_dir = os.path.join(OUTPUT_DIR, label)
    full_path = os.path.join(save_dir, filename)
    
    if os.path.exists(full_path) and os.path.getsize(full_path) > 1000:
        return None

    try:
        r = requests.get(url, headers=HEADERS, timeout=5)
        if r.status_code == 200:
            Image.open(BytesIO(r.content)).verify() # Check integrity
            Image.open(BytesIO(r.content)).convert('RGB').save(full_path)
            return [filename, label, lat, lon, provider_name, url]
    except Exception:
        pass
    return None

if __name__ == "__main__":
    print("🚀 Initializing Crawler (5 Classes)...")
    if not os.path.exists(CSV_FILE):
        os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["filename", "label", "latitude", "longitude", "provider", "url"])

    tasks = []
    for label, lat_min, lon_min, lat_max, lon_max, provider in TARGETS:
        points = generate_grid(lat_min, lon_min, lat_max, lon_max, STEP_SIZE)
        for lat, lon in points:
            tasks.append((lat, lon, label, provider))
            
    print(f"📡 Targets: {len(tasks)} tiles scheduled.")
    
    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(download_task, tasks)
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            for res in results:
                if res:
                    writer.writerow(res)
                    count += 1
                    if count % 200 == 0: print(f"  [+] Saved {count}...")

    print(f"\n✅ Crawl Complete. Total Images: {count}")
