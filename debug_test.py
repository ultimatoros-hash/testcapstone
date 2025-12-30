import requests
import os

# Test URLs for the 3 providers
TEST_URLS = {
    "ESRI": "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/16/24636/19024",
    "USGS": "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/16/24636/19024",
    "NASA": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/2022-12-01/GoogleMapsCompatible_Level8/6/20/30.jpg"
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

print("--- NETWORK DIAGNOSTIC ---")
for provider, url in TEST_URLS.items():
    print(f"\nTesting {provider}...")
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {r.status_code}")
        print(f"Content Type: {r.headers.get('Content-Type')}")
        print(f"File Size: {len(r.content)} bytes")
        
        if r.status_code == 200 and len(r.content) > 1000:
            print("✅ SUCCESS")
        else:
            print("❌ FAILURE (Blocked or Empty)")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")