import requests
import pandas as pd
import io
import os
from astropy.io import fits

# 1. Configuration - Set your survey (ZTF or LSST)
# For ZTF: https://api.fink-portal.org/api/v1
# For LSST: https://api.lsst.fink-portal.org/api/v1
BASE_URL = 'https://api.fink-portal.org/api/v1'
SAVE_DIR = "data/raw/fink_alerts"

# Create the directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Searching for SN Ia candidates...")

# 2. Get 50 SN Ia Object IDs using the /latests endpoint
# We use /latests because we are searching by CLASS, not specific ID
r = requests.post(
    f'{BASE_URL}/latests',
    json={
        'class': 'Early SN Ia candidate', 
        'n': '50' 
    }
)

if r.status_code != 200:
    print(f"Error fetching list: {r.text}")
else:
    objects = pd.read_json(io.BytesIO(r.content))
    print(f"Found {len(objects)} objects. Starting download...")

    # 3. Download their Science Cutouts as FITS
    for _, row in objects.iterrows():
        obj_id = row['i:objectId']
        
        # Download the cutout
        img_data = requests.post(
            f'{BASE_URL}/cutouts',
            json={
                'objectId': obj_id, 
                'kind': 'Science', 
                'output-format': 'FITS'
            }
        )
        
        if img_data.status_code == 200:
            file_path = os.path.join(SAVE_DIR, f"{obj_id}.fits")
            with open(file_path, "wb") as f:
                f.write(img_data.content)
            print(f" Successfully saved: {obj_id}.fits")
        else:
            print(f" Failed to download cutout for {obj_id}: {img_data.status_code}")

print("\nTask Complete!")