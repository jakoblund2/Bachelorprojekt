import os
import requests
from datetime import datetime
from shapely.geometry import Point
from dotenv import load_dotenv
from copernicusapi import QueryConstructor
from tqdm import tqdm

# ---- EDIT THESE ----
LAT = 56.17174766707655
LON = 10.191236328269078
START = "2025-08-19"   # start date (UTC)
END   = "2025-08-20"   # end date (UTC) - exclusive
COLLECTION = "sentinel-1"   # "sentinel-1" or "sentinel-2"
PRODUCT_TYPE = "grd"        # sentinel-1: "grd" ; sentinel-2: "l2a"
MAX_CLOUD = 30              # only used for sentinel-2 in percent
# --------------------

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

def get_token(user: str, pwd: str) -> str:
    r = requests.post(TOKEN_URL, data={"client_id": "cdse-public", "username": user, "password": pwd, "grant_type": "password",}, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]

def download(url: str, out_path: str, token: str):
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True) as r:
        r.raise_for_status()

        total = int(r.headers.get("Content-Length", 0)) or None
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc="Downloading")

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

        pbar.close()

def main():
    load_dotenv()
    user = os.environ["CDSE_USER"]
    pwd  = os.environ["CDSE_PASS"]

    # 1) Search (by point + time)
    qc = QueryConstructor()
    qc.add_collection_filter(COLLECTION)
    qc.add_product_type_filter(PRODUCT_TYPE)
    qc.add_aoi_filter(Point(LON, LAT))  # shapely Point is (lon, lat)
    qc.add_sensing_start_date_filter(datetime.fromisoformat(START), datetime.fromisoformat(END))

    if COLLECTION.lower().replace("-", "") in ("sentinel2", "sentinel-2", "s2"):
        qc.add_cloud_cover_filter(MAX_CLOUD)

    n = qc.check_query()
    print("Matching products:", n)
    if n == 0:
        return

    products, _ = qc.send_query()
    row = products.iloc[0]  # dead simple: pick the first match

    print("Chosen:", row["file_name"])
    print("Download URL:", row["download_url"])

    # 2) Download
    token = get_token(user, pwd)
    out_zip = row["file_name"] + ".zip"
    download(row["download_url"], out_zip, token)
    print("Saved:", out_zip)

if __name__ == "__main__":
    main()
