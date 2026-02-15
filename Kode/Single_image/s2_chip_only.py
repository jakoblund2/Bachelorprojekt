from pathlib import Path
import zipfile
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform

# --- EDIT ---
ZIP_PATH = Path("S2C_MSIL2A_20250819T104041_N0511_R008_T32VNH_20250819T155312.SAFE.zip")  # your .SAFE.zip filename
LAT = 56.17174766707655
LON = 10.191236328269078
CROP_KM = 3.0  # crop size: 3km x 3km
OUT_TIF = Path("data/chips/s2_crop.tif")
# ----------

def main():
    # km -> pixels at 10m
    chip = int((CROP_KM * 1000) / 10)

    out_dir = Path("data/raw") / ZIP_PATH.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(out_dir)

    safe = list(out_dir.glob("**/*.SAFE"))[0]
    r10 = list(safe.glob("**/IMG_DATA/R10m"))[0]

    b02 = list(r10.glob("*_B02_10m.jp2"))[0]
    b03 = list(r10.glob("*_B03_10m.jp2"))[0]
    b04 = list(r10.glob("*_B04_10m.jp2"))[0]
    b08 = list(r10.glob("*_B08_10m.jp2"))[0]

    with rasterio.open(b02) as ref:
        x, y = transform("EPSG:4326", ref.crs, [LON], [LAT])
        row, col = ref.index(x[0], y[0])

        half = chip // 2
        win = Window(col - half, row - half, chip, chip)
        tr = ref.window_transform(win)

        stack = []
        for p in [b02, b03, b04, b08]:
            with rasterio.open(p) as ds:
                stack.append(ds.read(1, window=win))

        data = np.stack(stack, axis=0)

        profile = ref.profile
        profile.update(driver="GTiff", count=4, height=chip, width=chip, transform=tr)

        OUT_TIF.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(OUT_TIF, "w", **profile) as dst:
            dst.write(data)

    print("Wrote:", OUT_TIF)

if __name__ == "__main__":
    main()