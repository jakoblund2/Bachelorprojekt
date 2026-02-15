import numpy as np
import rasterio
from PIL import Image

IN_TIF = "data/chips/s2_crop.tif"  # your 4-band file
OUT_PNG = "s2_rgb.png"

def stretch01(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (2, 98))
    x = (x - lo) / (hi - lo + 1e-9)
    return np.clip(x, 0, 1)

with rasterio.open(IN_TIF) as ds:
    b02 = ds.read(1)  # Blue
    b03 = ds.read(2)  # Green
    b04 = ds.read(3)  # Red

rgb = np.dstack([stretch01(b04), stretch01(b03), stretch01(b02)])
img = (rgb * 255).astype(np.uint8)

Image.fromarray(img).save(OUT_PNG)
print("Wrote:", OUT_PNG)