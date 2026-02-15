import numpy as np
import rasterio
from PIL import Image

src = "data/chips/S1C_IW_GRDH_1SDV_20250819T053147_20250819T053212_003737_007770_1C38_crop.tif"
out = "preview-s1.png"

with rasterio.open(src) as ds:
    a = ds.read(1)  # band 1
    a = np.nan_to_num(a)

    # simple contrast stretch (2–98%)
    lo, hi = np.percentile(a, (2, 98))
    a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    img = (a * 255).astype(np.uint8)

Image.fromarray(img).save(out)
print("Wrote", out)
