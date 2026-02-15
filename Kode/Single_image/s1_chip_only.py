from pathlib import Path
import zipfile
import subprocess
from pyproj import Transformer

ZIP_PATH = Path("S1C_IW_GRDH_1SDV_20250819T053147_20250819T053212_003737_007770_1C38.SAFE.zip")
SNAP_GPT = Path.home() / "esa-snap" / "bin" / "gpt"

GRAPH_TC   = Path("s1_grd_to_tc_dim.xml")      # SAFE -> TC .dim (EPSG:32632)
GRAPH_CROP = Path("tc_dim_to_chip_tif.xml")    # .dim -> subset(WKT) -> GeoTIFF

LAT = 56.17174766707655
LON = 10.191236328269078

CROP_KM = 3.0   # outputs ~3km x 3km crop around point

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/proc")
CHIP_DIR = Path("data/chips")

def unzip_safe(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    safes = list(out_dir.glob("*.SAFE")) or list(out_dir.glob("**/*.SAFE"))
    return safes[0]

def run(cmd):
    print("Running:", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)

def wkt_box_utm32(lon, lat, size_km):
    # WGS84 -> UTM32N (meters)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    x, y = tr.transform(lon, lat)

    half = (size_km * 1000.0) / 2.0
    minx, maxx = x - half, x + half
    miny, maxy = y - half, y + half

    return (
        f"POLYGON(({minx} {miny},"
        f"{maxx} {miny},"
        f"{maxx} {maxy},"
        f"{minx} {maxy},"
        f"{minx} {miny}))"
    )

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    CHIP_DIR.mkdir(parents=True, exist_ok=True)

    safe_root = RAW_DIR / ZIP_PATH.stem.replace(".SAFE", "")
    safe_path = unzip_safe(ZIP_PATH, safe_root)

    tc_dim = PROC_DIR / (ZIP_PATH.stem.replace(".SAFE", "") + "_tc.dim")
    run([SNAP_GPT, GRAPH_TC, f"-Pin={safe_path}", f"-Pout={tc_dim}"])

    wkt = wkt_box_utm32(LON, LAT, CROP_KM)
    chip_tif = CHIP_DIR / (ZIP_PATH.stem.replace(".SAFE", "") + f"_crop_{CROP_KM:.1f}km.tif")
    run([SNAP_GPT, GRAPH_CROP, f"-Pin={tc_dim}", f"-Pwkt={wkt}", f"-Pout={chip_tif}"])

    print("Wrote:", chip_tif)

if __name__ == "__main__":
    main()
