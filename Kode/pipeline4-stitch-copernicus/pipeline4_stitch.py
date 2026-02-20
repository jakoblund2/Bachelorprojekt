#!/usr/bin/env python3
"""
Stitch-first Sentinel-1/Sentinel-2 training dataset pipeline.

Workflow:
1) Search/download S2 products from CDSE and group by same acquisition pass.
2) Stitch all S2 tiles in each pass into one AOI-clipped mosaic grid.
3) Find nearest-in-time S1 scene, process with SNAP terrain correction.
4) Warp S1 to stitched S2 grid.
5) Extract strict-valid patches and write one NPZ per stitched mosaic.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import rasterio
import requests
from dotenv import load_dotenv
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window


# -----------------------------
# Paths / constants
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOWNLOAD_DIR = DATA_DIR / "downloads"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "proc"
MOSAIC_DIR = DATA_DIR / "mosaic"
OUT_DIR = DATA_DIR / "tiles_npz"
TMP_DIR = DATA_DIR / "tmp"

SNAP_GPT = Path.home() / "esa-snap" / "bin" / "gpt"
GRAPH_TC = BASE_DIR / "s1_grd_to_tc_dim.xml"

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_ROOT = "https://catalogue.dataspace.copernicus.eu/odata/v1"

SCL_INVALID = {3, 8, 9, 10, 11}

# Keep current 11-band layout (B08 omitted by decision).
S2_BANDS_11 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B09", "B11", "B12"]

S2_NAME_RE = re.compile(
    r"^(?P<prefix>S2[AB]_MSIL2A_(?P<sensing>\d{8}T\d{6})_(?P<baseline>N\d{4})_(?P<orbit>R\d{3}))_"
    r"(?P<tile>T\d{2}[A-Z]{3})_(?P<generation>\d{8}T\d{6})(?:_.+)?$"
)


# -----------------------------
# Config
# -----------------------------
@dataclass
class Job:
    name: str
    bbox_lonlat: Tuple[float, float, float, float]  # minLon, minLat, maxLon, maxLat
    date_start: str
    date_end: str
    max_s2: int = 500
    max_cloud: float = 20.0
    max_time_diff_hours: int = 36
    tile: int = 256
    stride: int = 256
    ocean_std_thr: float = 1.2
    output_mode: str = "per_patch"  # "per_patch" or "per_pass"


JOBS: List[Job] = [
    # Job(
    #     name="dk_test",
    #     bbox_lonlat=(7.9, 54.5, 12.8, 57.8),
    #     date_start="2025-06-12",
    #     date_end="2025-06-12",
    #     max_s2=999,
    #     max_cloud=30.0,
    #     max_time_diff_hours=36,
    #     tile=256,
    #     stride=256,
    #     ocean_std_thr=1.2,
    # ),
    Job(
        name="larger_test",
        bbox_lonlat=(8.8, 55.7, 9.5, 56.2),  # small AOI
        date_start="2025-06-12",
        date_end="2025-06-12",
        max_s2=30,                 # only one S2 product
        max_cloud=30.0,          # maximize chance of getting a product
        max_time_diff_hours=48,   # easier S1 match
        tile=128,                 # smaller patch for quick test
        stride=128,
        ocean_std_thr=0.0,        # don't reject flat areas during smoke test
        output_mode="per_patch",
    ),
]


for d in [DATA_DIR, DOWNLOAD_DIR, RAW_DIR, PROC_DIR, MOSAIC_DIR, OUT_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")


# -----------------------------
# Helpers: logging / shell
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: Iterable[str]) -> None:
    cmd = [str(x) for x in cmd]
    log("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# -----------------------------
# CDSE API helpers
# -----------------------------
def cdse_token() -> str:
    user = os.environ.get("CDSE_USER")
    pw = os.environ.get("CDSE_PASS")
    totp = os.environ.get("CDSE_TOTP")
    if not user or not pw:
        raise RuntimeError("Missing CDSE_USER/CDSE_PASS in environment or .env")

    payload = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": user,
        "password": pw,
    }
    if totp:
        payload["totp"] = totp

    response = requests.post(TOKEN_URL, data=payload, timeout=60)
    response.raise_for_status()
    return response.json()["access_token"]


def odata_get(
    url: str,
    token: str,
    params: Optional[dict] = None,
    timeout: int = 120,
    retries: int = 1,
) -> Tuple[requests.Response, str]:
    def _do(tok: str) -> requests.Response:
        return requests.get(
            url,
            headers={"Authorization": f"Bearer {tok}"},
            params=params,
            timeout=timeout,
        )

    resp = _do(token)
    if resp.status_code in (401, 403, 429) and retries > 0:
        token = cdse_token()
        resp = _do(token)

    resp.raise_for_status()
    return resp, token


def bbox_to_wkt(bbox_lonlat: Tuple[float, float, float, float]) -> str:
    minx, miny, maxx, maxy = bbox_lonlat
    return (
        "POLYGON(("
        f"{minx} {miny},"
        f"{maxx} {miny},"
        f"{maxx} {maxy},"
        f"{minx} {maxy},"
        f"{minx} {miny}"
        "))"
    )


def odata_search_s2(
    token: str,
    bbox_lonlat: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    max_cloud: float,
    top: int,
) -> Tuple[List[Dict[str, Any]], str]:
    wkt = bbox_to_wkt(bbox_lonlat)
    dt0 = f"{date_start}T00:00:00.000Z"
    dt1 = f"{date_end}T23:59:59.999Z"

    filt = " and ".join(
        [
            "Collection/Name eq 'SENTINEL-2'",
            "contains(Name,'MSIL2A')",
            f"ContentDate/Start gt {dt0}",
            f"ContentDate/Start lt {dt1}",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
            (
                "Attributes/OData.CSC.DoubleAttribute/any("
                f"a: a/Name eq 'cloudCover' and a/OData.CSC.DoubleAttribute/Value le {max_cloud})"
            ),
        ]
    )

    resp, token = odata_get(
        f"{ODATA_ROOT}/Products",
        token,
        params={"$filter": filt, "$top": str(top), "$orderby": "ContentDate/Start desc"},
        timeout=120,
        retries=1,
    )
    return resp.json().get("value", []), token


def odata_search_s1(
    token: str,
    bbox_lonlat: Tuple[float, float, float, float],
    dt_center_iso: str,
    hours: int,
    top: int = 10,
) -> Tuple[List[Dict[str, Any]], str]:
    wkt = bbox_to_wkt(bbox_lonlat)
    center = datetime.fromisoformat(dt_center_iso.replace("Z", "+00:00"))
    dt0 = (center - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    dt1 = (center + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    filt = " and ".join(
        [
            "Collection/Name eq 'SENTINEL-1'",
            "contains(Name,'IW_GRDH')",
            "contains(Name,'1SDV')",
            f"ContentDate/Start gt {dt0}",
            f"ContentDate/Start lt {dt1}",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
        ]
    )

    resp, token = odata_get(
        f"{ODATA_ROOT}/Products",
        token,
        params={"$filter": filt, "$top": str(top), "$orderby": "ContentDate/Start desc"},
        timeout=120,
        retries=1,
    )
    return resp.json().get("value", []), token


def parse_dt(prod: Dict[str, Any]) -> datetime:
    s = prod["ContentDate"]["Start"]
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def pick_nearest_by_time(candidates: List[Dict[str, Any]], target_dt: datetime) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    return min(candidates, key=lambda p: abs((parse_dt(p) - target_dt).total_seconds()))


def download_product_zip(
    get_token_fn,
    token: str,
    product: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Path, str]:
    pid = product["Id"]
    name = product["Name"]
    out = out_dir / (f"{name}.zip" if name.endswith(".SAFE") else f"{name}.SAFE.zip")

    if out.exists() and out.stat().st_size > 10_000_000:
        return out, token

    url = f"{ODATA_ROOT}/Products({pid})/$value"
    tmp = out.with_suffix(out.suffix + ".part")

    def _stream_download(url_to_get: str, bearer: str) -> None:
        headers = {"Authorization": f"Bearer {bearer}"}
        with requests.get(url_to_get, headers=headers, stream=True, timeout=300, allow_redirects=False) as r1:
            if r1.status_code in (301, 302, 303, 307, 308):
                loc = r1.headers.get("Location")
                if not loc:
                    r1.raise_for_status()
                with requests.get(loc, headers=headers, stream=True, timeout=300) as r2:
                    r2.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r2.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                return

            r1.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r1.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    try:
        _stream_download(url, token)
    except requests.HTTPError as exc:
        code = getattr(exc.response, "status_code", None)
        if code in (401, 403):
            token = get_token_fn()
            _stream_download(url, token)
        else:
            raise

    tmp.rename(out)
    return out, token


# -----------------------------
# Product name/grouping helpers
# -----------------------------
def parse_s2_name(name: str) -> Optional[Dict[str, str]]:
    cleaned = name.strip()
    for suffix in (".SAFE.zip", ".SAFE", ".zip"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]

    m = S2_NAME_RE.match(cleaned)
    if not m:
        return None

    prefix = m.group("prefix")
    generation = m.group("generation")
    return {
        "sensing": m.group("sensing"),
        "baseline": m.group("baseline"),
        "orbit": m.group("orbit"),
        "tile": m.group("tile"),
        "generation": generation,
        "pass_key": f"{prefix}_{generation}",
    }


def group_s2_by_pass(products: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    skipped = 0
    skipped_names: List[str] = []

    for prod in products:
        parsed = parse_s2_name(prod.get("Name", ""))
        if not parsed:
            skipped += 1
            skipped_names.append(prod.get("Name", ""))
            continue

        prod_copy = dict(prod)
        prod_copy["_s2_group"] = parsed
        grouped.setdefault(parsed["pass_key"], []).append(prod_copy)

    if skipped:
        log(f"Skipped {skipped} S2 product(s) with unexpected naming format")
        for n in skipped_names[:5]:
            log(f"  skipped-name: {n}")

    for key in grouped:
        grouped[key].sort(key=lambda p: p["_s2_group"]["tile"])

    return grouped


# -----------------------------
# Raster helpers
# -----------------------------
def utm_epsg_from_bbox(bbox_lonlat: Tuple[float, float, float, float]) -> str:
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0

    zone = int(math.floor((center_lon + 180.0) / 6.0) + 1)
    if center_lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return f"EPSG:{epsg}"


def aligned_aoi_grid(
    bbox_lonlat: Tuple[float, float, float, float],
    target_crs: str,
    resolution: float = 10.0,
) -> Dict[str, Any]:
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    # Densify bbox edges to avoid under-clipping in projected space.
    samples = []
    for t in np.linspace(0.0, 1.0, 21):
        lon = min_lon + t * (max_lon - min_lon)
        samples.append((lon, min_lat))
        samples.append((lon, max_lat))
    for t in np.linspace(0.0, 1.0, 21):
        lat = min_lat + t * (max_lat - min_lat)
        samples.append((min_lon, lat))
        samples.append((max_lon, lat))

    xs: List[float] = []
    ys: List[float] = []
    for lon, lat in samples:
        x, y = transformer.transform(lon, lat)
        xs.append(x)
        ys.append(y)

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    min_x = math.floor(min_x / resolution) * resolution
    min_y = math.floor(min_y / resolution) * resolution
    max_x = math.ceil(max_x / resolution) * resolution
    max_y = math.ceil(max_y / resolution) * resolution

    width = int(round((max_x - min_x) / resolution))
    height = int(round((max_y - min_y) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid aligned grid size: width={width}, height={height}")

    transform = from_origin(min_x, max_y, resolution, resolution)
    return {
        "transform": transform,
        "width": width,
        "height": height,
        "bounds": (min_x, min_y, max_x, max_y),
        "resolution": resolution,
        "crs": target_crs,
    }


def unzip_safe(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.SAFE")) or list(out_dir.glob("**/*.SAFE"))
    if existing:
        return existing[0]

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    safes = list(out_dir.glob("*.SAFE")) or list(out_dir.glob("**/*.SAFE"))
    if not safes:
        raise FileNotFoundError(f"No .SAFE folder found after unzip: {zip_path}")
    return safes[0]


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Pattern not found in SAFE: {pattern} (root={root})")
    return matches[0]


def find_s2_band_paths(s2_safe: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}

    r10 = find_one(s2_safe, "**/IMG_DATA/R10m")
    r20 = find_one(s2_safe, "**/IMG_DATA/R20m")
    r60 = find_one(s2_safe, "**/IMG_DATA/R60m")

    paths["B02"] = find_one(r10, "*_B02_10m.jp2")
    paths["B03"] = find_one(r10, "*_B03_10m.jp2")
    paths["B04"] = find_one(r10, "*_B04_10m.jp2")

    for band in ["B05", "B06", "B07", "B8A", "B11", "B12"]:
        paths[band] = find_one(r20, f"*_{band}_20m.jp2")

    for band in ["B01", "B09"]:
        paths[band] = find_one(r60, f"*_{band}_60m.jp2")

    paths["SCL"] = find_one(r20, "*_SCL_20m.jp2")
    return paths


def create_single_band_raster(
    out_path: Path,
    grid: Dict[str, Any],
    dtype: str,
    nodata: float,
) -> None:
    profile: Dict[str, Any] = {
        "driver": "GTiff",
        "height": int(grid["height"]),
        "width": int(grid["width"]),
        "count": 1,
        "dtype": dtype,
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if dtype == "float32":
        profile["predictor"] = 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile):
        pass


def stitch_band_to_grid(
    src_paths: List[Path],
    out_path: Path,
    grid: Dict[str, Any],
    resampling: Resampling,
    dtype: str,
    nodata: float,
) -> None:
    create_single_band_raster(out_path, grid, dtype=dtype, nodata=nodata)

    with rasterio.open(out_path, "r+") as dst:
        first = True
        for src_path in src_paths:
            with rasterio.open(src_path) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=0,
                    dst_transform=grid["transform"],
                    dst_crs=grid["crs"],
                    dst_nodata=nodata,
                    resampling=resampling,
                    init_dest_nodata=first,
                    num_threads=2,
                )
            first = False


def build_s2_mosaics(
    job: Job,
    s2_pass_key: str,
    s2_zips: List[Path],
    target_crs: str,
) -> Dict[str, Path]:
    grid = aligned_aoi_grid(job.bbox_lonlat, target_crs, resolution=10.0)
    mosaic_id = f"{job.name}__{s2_pass_key}"

    all_band_sources: Dict[str, List[Path]] = {band: [] for band in (S2_BANDS_11 + ["SCL"])}

    for s2_zip in s2_zips:
        unzip_dir = RAW_DIR / s2_zip.stem.replace(".SAFE", "")
        s2_safe = unzip_safe(s2_zip, unzip_dir)
        paths = find_s2_band_paths(s2_safe)
        for band in S2_BANDS_11:
            all_band_sources[band].append(paths[band])
        all_band_sources["SCL"].append(paths["SCL"])

    out_paths: Dict[str, Path] = {}
    for band in S2_BANDS_11:
        out = MOSAIC_DIR / f"{mosaic_id}__{band}.tif"
        log(f"  Stitch S2 {band}: {len(all_band_sources[band])} tile(s)")
        stitch_band_to_grid(
            src_paths=all_band_sources[band],
            out_path=out,
            grid=grid,
            resampling=Resampling.bilinear,
            dtype="float32",
            nodata=0.0,
        )
        out_paths[band] = out

    scl_out = MOSAIC_DIR / f"{mosaic_id}__SCL.tif"
    log(f"  Stitch S2 SCL: {len(all_band_sources['SCL'])} tile(s)")
    stitch_band_to_grid(
        src_paths=all_band_sources["SCL"],
        out_path=scl_out,
        grid=grid,
        resampling=Resampling.nearest,
        dtype="uint8",
        nodata=0,
    )
    out_paths["SCL"] = scl_out
    return out_paths


def process_s1_to_tc_imgs(s1_zip: Path, map_projection: str) -> Tuple[Path, Path]:
    if not SNAP_GPT.exists():
        raise FileNotFoundError(f"SNAP GPT not found: {SNAP_GPT}")
    if not GRAPH_TC.exists():
        raise FileNotFoundError(f"SNAP graph not found: {GRAPH_TC}")

    unzip_dir = RAW_DIR / s1_zip.stem.replace(".SAFE", "")
    s1_safe = unzip_safe(s1_zip, unzip_dir)

    base = s1_zip.stem.replace(".SAFE", "")
    tc_dim = PROC_DIR / f"{base}_tc.dim"

    if not tc_dim.exists():
        run([SNAP_GPT, GRAPH_TC, f"-Pin={s1_safe}", f"-Pout={tc_dim}", f"-PmapProjection={map_projection}"])

    tc_data = PROC_DIR / f"{base}_tc.data"
    vv = tc_data / "Sigma0_VV.img"
    vh = tc_data / "Sigma0_VH.img"
    if not vv.exists() or not vh.exists():
        raise FileNotFoundError(f"Missing Sigma0_VV/Sigma0_VH in {tc_data}")

    return vv, vh


def warp_s1_to_grid_db(vv_path: Path, vh_path: Path, ref_grid_path: Path, out_path: Path) -> Path:
    with rasterio.open(ref_grid_path) as ref:
        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            count=2,
            dtype="float32",
            nodata=np.nan,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            with rasterio.open(vv_path) as vv:
                reproject(
                    source=rasterio.band(vv, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=vv.transform,
                    src_crs=vv.crs,
                    src_nodata=0,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                    init_dest_nodata=True,
                    num_threads=2,
                )
            with rasterio.open(vh_path) as vh:
                reproject(
                    source=rasterio.band(vh, 1),
                    destination=rasterio.band(dst, 2),
                    src_transform=vh.transform,
                    src_crs=vh.crs,
                    src_nodata=0,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                    init_dest_nodata=True,
                    num_threads=2,
                )

    # Convert linear sigma0 -> dB in-place window by window.
    with rasterio.open(out_path, "r+") as ds:
        for _, window in ds.block_windows(1):
            block = ds.read([1, 2], window=window).astype(np.float32)
            block = 10.0 * np.log10(np.maximum(block, 1e-10))
            ds.write(block, window=window)

    return out_path


# -----------------------------
# Patch extraction / NPZ writing
# -----------------------------
def scan_valid_patch_indices(
    b02_path: Path,
    scl_path: Path,
    s1_path: Path,
    tile: int,
    stride: int,
    ocean_std_thr: float,
) -> Dict[str, Any]:
    accepted: List[Tuple[int, int]] = []

    total = 0
    skipped_valid = 0
    skipped_ocean = 0

    with rasterio.open(b02_path) as b02_ds, rasterio.open(scl_path) as scl_ds, rasterio.open(s1_path) as s1_ds:
        height, width = b02_ds.height, b02_ds.width

        for r0 in range(0, height - tile + 1, stride):
            for c0 in range(0, width - tile + 1, stride):
                total += 1
                win = Window(c0, r0, tile, tile)

                b02 = b02_ds.read(1, window=win)
                scl = scl_ds.read(1, window=win).astype(np.uint8)
                s1 = s1_ds.read([1, 2], window=win).astype(np.float32)

                clear = ~np.isin(scl, list(SCL_INVALID))
                nonzero = b02 > 0
                s1_ok = np.isfinite(s1[0]) & np.isfinite(s1[1]) & (s1[0] > -80) & (s1[1] > -80)

                valid = clear & nonzero & s1_ok
                if not bool(valid.all()):
                    skipped_valid += 1
                    continue

                vv_std = float(np.nanstd(s1[0][valid]))
                if vv_std < ocean_std_thr:
                    skipped_ocean += 1
                    continue

                accepted.append((r0, c0))

    return {
        "indices": accepted,
        "total": total,
        "skipped_valid": skipped_valid,
        "skipped_ocean": skipped_ocean,
    }


def write_npz_single_file(
    out_path: Path,
    s2_band_paths: Dict[str, Path],
    s1_path: Path,
    indices: List[Tuple[int, int]],
    tile: int,
    meta: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(indices)
    if n == 0:
        np.savez_compressed(
            out_path,
            s1=np.empty((0, 2, tile, tile), dtype=np.float32),
            s2=np.empty((0, len(S2_BANDS_11), tile, tile), dtype=np.float32),
            valid=np.empty((0, tile, tile), dtype=np.uint8),
            row0=np.empty((0,), dtype=np.int32),
            col0=np.empty((0,), dtype=np.int32),
            meta=np.array(json.dumps(meta)),
        )
        return

    tmp_root = TMP_DIR / (out_path.stem + "__memmaps")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    s1_mm = np.lib.format.open_memmap(
        tmp_root / "s1.npy", mode="w+", dtype=np.float32, shape=(n, 2, tile, tile)
    )
    s2_mm = np.lib.format.open_memmap(
        tmp_root / "s2.npy", mode="w+", dtype=np.float32, shape=(n, len(S2_BANDS_11), tile, tile)
    )
    valid_mm = np.lib.format.open_memmap(
        tmp_root / "valid.npy", mode="w+", dtype=np.uint8, shape=(n, tile, tile)
    )
    row_mm = np.lib.format.open_memmap(tmp_root / "row0.npy", mode="w+", dtype=np.int32, shape=(n,))
    col_mm = np.lib.format.open_memmap(tmp_root / "col0.npy", mode="w+", dtype=np.int32, shape=(n,))

    s2_handles: Dict[str, rasterio.DatasetReader] = {}
    try:
        with rasterio.open(s1_path) as s1_ds:
            for band in S2_BANDS_11:
                s2_handles[band] = rasterio.open(s2_band_paths[band])

            for i, (r0, c0) in enumerate(indices):
                win = Window(c0, r0, tile, tile)

                s1_mm[i] = s1_ds.read([1, 2], window=win).astype(np.float32)
                for b_idx, band in enumerate(S2_BANDS_11):
                    s2_mm[i, b_idx] = s2_handles[band].read(1, window=win).astype(np.float32)

                valid_mm[i].fill(1)
                row_mm[i] = r0
                col_mm[i] = c0

                if (i + 1) % 500 == 0 or (i + 1) == n:
                    log(f"    Filled patches: {i + 1}/{n}")

        np.savez_compressed(
            out_path,
            s1=s1_mm,
            s2=s2_mm,
            valid=valid_mm,
            row0=row_mm,
            col0=col_mm,
            meta=np.array(json.dumps(meta)),
        )
    finally:
        for ds in s2_handles.values():
            ds.close()

        del s1_mm
        del s2_mm
        del valid_mm
        del row_mm
        del col_mm
        shutil.rmtree(tmp_root, ignore_errors=True)


def write_npz_per_patch(
    out_dir: Path,
    base_name: str,
    s2_band_paths: Dict[str, Path],
    s1_path: Path,
    indices: List[Tuple[int, int]],
    tile: int,
    meta_base: Dict[str, Any],
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    wrote = 0

    if not indices:
        return wrote

    s2_handles: Dict[str, rasterio.DatasetReader] = {}
    try:
        with rasterio.open(s1_path) as s1_ds:
            for band in S2_BANDS_11:
                s2_handles[band] = rasterio.open(s2_band_paths[band])

            for i, (r0, c0) in enumerate(indices):
                win = Window(c0, r0, tile, tile)
                s1 = s1_ds.read([1, 2], window=win).astype(np.float32)

                s2 = np.empty((len(S2_BANDS_11), tile, tile), dtype=np.float32)
                for b_idx, band in enumerate(S2_BANDS_11):
                    s2[b_idx] = s2_handles[band].read(1, window=win).astype(np.float32)

                valid = np.ones((tile, tile), dtype=np.uint8)

                meta = dict(meta_base)
                meta["row0"] = int(r0)
                meta["col0"] = int(c0)
                meta["patch_index"] = int(i)

                out = out_dir / f"{base_name}_r{r0}_c{c0}.npz"
                np.savez_compressed(
                    out,
                    s1=s1,
                    s2=s2,
                    valid=valid,
                    row0=np.array(int(r0), dtype=np.int32),
                    col0=np.array(int(c0), dtype=np.int32),
                    meta=np.array(json.dumps(meta)),
                )
                wrote += 1

                if (i + 1) % 500 == 0 or (i + 1) == len(indices):
                    log(f"    Wrote patch files: {i + 1}/{len(indices)}")
    finally:
        for ds in s2_handles.values():
            ds.close()

    return wrote


# -----------------------------
# Pipeline core
# -----------------------------
def build_stitched_npz_for_pass(
    job: Job,
    s2_pass_key: str,
    s2_products: List[Dict[str, Any]],
    s1_product: Dict[str, Any],
    token: str,
) -> str:
    # Download all S2 tiles in pass + one S1 product.
    s2_zips: List[Path] = []
    for p in s2_products:
        zip_path, token = download_product_zip(cdse_token, token, p, DOWNLOAD_DIR)
        s2_zips.append(zip_path)

    s1_zip, token = download_product_zip(cdse_token, token, s1_product, DOWNLOAD_DIR)

    target_crs = utm_epsg_from_bbox(job.bbox_lonlat)
    log(f"  Target CRS: {target_crs}")

    # 1) Stitch S2 AOI mosaics.
    s2_mosaic = build_s2_mosaics(job, s2_pass_key, s2_zips, target_crs)

    # 2) Process and align S1.
    vv_img, vh_img = process_s1_to_tc_imgs(s1_zip, target_crs)
    s1_on_s2 = MOSAIC_DIR / f"{job.name}__{s2_pass_key}__{s1_product['Name']}__S1_on_S2.tif"
    warp_s1_to_grid_db(vv_img, vh_img, s2_mosaic["B02"], s1_on_s2)

    # 3) Scan valid windows (strict 100%).
    scan = scan_valid_patch_indices(
        b02_path=s2_mosaic["B02"],
        scl_path=s2_mosaic["SCL"],
        s1_path=s1_on_s2,
        tile=job.tile,
        stride=job.stride,
        ocean_std_thr=job.ocean_std_thr,
    )

    indices = scan["indices"]
    with rasterio.open(s2_mosaic["B02"]) as b02_ds:
        mosaic_shape = (b02_ds.height, b02_ds.width)
        mosaic_bounds = tuple(float(x) for x in b02_ds.bounds)
        mosaic_crs = str(b02_ds.crs)

    base_name = f"{job.name}__{s2_pass_key}__{s1_product['Name']}"

    meta = {
        "job": job.name,
        "s2_pass": s2_pass_key,
        "s2_tiles": [p["Name"] for p in s2_products],
        "s1_scene": s1_product["Name"],
        "tile": int(job.tile),
        "stride": int(job.stride),
        "strict_valid": True,
        "s2_bands": S2_BANDS_11,
        "scl_invalid": sorted(SCL_INVALID),
        "ocean_std_thr": float(job.ocean_std_thr),
        "target_crs": mosaic_crs,
        "mosaic_shape_hw": [int(mosaic_shape[0]), int(mosaic_shape[1])],
        "mosaic_bounds": list(mosaic_bounds),
        "window_total": int(scan["total"]),
        "accepted": int(len(indices)),
        "skipped_valid": int(scan["skipped_valid"]),
        "skipped_ocean": int(scan["skipped_ocean"]),
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # 4) Write outputs in selected mode.
    if job.output_mode == "per_pass":
        out_path = OUT_DIR / f"{base_name}.npz"
        write_npz_single_file(
            out_path=out_path,
            s2_band_paths=s2_mosaic,
            s1_path=s1_on_s2,
            indices=indices,
            tile=job.tile,
            meta=meta,
        )
        size_mb = out_path.stat().st_size / (1024 * 1024)
        log(
            f"  Wrote NPZ: {out_path.name} | "
            f"patches={len(indices)} / windows={scan['total']} | "
            f"skipped_valid={scan['skipped_valid']} skipped_ocean={scan['skipped_ocean']} | "
            f"size={size_mb:.1f} MB"
        )
    else:
        wrote = write_npz_per_patch(
            out_dir=OUT_DIR,
            base_name=base_name,
            s2_band_paths=s2_mosaic,
            s1_path=s1_on_s2,
            indices=indices,
            tile=job.tile,
            meta_base=meta,
        )
        log(
            f"  Wrote patch NPZ files: {wrote} | "
            f"windows={scan['total']} skipped_valid={scan['skipped_valid']} "
            f"skipped_ocean={scan['skipped_ocean']}"
        )

    return token


def run_batch() -> None:
    token = cdse_token()

    for job in JOBS:
        log(
            f"\n=== JOB {job.name} | dates={job.date_start}..{job.date_end} "
            f"bbox={job.bbox_lonlat} ==="
        )

        s2_products, token = odata_search_s2(
            token=token,
            bbox_lonlat=job.bbox_lonlat,
            date_start=job.date_start,
            date_end=job.date_end,
            max_cloud=job.max_cloud,
            top=job.max_s2,
        )

        if not s2_products:
            log("No S2 products found.")
            continue

        grouped = group_s2_by_pass(s2_products)
        if not grouped:
            log("No S2 pass groups created from search results.")
            continue

        ordered_groups = sorted(grouped.items(), key=lambda kv: parse_dt(kv[1][0]), reverse=True)
        log(f"Found {len(s2_products)} S2 products -> {len(ordered_groups)} pass group(s)")

        for s2_pass_key, pass_products in ordered_groups:
            s2_center = parse_dt(pass_products[0])
            s1_cands, token = odata_search_s1(
                token=token,
                bbox_lonlat=job.bbox_lonlat,
                dt_center_iso=pass_products[0]["ContentDate"]["Start"],
                hours=job.max_time_diff_hours,
                top=10,
            )

            s1_best = pick_nearest_by_time(s1_cands, s2_center)
            if not s1_best:
                log(f"Skip pass {s2_pass_key}: no S1 candidate")
                continue

            gap_h = abs((parse_dt(s1_best) - s2_center).total_seconds()) / 3600.0
            if gap_h > job.max_time_diff_hours:
                log(f"Skip pass {s2_pass_key}: nearest S1 gap={gap_h:.1f}h exceeds {job.max_time_diff_hours}h")
                continue

            log(
                f"PASS {s2_pass_key} | S2 tiles={len(pass_products)} | "
                f"S1={s1_best['Name']} | gap={gap_h:.1f}h"
            )

            token = build_stitched_npz_for_pass(
                job=job,
                s2_pass_key=s2_pass_key,
                s2_products=pass_products,
                s1_product=s1_best,
                token=token,
            )


if __name__ == "__main__":
    run_batch()
