#!/usr/bin/env python3
"""
Pipeline5 mosaic variant:
- Sentinel-1 IW Monthly Mosaics
- Sentinel-2 Quarterly Mosaics
- aerial enrichment from Datafordeler

This uses Sentinel Hub BYOC collections rather than CDSE OData SAFE downloads.
The S2 output schema changes from 11 bands to 4 bands: B02, B03, B04, B08.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from getpass import getpass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import requests
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window


THIS_DIR = Path(__file__).resolve().parent
PARENT_PIPELINE = THIS_DIR.parent / "pipeline5_aerial.py"

_SPEC = importlib.util.spec_from_file_location("pipeline5_aerial_base", PARENT_PIPELINE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load base pipeline from {PARENT_PIPELINE}")
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)


BASE_DIR = THIS_DIR
DATA_DIR = BASE_DIR / "data_s1_s2_mosaics"
DOWNLOAD_DIR = DATA_DIR / "downloads"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "proc"
MOSAIC_DIR = DATA_DIR / "mosaic"
OUT_DIR = DATA_DIR / "tiles_npz"
TMP_DIR = DATA_DIR / "tmp"
AERIAL_INDEX_DIR = DATA_DIR / "aerial_index"
AERIAL_MOSAIC_DIR = DATA_DIR / "aerial_mosaic_1m"
AERIAL_TMP_DIR = DATA_DIR / "aerial_tmp"
AERIAL_CACHE_DIR = DATA_DIR / "aerial_cache_1m"

for d in [
    DATA_DIR,
    DOWNLOAD_DIR,
    RAW_DIR,
    PROC_DIR,
    MOSAIC_DIR,
    OUT_DIR,
    TMP_DIR,
    AERIAL_INDEX_DIR,
    AERIAL_MOSAIC_DIR,
    AERIAL_TMP_DIR,
    AERIAL_CACHE_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

base.BASE_DIR = BASE_DIR
base.DATA_DIR = DATA_DIR
base.DOWNLOAD_DIR = DOWNLOAD_DIR
base.RAW_DIR = RAW_DIR
base.PROC_DIR = PROC_DIR
base.MOSAIC_DIR = MOSAIC_DIR
base.OUT_DIR = OUT_DIR
base.TMP_DIR = TMP_DIR
base.AERIAL_INDEX_DIR = AERIAL_INDEX_DIR
base.AERIAL_MOSAIC_DIR = AERIAL_MOSAIC_DIR
base.AERIAL_TMP_DIR = AERIAL_TMP_DIR
base.AERIAL_CACHE_DIR = AERIAL_CACHE_DIR
base.DF_CATALOG_JSONL = AERIAL_INDEX_DIR / "geodko_catalog.jsonl"
base.DF_CATALOG_PARQUET = AERIAL_INDEX_DIR / "geodko_catalog.parquet"

base.load_dotenv(THIS_DIR.parent / ".env", override=False)
base.load_dotenv(THIS_DIR / ".env", override=False)

CPU_LIMIT: Optional[int] = None  # e.g. 4 to keep the process on 4 cores

def _env_int_optional(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
        return value if value > 0 else None
    except Exception:
        return None


def apply_cpu_limit() -> None:
    limit = CPU_LIMIT if CPU_LIMIT is not None else _env_int_optional("PIPELINE5_CPU_LIMIT")
    if limit is None:
        return

    detected = max(1, int(os.cpu_count() or 1))
    target = max(1, min(limit, detected))
    actual = target

    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        try:
            allowed = sorted(os.sched_getaffinity(0))
            if allowed:
                keep = set(allowed[:target])
                os.sched_setaffinity(0, keep)
                actual = len(os.sched_getaffinity(0))
        except Exception:
            actual = target

    base.CPU_COUNT = max(1, int(actual))
    base.AUTO_REPROJECT_THREADS = max(1, min(8, base.CPU_COUNT // 2))
    base.AUTO_S1_WORKERS = max(1, min(4, base.CPU_COUNT // 4))
    base.AUTO_CDSE_DOWNLOAD_WORKERS = max(1, min(4, base.CPU_COUNT))
    base.REPROJECT_THREADS = min(base._env_int("PIPELINE5_REPROJECT_THREADS", base.AUTO_REPROJECT_THREADS), base.CPU_COUNT)
    base.S1_SCENE_WORKERS = min(base._env_int("PIPELINE5_S1_WORKERS", base.AUTO_S1_WORKERS), base.CPU_COUNT)
    base.CDSE_DOWNLOAD_WORKERS = min(
        base._env_int("PIPELINE5_CDSE_DOWNLOAD_WORKERS", base.AUTO_CDSE_DOWNLOAD_WORKERS),
        base.CPU_COUNT,
    )


def apply_job_concurrency_caps(job: base.Job) -> None:
    cpu = max(1, int(base.CPU_COUNT))
    job.aerial_workers = min(int(job.aerial_workers), max(1, cpu * 2))
    job.aerial_chunk_workers = min(int(job.aerial_chunk_workers), cpu)

apply_cpu_limit()

# Sentinel-2 quarterly mosaics only expose these spectral bands via the BYOC collection.
base.S2_BANDS_11 = ["B02", "B03", "B04", "B08"]

SH_TOKEN_URL = base.TOKEN_URL
SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
S1_MONTHLY_BYOC = "byoc-3c662330-108b-4378-8899-525fd5a225cb"
S2_QUARTERLY_BYOC = "byoc-5460de54-082e-473a-b6ea-d5cbe3c17cca"
SH_MAX_DIM = 2500

S2_QUARTERLY_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B08", "observations", "dataMask"],
    output: { bands: 6, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  if (sample.dataMask <= 0 || sample.observations <= 0) {
    return [0, 0, 0, 0, 0, 0];
  }
  return [sample.B02, sample.B03, sample.B04, sample.B08, sample.observations, sample.dataMask];
}
""".strip()

S1_MONTHLY_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["VV", "VH", "dataMask"],
    output: { bands: 3, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  return [sample.VV, sample.VH, sample.dataMask];
}
""".strip()

def google_maps_coords(coord: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    lat1, lon1 = coord[0], coord[1]
    lat2, lon2 = coord[2], coord[3]
    return (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))


def bbox_cache_tag(bbox: Tuple[float, float, float, float]) -> str:
    payload = ",".join(f"{float(v):.8f}" for v in bbox)
    return hashlib.sha1(payload.encode("ascii")).hexdigest()[:10]


def apply_bbox_cache_suffix(job: base.Job) -> None:
    suffix = f"__bbox_{bbox_cache_tag(job.bbox_lonlat)}"
    if suffix not in job.name:
        job.name = f"{job.name}{suffix}"


JOBS: List[base.Job] = [
    base.Job(
        name="larger_test_s1_s2_mosaics",
        bbox_lonlat=google_maps_coords((56.19276935659911, 10.070327136252029, 56.08083648522022, 10.253216928049108)),
        date_start="2025-04-01",
        date_end="2025-04-01",
        max_s2=1,
        max_cloud=100.0,
        max_time_diff_hours=24,
        max_s1_scenes=1,
        tile=256,
        stride=256,
        min_valid_frac=0.9,
        output_mode="per_patch",
        aerial_target_res_m=10.0,
        aerial_min_cov_frac=1.0,
        aerial_download_batch_gb=4.0,
        aerial_workers=32,
        aerial_chunk_workers=4,
    ),
]

def log(msg: str) -> None:
    base.log(msg)


def stage(msg: str) -> None:
    log(f"[stage] {msg}")


def _gtiff_layout(width: int, height: int) -> Dict[str, Any]:
    width = max(1, int(width))
    height = max(1, int(height))

    # Small coarse-resolution tiles can render badly with oversized 256x256 blocks.
    # Use untiled output for tiny rasters and clamp tiled block sizes otherwise.
    if width < 64 or height < 64:
        return {"tiled": False}

    def _block(dim: int) -> int:
        if dim >= 256:
            return 256
        return max(16, (dim // 16) * 16)

    return {
        "tiled": True,
        "blockxsize": _block(width),
        "blockysize": _block(height),
    }


def _set_aerial_colorinterp(ds: rasterio.io.DatasetWriter) -> None:
    ds.colorinterp = (
        ColorInterp.red,
        ColorInterp.green,
        ColorInterp.blue,
        ColorInterp.nir,
    )


def downsample_to_cache_1m(src_tif: Path, cache_path: Path, target_res_m: float) -> Path:
    if cache_path.exists():
        if base._is_valid_cached_aerial_tif(cache_path):
            return cache_path
        cache_path.unlink(missing_ok=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)

    with rasterio.open(src_tif) as src:
        inferred_src = None
        if not base._has_valid_georef(src):
            inferred_src = base._infer_geodko_georef(src_tif.name, src.width, src.height)
            if inferred_src is None:
                raise RuntimeError(f"Source TIFF has no valid georeference: {src_tif.name}")
        if src.count < 4:
            raise RuntimeError(f"Expected RGBNIR (>=4 bands), got {src.count} bands in {src_tif.name}")

        if inferred_src is None:
            src_transform = src.transform
            src_crs = src.crs
            left, bottom, right, top = src.bounds
        else:
            src_transform, src_crs, bounds = inferred_src
            left, bottom, right, top = bounds

        width = max(1, int(np.ceil((right - left) / target_res_m)))
        height = max(1, int(np.ceil((top - bottom) / target_res_m)))
        transform = from_origin(left, top, target_res_m, target_res_m)

        profile: Dict[str, Any] = {
            "driver": "GTiff",
            "count": 4,
            "dtype": "uint8",
            "width": width,
            "height": height,
            "crs": src_crs,
            "transform": transform,
            "compress": "deflate",
            "predictor": 2,
            "interleave": "pixel",
        }
        profile.update(_gtiff_layout(width, height))

        with rasterio.open(tmp, "w", **profile) as dst:
            _set_aerial_colorinterp(dst)
            src_nodata = src.nodata
            for out_band, src_band in enumerate([1, 2, 3, 4], start=1):
                base._reproject_strict(
                    source=rasterio.band(src, src_band),
                    destination=rasterio.band(dst, out_band),
                    src_transform=src_transform,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    dst_transform=transform,
                    dst_crs=src_crs,
                    resampling=Resampling.bilinear,
                    init_dest_nodata=False,
                    num_threads=base.REPROJECT_THREADS,
                )

            src_mask = (src.read_masks(1) > 0).astype(np.uint8)
            dst_mask = np.zeros((height, width), dtype=np.uint8)
            base._reproject_strict(
                source=src_mask,
                destination=dst_mask,
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=0,
                dst_transform=transform,
                dst_crs=src_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
                init_dest_nodata=True,
                num_threads=base.REPROJECT_THREADS,
            )
            dst.write_mask((dst_mask > 0).astype(np.uint8) * 255)

    tmp.replace(cache_path)
    return cache_path


def create_aerial_rasters(aerial_path: Path, cov_path: Path, grid: Dict[str, Any]) -> None:
    height = int(grid["height"])
    width = int(grid["width"])
    profile: Dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "uint8",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "compress": "deflate",
        "predictor": 2,
        "interleave": "pixel",
    }
    profile.update(_gtiff_layout(width, height))

    aerial_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(aerial_path, "w", **base.with_bigtiff(profile, mode="YES")) as ds:
        _set_aerial_colorinterp(ds)

    cov_profile = dict(profile)
    cov_profile.pop("predictor", None)
    cov_profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(cov_path, "w", **base.with_bigtiff(cov_profile, mode="YES")):
        pass


base.downsample_to_cache_1m = downsample_to_cache_1m
base.create_aerial_rasters = create_aerial_rasters


def format_duration(seconds: float) -> str:
    total = int(round(max(0.0, float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def parse_job_datetime(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def exact_day_range(date_str: str) -> Tuple[str, str]:
    dt = parse_job_datetime(date_str)
    return (
        dt.strftime("%Y-%m-%dT00:00:00Z"),
        (dt + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
    )


def quarter_label(dt: datetime) -> str:
    q = ((dt.month - 1) // 3) + 1
    return f"{dt.year}Q{q}"


def pass_key_for_date(dt: datetime) -> str:
    ts = dt.strftime("%Y%m%dT000000")
    return f"S2A_MSIL2A_{ts}_N9999_R000_T00XXX_{ts}"


def sh_client_credentials() -> Tuple[str, str]:
    client_id = (
        os.environ.get("SH_CLIENT_ID")
        or os.environ.get("CDSE_SH_CLIENT_ID")
        or os.environ.get("OAUTH_CLIENT_ID")
    )
    client_secret = (
        os.environ.get("SH_CLIENT_SECRET")
        or os.environ.get("CDSE_SH_CLIENT_SECRET")
        or os.environ.get("OAUTH_CLIENT_SECRET")
    )
    if not client_id:
        try:
            client_id = getpass("Sentinel Hub client id: ").strip()
        except Exception:
            client_id = ""
    if not client_secret:
        try:
            client_secret = getpass("Sentinel Hub client secret: ").strip()
        except Exception:
            client_secret = ""
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Sentinel Hub OAuth client credentials. "
            "Set SH_CLIENT_ID/SH_CLIENT_SECRET, "
            "CDSE_SH_CLIENT_ID/CDSE_SH_CLIENT_SECRET, "
            "or OAUTH_CLIENT_ID/OAUTH_CLIENT_SECRET, "
            "or enter them interactively when prompted."
        )
    return client_id, client_secret


def sh_token() -> str:
    client_id, client_secret = sh_client_credentials()
    resp = requests.post(
        SH_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _epsg_code(crs: str) -> str:
    if ":" not in crs:
        raise ValueError(f"Unsupported CRS format: {crs}")
    return crs.split(":", 1)[1]


def _subgrid(grid: Dict[str, Any], row_off: int, col_off: int, height: int, width: int) -> Dict[str, Any]:
    transform = grid["transform"] * base.Affine.translation(col_off, row_off)
    minx = float(transform.c)
    maxy = float(transform.f)
    res = float(grid["resolution"])
    maxx = minx + (width * res)
    miny = maxy - (height * res)
    return {
        "transform": transform,
        "width": int(width),
        "height": int(height),
        "bounds": (minx, miny, maxx, maxy),
        "resolution": res,
        "crs": grid["crs"],
    }


def _grid_chunks(grid: Dict[str, Any], max_dim: int = SH_MAX_DIM) -> List[Tuple[int, int, Dict[str, Any]]]:
    chunks: List[Tuple[int, int, Dict[str, Any]]] = []
    total_h = int(grid["height"])
    total_w = int(grid["width"])
    for row_off in range(0, total_h, max_dim):
        h = min(max_dim, total_h - row_off)
        for col_off in range(0, total_w, max_dim):
            w = min(max_dim, total_w - col_off)
            chunks.append((row_off, col_off, _subgrid(grid, row_off, col_off, h, w)))
    return chunks


def _sh_process_request(
    token: str,
    collection_id: str,
    grid: Dict[str, Any],
    evalscript: str,
    time_from: str,
    time_to: str,
) -> bytes:
    minx, miny, maxx, maxy = [float(x) for x in grid["bounds"]]
    epsg = _epsg_code(str(grid["crs"]))
    payload = {
        "input": {
            "bounds": {
                "bbox": [minx, miny, maxx, maxy],
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"},
            },
            "data": [
                {
                    "type": collection_id,
                    "dataFilter": {
                        "timeRange": {
                            "from": time_from,
                            "to": time_to,
                        }
                    },
                }
            ],
        },
        "output": {
            "width": int(grid["width"]),
            "height": int(grid["height"]),
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    last_exc: Optional[Exception] = None
    for attempt in range(1, base.CDSE_DOWNLOAD_ATTEMPTS + 1):
        try:
            resp = requests.post(SH_PROCESS_URL, headers=headers, json=payload, timeout=600)
            if resp.status_code in (401, 403) and attempt < base.CDSE_DOWNLOAD_ATTEMPTS:
                token = sh_token()
                headers["Authorization"] = f"Bearer {token}"
                continue
            if resp.status_code == 429 and attempt < base.CDSE_DOWNLOAD_ATTEMPTS:
                wait_s = max(1, base.CDSE_DOWNLOAD_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
                sleep(float(wait_s))
                continue
            if resp.status_code >= 400:
                body = resp.text.strip()
                snippet = body[:1200] if body else "<empty body>"
                raise requests.HTTPError(
                    f"SH process error {resp.status_code}: {snippet}",
                    response=resp,
                )
            return resp.content
        except Exception as exc:
            last_exc = exc
            if attempt < base.CDSE_DOWNLOAD_ATTEMPTS:
                wait_s = max(1, base.CDSE_DOWNLOAD_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
                sleep(float(wait_s))
                continue
            break

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed SH process request for {collection_id}")


def sh_process_to_tiff(
    token: str,
    collection_id: str,
    grid: Dict[str, Any],
    evalscript: str,
    time_from: str,
    time_to: str,
    out_path: Path,
) -> Path:
    if base.file_is_ready(out_path, min_bytes=1024 * 1024):
        log(f"  Reuse SH cache: {out_path.name}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    chunk_dir = TMP_DIR / "sh_chunks" / out_path.stem
    chunk_dir.mkdir(parents=True, exist_ok=True)

    width = int(grid["width"])
    height = int(grid["height"])
    if width <= SH_MAX_DIM and height <= SH_MAX_DIM:
        stage(f"Request Sentinel Hub raster: {out_path.name} ({width}x{height})")
        content = _sh_process_request(token, collection_id, grid, evalscript, time_from, time_to)
        with tmp.open("wb") as f:
            f.write(content)
        tmp.replace(out_path)
        return out_path

    chunks = _grid_chunks(grid, SH_MAX_DIM)
    stage(f"Request Sentinel Hub raster in chunks: {out_path.name}")
    log(
        "  SH tiling: "
        f"{len(chunks)} chunk(s) for {out_path.name} "
        f"(full={width}x{height}, max={SH_MAX_DIM})"
    )

    first_chunk_path: Optional[Path] = None
    for idx, (row_off, col_off, chunk_grid) in enumerate(chunks):
        chunk_path = chunk_dir / f"r{row_off}_c{col_off}.tif"
        if not base.file_is_ready(chunk_path, min_bytes=1024):
            content = _sh_process_request(token, collection_id, chunk_grid, evalscript, time_from, time_to)
            chunk_tmp = chunk_path.with_suffix(".tif.part")
            with chunk_tmp.open("wb") as f:
                f.write(content)
            chunk_tmp.replace(chunk_path)
        if first_chunk_path is None:
            first_chunk_path = chunk_path
        if base.is_progress_tick(idx + 1, len(chunks), 2):
            log(f"  SH tile progress: {idx + 1}/{len(chunks)}")

    if first_chunk_path is None:
        raise RuntimeError(f"No SH chunks were generated for {collection_id}")

    with rasterio.open(first_chunk_path) as first_ds:
        profile = first_ds.profile.copy()
    profile.update(
        width=width,
        height=height,
        transform=grid["transform"],
        crs=str(grid["crs"]),
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    with rasterio.open(tmp, "w", **base.with_bigtiff(profile)) as dst:
        stage(f"Stitch Sentinel Hub chunks: {out_path.name}")
        for row_off, col_off, _chunk_grid in chunks:
            chunk_path = chunk_dir / f"r{row_off}_c{col_off}.tif"
            with rasterio.open(chunk_path) as src:
                data = src.read()
            dst.write(
                data,
                window=Window(col_off, row_off, int(data.shape[2]), int(data.shape[1])),
            )

    tmp.replace(out_path)
    return out_path


def write_single_band(
    out_path: Path,
    grid: Dict[str, Any],
    arr: np.ndarray,
    dtype: str,
    nodata: Optional[float],
) -> Path:
    profile = {
        "driver": "GTiff",
        "height": int(grid["height"]),
        "width": int(grid["width"]),
        "count": 1,
        "dtype": dtype,
        "crs": str(grid["crs"]),
        "transform": grid["transform"],
        "compress": "deflate",
        "predictor": 2 if "float" in dtype else 1,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": nodata,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(arr, 1)
    return out_path


def build_s2_quarterly_mosaic(job: base.Job, grid: Dict[str, Any], token: str, ref_dt: datetime) -> Dict[str, Path]:
    pass_key = pass_key_for_date(ref_dt)
    raw_path = MOSAIC_DIR / f"{job.name}__{pass_key}__S2_quarterly_raw.tif"
    time_from, time_to = exact_day_range(job.date_start)
    stage(f"Prepare S2 quarterly mosaic for {quarter_label(ref_dt)}")
    sh_process_to_tiff(
        token=token,
        collection_id=S2_QUARTERLY_BYOC,
        grid=grid,
        evalscript=S2_QUARTERLY_EVALSCRIPT,
        time_from=time_from,
        time_to=time_to,
        out_path=raw_path,
    )

    out_paths = {band: MOSAIC_DIR / f"{job.name}__{pass_key}__{band}.tif" for band in base.S2_BANDS_11}
    out_paths["SCL"] = MOSAIC_DIR / f"{job.name}__{pass_key}__SCL.tif"
    if all(base.file_is_ready(p) for p in out_paths.values()):
        log(f"  Reuse S2 quarterly derived cache: {pass_key}")
        return out_paths

    stage("Split S2 quarterly mosaic into per-band rasters")
    with rasterio.open(raw_path) as ds:
        band_arrays = [ds.read(i).astype(np.float32) for i in range(1, 5)]
        observations = ds.read(5).astype(np.float32)
        data_mask = ds.read(6).astype(np.float32)

    invalid = (observations <= 0) | (data_mask <= 0)
    for band_name, arr in zip(base.S2_BANDS_11, band_arrays):
        band_arr = arr.copy()
        band_arr[invalid] = 0.0
        write_single_band(out_paths[band_name], grid, band_arr.astype(np.float32), "float32", 0.0)

    # Synthetic SCL: 4 means clear vegetation/land, 9 means invalid/no data.
    scl = np.where(invalid, 9, 4).astype(np.uint8)
    write_single_band(out_paths["SCL"], grid, scl, "uint8", 0)
    return out_paths


def build_s1_monthly_mosaic(job: base.Job, grid: Dict[str, Any], token: str, ref_dt: datetime, pass_key: str) -> Path:
    raw_path = MOSAIC_DIR / f"{job.name}__{pass_key}__S1_monthly_raw.tif"
    out_path = MOSAIC_DIR / f"{job.name}__{pass_key}__S1_on_S2.tif"
    time_from, time_to = exact_day_range(job.date_start)
    stage(f"Prepare S1 monthly mosaic for {ref_dt.strftime('%Y-%m')}")
    sh_process_to_tiff(
        token=token,
        collection_id=S1_MONTHLY_BYOC,
        grid=grid,
        evalscript=S1_MONTHLY_EVALSCRIPT,
        time_from=time_from,
        time_to=time_to,
        out_path=raw_path,
    )
    if base.file_is_ready(out_path):
        log(f"  Reuse S1 monthly cache: {out_path.name}")
        return out_path

    stage("Convert S1 monthly mosaic to VV/VH dB raster")
    with rasterio.open(raw_path) as ds:
        vv = ds.read(1).astype(np.float32)
        vh = ds.read(2).astype(np.float32)
        data_mask = ds.read(3).astype(np.float32)
        profile = ds.profile.copy()

    valid = (data_mask > 0) & np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
    out = np.full((2, vv.shape[0], vv.shape[1]), np.nan, dtype=np.float32)
    out[0, valid] = 10.0 * np.log10(vv[valid])
    out[1, valid] = 10.0 * np.log10(vh[valid])

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
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(out)
    return out_path


def build_npz_for_job(
    job: base.Job,
    token: str,
    aerial_catalog: List[base.CatalogRecord],
    df_session: requests.Session,
    df_api_key: str,
) -> str:
    ref_dt = parse_job_datetime(job.date_start)
    target_crs = base.utm_epsg_from_bbox(job.bbox_lonlat)
    grid = base.aligned_aoi_grid(job.bbox_lonlat, target_crs, resolution=10.0)
    pass_key = pass_key_for_date(ref_dt)

    log(f"  Target CRS: {target_crs}")
    stage("Build S2 mosaic")
    s2_mosaic = build_s2_quarterly_mosaic(job, grid, token, ref_dt)
    stage("Build S1 mosaic")
    s1_on_s2 = build_s1_monthly_mosaic(job, grid, token, ref_dt, pass_key)

    stage("Scan valid patch windows")
    scan = base.scan_valid_patch_indices(
        b02_path=s2_mosaic["B02"],
        scl_path=s2_mosaic["SCL"],
        s1_path=s1_on_s2,
        tile=job.tile,
        stride=job.stride,
        min_valid_frac=job.min_valid_frac,
    )
    log(f"  valid threshold: min_valid_frac={job.min_valid_frac:.2f}")
    log(
        "  S2 prefilter: "
        f"candidate_windows={scan['total_windows']} "
        f"scanned={scan['total_scanned']} "
        f"prefilter_nonzero={scan['prefilter_nonzero']}"
    )

    with rasterio.open(s2_mosaic["B02"]) as b02_ds:
        mosaic_shape = (b02_ds.height, b02_ds.width)
        mosaic_bounds = tuple(float(x) for x in b02_ds.bounds)
        mosaic_crs = str(b02_ds.crs)

    base_name = f"{job.name}__{pass_key}__S1monthly__S2quarterly"
    meta = {
        "job": job.name,
        "s2_pass": pass_key,
        "s2_tiles": [f"S2 Quarterly Mosaic {quarter_label(ref_dt)}"],
        "s1_scenes": [f"S1 IW Monthly Mosaic {ref_dt.strftime('%Y-%m')}"],
        "s1_source_type": "Sentinel-1 IW Monthly Mosaics (Sentinel Hub BYOC)",
        "s2_source_type": "Sentinel-2 Quarterly Mosaics (Sentinel Hub BYOC)",
        "s2_bands": base.S2_BANDS_11,
        "tile": int(job.tile),
        "stride": int(job.stride),
        "min_valid_frac": float(job.min_valid_frac),
        "scl_invalid": sorted(base.SCL_INVALID),
        "target_crs": mosaic_crs,
        "mosaic_shape_hw": [int(mosaic_shape[0]), int(mosaic_shape[1])],
        "mosaic_bounds": list(mosaic_bounds),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    staging_patch_dir = TMP_DIR / "pending_npz" / base_name
    stage("Write patch NPZ files")
    patch_paths = base.write_npz_per_patch(
        out_dir=OUT_DIR,
        base_name=base_name,
        s2_band_paths=s2_mosaic,
        scl_path=s2_mosaic["SCL"],
        s1_path=s1_on_s2,
        indices=scan["indices"],
        tile=job.tile,
        meta_base=meta,
        require_aerial=True,
        staging_out_dir=staging_patch_dir,
    )
    stage("Enrich patch NPZ files with aerial imagery")
    kept, dropped_aerial = base.enrich_pass_patch_files_with_aerial(
        job=job,
        s2_pass_key=pass_key,
        s2_b02_path=s2_mosaic["B02"],
        patch_paths=patch_paths,
        catalog=aerial_catalog,
        df_session=df_session,
        df_api_key=df_api_key,
        final_out_dir=OUT_DIR,
    )
    log(
        f"  Wrote patch NPZ files: {kept} | "
        f"windows={scan['total_scanned']} skipped_valid={scan['skipped_valid']} "
        f"dropped_aerial={dropped_aerial}"
    )
    return token


def run_batch() -> None:
    stage("Authenticate against Sentinel Hub")
    token = sh_token()

    for job in JOBS:
        apply_bbox_cache_suffix(job)
        apply_job_concurrency_caps(job)
        if job.output_mode != "per_patch":
            raise ValueError(
                f"JOB {job.name}: pipeline5 requires output_mode='per_patch' "
                f"(got '{job.output_mode}')"
            )
        if job.date_start != job.date_end:
            log(
                f"JOB {job.name}: using exact mosaic date {job.date_start}; "
                f"date_end={job.date_end} is ignored for this variant"
            )

        df_api_key = base.datafordeler_api_key()
        df_session = base.build_retry_session()
        stage("Load or build aerial catalog")
        aerial_catalog = base.build_aerial_catalog(
            session=df_session,
            api_key=df_api_key,
            datasets=base.AERIAL_DATASET_PRIORITY,
            rebuild=base.AERIAL_REBUILD_INDEX,
        )
        if not aerial_catalog:
            raise RuntimeError("Aerial catalog is empty; cannot enrich with aerial")

        log(
            f"\n=== JOB {job.name} | date={job.date_start} "
            f"bbox={job.bbox_lonlat} ==="
        )
        log(
            "Runtime tuning: "
            f"cpu={base.CPU_COUNT} "
            f"reproject_threads={base.REPROJECT_THREADS} "
            f"aerial_download_workers={max(1, int(job.aerial_workers))} "
            f"aerial_chunk_workers={max(1, int(job.aerial_chunk_workers))} "
            f"aerial_download_attempts={base.AERIAL_DOWNLOAD_ATTEMPTS} "
            f"aerial_retry_base_s={base.AERIAL_DOWNLOAD_RETRY_BASE_SECONDS}"
        )
        stage("Run mosaic pipeline for job")
        build_npz_for_job(
            job=job,
            token=token,
            aerial_catalog=aerial_catalog,
            df_session=df_session,
            df_api_key=df_api_key,
        )


if __name__ == "__main__":
    t0 = perf_counter()
    try:
        run_batch()
    except Exception:
        log(f"[stage] Total runtime before failure: {format_duration(perf_counter() - t0)}")
        raise
    log(f"[stage] Total runtime: {format_duration(perf_counter() - t0)}")
