#!/usr/bin/env python3
"""
Pipeline5 variant that swaps single-scene Sentinel-1 GRD inputs for
Sentinel-1 IW Monthly Mosaics from CDSE GLOBAL-MOSAICS.

This keeps the existing Sentinel-2 stitching and aerial enrichment flow,
but replaces the S1 search/download/processing path with monthly mosaic
tiles that are already terrain-corrected. SNAP is therefore not used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject


THIS_DIR = Path(__file__).resolve().parent
PARENT_PIPELINE = THIS_DIR.parent / "pipeline5_aerial.py"

_SPEC = importlib.util.spec_from_file_location("pipeline5_aerial_base", PARENT_PIPELINE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load base pipeline from {PARENT_PIPELINE}")
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)


# -----------------------------
# Local paths / config
# -----------------------------
BASE_DIR = THIS_DIR
DATA_DIR = BASE_DIR / "data"
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

# Keep compatibility with the parent `.env`, but allow an optional local one too.
base.load_dotenv(THIS_DIR.parent / ".env", override=False)
base.load_dotenv(THIS_DIR / ".env", override=False)

CPU_LIMIT: Optional[int] = None  # e.g. 4 to keep the process on 4 cores
COORDINATE1: Optional[Tuple[float, float]] = None  # (lat, lon)
COORDINATE2: Optional[Tuple[float, float]] = None  # (lat, lon)


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


def bbox_from_latlon_corners(
    coordinate1: Tuple[float, float],
    coordinate2: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    lat1, lon1 = float(coordinate1[0]), float(coordinate1[1])
    lat2, lon2 = float(coordinate2[0]), float(coordinate2[1])
    return (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))


def resolve_bbox(default_bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    if COORDINATE1 is None or COORDINATE2 is None:
        return default_bbox
    return bbox_from_latlon_corners(COORDINATE1, COORDINATE2)


def bbox_cache_tag(bbox: Tuple[float, float, float, float]) -> str:
    payload = ",".join(f"{float(v):.8f}" for v in bbox)
    return hashlib.sha1(payload.encode("ascii")).hexdigest()[:10]


def apply_bbox_cache_suffix(job: base.Job) -> None:
    suffix = f"__bbox_{bbox_cache_tag(job.bbox_lonlat)}"
    if suffix not in job.name:
        job.name = f"{job.name}{suffix}"


apply_cpu_limit()

MONTHLY_COLLECTION = "GLOBAL-MOSAICS"
MONTHLY_S3_HINT = "/Global-Mosaics/Sentinel-1/S1SAR_L3_IW_MCM/"
MONTHLY_NAME_HINT = "S1SAR_L3_IW_MCM"
MONTHLY_TOP = 200
RASTER_EXTS = (".tif", ".tiff", ".img")

JOBS: List[base.Job] = [
    base.Job(
        name="larger_test_iw_monthly",
        bbox_lonlat=resolve_bbox((8.8, 55.7, 9.5, 56.2)),
        date_start="2025-06-12",
        date_end="2025-06-12",
        max_s2=200,
        max_cloud=10.0,
        max_time_diff_hours=36,
        max_s1_scenes=24,
        tile=128,
        stride=128,
        min_valid_frac=0.9,
        output_mode="per_patch",
        aerial_target_res_m=1.0,
        aerial_min_cov_frac=1.0,
        aerial_download_batch_gb=4.0,
        aerial_workers=32,
        aerial_chunk_workers=24,
    ),
]


def log(msg: str) -> None:
    base.log(msg)


def stage(msg: str) -> None:
    log(f"[stage] {msg}")


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


def month_bounds_utc(dt: datetime) -> Tuple[datetime, datetime]:
    dt_utc = dt.astimezone(timezone.utc)
    start = dt_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


def format_odata_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def odata_search_s1_iw_monthly_mosaics(
    token: str,
    bbox_lonlat: Tuple[float, float, float, float],
    target_dt: datetime,
    top: int = MONTHLY_TOP,
) -> Tuple[List[Dict[str, Any]], str]:
    month_start, month_end = month_bounds_utc(target_dt)
    wkt = base.bbox_to_wkt(bbox_lonlat)
    stage(f"Search Sentinel-1 IW monthly mosaics for {month_start.strftime('%Y-%m')}")
    filt = " and ".join(
        [
            f"Collection/Name eq '{MONTHLY_COLLECTION}'",
            f"ContentDate/Start gt {format_odata_dt(month_start)}",
            f"ContentDate/Start lt {format_odata_dt(month_end)}",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
        ]
    )
    target = max(1, int(top))
    skip = 0
    products: List[Dict[str, Any]] = []
    cur_token = token
    pages_without_match = 0

    # S3Path is present in results, but using contains(S3Path, ...) in the
    # server-side filter currently triggers HTTP 400 for GLOBAL-MOSAICS.
    # Query broadly, then filter the monthly IW path client-side.
    while len(products) < target:
        page_top = min(base.ODATA_TOP_MAX, max(target * 2, 1000))
        resp, cur_token = base.odata_get(
            f"{base.ODATA_ROOT}/Products",
            cur_token,
            params={
                "$filter": filt,
                "$top": str(page_top),
                "$skip": str(skip),
                "$orderby": "ContentDate/Start desc",
            },
            timeout=120,
            retries=1,
        )
        vals = resp.json().get("value", [])
        if not vals:
            break

        keep_before = len(products)
        for prod in vals:
            s3_path = str(prod.get("S3Path", ""))
            name = str(prod.get("Name", ""))
            if MONTHLY_S3_HINT in s3_path or MONTHLY_NAME_HINT in name:
                products.append(prod)
                if len(products) >= target:
                    break

        skip += len(vals)
        if len(products) == keep_before:
            pages_without_match += 1
        else:
            pages_without_match = 0
        if len(vals) < page_top or pages_without_match >= 5 or skip >= 10_000:
            break

    return products[:target], cur_token


def pick_s1_monthly_tiles(
    candidates: List[Dict[str, Any]],
    target_geom: Any,
    max_tiles: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    ranked: List[Tuple[float, str, Dict[str, Any]]] = []
    for prod in candidates:
        name = str(prod.get("Name", ""))
        s3_path = str(prod.get("S3Path", ""))
        if MONTHLY_NAME_HINT not in name and MONTHLY_NAME_HINT not in s3_path:
            continue
        geom = base.parse_product_geometry(prod)
        overlap = base.overlap_ratio(geom, target_geom)
        if geom is not None and overlap <= 0.0:
            continue
        prod_copy = dict(prod)
        prod_copy["_s1_overlap_ratio"] = float(overlap)
        ranked.append((-float(overlap), name, prod_copy))

    ranked.sort(key=lambda x: (x[0], x[1]))
    selected = [prod for _, _, prod in ranked[: max(1, int(max_tiles))]]
    stats = {
        "raw": len(candidates),
        "within_time": len(ranked),
        "deduped": len(selected),
    }
    return selected, stats


def _header_filename(headers: requests.structures.CaseInsensitiveDict[str]) -> Optional[str]:
    raw = headers.get("Content-Disposition")
    if not raw:
        return None
    msg = Message()
    msg["content-disposition"] = raw
    filename = msg.get_param("filename", header="content-disposition")
    if not filename:
        return None
    return os.path.basename(filename.strip().strip("\""))


def _guess_ext(content_type: str, fallback_name: str) -> str:
    ct = (content_type or "").lower()
    if "tiff" in ct:
        return ".tif"
    if "zip" in ct:
        return ".zip"
    if fallback_name.endswith(".SAFE"):
        return ".zip"
    return ""


def _candidate_name(product: Dict[str, Any], headers: requests.structures.CaseInsensitiveDict[str], s3_hint: str) -> str:
    from_header = _header_filename(headers)
    if from_header:
        return from_header

    hint_name = os.path.basename((s3_hint or "").rstrip("/"))
    if hint_name and Path(hint_name).suffix:
        return hint_name

    base_name = sanitize_name(str(product.get("Name", product.get("Id", "product"))))
    ext = _guess_ext(headers.get("Content-Type", ""), base_name)
    return f"{base_name}{ext}"


def _stream_cdse_download(url: str, token: str) -> requests.Response:
    headers = {"Authorization": f"Bearer {token}"}
    r1 = requests.get(url, headers=headers, stream=True, timeout=300, allow_redirects=False)
    if r1.status_code in (301, 302, 303, 307, 308):
        loc = r1.headers.get("Location")
        if not loc:
            r1.raise_for_status()
        r1.close()
        return requests.get(loc, headers=headers, stream=True, timeout=300)
    return r1


def _download_single_cdse_file(
    token: str,
    product: Dict[str, Any],
    url: str,
    out_dir: Path,
    s3_hint: str = "",
) -> Tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    last_exc: Optional[Exception] = None
    current_token = token
    tmp = out_dir / f"{sanitize_name(str(product.get('Id', product.get('Name', 'product'))))}.part"

    for attempt in range(1, base.CDSE_DOWNLOAD_ATTEMPTS + 1):
        try:
            tmp.unlink(missing_ok=True)
            with _stream_cdse_download(url, current_token) as resp:
                if resp.status_code in (401, 403):
                    raise requests.HTTPError(response=resp)
                resp.raise_for_status()
                out_name = _candidate_name(product, resp.headers, s3_hint)
                out_path = out_dir / out_name
                if base.file_is_ready(out_path, min_bytes=1024 * 1024):
                    return out_path, current_token
                with tmp.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out_path)
            return out_path, current_token
        except requests.HTTPError as exc:
            last_exc = exc
            code = getattr(exc.response, "status_code", None)
            if code in (401, 403) and attempt < base.CDSE_DOWNLOAD_ATTEMPTS:
                current_token = base.cdse_token()
                continue
            if code == 429 and attempt < base.CDSE_DOWNLOAD_ATTEMPTS:
                wait_s = max(1, base.CDSE_DOWNLOAD_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
                sleep(float(wait_s))
                continue
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < base.CDSE_DOWNLOAD_ATTEMPTS:
                wait_s = max(1, base.CDSE_DOWNLOAD_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
                sleep(float(wait_s))
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to download monthly mosaic product {product.get('Name', product.get('Id', 'unknown'))}")


def _iter_raster_asset_links(product: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    for asset in product.get("Assets") or []:
        if not isinstance(asset, dict):
            continue
        link = asset.get("DownloadLink")
        s3_path = str(asset.get("S3Path", ""))
        title = str(asset.get("Name", asset.get("Id", "")))
        type_hint = str(asset.get("Type", ""))
        token = f"{title} {s3_path} {type_hint}".lower()
        if "quicklook" in token or "thumbnail" in token:
            continue
        if Path(s3_path).suffix.lower() in RASTER_EXTS or Path(title).suffix.lower() in RASTER_EXTS:
            if link:
                yield str(link), s3_path


def download_s1_monthly_products_parallel(
    products: Sequence[Dict[str, Any]],
    token: str,
) -> Tuple[List[List[Path]], str]:
    slots: List[Optional[List[Path]]] = [None] * len(products)
    if not products:
        return [], token

    workers = max(1, min(base.CDSE_DOWNLOAD_WORKERS, len(products)))
    stage(f"Download {len(products)} S1 monthly mosaic product(s)")
    log(f"  CDSE monthly download tuning: workers={workers} (auto={base.AUTO_CDSE_DOWNLOAD_WORKERS})")

    def _job(idx: int, prod: Dict[str, Any], cur_token: str) -> Tuple[int, List[Path], str]:
        asset_links = list(_iter_raster_asset_links(prod))
        downloaded: List[Path] = []
        out_dir = DOWNLOAD_DIR / sanitize_name(str(prod.get("Name", prod.get("Id", f"prod_{idx}"))))
        token_out = cur_token
        if asset_links:
            for link, s3_hint in asset_links:
                path, token_out = _download_single_cdse_file(token_out, prod, link, out_dir=out_dir, s3_hint=s3_hint)
                downloaded.append(path)
        else:
            url = f"{base.ODATA_ROOT}/Products({prod['Id']})/$value"
            path, token_out = _download_single_cdse_file(token_out, prod, url, out_dir=DOWNLOAD_DIR, s3_hint=str(prod.get("S3Path", "")))
            downloaded.append(path)
        return idx, downloaded, token_out

    if workers == 1:
        out_token = token
        for idx, prod in enumerate(products):
            slot_idx, paths, out_token = _job(idx, prod, out_token)
            slots[slot_idx] = paths
        return [p for p in slots if p is not None], out_token

    out_token = token
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, prod in enumerate(products):
            fut = pool.submit(_job, idx, prod, token)
            futures[fut] = (idx, str(prod.get("Name", "")))

        done = 0
        for fut in as_completed(futures):
            idx, name = futures[fut]
            try:
                slot_idx, paths, token_out = fut.result()
            except Exception as exc:
                raise RuntimeError(f"CDSE monthly mosaic download failed for {name}") from exc
            slots[slot_idx] = paths
            if token_out:
                out_token = token_out
            done += 1
            if base.is_progress_tick(done, len(futures), 5):
                log(f"  CDSE monthly download progress: {done}/{len(futures)}")

    if any(v is None for v in slots):
        raise RuntimeError("Missing monthly mosaic download result(s)")
    return [p for p in slots if p is not None], out_token


def _extract_zip_if_needed(path: Path, extract_root: Path) -> List[Path]:
    if path.is_dir():
        return [path]
    if not zipfile.is_zipfile(path):
        return [path]
    target_dir = extract_root / path.stem
    marker = target_dir / ".done"
    if not marker.exists():
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target_dir)
        marker.write_text("ok", encoding="ascii")
    return [target_dir]


def _collect_raster_paths(paths: Sequence[Path]) -> List[Path]:
    rasters: List[Path] = []
    for p in paths:
        if p.is_dir():
            for ext in RASTER_EXTS:
                rasters.extend(sorted(p.rglob(f"*{ext}")))
                rasters.extend(sorted(p.rglob(f"*{ext.upper()}")))
            continue
        if p.suffix.lower() in RASTER_EXTS:
            rasters.append(p)
    dedup: Dict[Path, None] = {}
    for p in rasters:
        dedup[p] = None
    return list(dedup.keys())


def _band_scale_offset(ds: rasterio.DatasetReader, band_index: int) -> Tuple[float, float]:
    scale = 1.0
    offset = 0.0
    if ds.scales and len(ds.scales) >= band_index and ds.scales[band_index - 1] not in (None, 0):
        scale = float(ds.scales[band_index - 1])
    if ds.offsets and len(ds.offsets) >= band_index and ds.offsets[band_index - 1] is not None:
        offset = float(ds.offsets[band_index - 1])
    return scale, offset


def _pick_monthly_raster_sources(downloaded_paths: Sequence[Path], product: Dict[str, Any]) -> Tuple[Path, int, Path, int]:
    expanded: List[Path] = []
    extract_root = RAW_DIR / sanitize_name(str(product.get("Name", product.get("Id", "monthly_product"))))
    for p in downloaded_paths:
        expanded.extend(_extract_zip_if_needed(p, extract_root))
    rasters = _collect_raster_paths(expanded)
    if not rasters:
        raise FileNotFoundError(f"No raster asset found for monthly mosaic product {product.get('Name', product.get('Id', 'unknown'))}")

    multi_band: List[Tuple[Path, int]] = []
    vv_candidates: List[Path] = []
    vh_candidates: List[Path] = []
    for raster_path in rasters:
        lname = raster_path.name.lower()
        if "vv" in lname:
            vv_candidates.append(raster_path)
        if "vh" in lname:
            vh_candidates.append(raster_path)
        try:
            with rasterio.open(raster_path) as ds:
                if ds.count >= 2:
                    multi_band.append((raster_path, ds.count))
                    descs = [str(d or "").upper() for d in ds.descriptions]
                    if len(descs) >= 2 and "VV" in descs[0] and "VH" in descs[1]:
                        return raster_path, 1, raster_path, 2
        except Exception:
            continue

    if multi_band:
        return multi_band[0][0], 1, multi_band[0][0], 2
    if vv_candidates and vh_candidates:
        return sorted(vv_candidates)[0], 1, sorted(vh_candidates)[0], 1
    if len(rasters) == 2:
        ordered = sorted(rasters)
        return ordered[0], 1, ordered[1], 1

    raise RuntimeError(
        "Could not resolve VV/VH raster inputs for monthly mosaic product "
        f"{product.get('Name', product.get('Id', 'unknown'))}: {[p.name for p in rasters]}"
    )


def warp_monthly_raster_pair_to_grid_db(
    vv_path: Path,
    vv_band: int,
    vh_path: Path,
    vh_band: int,
    ref_grid_path: Path,
    out_path: Path,
) -> Path:
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
                    source=rasterio.band(vv, vv_band),
                    destination=rasterio.band(dst, 1),
                    src_transform=vv.transform,
                    src_crs=vv.crs,
                    src_nodata=(vv.nodata if vv.nodata is not None else 0),
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                    init_dest_nodata=True,
                    num_threads=base.REPROJECT_THREADS,
                )
                vv_scale, vv_offset = _band_scale_offset(vv, vv_band)
            with rasterio.open(vh_path) as vh:
                reproject(
                    source=rasterio.band(vh, vh_band),
                    destination=rasterio.band(dst, 2),
                    src_transform=vh.transform,
                    src_crs=vh.crs,
                    src_nodata=(vh.nodata if vh.nodata is not None else 0),
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                    init_dest_nodata=True,
                    num_threads=base.REPROJECT_THREADS,
                )
                vh_scale, vh_offset = _band_scale_offset(vh, vh_band)

    with rasterio.open(out_path, "r+") as ds:
        for _, window in ds.block_windows(1):
            block = ds.read([1, 2], window=window).astype(np.float32)
            finite = np.isfinite(block)
            scaled = np.full_like(block, np.nan, dtype=np.float32)
            scaled[0] = np.where(finite[0], block[0] * vv_scale + vv_offset, np.nan)
            scaled[1] = np.where(finite[1], block[1] * vh_scale + vh_offset, np.nan)
            scaled = np.where(np.isfinite(scaled), np.maximum(scaled, 1e-10), np.nan)
            scaled = 10.0 * np.log10(scaled)
            ds.write(scaled.astype(np.float32), window=window)

    return out_path


def process_and_warp_single_monthly_tile(
    job: base.Job,
    s2_pass_key: str,
    s2_ref_grid_path: Path,
    product: Dict[str, Any],
    downloaded_paths: Sequence[Path],
) -> Path:
    scene_tag = sanitize_name(str(product["Name"]))
    warped = MOSAIC_DIR / f"{job.name}__{s2_pass_key}__{scene_tag}__S1_on_S2_src.tif"
    if base.file_is_ready(warped):
        log(f"  Reuse S1 monthly warped cache: {warped.name}")
        return warped

    vv_path, vv_band, vh_path, vh_band = _pick_monthly_raster_sources(downloaded_paths, product)
    warp_monthly_raster_pair_to_grid_db(vv_path, vv_band, vh_path, vh_band, s2_ref_grid_path, warped)
    return warped


def build_stitched_npz_for_pass_monthly(
    job: base.Job,
    s2_pass_key: str,
    s2_products: List[Dict[str, Any]],
    s1_monthly_products: List[Dict[str, Any]],
    token: str,
    s2_target_geom: Optional[Any] = None,
    aerial_catalog: Optional[List[base.CatalogRecord]] = None,
    df_session: Optional[requests.Session] = None,
    df_api_key: str = "",
) -> str:
    stage("Download S2 tiles")
    s2_zips, _unused_s1, token = base.download_s1_s2_parallel(
        s2_products=s2_products,
        s1_products=[],
        token=token,
    )
    stage("Download S1 monthly mosaic products")
    s1_downloads, token = download_s1_monthly_products_parallel(s1_monthly_products, token)

    target_crs = base.utm_epsg_from_bbox(job.bbox_lonlat)
    log(f"  Target CRS: {target_crs}")

    stage("Build stitched S2 mosaic")
    s2_mosaic = base.build_s2_mosaics(
        job,
        s2_pass_key,
        s2_zips,
        target_crs,
        s2_target_geom=s2_target_geom,
    )

    s1_workers = max(1, min(len(s1_monthly_products), base.S1_SCENE_WORKERS))
    log(
        "  S1 monthly processing tuning: "
        f"workers={s1_workers} (auto={base.AUTO_S1_WORKERS}) "
        f"reproject_threads={base.REPROJECT_THREADS}"
    )
    stage("Warp S1 monthly mosaics onto S2 grid")
    warped_slots: List[Optional[Path]] = [None] * len(s1_monthly_products)
    if s1_workers == 1:
        for idx, (product, paths) in enumerate(zip(s1_monthly_products, s1_downloads)):
            warped_slots[idx] = process_and_warp_single_monthly_tile(
                job=job,
                s2_pass_key=s2_pass_key,
                s2_ref_grid_path=s2_mosaic["B02"],
                product=product,
                downloaded_paths=paths,
            )
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=s1_workers) as pool:
            for idx, (product, paths) in enumerate(zip(s1_monthly_products, s1_downloads)):
                fut = pool.submit(
                    process_and_warp_single_monthly_tile,
                    job,
                    s2_pass_key,
                    s2_mosaic["B02"],
                    product,
                    paths,
                )
                futures[fut] = idx

            done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                warped_slots[idx] = fut.result()
                done += 1
                if base.is_progress_tick(done, len(futures), 2):
                    log(f"  S1 monthly processing progress: {done}/{len(futures)}")

    if any(p is None for p in warped_slots):
        raise RuntimeError("Failed to build all monthly mosaic warped sources")
    warped_s1_paths = [p for p in warped_slots if p is not None]

    month_tag = month_bounds_utc(base.parse_dt(s2_products[0]))[0].strftime("%Y%m")
    s1_merge_tag = f"S1_IW_MONTHLY_{month_tag}_tiles{len(s1_monthly_products)}"
    s1_on_s2 = MOSAIC_DIR / f"{job.name}__{s2_pass_key}__{s1_merge_tag}__S1_on_S2.tif"
    if base.file_is_ready(s1_on_s2):
        log(f"  Reuse S1 monthly merged cache: {s1_on_s2.name}")
    else:
        stage("Merge warped S1 monthly mosaics")
        base.merge_warped_s1_scenes(warped_s1_paths, s1_on_s2)

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
    if scan["skipped_valid"] > 0:
        log(
            "  skipped_valid breakdown: "
            f"s1={scan['fail_s1ok']} "
            f"nonzero={scan['fail_nonzero']} "
            f"clear={scan['fail_clear']}"
        )

    indices = scan["indices"]
    with rasterio.open(s2_mosaic["B02"]) as b02_ds:
        mosaic_shape = (b02_ds.height, b02_ds.width)
        mosaic_bounds = tuple(float(x) for x in b02_ds.bounds)
        mosaic_crs = str(b02_ds.crs)

    month_start = month_bounds_utc(base.parse_dt(s2_products[0]))[0]
    base_name = f"{job.name}__{s2_pass_key}__{s1_merge_tag}"
    meta = {
        "job": job.name,
        "s2_pass": s2_pass_key,
        "s2_tiles": [p["Name"] for p in s2_products],
        "s1_scenes": [p["Name"] for p in s1_monthly_products],
        "s1_source_type": "Sentinel-1 IW Monthly Mosaics",
        "s1_mosaic_month": month_start.strftime("%Y-%m"),
        "tile": int(job.tile),
        "stride": int(job.stride),
        "min_valid_frac": float(job.min_valid_frac),
        "s2_bands": base.S2_BANDS_11,
        "scl_invalid": sorted(base.SCL_INVALID),
        "target_crs": mosaic_crs,
        "mosaic_shape_hw": [int(mosaic_shape[0]), int(mosaic_shape[1])],
        "mosaic_bounds": list(mosaic_bounds),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if job.output_mode == "per_pass":
        out_path = OUT_DIR / f"{base_name}.npz"
        base.write_npz_single_file(
            out_path=out_path,
            s2_band_paths=s2_mosaic,
            scl_path=s2_mosaic["SCL"],
            s1_path=s1_on_s2,
            indices=indices,
            tile=job.tile,
            meta=meta,
        )
        size_mb = out_path.stat().st_size / (1024 * 1024)
        log(
            f"  Wrote NPZ: {out_path.name} | "
            f"patches={len(indices)} / windows={scan['total_scanned']} | "
            f"skipped_valid={scan['skipped_valid']} | "
            f"size={size_mb:.1f} MB"
        )
    else:
        staging_patch_dir = TMP_DIR / "pending_npz" / base_name
        stage("Write patch NPZ files")
        patch_paths = base.write_npz_per_patch(
            out_dir=OUT_DIR,
            base_name=base_name,
            s2_band_paths=s2_mosaic,
            scl_path=s2_mosaic["SCL"],
            s1_path=s1_on_s2,
            indices=indices,
            tile=job.tile,
            meta_base=meta,
            require_aerial=True,
            staging_out_dir=staging_patch_dir,
        )
        if (not aerial_catalog) or (df_session is None) or (not df_api_key):
            raise RuntimeError("Aerial catalog/session/api key missing for mandatory aerial enrichment")
        stage("Enrich patch NPZ files with aerial imagery")
        kept, dropped_aerial = base.enrich_pass_patch_files_with_aerial(
            job=job,
            s2_pass_key=s2_pass_key,
            s2_b02_path=s2_mosaic["B02"],
            patch_paths=patch_paths,
            catalog=aerial_catalog,
            df_session=df_session,
            df_api_key=df_api_key,
            final_out_dir=OUT_DIR,
        )
        shutil.rmtree(staging_patch_dir, ignore_errors=True)
        log(
            f"  Wrote patch NPZ files: {kept} | "
            f"windows={scan['total_scanned']} skipped_valid={scan['skipped_valid']} "
            f"dropped_aerial={dropped_aerial}"
        )

    return token


def run_batch() -> None:
    stage("Authenticate against CDSE")
    token = base.cdse_token()

    for job in JOBS:
        apply_bbox_cache_suffix(job)
        apply_job_concurrency_caps(job)
        if job.output_mode != "per_patch":
            raise ValueError(
                f"JOB {job.name}: pipeline5 requires output_mode='per_patch' "
                f"(got '{job.output_mode}')"
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
            f"\n=== JOB {job.name} | dates={job.date_start}..{job.date_end} "
            f"bbox={job.bbox_lonlat} ==="
        )
        log(
            "Runtime tuning: "
            f"cpu={base.CPU_COUNT} "
            f"reproject_threads={base.REPROJECT_THREADS} "
            f"s1_workers={base.S1_SCENE_WORKERS} "
            f"cdse_download_workers={base.CDSE_DOWNLOAD_WORKERS} "
            f"cdse_download_attempts={base.CDSE_DOWNLOAD_ATTEMPTS} "
            f"cdse_retry_base_s={base.CDSE_DOWNLOAD_RETRY_BASE_SECONDS} "
            f"aerial_download_workers={max(1, int(job.aerial_workers))} "
            f"aerial_chunk_workers={max(1, int(job.aerial_chunk_workers))} "
            f"aerial_download_attempts={base.AERIAL_DOWNLOAD_ATTEMPTS} "
            f"aerial_retry_base_s={base.AERIAL_DOWNLOAD_RETRY_BASE_SECONDS}"
        )

        s2_products, token = base.odata_search_s2(
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

        stage("Group S2 products into passes")
        grouped = base.group_s2_by_pass(s2_products)
        if not grouped:
            log("No S2 pass groups created from search results.")
            continue

        ordered_groups = sorted(grouped.items(), key=lambda kv: base.parse_dt(kv[1][0]), reverse=True)
        log(f"Found {len(s2_products)} S2 products -> {len(ordered_groups)} pass group(s)")

        for s2_pass_key, pass_products in ordered_groups:
            s2_center = base.parse_dt(pass_products[0])

            s2_target_geom = base.build_s2_pass_target_geometry(pass_products, job.bbox_lonlat)
            s1_cands, token = odata_search_s1_iw_monthly_mosaics(
                token=token,
                bbox_lonlat=job.bbox_lonlat,
                target_dt=s2_center,
                top=max(MONTHLY_TOP, job.max_s1_scenes * 4),
            )
            s1_selected, s1_stats = pick_s1_monthly_tiles(
                candidates=s1_cands,
                target_geom=s2_target_geom,
                max_tiles=job.max_s1_scenes,
            )
            if not s1_selected:
                log(f"Skip pass {s2_pass_key}: no Sentinel-1 IW monthly mosaic tile overlaps {s2_center.strftime('%Y-%m')}")
                continue

            overlaps = [float(p.get("_s1_overlap_ratio", 0.0)) for p in s1_selected]
            ov_min, ov_max = min(overlaps), max(overlaps)
            log(
                f"PASS {s2_pass_key} | S2 tiles={len(pass_products)} | "
                f"S1 monthly tiles={len(s1_selected)} | month={s2_center.strftime('%Y-%m')} | "
                f"ov_range={ov_min:.2f}-{ov_max:.2f} | "
                f"S1 raw/filtered/selected={s1_stats['raw']}/{s1_stats['within_time']}/{s1_stats['deduped']}"
            )

            stage(f"Build dataset for pass {s2_pass_key}")
            token = build_stitched_npz_for_pass_monthly(
                job=job,
                s2_pass_key=s2_pass_key,
                s2_products=pass_products,
                s1_monthly_products=s1_selected,
                token=token,
                s2_target_geom=s2_target_geom,
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
