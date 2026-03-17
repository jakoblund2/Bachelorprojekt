#!/usr/bin/env python3
"""
Stitch-first Sentinel-1/Sentinel-2 training dataset pipeline.

Workflow:
1) Search/download S2 products from CDSE and group by same acquisition pass.
2) Stitch all S2 tiles in each pass into one AOI-clipped mosaic grid.
3) Find best-overlap/nearest-time S1 scenes, process with SNAP terrain correction.
4) Warp S1 to stitched S2 grid.
5) Extract patches using `min_valid_frac` and write NPZ output (`per_patch` or `per_pass`).
"""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import rasterio
import requests
from affine import Affine
from dotenv import load_dotenv
from pyproj import Transformer
from rasterio.enums import ColorInterp, Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from requests.adapters import HTTPAdapter
from shapely import wkt as shapely_wkt
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform, unary_union
from urllib3.util.retry import Retry


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
AERIAL_INDEX_DIR = DATA_DIR / "aerial_index"
AERIAL_MOSAIC_DIR = DATA_DIR / "aerial_mosaic_1m"
AERIAL_TMP_DIR = DATA_DIR / "aerial_tmp"
AERIAL_CACHE_DIR = DATA_DIR / "aerial_cache_1m"

SNAP_GPT = Path.home() / "esa-snap" / "bin" / "gpt"
GRAPH_TC = BASE_DIR / "s1_grd_to_tc_dim.xml"

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_ROOT = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DF_BASE = "https://api.datafordeler.dk/FileDownloads"
DF_REGISTER = "GeoDKO"
DF_CATALOG_JSONL = AERIAL_INDEX_DIR / "geodko_catalog.jsonl"
DF_CATALOG_PARQUET = AERIAL_INDEX_DIR / "geodko_catalog.parquet"
ODATA_TOP_MAX = 999
DF_FALLBACK_BBOX_EPSG = 25832
AERIAL_MAX_FILES_PER_CHUNK = 64

# 6 is water
SCL_INVALID = {3, 8, 9, 10, 11}

# Keep current 11-band layout (B08 and B10 omitted).
S2_BANDS_11 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B09", "B11", "B12"]

S2_NAME_RE = re.compile(
    r"^(?P<prefix>S2[AB]_MSIL2A_(?P<sensing>\d{8}T\d{6})_(?P<baseline>N\d{4})_(?P<orbit>R\d{3}))_"
    r"(?P<tile>T\d{2}[A-Z]{3})_(?P<generation>\d{8}T\d{6})(?:_.+)?$"
)
S1_ACQ_KEY_RE = re.compile(
    r"^(S1[AB]_IW_GRDH_1SDV_\d{8}T\d{6}_\d{8}T\d{6}_[0-9A-Z]{6}_[0-9A-Z]{6})(?:_.+)?$"
)
S2_PASS_DT_RE = re.compile(r"^S2[AB]_MSIL2A_(\d{8}T\d{6})_")
GEODKO_TILE_RE = re.compile(r"^\d{4}_1km_(\d+)_(\d+)$")
YEAR_RE = re.compile(r"(19|20)\d{2}")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        v = int(raw)
        return v if v > 0 else int(default)
    except Exception:
        return int(default)


CPU_COUNT = max(1, int(os.cpu_count() or 1))
AUTO_REPROJECT_THREADS = max(1, min(8, CPU_COUNT // 2))
AUTO_S1_WORKERS = max(1, min(4, CPU_COUNT // 4))
AUTO_CDSE_DOWNLOAD_WORKERS = max(1, min(4, CPU_COUNT))
REPROJECT_THREADS = _env_int("PIPELINE5_REPROJECT_THREADS", AUTO_REPROJECT_THREADS)
S1_SCENE_WORKERS = _env_int("PIPELINE5_S1_WORKERS", AUTO_S1_WORKERS)
CDSE_DOWNLOAD_WORKERS = _env_int("PIPELINE5_CDSE_DOWNLOAD_WORKERS", AUTO_CDSE_DOWNLOAD_WORKERS)
CDSE_DOWNLOAD_ATTEMPTS = _env_int("PIPELINE5_CDSE_DOWNLOAD_ATTEMPTS", 5)
CDSE_DOWNLOAD_RETRY_BASE_SECONDS = _env_int("PIPELINE5_CDSE_DOWNLOAD_RETRY_BASE_SECONDS", 4)
AERIAL_DOWNLOAD_ATTEMPTS = _env_int("PIPELINE5_AERIAL_DOWNLOAD_ATTEMPTS", 3)
AERIAL_DOWNLOAD_RETRY_BASE_SECONDS = _env_int("PIPELINE5_AERIAL_DOWNLOAD_RETRY_BASE_SECONDS", 5)


# -----------------------------
# Config
# -----------------------------
AERIAL_DATE_POLICY = "nearest_year"
AERIAL_DATASET_PRIORITY: Tuple[str, ...] = ("GeoDKO12,5cm", "GeoDKO10cm")
AERIAL_REBUILD_INDEX = False
AERIAL_REBUILD_MOSAIC = False
AERIAL_MISSING_POLICY = "drop"


@dataclass
class Job:
    name: str
    bbox_lonlat: Tuple[float, float, float, float]  # minLon, minLat, maxLon, maxLat
    date_start: str
    date_end: str
    max_s2: int = 30
    max_cloud: float = 20.0
    max_time_diff_hours: int = 36
    max_s1_scenes: int = 10
    tile: int = 256
    stride: int = 256
    min_valid_frac: float = 0.9
    output_mode: str = "per_patch"  # "per_patch" or "per_pass"
    aerial_target_res_m: float = 1.0
    aerial_min_cov_frac: float = 0.98
    aerial_download_batch_gb: float = 4.0
    aerial_workers: int = 32  # concurrent aerial file download/downsample workers
    aerial_chunk_workers: int = 24  # concurrent chunk-mosaic build workers


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
    # ),
    Job(
        name="larger_test",
        bbox_lonlat=(10.00, 56.09, 10.23, 56.18),  # small AOI
        date_start="2025-04-01",
        date_end="2025-04-01",
        max_s2=200,
        max_cloud=10.0,
        max_time_diff_hours=36,
        max_s1_scenes=10,
        tile=128,
        stride=128,
        min_valid_frac=0.9,
        output_mode="per_patch",
        aerial_target_res_m=1.0,
        aerial_min_cov_frac=1,
        aerial_download_batch_gb=4.0,
        aerial_workers=32,
        aerial_chunk_workers=24,
    ),
]


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

load_dotenv(BASE_DIR / ".env")


# -----------------------------
# Helpers: logging / shell
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def is_progress_tick(done: int, total: int, every: int) -> bool:
    if total <= 0:
        return False
    if done >= total:
        return True
    return every > 0 and (done % every == 0)


def with_bigtiff(profile: Dict[str, Any], mode: str = "IF_SAFER") -> Dict[str, Any]:
    out = dict(profile)
    # Avoid classic TIFF 4 GiB limits on large mosaics.
    out["BIGTIFF"] = mode
    return out


def run(cmd: Iterable[str]) -> None:
    cmd = [str(x) for x in cmd]
    log("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def file_is_ready(path: Path, min_bytes: int = 4096) -> bool:
    try:
        return path.exists() and path.stat().st_size >= min_bytes
    except Exception:
        return False


def npz_has_aerial(path: Path) -> bool:
    if not file_is_ready(path):
        return False
    try:
        with np.load(path, allow_pickle=True) as d:
            if "aerial" not in d.files:
                return False
            arr = d["aerial"]
            return isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[0] == 4
    except Exception:
        return False


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

    req_top = max(1, int(top))
    query_top = min(req_top, ODATA_TOP_MAX)

    resp, token = odata_get(
        f"{ODATA_ROOT}/Products",
        token,
        params={"$filter": filt, "$top": str(query_top), "$orderby": "ContentDate/Start desc"},
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

    req_top = max(1, int(top))
    query_top = min(req_top, ODATA_TOP_MAX)

    resp, token = odata_get(
        f"{ODATA_ROOT}/Products",
        token,
        params={"$filter": filt, "$top": str(query_top), "$orderby": "ContentDate/Start desc"},
        timeout=120,
        retries=1,
    )
    return resp.json().get("value", []), token


def parse_dt(prod: Dict[str, Any]) -> datetime:
    s = prod["ContentDate"]["Start"]
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def strip_safe_suffix(name: str) -> str:
    cleaned = name.strip()
    for suffix in (".SAFE.zip", ".SAFE", ".zip"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned


def s1_acquisition_key(name: str) -> str:
    cleaned = strip_safe_suffix(name)
    m = S1_ACQ_KEY_RE.match(cleaned)
    if m:
        return m.group(1)
    return cleaned


def parse_product_geometry(prod: Dict[str, Any]) -> Optional[BaseGeometry]:
    raw = (
        prod.get("GeoFootprint")
        or prod.get("Footprint")
        or prod.get("footprint")
        or prod.get("Geometry")
        or prod.get("geometry")
    )
    if raw is None:
        return None

    try:
        geom: BaseGeometry
        if isinstance(raw, dict):
            geom = shape(raw)
        elif isinstance(raw, str):
            txt = raw.strip()
            if txt.startswith("geography'") and txt.endswith("'"):
                txt = txt[len("geography'") : -1]
            if txt.upper().startswith("SRID=") and ";" in txt:
                txt = txt.split(";", 1)[1]
            geom = shapely_wkt.loads(txt)
        else:
            return None

        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            return None
        return geom
    except Exception:
        return None


def build_s2_pass_target_geometry(
    pass_products: List[Dict[str, Any]],
    bbox_lonlat: Tuple[float, float, float, float],
) -> BaseGeometry:
    aoi = box(*bbox_lonlat)
    geoms: List[BaseGeometry] = []
    for p in pass_products:
        g = parse_product_geometry(p)
        if g is None:
            continue
        gi = g.intersection(aoi)
        if not gi.is_empty:
            geoms.append(gi)

    if not geoms:
        return aoi
    return unary_union(geoms)


def overlap_ratio(product_geom: Optional[BaseGeometry], target_geom: BaseGeometry) -> float:
    if product_geom is None or target_geom.is_empty:
        return 0.0
    denom = float(target_geom.area)
    if denom <= 0.0:
        return 0.0
    inter = product_geom.intersection(target_geom)
    if inter.is_empty:
        return 0.0
    return max(0.0, min(1.0, float(inter.area) / denom))


def pick_s1_scenes_by_overlap_time(
    candidates: List[Dict[str, Any]],
    target_dt: datetime,
    max_time_diff_hours: int,
    max_scenes: int,
    target_geom: BaseGeometry,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not candidates or max_scenes <= 0:
        return [], {"raw": len(candidates), "within_time": 0, "deduped": 0}

    max_sec = max_time_diff_hours * 3600.0
    within_time = 0
    best_by_acq: Dict[str, Dict[str, Any]] = {}

    for p in candidates:
        dt = parse_dt(p)
        diff_sec = abs((dt - target_dt).total_seconds())
        if diff_sec > max_sec:
            continue
        within_time += 1

        name = p.get("Name", "")
        acq_key = s1_acquisition_key(name)
        geom = parse_product_geometry(p)
        overlap = overlap_ratio(geom, target_geom)
        prefer_non_cog = 0 if "_COG" in name else 1

        entry = {
            "product": dict(p),
            "acq_key": acq_key,
            "diff_sec": float(diff_sec),
            "overlap": float(overlap),
            "prefer_non_cog": int(prefer_non_cog),
        }
        score = (entry["overlap"], -entry["diff_sec"], entry["prefer_non_cog"])

        prev = best_by_acq.get(acq_key)
        if prev is None:
            best_by_acq[acq_key] = entry
            continue
        prev_score = (prev["overlap"], -prev["diff_sec"], prev["prefer_non_cog"])
        if score > prev_score:
            best_by_acq[acq_key] = entry

    deduped = list(best_by_acq.values())
    deduped.sort(
        key=lambda e: (
            -e["overlap"],
            e["diff_sec"],
            -e["prefer_non_cog"],
            e["product"].get("Name", ""),
        )
    )

    selected: List[Dict[str, Any]] = []
    for e in deduped[:max_scenes]:
        prod = e["product"]
        prod["_s1_overlap_ratio"] = float(e["overlap"])
        prod["_s1_time_gap_hours"] = float(e["diff_sec"] / 3600.0)
        prod["_s1_acq_key"] = e["acq_key"]
        selected.append(prod)

    stats = {"raw": len(candidates), "within_time": within_time, "deduped": len(deduped)}
    return selected, stats


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

    last_exc: Optional[Exception] = None
    for attempt in range(1, CDSE_DOWNLOAD_ATTEMPTS + 1):
        try:
            tmp.unlink(missing_ok=True)
            _stream_download(url, token)
            break
        except requests.HTTPError as exc:
            last_exc = exc
            code = getattr(exc.response, "status_code", None)
            if code in (401, 403):
                token = get_token_fn()
                if attempt < CDSE_DOWNLOAD_ATTEMPTS:
                    continue
            elif code == 429:
                if attempt < CDSE_DOWNLOAD_ATTEMPTS:
                    wait_s = max(1, CDSE_DOWNLOAD_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
                    sleep(float(wait_s))
                    continue
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < CDSE_DOWNLOAD_ATTEMPTS:
                wait_s = max(1, CDSE_DOWNLOAD_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
                sleep(float(wait_s))
                continue
            raise
    else:
        if last_exc is not None:
            raise last_exc

    tmp.rename(out)
    return out, token


def _download_product_job(
    kind: str,
    idx: int,
    product: Dict[str, Any],
    token: str,
) -> Tuple[str, int, Path, str]:
    zip_path, token_out = download_product_zip(
        get_token_fn=cdse_token,
        token=token,
        product=product,
        out_dir=DOWNLOAD_DIR,
    )
    return kind, idx, zip_path, token_out


def download_s1_s2_parallel(
    s2_products: List[Dict[str, Any]],
    s1_products: List[Dict[str, Any]],
    token: str,
) -> Tuple[List[Path], List[Path], str]:
    s2_zips: List[Optional[Path]] = [None] * len(s2_products)
    s1_zips: List[Optional[Path]] = [None] * len(s1_products)

    jobs: List[Tuple[str, int, Dict[str, Any]]] = []
    jobs.extend(("s2", i, p) for i, p in enumerate(s2_products))
    jobs.extend(("s1", i, p) for i, p in enumerate(s1_products))
    if not jobs:
        return [], [], token

    workers = max(1, min(CDSE_DOWNLOAD_WORKERS, len(jobs)))
    log(f"  CDSE download tuning: workers={workers} (auto={AUTO_CDSE_DOWNLOAD_WORKERS})")

    if workers == 1:
        out_token = token
        for kind, idx, product in jobs:
            _, _, path, out_token = _download_product_job(kind, idx, product, out_token)
            if kind == "s2":
                s2_zips[idx] = path
            else:
                s1_zips[idx] = path
        return [p for p in s2_zips if p is not None], [p for p in s1_zips if p is not None], out_token

    out_token = token
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for kind, idx, product in jobs:
            fut = pool.submit(_download_product_job, kind, idx, product, token)
            futures[fut] = (kind, idx, str(product.get("Name", "")))

        done = 0
        for fut in as_completed(futures):
            kind, idx, name = futures[fut]
            try:
                kind, idx, path, tok = fut.result()
            except Exception as exc:
                raise RuntimeError(
                    f"CDSE download failed for {kind.upper()} product {name}"
                ) from exc
            if tok:
                out_token = tok
            if kind == "s2":
                s2_zips[idx] = path
            else:
                s1_zips[idx] = path
            done += 1
            if is_progress_tick(done, len(futures), 5):
                log(f"  CDSE download progress: {done}/{len(futures)}")

    if any(p is None for p in s2_zips):
        raise RuntimeError("Missing S2 download result(s)")
    if any(p is None for p in s1_zips):
        raise RuntimeError("Missing S1 download result(s)")

    return [p for p in s2_zips if p is not None], [p for p in s1_zips if p is not None], out_token


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
# Aerial helpers (Datafordeler)
# -----------------------------
@dataclass
class CatalogRecord:
    dataset: str
    file_name: str
    file_size: int
    minx: float
    miny: float
    maxx: float
    maxy: float
    year: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "file_name": self.file_name,
            "file_size": int(self.file_size),
            "minx": float(self.minx),
            "miny": float(self.miny),
            "maxx": float(self.maxx),
            "maxy": float(self.maxy),
            "year": (int(self.year) if self.year is not None else None),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CatalogRecord":
        y = d.get("year")
        return CatalogRecord(
            dataset=str(d["dataset"]),
            file_name=str(d["file_name"]),
            file_size=int(d.get("file_size", 0) or 0),
            minx=float(d["minx"]),
            miny=float(d["miny"]),
            maxx=float(d["maxx"]),
            maxy=float(d["maxy"]),
            year=(int(y) if y is not None else None),
        )


def datafordeler_api_key() -> str:
    key = os.environ.get("DATAFORDELER_APIKEY")
    if not key:
        raise RuntimeError("Missing DATAFORDELER_APIKEY in environment or pipeline5-aerial/.env")
    return key


def build_retry_session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _nk(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _nm(d: Dict[str, Any]) -> Dict[str, Any]:
    return {_nk(k): v for k, v in d.items()}


def _ga(item: Dict[str, Any], *keys: str) -> Any:
    nm = _nm(item)
    for k in keys:
        nk = _nk(k)
        if nk in nm:
            return nm[nk]
    return None


def _sf(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip().replace(",", ".")
            if not v:
                return None
        return float(v)
    except Exception:
        return None


def _si(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, str):
            v = v.strip().replace(" ", "")
            if not v:
                return default
        return int(float(v))
    except Exception:
        return default


def _parse_year(item: Dict[str, Any], file_name: str) -> Optional[int]:
    for key in ("captureDate", "acquisitionDate", "photoDate", "date", "timestamp", "dato"):
        v = _ga(item, key)
        if v is None:
            continue
        m = YEAR_RE.search(str(v))
        if m:
            y = _si(m.group(0), 0)
            if 1900 <= y <= 2100:
                return y
    m = YEAR_RE.search(file_name)
    if m:
        y = _si(m.group(0), 0)
        if 1900 <= y <= 2100:
            return y
    return None


def _bbox_from_obj(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    nm = _nm(obj)
    minx = _sf(nm.get("minx") or nm.get("xmin") or nm.get("left") or nm.get("west"))
    miny = _sf(nm.get("miny") or nm.get("ymin") or nm.get("bottom") or nm.get("south"))
    maxx = _sf(nm.get("maxx") or nm.get("xmax") or nm.get("right") or nm.get("east"))
    maxy = _sf(nm.get("maxy") or nm.get("ymax") or nm.get("top") or nm.get("north"))
    if None not in (minx, miny, maxx, maxy) and minx < maxx and miny < maxy:
        return float(minx), float(miny), float(maxx), float(maxy)

    for key in ("bbox", "bounds", "extent"):
        arr = nm.get(key)
        if isinstance(arr, (list, tuple)) and len(arr) == 4:
            vals = [_sf(x) for x in arr]
            if all(v is not None for v in vals):
                a, b, c, d = vals
                if a < c and b < d:
                    return float(a), float(b), float(c), float(d)
    return None


def _bbox_looks_wgs84(bbox: Tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bbox
    return (
        -180.0 <= minx <= 180.0
        and -180.0 <= maxx <= 180.0
        and -90.0 <= miny <= 90.0
        and -90.0 <= maxy <= 90.0
    )


def _extract_bbox_wgs84(item: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    bbox = _bbox_from_obj(item)
    if bbox is None:
        for key in ("boundingBox", "bbox", "extent", "tileBoundingBox", "envelope", "geo"):
            nested = _ga(item, key)
            if isinstance(nested, dict):
                bbox = _bbox_from_obj(nested)
                if bbox is not None:
                    break
    if bbox is None:
        return None

    epsg: Optional[int] = None
    crs_raw = _ga(item, "epsg", "srid", "crs", "coordsystem", "projection")
    if crs_raw is not None:
        m = re.search(r"(\d{4,6})", str(crs_raw))
        if m:
            epsg = _si(m.group(1), 4326)

    if epsg is None:
        if _bbox_looks_wgs84(bbox):
            epsg = 4326
        else:
            epsg = DF_FALLBACK_BBOX_EPSG

    if epsg == 4326:
        return bbox

    try:
        tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        minx, miny, maxx, maxy = bbox
        xs: List[float] = []
        ys: List[float] = []
        for t in np.linspace(0.0, 1.0, 11):
            x = minx + t * (maxx - minx)
            x1, y1 = tr.transform(x, miny)
            x2, y2 = tr.transform(x, maxy)
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        for t in np.linspace(0.0, 1.0, 11):
            y = miny + t * (maxy - miny)
            x1, y1 = tr.transform(minx, y)
            x2, y2 = tr.transform(maxx, y)
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        return min(xs), min(ys), max(xs), max(ys)
    except Exception:
        return None


def _extract_file_size_bytes(item: Dict[str, Any]) -> int:
    # Datafordeler fields vary by endpoint/key casing.
    v = _ga(
        item,
        "fileSize",
        "size",
        "byteSize",
        "fileByteSize",
        "sizeBytes",
        "fileSizeInBytes",
        "downloadSize",
        "downloadFileSize",
        "bytes",
    )
    size = _si(v, 0)
    if size > 0:
        return size

    # Fallback: scan nested dict values for common size-like keys.
    for key in ("file", "download", "metadata", "meta"):
        nested = _ga(item, key)
        if not isinstance(nested, dict):
            continue
        n = _nm(nested)
        for nk, nv in n.items():
            if "size" in nk or "bytes" in nk:
                size2 = _si(nv, 0)
                if size2 > 0:
                    return size2
    return 0


def fetch_df_available_page(
    session: requests.Session,
    api_key: str,
    dataset_name: str,
    page_number: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    params = {
        "apiKey": api_key,
        "Register": DF_REGISTER,
        "DataSetName": dataset_name,
        "Version": 1,
        "FileFormat": "tif",
        "PageNumber": page_number,
    }
    r = session.get(f"{DF_BASE}/GetAvailableRasterFileDownloads", params=params, timeout=(30, 300))
    r.raise_for_status()
    payload = r.json()
    items = payload.get("availableFileDownloads") or payload.get("AvailableFileDownloads") or []
    meta = payload.get("paginationMetadata") or payload.get("PaginationMetadata") or {}
    return list(items), dict(meta)


def _normalize_catalog_item(dataset: str, item: Dict[str, Any]) -> Optional[CatalogRecord]:
    file_name = _ga(item, "fileName")
    if not file_name:
        return None
    file_name = str(file_name)
    bbox = _extract_bbox_wgs84(item)
    if bbox is None:
        return None
    minx, miny, maxx, maxy = bbox
    if not (minx < maxx and miny < maxy):
        return None
    return CatalogRecord(
        dataset=dataset,
        file_name=file_name,
        file_size=_extract_file_size_bytes(item),
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        year=_parse_year(item, file_name),
    )


def _save_catalog(records: List[CatalogRecord]) -> Path:
    rows = [r.to_dict() for r in records]
    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(rows).to_parquet(DF_CATALOG_PARQUET, index=False)
        return DF_CATALOG_PARQUET
    except Exception:
        with DF_CATALOG_JSONL.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return DF_CATALOG_JSONL


def _load_catalog() -> Optional[List[CatalogRecord]]:
    if DF_CATALOG_PARQUET.exists():
        try:
            import pandas as pd  # type: ignore

            df = pd.read_parquet(DF_CATALOG_PARQUET)
            return [CatalogRecord.from_dict(r) for r in df.to_dict("records")]
        except Exception:
            pass
    if DF_CATALOG_JSONL.exists():
        out: List[CatalogRecord] = []
        with DF_CATALOG_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(CatalogRecord.from_dict(json.loads(line)))
        return out
    return None


def build_aerial_catalog(
    session: requests.Session,
    api_key: str,
    datasets: Sequence[str],
    rebuild: bool,
) -> List[CatalogRecord]:
    if not rebuild:
        existing = _load_catalog()
        if existing is not None:
            log(f"Loaded aerial catalog: {len(existing)} records")
            return existing

    records: List[CatalogRecord] = []
    log("Building aerial catalog index...")
    for dataset in datasets:
        items1, meta = fetch_df_available_page(session, api_key, dataset, 1)
        total_pages = _si(meta.get("totalPages") or meta.get("TotalPages"), 1)
        if total_pages <= 0:
            total_pages = 1
        total = 0
        kept = 0

        def _ingest(items: List[Dict[str, Any]]) -> None:
            nonlocal total, kept
            total += len(items)
            for item in items:
                rec = _normalize_catalog_item(dataset, item)
                if rec is None:
                    continue
                records.append(rec)
                kept += 1

        progress_every = 1 if total_pages <= 20 else 20
        _ingest(items1)
        log(f"  {dataset}: indexed 1/{total_pages} pages ({(1 / total_pages) * 100:.0f}%)")
        for page in range(2, total_pages + 1):
            items, _m = fetch_df_available_page(session, api_key, dataset, page)
            _ingest(items)
            if is_progress_tick(page, total_pages, progress_every):
                log(f"  {dataset}: indexed {page}/{total_pages} pages ({(page / total_pages) * 100:.0f}%)")

        log(f"  {dataset}: total={total} indexed={kept}")

    dedup: Dict[Tuple[str, str], CatalogRecord] = {}
    for r in records:
        dedup[(r.dataset, r.file_name)] = r
    out = sorted(dedup.values(), key=lambda r: (r.dataset, r.file_name))
    path = _save_catalog(out)
    log(f"Aerial catalog saved: {path} ({len(out)} records)")
    return out


def select_aerial_records_for_pass(
    records: List[CatalogRecord],
    dataset_priority: Sequence[str],
    aoi_wgs84: BaseGeometry,
    target_year: Optional[int],
    date_policy: str,
) -> Tuple[List[CatalogRecord], Optional[str], Optional[int]]:
    def _tile_geom(rec: CatalogRecord) -> BaseGeometry:
        return box(rec.minx, rec.miny, rec.maxx, rec.maxy)

    def _intersect_with_aoi(aoi_geom: BaseGeometry) -> List[CatalogRecord]:
        return [r for r in records if _tile_geom(r).intersects(aoi_geom)]

    aoi_geom = aoi_wgs84
    inter = _intersect_with_aoi(aoi_geom)
    if not inter and records:
        sample = records[: min(200, len(records))]
        projected_hint = any(
            (
                abs(r.minx) > 180.0
                or abs(r.maxx) > 180.0
                or abs(r.miny) > 90.0
                or abs(r.maxy) > 90.0
            )
            for r in sample
        )
        if projected_hint:
            try:
                tr = Transformer.from_crs("EPSG:4326", f"EPSG:{DF_FALLBACK_BBOX_EPSG}", always_xy=True)
                aoi_projected = shapely_transform(tr.transform, aoi_wgs84)
                inter = _intersect_with_aoi(aoi_projected)
                if inter:
                    aoi_geom = aoi_projected
                    log(
                        f"  Aerial catalog intersection used projected CRS fallback "
                        f"(EPSG:{DF_FALLBACK_BBOX_EPSG})"
                    )
            except Exception:
                pass

    if not inter:
        return [], None, None

    selected: List[CatalogRecord] = []
    used_datasets: List[str] = []
    used_years: List[int] = []
    filtered_by_dataset: Dict[str, List[CatalogRecord]] = {}
    remaining_geom = aoi_geom

    for dataset in dataset_priority:
        if remaining_geom.is_empty:
            break
        ds = [r for r in inter if r.dataset == dataset]
        if not ds:
            continue

        ds_sel = ds
        if date_policy == "nearest_year" and target_year is not None:
            years = sorted({r.year for r in ds if r.year is not None})
            if years:
                best_year = min(years, key=lambda y: abs(y - target_year))
                ds_sel = [r for r in ds if r.year == best_year]
                used_years.append(int(best_year))
        filtered_by_dataset[dataset] = ds_sel

        if ds_sel:
            # Prefer fewer tiles by covering largest remaining AOI area first.
            ranked: List[Tuple[float, CatalogRecord]] = []
            for rec in ds_sel:
                tile = _tile_geom(rec)
                try:
                    overlap = float(tile.intersection(remaining_geom).area)
                except Exception:
                    overlap = 0.0
                if overlap > 0.0:
                    ranked.append((overlap, rec))

            ranked.sort(key=lambda x: x[0], reverse=True)
            added = 0
            for _, rec in ranked:
                tile = _tile_geom(rec)
                if not tile.intersects(remaining_geom):
                    continue
                selected.append(rec)
                added += 1
                try:
                    remaining_geom = remaining_geom.difference(tile)
                except Exception:
                    pass
                if remaining_geom.is_empty:
                    break

            if added > 0:
                used_datasets.append(dataset)
        if remaining_geom.is_empty:
            break

    if selected:
        # If tiny topology slivers remain, include touching tiles from remaining datasets.
        if not remaining_geom.is_empty:
            for dataset in dataset_priority:
                if remaining_geom.is_empty:
                    break
                extra = [r for r in filtered_by_dataset.get(dataset, []) if _tile_geom(r).intersects(remaining_geom)]
                for rec in extra:
                    selected.append(rec)
                    try:
                        remaining_geom = remaining_geom.difference(_tile_geom(rec))
                    except Exception:
                        pass
                    if remaining_geom.is_empty:
                        break
                if extra and dataset not in used_datasets:
                    used_datasets.append(dataset)

    if not selected:
        return [], None, None

    dedup: Dict[Tuple[str, str], CatalogRecord] = {}
    for r in selected:
        dedup[(r.dataset, r.file_name)] = r
    selected = sorted(dedup.values(), key=lambda r: (r.dataset, r.file_name))

    dataset_label = "+".join(used_datasets) if used_datasets else None
    if target_year is not None:
        year_label: Optional[int] = int(target_year)
    else:
        uniq_years = sorted({int(y) for y in used_years})
        year_label = uniq_years[0] if len(uniq_years) == 1 else None
    return selected, dataset_label, year_label


def _is_zip_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 4:
        return False
    with path.open("rb") as f:
        return f.read(4) == b"PK\x03\x04"


def _is_tiff_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 4:
        return False
    with path.open("rb") as f:
        sig = f.read(4)
    return sig in (b"II*\x00", b"MM\x00*")


def _has_valid_georef(ds: rasterio.DatasetReader) -> bool:
    if ds.crs is None:
        return False
    # Some malformed TIFFs trigger NotGeoreferencedWarning on transform access.
    # Treat those files as invalid georeference instead of surfacing warning spam.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        tr = ds.transform
    if tr is None:
        return False
    if tr == Affine.identity():
        return False
    if abs(float(tr.a)) < 1e-12 or abs(float(tr.e)) < 1e-12:
        return False
    return True


def _infer_geodko_bounds_from_name(file_name: str) -> Optional[Tuple[float, float, float, float]]:
    stem = Path(file_name).stem
    stem = stem.split("__", 1)[0]
    m = GEODKO_TILE_RE.match(stem)
    if not m:
        return None
    north_km = int(m.group(1))
    east_km = int(m.group(2))
    minx = float(east_km * 1000)
    miny = float(north_km * 1000)
    maxx = minx + 1000.0
    maxy = miny + 1000.0
    return minx, miny, maxx, maxy


def _infer_geodko_georef(file_name: str, width: int, height: int) -> Optional[Tuple[Affine, str, Tuple[float, float, float, float]]]:
    if width <= 0 or height <= 0:
        return None
    bounds = _infer_geodko_bounds_from_name(file_name)
    if bounds is None:
        return None
    minx, miny, maxx, maxy = bounds
    resx = (maxx - minx) / float(width)
    resy = (maxy - miny) / float(height)
    transform = from_origin(minx, maxy, resx, resy)
    return transform, "EPSG:25832", bounds


def _repair_cached_aerial_georef(cache_path: Path, source_name: str) -> bool:
    try:
        with rasterio.open(cache_path, "r+") as ds:
            inferred = _infer_geodko_georef(source_name, ds.width, ds.height)
            if inferred is None:
                return False
            transform, crs, _ = inferred
            ds.transform = transform
            ds.crs = crs
    except Exception:
        return False
    return _is_valid_cached_aerial_tif(cache_path)


def _is_valid_cached_aerial_tif(path: Path) -> bool:
    if not file_is_ready(path):
        return False
    if not _is_tiff_file(path):
        return False
    try:
        with rasterio.open(path) as ds:
            if ds.count < 4:
                return False
            if not _has_valid_georef(ds):
                return False
    except Exception:
        return False
    return True


def _set_aerial_colorinterp(ds: rasterio.io.DatasetWriter) -> None:
    ds.colorinterp = (
        ColorInterp.red,
        ColorInterp.green,
        ColorInterp.blue,
        ColorInterp.nir,
    )


def _reproject_strict(*args: Any, **kwargs: Any) -> Any:
    # Fail fast on non-georeferenced sources so retry/skip logic can handle them.
    with warnings.catch_warnings():
        warnings.simplefilter("error", NotGeoreferencedWarning)
        return reproject(*args, **kwargs)


def _download_stream(resp: requests.Response, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def _download_df_single(
    session: requests.Session,
    api_key: str,
    dataset: str,
    file_name: str,
    out_path: Path,
) -> Path:
    params = {
        "apiKey": api_key,
        "Register": DF_REGISTER,
        "DataSetName": dataset,
        "Version": 1,
        "Filename": file_name,
    }
    with session.get(f"{DF_BASE}/GetRasterFile", params=params, stream=True, timeout=(30, 1200)) as r:
        r.raise_for_status()
        _download_stream(r, out_path)
    return out_path


def _extract_tifs(archive_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _is_zip_file(archive_path):
        out: List[Path] = []
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.namelist():
                if not member.lower().endswith((".tif", ".tiff")):
                    continue
                dst = out_dir / Path(member).name
                with zf.open(member, "r") as src, dst.open("wb") as f:
                    shutil.copyfileobj(src, f)
                out.append(dst)
        if not out:
            raise RuntimeError(f"No tif files found in archive: {archive_path}")
        return out
    if archive_path.suffix.lower() in (".tif", ".tiff"):
        if not _is_tiff_file(archive_path):
            raise RuntimeError(f"Downloaded file is not a TIFF: {archive_path.name}")
        return [archive_path]
    if not _is_tiff_file(archive_path):
        raise RuntimeError(f"Downloaded file has unsupported format: {archive_path.name}")
    tif = archive_path.with_suffix(".tif")
    archive_path.rename(tif)
    return [tif]


def _sanitize_cache_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", token)


def _aerial_scope_key(job: Job, year: Optional[int], target_crs: str) -> str:
    year_token = str(int(year)) if year is not None else "unknown_year"
    res_token = str(float(job.aerial_target_res_m)).replace(".", "p")
    bbox_sig = ",".join(f"{v:.6f}" for v in job.bbox_lonlat)
    bbox_hash = hashlib.sha1(bbox_sig.encode("utf-8")).hexdigest()[:10]
    crs_token = _sanitize_cache_token(str(target_crs))
    return f"{job.name}__AERIAL__{year_token}__{crs_token}__{res_token}m__bbox{bbox_hash}"


def aerial_cache_1m_path(dataset: str, file_name: str, target_res_m: float) -> Path:
    ds = _sanitize_cache_token(dataset)
    stem = _sanitize_cache_token(Path(file_name).stem)
    res = str(target_res_m).replace(".", "p")
    return AERIAL_CACHE_DIR / ds / f"{stem}__{res}m.tif"


def cached_aerial_1m_paths(dataset: str, file_name: str, target_res_m: float) -> List[Path]:
    base = aerial_cache_1m_path(dataset, file_name, target_res_m)
    out: List[Path] = []
    if _is_valid_cached_aerial_tif(base):
        out.append(base)
    elif base.exists():
        base.unlink(missing_ok=True)
    for p in sorted(base.parent.glob(f"{base.stem}__*.tif")):
        if _is_valid_cached_aerial_tif(p):
            out.append(p)
        else:
            p.unlink(missing_ok=True)
    return out


def downsample_to_cache_1m(src_tif: Path, cache_path: Path, target_res_m: float) -> Path:
    if cache_path.exists():
        if _is_valid_cached_aerial_tif(cache_path):
            return cache_path
        cache_path.unlink(missing_ok=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)

    with rasterio.open(src_tif) as src:
        inferred_src = None
        if not _has_valid_georef(src):
            inferred_src = _infer_geodko_georef(src_tif.name, src.width, src.height)
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
        width = max(1, int(math.ceil((right - left) / target_res_m)))
        height = max(1, int(math.ceil((top - bottom) / target_res_m)))
        transform = from_origin(left, top, target_res_m, target_res_m)

        profile = src.profile.copy()
        profile.pop("nodata", None)
        profile.update(
            driver="GTiff",
            count=4,
            dtype="uint8",
            width=width,
            height=height,
            transform=transform,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        with rasterio.open(tmp, "w", **profile) as dst:
            _set_aerial_colorinterp(dst)
            src_nodata = src.nodata
            for out_band, src_band in enumerate([1, 2, 3, 4], start=1):
                _reproject_strict(
                    source=rasterio.band(src, src_band),
                    destination=rasterio.band(dst, out_band),
                    src_transform=src_transform,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    dst_transform=transform,
                    dst_crs=src_crs,
                    resampling=Resampling.bilinear,
                    init_dest_nodata=False,
                    num_threads=REPROJECT_THREADS,
                )

            # Preserve source validity as dataset mask (not nodata value),
            # so black pixels (DN=0) are not conflated with missing data.
            src_mask = (src.read_masks(1) > 0).astype(np.uint8)
            dst_mask = np.zeros((height, width), dtype=np.uint8)
            _reproject_strict(
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
                num_threads=REPROJECT_THREADS,
            )
            dst.write_mask((dst_mask > 0).astype(np.uint8) * 255)

    tmp.replace(cache_path)
    return cache_path


def _split_records_batches(records: Sequence[CatalogRecord], max_batch_bytes: int) -> List[List[CatalogRecord]]:
    # max_batch_bytes <= 0 means "no chunking": treat all records as one chunk.
    if max_batch_bytes <= 0:
        return [list(records)]
    batches: List[List[CatalogRecord]] = []
    cur: List[CatalogRecord] = []
    cur_bytes = 0
    for rec in records:
        sz = rec.file_size if rec.file_size > 0 else 0
        if cur and ((cur_bytes + sz > max_batch_bytes) or (len(cur) >= AERIAL_MAX_FILES_PER_CHUNK)):
            batches.append(cur)
            cur = [rec]
            cur_bytes = sz
        else:
            cur.append(rec)
            cur_bytes += sz
    if cur:
        batches.append(cur)
    return batches


def parse_s2_pass_datetime(s2_pass_key: str) -> Optional[datetime]:
    m = S2_PASS_DT_RE.match(s2_pass_key)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def dedupe_catalog_records(records: Sequence[CatalogRecord]) -> List[CatalogRecord]:
    out: List[CatalogRecord] = []
    seen: Set[Tuple[str, str]] = set()
    for rec in records:
        key = (rec.dataset, rec.file_name)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def create_aligned_aerial_grid_from_job(job: Job, target_crs: str) -> Dict[str, Any]:
    return aligned_aoi_grid(
        bbox_lonlat=job.bbox_lonlat,
        target_crs=target_crs,
        resolution=float(job.aerial_target_res_m),
    )


def create_aerial_rasters(aerial_path: Path, cov_path: Path, grid: Dict[str, Any]) -> None:
    profile = {
        "driver": "GTiff",
        "height": int(grid["height"]),
        "width": int(grid["width"]),
        "count": 4,
        "dtype": "uint8",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    aerial_path.parent.mkdir(parents=True, exist_ok=True)
    # Aerial mosaics can exceed 4 GiB after chunk merges; force BigTIFF.
    with rasterio.open(aerial_path, "w", **with_bigtiff(profile, mode="YES")) as ds:
        _set_aerial_colorinterp(ds)

    cov_profile = dict(profile)
    cov_profile.pop("predictor", None)
    cov_profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(cov_path, "w", **with_bigtiff(cov_profile, mode="YES")):
        pass


def ingest_aerial_tif_to_grid(src_tif: Path, aerial_path: Path, cov_path: Path) -> None:
    with rasterio.open(src_tif) as src, rasterio.open(aerial_path, "r+") as dst, rasterio.open(cov_path, "r+") as cov:
        inferred_src = None
        if not _has_valid_georef(src):
            inferred_src = _infer_geodko_georef(src_tif.name, src.width, src.height)
            if inferred_src is None:
                raise RuntimeError(f"Source TIFF has no valid georeference: {src_tif.name}")
        if src.count < 4:
            raise RuntimeError(f"Expected RGBNIR (>=4 bands), got {src.count} bands in {src_tif.name}")

        if inferred_src is None:
            src_transform = src.transform
            src_crs = src.crs
        else:
            src_transform, src_crs, _ = inferred_src

        src_valid = (src.read_masks(1) > 0).astype(np.uint8)
        _reproject_strict(
            source=src_valid,
            destination=rasterio.band(cov, 1),
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=0,
            dst_transform=dst.transform,
            dst_crs=dst.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
            init_dest_nodata=False,
            num_threads=REPROJECT_THREADS,
        )

        src_nodata = src.nodata
        for out_band, src_band in enumerate([1, 2, 3, 4], start=1):
            _reproject_strict(
                source=rasterio.band(src, src_band),
                destination=rasterio.band(dst, out_band),
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=src_nodata,
                dst_transform=dst.transform,
                dst_crs=dst.crs,
                resampling=Resampling.bilinear,
                init_dest_nodata=False,
                num_threads=REPROJECT_THREADS,
            )


def _prepare_aerial_cache_for_selected(
    job: Job,
    scope_id: str,
    selected: Sequence[CatalogRecord],
    df_session: requests.Session,
    df_api_key: str,
) -> Dict[str, Any]:
    t_cache_start = perf_counter()
    chunk_target_gb = max(0.0, float(job.aerial_download_batch_gb))
    chunk_target_bytes = int(chunk_target_gb * 1024**3)
    chunks = _split_records_batches(selected, chunk_target_bytes)

    record_cache_paths: Dict[Tuple[str, str], List[Path]] = {}
    downloaded_bytes = 0
    single_ok = 0
    single_failed = 0
    cache_hits = 0
    cache_misses = 0

    for i, chunk in enumerate(chunks, start=1):
        t_chunk_start = perf_counter()
        cached: List[Tuple[CatalogRecord, List[Path]]] = []
        to_download: List[CatalogRecord] = []

        for rec in chunk:
            rec_key = (rec.dataset, rec.file_name)
            paths = cached_aerial_1m_paths(rec.dataset, rec.file_name, job.aerial_target_res_m)
            if paths:
                cached.append((rec, paths))
                record_cache_paths[rec_key] = paths
            else:
                to_download.append(rec)

        known_est = sum(max(0, int(r.file_size)) for r in chunk)
        unknown_size_files = sum(1 for r in chunk if int(r.file_size) <= 0)
        log(
            f"    Aerial chunk {i}/{len(chunks)} files={len(chunk)} "
            f"known_est={(known_est / (1024**3)):.2f} GiB "
            f"unknown_size_files={unknown_size_files} "
            f"(cached={len(cached)} download={len(to_download)})"
        )
        cache_hits += len(cached)

        if not to_download:
            continue

        futures = {}
        with ThreadPoolExecutor(max_workers=max(1, int(job.aerial_workers))) as pool:
            for rec in to_download:
                fut = pool.submit(
                    _download_extract_downsample_record,
                    session=df_session,
                    api_key=df_api_key,
                    rec=rec,
                    target_res_m=job.aerial_target_res_m,
                    scope_id=scope_id,
                    chunk_index=i,
                )
                futures[fut] = (rec.dataset, rec.file_name)

            done = 0
            for fut in as_completed(futures):
                rec_dataset, file_name = futures[fut]
                rec_key = (rec_dataset, file_name)
                cache_misses += 1
                try:
                    result = fut.result()
                except Exception as exc:
                    single_failed += 1
                    log(f"    Aerial single-file download failed for {rec_dataset}/{file_name}: {exc}")
                    continue

                downloaded_bytes += int(result.get("downloaded_bytes", 0) or 0)
                output_paths = list(result.get("cache_paths", []))
                if result.get("ok") and output_paths:
                    record_cache_paths[rec_key] = output_paths
                    single_ok += 1
                    attempts_used = int(result.get("attempts", 1) or 1)
                    if attempts_used > 1:
                        log(
                            f"    Aerial download retry succeeded for {rec_dataset}/{file_name}: "
                            f"attempt {attempts_used}/{AERIAL_DOWNLOAD_ATTEMPTS}"
                        )
                else:
                    single_failed += 1
                    err = result.get("error", "unknown error")
                    attempts_used = int(result.get("attempts", AERIAL_DOWNLOAD_ATTEMPTS) or AERIAL_DOWNLOAD_ATTEMPTS)
                    log(
                        f"    Aerial single-file download failed for {rec_dataset}/{file_name} "
                        f"after {attempts_used} attempt(s): {err}"
                    )
                done += 1
                if is_progress_tick(done, len(futures), 10):
                    log(
                        f"    Aerial download progress {i}/{len(chunks)}: "
                        f"{done}/{len(futures)}"
                    )

        t_chunk = perf_counter() - t_chunk_start
        log(
            f"    Aerial chunk timing {i}/{len(chunks)}: "
            f"{t_chunk:.1f}s (cached={len(cached)} downloaded={len(to_download)})"
        )

    cache_stage_seconds = perf_counter() - t_cache_start
    log(
        "    Aerial cache prep timing: "
        f"{cache_stage_seconds:.1f}s total "
        f"(hits={cache_hits} misses={cache_misses} ok={single_ok} fail={single_failed})"
    )

    return {
        "record_cache_paths": record_cache_paths,
        "downloaded_bytes": downloaded_bytes,
        "single_downloads_ok": single_ok,
        "single_downloads_failed": single_failed,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_stage_seconds": cache_stage_seconds,
    }


def _download_extract_downsample_record(
    session: requests.Session,
    api_key: str,
    rec: CatalogRecord,
    target_res_m: float,
    scope_id: str,
    chunk_index: int,
) -> Dict[str, Any]:
    dl = AERIAL_TMP_DIR / f"{scope_id}__{chunk_index:04d}__{Path(rec.file_name).name}.download"
    ext_dir = AERIAL_TMP_DIR / f"{scope_id}__single_{Path(rec.file_name).stem}"
    downloaded_bytes = 0
    last_error = "unknown error"

    for attempt in range(1, AERIAL_DOWNLOAD_ATTEMPTS + 1):
        extracted_tifs: List[Path] = []
        output_paths: List[Path] = []
        try:
            archive_path = _download_df_single(session, api_key, rec.dataset, rec.file_name, dl)
            if archive_path.exists():
                downloaded_bytes = max(downloaded_bytes, archive_path.stat().st_size)

            tifs = _extract_tifs(archive_path, ext_dir)
            extracted_tifs.extend(tifs)

            base_cache = aerial_cache_1m_path(rec.dataset, rec.file_name, target_res_m)
            for ti, tif in enumerate(tifs, start=1):
                if len(tifs) == 1:
                    cache_target = base_cache
                else:
                    cache_target = base_cache.with_name(f"{base_cache.stem}__{ti:02d}.tif")
                cached_tif = downsample_to_cache_1m(
                    src_tif=tif,
                    cache_path=cache_target,
                    target_res_m=target_res_m,
                )
                output_paths.append(cached_tif)

            return {
                "file_name": rec.file_name,
                "ok": len(output_paths) > 0,
                "cache_paths": output_paths,
                "downloaded_bytes": downloaded_bytes,
                "attempts": attempt,
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt < AERIAL_DOWNLOAD_ATTEMPTS:
                wait_s = max(1, AERIAL_DOWNLOAD_RETRY_BASE_SECONDS) * (2 ** (attempt - 1))
                sleep(float(wait_s))
        finally:
            dl.unlink(missing_ok=True)
            for tif in extracted_tifs:
                tif.unlink(missing_ok=True)
            shutil.rmtree(ext_dir, ignore_errors=True)

    return {
        "file_name": rec.file_name,
        "ok": False,
        "cache_paths": [],
        "downloaded_bytes": downloaded_bytes,
        "attempts": AERIAL_DOWNLOAD_ATTEMPTS,
        "error": last_error,
    }


def _chunk_grid_from_cache_paths(cache_paths: Sequence[Path], final_grid: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not cache_paths:
        return None

    final_crs = final_grid["crs"]
    final_transform = final_grid["transform"]
    final_width = int(final_grid["width"])
    final_height = int(final_grid["height"])
    final_minx, final_miny, final_maxx, final_maxy = [float(v) for v in final_grid["bounds"]]

    minx = float("inf")
    miny = float("inf")
    maxx = float("-inf")
    maxy = float("-inf")
    for p in cache_paths:
        if not file_is_ready(p):
            continue
        try:
            with rasterio.open(p) as src:
                bx = transform_bounds(src.crs, final_crs, *src.bounds, densify_pts=21)
            minx = min(minx, float(bx[0]))
            miny = min(miny, float(bx[1]))
            maxx = max(maxx, float(bx[2]))
            maxy = max(maxy, float(bx[3]))
        except Exception:
            continue

    if not np.isfinite(minx) or not np.isfinite(miny) or not np.isfinite(maxx) or not np.isfinite(maxy):
        return None

    minx = max(minx, final_minx)
    miny = max(miny, final_miny)
    maxx = min(maxx, final_maxx)
    maxy = min(maxy, final_maxy)
    if not (minx < maxx and miny < maxy):
        return None

    x0 = float(final_transform.c)
    y0 = float(final_transform.f)
    resx = float(final_transform.a)
    resy = abs(float(final_transform.e))

    c0 = int(math.floor((minx - x0) / resx))
    c1 = int(math.ceil((maxx - x0) / resx))
    r0 = int(math.floor((y0 - maxy) / resy))
    r1 = int(math.ceil((y0 - miny) / resy))

    c0 = max(0, min(c0, final_width))
    c1 = max(0, min(c1, final_width))
    r0 = max(0, min(r0, final_height))
    r1 = max(0, min(r1, final_height))
    if c1 <= c0 or r1 <= r0:
        return None

    transform = final_transform * Affine.translation(c0, r0)
    return {
        "crs": final_crs,
        "transform": transform,
        "width": int(c1 - c0),
        "height": int(r1 - r0),
        "bounds": (
            float(transform.c),
            float(transform.f + transform.e * (r1 - r0)),
            float(transform.c + transform.a * (c1 - c0)),
            float(transform.f),
        ),
        "row_off": int(r0),
        "col_off": int(c0),
    }


def _scope_chunk_paths(scope_id: str, chunk_index: int) -> Tuple[Path, Path]:
    chunk_dir = AERIAL_MOSAIC_DIR / f"{scope_id}__chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_aerial = chunk_dir / f"{scope_id}__chunk_{chunk_index:04d}__aerial.tif"
    chunk_cov = chunk_dir / f"{scope_id}__chunk_{chunk_index:04d}__cov.tif"
    return chunk_aerial, chunk_cov


def _build_chunk_mosaic_for_records(
    chunk_index: int,
    chunk_records: Sequence[CatalogRecord],
    target_res_m: float,
    record_cache_paths: Dict[Tuple[str, str], List[Path]],
    grid: Dict[str, Any],
    scope_id: str,
) -> Dict[str, Any]:
    t_chunk_build_start = perf_counter()
    chunk_aerial, chunk_cov = _scope_chunk_paths(scope_id, chunk_index)
    chunk_aerial.unlink(missing_ok=True)
    chunk_cov.unlink(missing_ok=True)

    source_files_used: List[str] = []
    missing_sources = 0
    ingest_fail_total = 0
    ingest_fail_reasons: Dict[str, int] = {}
    ingest_failed_files: List[Tuple[str, str, str]] = []
    chunk_cache_paths: List[Path] = []
    rec_paths: List[Tuple[str, str, List[Path]]] = []
    for rec in chunk_records:
        rec_key = (rec.dataset, rec.file_name)
        cache_paths = list(record_cache_paths.get(rec_key, []))
        if not cache_paths:
            cache_paths = cached_aerial_1m_paths(rec.dataset, rec.file_name, target_res_m)
        if not cache_paths:
            missing_sources += 1
            continue
        rec_paths.append((rec.dataset, rec.file_name, cache_paths))
        chunk_cache_paths.extend(cache_paths)

    chunk_grid = _chunk_grid_from_cache_paths(chunk_cache_paths, grid)
    if chunk_grid is None:
        return {
            "chunk_index": int(chunk_index),
            "aerial_path": chunk_aerial,
            "cov_path": chunk_cov,
            "source_files_used": [],
            "missing_sources": int(missing_sources),
            "row_off": 0,
            "col_off": 0,
            "width": 0,
            "height": 0,
            "has_data": False,
            "build_seconds": perf_counter() - t_chunk_build_start,
        }

    create_aerial_rasters(chunk_aerial, chunk_cov, chunk_grid)
    for rec_dataset, file_name, cache_paths in rec_paths:
        rec_ok = False
        for cache_path in cache_paths:
            last_exc: Optional[Exception] = None
            for attempt in range(1, 3):
                try:
                    ingest_aerial_tif_to_grid(cache_path, chunk_aerial, chunk_cov)
                    rec_ok = True
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if isinstance(exc, NotGeoreferencedWarning):
                        if _repair_cached_aerial_georef(cache_path, file_name):
                            log(
                                "      repaired_georef: "
                                f"dataset={rec_dataset} source={file_name} cache={cache_path.name}"
                            )
                            continue
                    if attempt < 2:
                        sleep(0.25)
            if rec_ok:
                continue

            reason = type(last_exc).__name__ if last_exc is not None else "IngestError"
            ingest_fail_total += 1
            ingest_fail_reasons[reason] = ingest_fail_reasons.get(reason, 0) + 1
            ingest_failed_files.append((rec_dataset, file_name, cache_path.name))

            # Do not aggressively delete on transient ingest failures.
            # Only drop cache files that fail basic integrity checks.
            if not _is_valid_cached_aerial_tif(cache_path):
                cache_path.unlink(missing_ok=True)
        if rec_ok:
            source_files_used.append(f"{rec_dataset}/{file_name}")
        else:
            missing_sources += 1

    if ingest_fail_total > 0:
        top = sorted(ingest_fail_reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_str = ", ".join(f"{k}:{v}" for k, v in top)
        log(
            f"    Aerial ingest failures in chunk {chunk_index}: "
            f"{ingest_fail_total} ({top_str})"
        )
        for ds_name, src_name, cache_name in ingest_failed_files:
            log(
                "      failed_file: "
                f"dataset={ds_name} source={src_name} cache={cache_name}"
            )

    return {
        "chunk_index": int(chunk_index),
        "aerial_path": chunk_aerial,
        "cov_path": chunk_cov,
        "source_files_used": sorted(set(source_files_used)),
        "missing_sources": int(missing_sources),
        "ingest_fail_total": int(ingest_fail_total),
        "ingest_fail_reasons": dict(ingest_fail_reasons),
        "row_off": int(chunk_grid["row_off"]),
        "col_off": int(chunk_grid["col_off"]),
        "width": int(chunk_grid["width"]),
        "height": int(chunk_grid["height"]),
        "has_data": True,
        "build_seconds": perf_counter() - t_chunk_build_start,
    }


def _build_scope_chunk_mosaics(
    job: Job,
    selected: Sequence[CatalogRecord],
    target_res_m: float,
    record_cache_paths: Dict[Tuple[str, str], List[Path]],
    grid: Dict[str, Any],
    scope_id: str,
) -> Tuple[List[Dict[str, Any]], List[str], int, float, float]:
    t_ingest_total_start = perf_counter()
    total = len(selected)
    if not total:
        return [], [], 0, 0.0, 0.0

    chunk_target_gb = max(0.0, float(job.aerial_download_batch_gb))
    chunk_target_bytes = int(chunk_target_gb * 1024**3)
    chunks = _split_records_batches(selected, chunk_target_bytes)
    chunk_workers = max(1, int(job.aerial_chunk_workers))
    log(
        f"    Build chunk mosaics in parallel: chunks={len(chunks)} "
        f"workers={chunk_workers}"
    )

    chunk_results: List[Dict[str, Any]] = []
    t_chunk_build_start = perf_counter()
    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, min(chunk_workers, len(chunks)))) as pool:
        for i, chunk in enumerate(chunks, start=1):
            fut = pool.submit(
                _build_chunk_mosaic_for_records,
                chunk_index=i,
                chunk_records=chunk,
                target_res_m=target_res_m,
                record_cache_paths=record_cache_paths,
                grid=grid,
                scope_id=scope_id,
            )
            futures[fut] = i

        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            chunk_results.append(res)
            chunk_idx = int(res.get("chunk_index", -1))
            chunk_s = float(res.get("build_seconds", 0.0))
            log(
                f"    Chunk mosaic built: {done}/{len(chunks)} "
                f"(chunk={chunk_idx} chunk_time={chunk_s:.1f}s)"
            )
    chunk_build_seconds = perf_counter() - t_chunk_build_start
    avg_chunk_seconds = (
        float(np.mean([float(r.get("build_seconds", 0.0)) for r in chunk_results])) if chunk_results else 0.0
    )
    log(
        "    Chunk build timing: "
        f"{chunk_build_seconds:.1f}s wall-clock, "
        f"{avg_chunk_seconds:.1f}s avg/chunk"
    )

    chunk_results.sort(key=lambda x: int(x["chunk_index"]))
    source_files_used: List[str] = []
    missing_sources = 0
    for res in chunk_results:
        missing_sources += int(res.get("missing_sources", 0))
        source_files_used.extend(list(res.get("source_files_used", [])))

    ingest_total_seconds = perf_counter() - t_ingest_total_start
    log(f"    Aerial ingest timing: build={chunk_build_seconds:.1f}s total={ingest_total_seconds:.1f}s")

    return chunk_results, sorted(set(source_files_used)), int(missing_sources), chunk_build_seconds, ingest_total_seconds


def build_or_reuse_aerial_mosaic_for_scope(
    job: Job,
    target_crs: str,
    selected: Sequence[CatalogRecord],
    year: Optional[int],
    df_session: requests.Session,
    df_api_key: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    t_scope_start = perf_counter()
    scope_id = _aerial_scope_key(job, year, target_crs)
    sidecar = AERIAL_MOSAIC_DIR / f"{scope_id}__aerial_sources.json"
    selected_datasets = sorted({r.dataset for r in selected})
    dataset_label = "+".join(selected_datasets) if selected_datasets else "unknown"
    required_sources = {f"{r.dataset}/{r.file_name}" for r in selected}

    grid = create_aligned_aerial_grid_from_job(job, target_crs)

    def _sidecar_chunks_to_paths(raw_chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for ch in raw_chunks:
            out.append(
                {
                    "chunk_index": int(ch.get("chunk_index", 0)),
                    "aerial_path": Path(str(ch.get("aerial_path", ""))),
                    "cov_path": Path(str(ch.get("cov_path", ""))),
                    "source_files_used": list(ch.get("source_files_used", [])),
                    "missing_sources": int(ch.get("missing_sources", 0)),
                    "row_off": int(ch.get("row_off", 0)),
                    "col_off": int(ch.get("col_off", 0)),
                    "width": int(ch.get("width", 0)),
                    "height": int(ch.get("height", 0)),
                    "has_data": bool(ch.get("has_data", False)),
                    "build_seconds": float(ch.get("build_seconds", 0.0)),
                }
            )
        return out

    if not AERIAL_REBUILD_MOSAIC and sidecar.exists():
        info = {}
        try:
            info = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            info = {}
        existing_sources = set(info.get("source_files", []))
        chunk_results = _sidecar_chunks_to_paths(info.get("chunks", []))
        chunks_ready = bool(chunk_results) and all(
            (not ch.get("has_data"))
            or (file_is_ready(ch["aerial_path"]) and file_is_ready(ch["cov_path"]))
            for ch in chunk_results
        )
        if not chunks_ready:
            log("  Rebuild aerial shared chunks: cached chunk files missing or incomplete")
        else:
            missing_count = len(required_sources - existing_sources) if required_sources else 0
            info.setdefault("scope_id", scope_id)
            info.setdefault("dataset", dataset_label)
            info.setdefault("datasets", selected_datasets)
            info.setdefault("year", year)
            info.setdefault("source_files", [f"{r.dataset}/{r.file_name}" for r in selected])
            info["reused"] = True
            if missing_count > 0:
                log(
                    f"  Reuse aerial shared chunks (partial): {scope_id} "
                    f"(missing_selected_sources={missing_count})"
                )
            else:
                log(f"  Reuse aerial shared chunks: {scope_id}")
            log(f"  Aerial scope timing: {perf_counter() - t_scope_start:.1f}s (reused)")
            return grid, chunk_results, info

    log(f"  Build aerial shared chunks: {scope_id}")
    cache_stats = _prepare_aerial_cache_for_selected(
        job=job,
        scope_id=scope_id,
        selected=selected,
        df_session=df_session,
        df_api_key=df_api_key,
    )
    chunk_results, source_files_used, missing_sources, chunk_build_seconds, ingest_total_seconds = _build_scope_chunk_mosaics(
        job=job,
        selected=selected,
        target_res_m=job.aerial_target_res_m,
        record_cache_paths=cache_stats["record_cache_paths"],
        grid=grid,
        scope_id=scope_id,
    )
    total_scope_seconds = perf_counter() - t_scope_start
    log(
        "  Aerial scope timing: "
        f"cache={float(cache_stats.get('cache_stage_seconds', 0.0)):.1f}s "
        f"build={chunk_build_seconds:.1f}s "
        f"total={total_scope_seconds:.1f}s"
    )

    chunks_sidecar = [
        {
            "chunk_index": int(ch.get("chunk_index", 0)),
            "aerial_path": str(ch.get("aerial_path", "")),
            "cov_path": str(ch.get("cov_path", "")),
            "source_files_used": list(ch.get("source_files_used", [])),
            "missing_sources": int(ch.get("missing_sources", 0)),
            "row_off": int(ch.get("row_off", 0)),
            "col_off": int(ch.get("col_off", 0)),
            "width": int(ch.get("width", 0)),
            "height": int(ch.get("height", 0)),
            "has_data": bool(ch.get("has_data", False)),
            "build_seconds": float(ch.get("build_seconds", 0.0)),
        }
        for ch in chunk_results
    ]

    info = {
        "scope_id": scope_id,
        "dataset": dataset_label,
        "datasets": selected_datasets,
        "year": year,
        "chunks": chunks_sidecar,
        "source_files": source_files_used,
        "missing_sources": int(missing_sources),
        "downloaded_gib": float(cache_stats["downloaded_bytes"]) / (1024**3),
        "download_mode": "single_parallel",
        "single_downloads_ok": int(cache_stats["single_downloads_ok"]),
        "single_downloads_failed": int(cache_stats["single_downloads_failed"]),
        "cache_hits": int(cache_stats["cache_hits"]),
        "cache_misses": int(cache_stats["cache_misses"]),
        "cache_stage_seconds": float(cache_stats.get("cache_stage_seconds", 0.0)),
        "ingest_stage_seconds": float(ingest_total_seconds),
        "scope_total_seconds": float(total_scope_seconds),
        "reused": False,
    }
    sidecar.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return grid, chunk_results, info


def enrich_npz_with_aerial_inplace(
    npz_path: Path,
    chunk_sources: Sequence[Dict[str, Any]],
    s2_ref_ds: rasterio.DatasetReader,
    aerial_grid: Dict[str, Any],
    target_res_m: float,
    min_cov_frac: float,
    meta_updates: Dict[str, Any],
    missing_policy: str,
) -> Tuple[bool, str]:
    if not chunk_sources:
        return False, "no_chunk_sources"

    aerial_transform = aerial_grid["transform"]
    aerial_crs = aerial_grid["crs"]
    aerial_width = int(aerial_grid["width"])
    aerial_height = int(aerial_grid["height"])

    with np.load(npz_path, allow_pickle=True) as d:
        if "s1" not in d or "s2" not in d or "valid" not in d:
            return False, "missing_keys"

        s1 = d["s1"].astype(np.float32)
        s2 = d["s2"].astype(np.float32)
        valid = d["valid"].astype(np.uint8)
        row0_arr = np.asarray(d["row0"]).reshape(-1) if "row0" in d else np.array([0], dtype=np.int32)
        col0_arr = np.asarray(d["col0"]).reshape(-1) if "col0" in d else np.array([0], dtype=np.int32)
        row0 = int(row0_arr[0])
        col0 = int(col0_arr[0])
        tile = int(s2.shape[-1])

        s2_res_x = abs(float(s2_ref_ds.transform.a))
        s2_res_y = abs(float(s2_ref_ds.transform.e))
        a_res_x = abs(float(aerial_transform.a))
        a_res_y = abs(float(aerial_transform.e))
        sx = s2_res_x / a_res_x if a_res_x > 0 else 0.0
        sy = s2_res_y / a_res_y if a_res_y > 0 else 0.0
        if not (abs(sx - round(sx)) < 1e-6 and abs(sy - round(sy)) < 1e-6):
            return False, "non_integer_scale"
        if abs(sx - sy) > 1e-6:
            return False, "scale_mismatch"
        scale = int(round(sx))
        if scale <= 0:
            return False, "invalid_scale"

        x0, y0 = s2_ref_ds.transform * (col0, row0)
        if str(s2_ref_ds.crs) == str(aerial_crs):
            c1f, r1f = (~aerial_transform) * (x0, y0)
        else:
            tr = Transformer.from_crs(s2_ref_ds.crs, aerial_crs, always_xy=True)
            x0a, y0a = tr.transform(x0, y0)
            c1f, r1f = (~aerial_transform) * (x0a, y0a)
        c1 = int(round(c1f))
        r1 = int(round(r1f))
        size1 = tile * scale
        if r1 < 0 or c1 < 0 or (r1 + size1) > aerial_height or (c1 + size1) > aerial_width:
            return False, "out_of_bounds"

        aerial = np.zeros((4, size1, size1), dtype=np.uint8)
        cov = np.zeros((size1, size1), dtype=np.uint8)
        patch_r0, patch_c0 = r1, c1
        patch_r1, patch_c1 = r1 + size1, c1 + size1

        for chunk in chunk_sources:
            row_off = int(chunk["row_off"])
            col_off = int(chunk["col_off"])
            h = int(chunk["height"])
            w = int(chunk["width"])
            if h <= 0 or w <= 0:
                continue

            chunk_r0, chunk_c0 = row_off, col_off
            chunk_r1, chunk_c1 = row_off + h, col_off + w
            ov_r0 = max(patch_r0, chunk_r0)
            ov_c0 = max(patch_c0, chunk_c0)
            ov_r1 = min(patch_r1, chunk_r1)
            ov_c1 = min(patch_c1, chunk_c1)
            if ov_r0 >= ov_r1 or ov_c0 >= ov_c1:
                continue

            src_win = Window(
                int(ov_c0 - chunk_c0),
                int(ov_r0 - chunk_r0),
                int(ov_c1 - ov_c0),
                int(ov_r1 - ov_r0),
            )
            src_cov = chunk["cov_ds"].read(1, window=src_win).astype(np.uint8)
            mask = src_cov == 1
            if not np.any(mask):
                continue
            src_aerial = chunk["aerial_ds"].read([1, 2, 3, 4], window=src_win).astype(np.uint8)

            dst_r0 = int(ov_r0 - patch_r0)
            dst_c0 = int(ov_c0 - patch_c0)
            dst_r1 = dst_r0 + int(ov_r1 - ov_r0)
            dst_c1 = dst_c0 + int(ov_c1 - ov_c0)

            cov_block = cov[dst_r0:dst_r1, dst_c0:dst_c1]
            cov_block[mask] = 1
            cov[dst_r0:dst_r1, dst_c0:dst_c1] = cov_block
            for bi in range(4):
                dst_block = aerial[bi, dst_r0:dst_r1, dst_c0:dst_c1]
                dst_block[mask] = src_aerial[bi][mask]
                aerial[bi, dst_r0:dst_r1, dst_c0:dst_c1] = dst_block

        cov_frac = float((cov == 1).mean())
        if missing_policy == "drop" and cov_frac < float(min_cov_frac):
            return False, "missing_aerial"

        meta = {}
        try:
            meta = json.loads(str(d["meta"]))
        except Exception:
            meta = {}
        meta.update(meta_updates)
        meta.update(
            {
                "aerial_bands": ["R", "G", "B", "NIR"],
                "aerial_res_m": float(target_res_m),
                "aerial_scale": int(scale),
                "aerial_mosaic_crs": str(aerial_crs),
                "aerial_mosaic_transform": [float(x) for x in aerial_transform[:6]],
                "aerial_tile_hw": [int(size1), int(size1)],
                "aerial_cov_frac": float(cov_frac),
            }
        )

    tmp = npz_path.with_name(npz_path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        s1=s1,
        s2=s2,
        aerial=aerial,
        valid=valid,
        row0=np.array(int(row0), dtype=np.int32),
        col0=np.array(int(col0), dtype=np.int32),
        meta=np.array(json.dumps(meta)),
    )
    tmp.replace(npz_path)
    return True, "ok"


# -----------------------------
# Aerial enrichment per pass
# -----------------------------
def enrich_pass_patch_files_with_aerial(
    job: Job,
    s2_pass_key: str,
    s2_b02_path: Path,
    patch_paths: List[Path],
    catalog: List[CatalogRecord],
    df_session: requests.Session,
    df_api_key: str,
    final_out_dir: Optional[Path] = None,
) -> Tuple[int, int]:
    if not patch_paths:
        return 0, 0

    with rasterio.open(s2_b02_path) as ds:
        target_crs = str(ds.crs)
    aoi = box(*job.bbox_lonlat)

    pass_dt = parse_s2_pass_datetime(s2_pass_key)
    target_year = pass_dt.year if pass_dt else None
    selected, dataset, sel_year = select_aerial_records_for_pass(
        records=catalog,
        dataset_priority=AERIAL_DATASET_PRIORITY,
        aoi_wgs84=aoi,
        target_year=target_year,
        date_policy=AERIAL_DATE_POLICY,
    )
    selected = dedupe_catalog_records(selected)

    if not selected or not dataset:
        log(f"  Aerial: no intersecting tiles for pass {s2_pass_key}; dropping {len(patch_paths)} patch files")
        for p in patch_paths:
            p.unlink(missing_ok=True)
        return 0, len(patch_paths)

    counts_by_dataset: Dict[str, int] = {}
    for r in selected:
        counts_by_dataset[r.dataset] = counts_by_dataset.get(r.dataset, 0) + 1
    mix_str = ", ".join(f"{k}:{v}" for k, v in counts_by_dataset.items())
    log(f"  Aerial dataset mix: {mix_str}")

    aerial_grid, chunk_results, info = build_or_reuse_aerial_mosaic_for_scope(
        job=job,
        target_crs=target_crs,
        selected=selected,
        year=sel_year,
        df_session=df_session,
        df_api_key=df_api_key,
    )
    if not chunk_results:
        log(f"  Aerial: no usable chunk mosaics for pass {s2_pass_key}; dropping {len(patch_paths)} patch files")
        for p in patch_paths:
            p.unlink(missing_ok=True)
        return 0, len(patch_paths)

    min_cov_frac = max(0.0, min(1.0, float(job.aerial_min_cov_frac)))

    kept = 0
    dropped = 0
    datasets_used = list(counts_by_dataset.keys())
    meta_updates = {
        "aerial_dataset": dataset,
        "aerial_datasets": datasets_used,
        "aerial_year": sel_year,
        "aerial_scope_id": info.get("scope_id"),
        "aerial_source_files": list(info.get("source_files", [f"{r.dataset}/{r.file_name}" for r in selected])),
        "aerial_date_policy": AERIAL_DATE_POLICY,
        "aerial_missing_policy": AERIAL_MISSING_POLICY,
        "aerial_min_cov_frac": float(min_cov_frac),
    }
    log(f"  Aerial coverage threshold: min_cov_frac={min_cov_frac:.2f}")

    chunk_sources: List[Dict[str, Any]] = []
    try:
        for ch in sorted(chunk_results, key=lambda x: int(x.get("chunk_index", 0))):
            if not bool(ch.get("has_data", False)):
                continue
            ap = Path(ch["aerial_path"])
            cp = Path(ch["cov_path"])
            if not (file_is_ready(ap) and file_is_ready(cp)):
                continue
            chunk_sources.append(
                {
                    "row_off": int(ch.get("row_off", 0)),
                    "col_off": int(ch.get("col_off", 0)),
                    "width": int(ch.get("width", 0)),
                    "height": int(ch.get("height", 0)),
                    "aerial_ds": rasterio.open(ap),
                    "cov_ds": rasterio.open(cp),
                }
            )

        if not chunk_sources:
            log(f"  Aerial: chunk files missing for pass {s2_pass_key}; dropping {len(patch_paths)} patch files")
            for p in patch_paths:
                p.unlink(missing_ok=True)
            return 0, len(patch_paths)

        with rasterio.open(s2_b02_path) as s2_ref_ds:
            for npz_path in patch_paths:
                ok, _ = enrich_npz_with_aerial_inplace(
                    npz_path=npz_path,
                    chunk_sources=chunk_sources,
                    s2_ref_ds=s2_ref_ds,
                    aerial_grid=aerial_grid,
                    target_res_m=job.aerial_target_res_m,
                    min_cov_frac=min_cov_frac,
                    meta_updates=meta_updates,
                    missing_policy=AERIAL_MISSING_POLICY,
                )
                if ok:
                    if final_out_dir is not None and npz_path.parent != final_out_dir:
                        final_out_dir.mkdir(parents=True, exist_ok=True)
                        final_path = final_out_dir / npz_path.name
                        npz_path.replace(final_path)
                    kept += 1
                else:
                    dropped += 1
                    npz_path.unlink(missing_ok=True)
    finally:
        for ch in chunk_sources:
            try:
                ch["aerial_ds"].close()
            except Exception:
                pass
            try:
                ch["cov_ds"].close()
            except Exception:
                pass

    return kept, dropped


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


def aligned_geom_grid(
    geom_lonlat: BaseGeometry,
    target_crs: str,
    resolution: float = 10.0,
) -> Dict[str, Any]:
    if geom_lonlat.is_empty:
        raise ValueError("Cannot build grid from empty geometry")

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    geom_proj = shapely_transform(transformer.transform, geom_lonlat)
    if geom_proj.is_empty:
        raise ValueError("Projected geometry is empty")

    min_x, min_y, max_x, max_y = geom_proj.bounds

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
    with rasterio.open(out_path, "w", **with_bigtiff(profile)):
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
                    num_threads=REPROJECT_THREADS,
                )
            first = False


def build_s2_mosaics(
    job: Job,
    s2_pass_key: str,
    s2_zips: List[Path],
    target_crs: str,
    s2_target_geom: Optional[BaseGeometry] = None,
) -> Dict[str, Path]:
    if s2_target_geom is not None:
        try:
            grid = aligned_geom_grid(s2_target_geom, target_crs, resolution=10.0)
        except Exception:
            grid = aligned_aoi_grid(job.bbox_lonlat, target_crs, resolution=10.0)
    else:
        grid = aligned_aoi_grid(job.bbox_lonlat, target_crs, resolution=10.0)
    mosaic_id = f"{job.name}__{s2_pass_key}"

    expected_out: Dict[str, Path] = {band: MOSAIC_DIR / f"{mosaic_id}__{band}.tif" for band in S2_BANDS_11}
    expected_out["SCL"] = MOSAIC_DIR / f"{mosaic_id}__SCL.tif"
    if all(file_is_ready(p) for p in expected_out.values()):
        log(f"  Reuse S2 stitched mosaic cache: {mosaic_id}")
        return expected_out

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
                    num_threads=REPROJECT_THREADS,
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
                    num_threads=REPROJECT_THREADS,
                )

    # Convert linear sigma0 -> dB in-place window by window.
    with rasterio.open(out_path, "r+") as ds:
        for _, window in ds.block_windows(1):
            block = ds.read([1, 2], window=window).astype(np.float32)
            block = 10.0 * np.log10(np.maximum(block, 1e-10))
            ds.write(block, window=window)

    return out_path


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def process_and_warp_single_s1_scene(
    job: Job,
    s2_pass_key: str,
    target_crs: str,
    s2_ref_grid_path: Path,
    s1_product: Dict[str, Any],
    s1_zip: Path,
) -> Path:
    scene_tag = sanitize_name(s1_product["Name"])
    warped = MOSAIC_DIR / f"{job.name}__{s2_pass_key}__{scene_tag}__S1_on_S2_src.tif"
    if file_is_ready(warped):
        log(f"  Reuse S1 warped cache: {warped.name}")
        return warped

    vv_img, vh_img = process_s1_to_tc_imgs(s1_zip, target_crs)
    warp_s1_to_grid_db(vv_img, vh_img, s2_ref_grid_path, warped)
    return warped


def merge_warped_s1_scenes(warped_s1_paths: List[Path], out_path: Path) -> Path:
    if not warped_s1_paths:
        raise ValueError("No warped S1 scenes provided for merge")

    if len(warped_s1_paths) == 1:
        src = warped_s1_paths[0]
        if src != out_path:
            shutil.copyfile(src, out_path)
        return out_path

    handles = [rasterio.open(p) for p in warped_s1_paths]
    try:
        ref = handles[0]
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
        with rasterio.open(out_path, "w", **with_bigtiff(profile)) as out:
            for _, window in ref.block_windows(1):
                h = int(window.height)
                w = int(window.width)
                acc = np.zeros((2, h, w), dtype=np.float64)
                cnt = np.zeros((2, h, w), dtype=np.uint16)

                for ds in handles:
                    blk = ds.read([1, 2], window=window).astype(np.float32)
                    ok = np.isfinite(blk)
                    acc[ok] += blk[ok]
                    cnt[ok] += 1

                merged = np.full((2, h, w), np.nan, dtype=np.float32)
                valid = cnt > 0
                merged[valid] = (acc[valid] / cnt[valid]).astype(np.float32)
                out.write(merged, window=window)
    finally:
        for ds in handles:
            ds.close()

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
    min_valid_frac: float,
) -> Dict[str, Any]:
    accepted: List[Tuple[int, int]] = []

    total_windows = 0
    total_scanned = 0
    prefilter_nonzero = 0
    skipped_valid = 0
    fail_clear = 0
    fail_nonzero = 0
    fail_s1ok = 0
    fail_combo = {
        "clear_only": 0,
        "nonzero_only": 0,
        "s1_only": 0,
        "clear_nonzero": 0,
        "clear_s1": 0,
        "nonzero_s1": 0,
        "clear_nonzero_s1": 0,
    }

    with rasterio.open(b02_path) as b02_ds, rasterio.open(scl_path) as scl_ds, rasterio.open(s1_path) as s1_ds:
        height, width = b02_ds.height, b02_ds.width
        if height < tile or width < tile:
            return {
                "indices": accepted,
                "total_windows": 0,
                "total_scanned": 0,
                "prefilter_nonzero": 0,
                "skipped_valid": 0,
                "fail_clear": 0,
                "fail_nonzero": 0,
                "fail_s1ok": 0,
                "fail_combo": fail_combo,
            }

        # Estimate valid-S2 footprint (B02 > 0) and scan only windows intersecting it.
        row_min = height
        row_max = -1
        col_min = width
        col_max = -1
        for _, win in b02_ds.block_windows(1):
            block = b02_ds.read(1, window=win)
            nz = block > 0
            if not np.any(nz):
                continue
            rr = np.where(np.any(nz, axis=1))[0]
            cc = np.where(np.any(nz, axis=0))[0]
            if rr.size == 0 or cc.size == 0:
                continue
            row_min = min(row_min, int(win.row_off) + int(rr[0]))
            row_max = max(row_max, int(win.row_off) + int(rr[-1]))
            col_min = min(col_min, int(win.col_off) + int(cc[0]))
            col_max = max(col_max, int(win.col_off) + int(cc[-1]))

        if row_max < 0 or col_max < 0:
            return {
                "indices": accepted,
                "total_windows": 0,
                "total_scanned": 0,
                "prefilter_nonzero": 0,
                "skipped_valid": 0,
                "fail_clear": 0,
                "fail_nonzero": 0,
                "fail_s1ok": 0,
                "fail_combo": fail_combo,
            }

        row_starts = [
            r0
            for r0 in range(0, height - tile + 1, stride)
            if not (r0 + tile - 1 < row_min or r0 > row_max)
        ]
        col_starts = [
            c0
            for c0 in range(0, width - tile + 1, stride)
            if not (c0 + tile - 1 < col_min or c0 > col_max)
        ]
        total_windows = len(row_starts) * len(col_starts)

        for r0 in row_starts:
            for c0 in col_starts:
                win = Window(c0, r0, tile, tile)

                b02 = b02_ds.read(1, window=win)
                nonzero = b02 > 0
                nonzero_frac = float(nonzero.mean())
                if nonzero_frac < min_valid_frac:
                    prefilter_nonzero += 1
                    fail_nonzero += 1
                    fail_combo["nonzero_only"] += 1
                    continue

                total_scanned += 1
                scl = scl_ds.read(1, window=win).astype(np.uint8)
                s1 = s1_ds.read([1, 2], window=win).astype(np.float32)

                clear = ~np.isin(scl, list(SCL_INVALID))
                s1_ok = np.isfinite(s1[0]) & np.isfinite(s1[1]) & (s1[0] > -80) & (s1[1] > -80)
                clear_frac = float(clear.mean())
                s1ok_frac = float(s1_ok.mean())

                valid = clear & nonzero & s1_ok
                valid_frac = float(valid.mean())
                if valid_frac < min_valid_frac:
                    skipped_valid += 1
                    clear_fail = clear_frac < min_valid_frac
                    nonzero_fail = False
                    s1ok_fail = s1ok_frac < min_valid_frac
                    if clear_fail:
                        fail_clear += 1
                    if nonzero_fail:
                        fail_nonzero += 1
                    if s1ok_fail:
                        fail_s1ok += 1

                    flags = (clear_fail, nonzero_fail, s1ok_fail)
                    if flags == (True, False, False):
                        fail_combo["clear_only"] += 1
                    elif flags == (False, True, False):
                        fail_combo["nonzero_only"] += 1
                    elif flags == (False, False, True):
                        fail_combo["s1_only"] += 1
                    elif flags == (True, True, False):
                        fail_combo["clear_nonzero"] += 1
                    elif flags == (True, False, True):
                        fail_combo["clear_s1"] += 1
                    elif flags == (False, True, True):
                        fail_combo["nonzero_s1"] += 1
                    elif flags == (True, True, True):
                        fail_combo["clear_nonzero_s1"] += 1
                    continue

                accepted.append((r0, c0))

    return {
        "indices": accepted,
        "total_windows": total_windows,
        "total_scanned": total_scanned,
        "prefilter_nonzero": prefilter_nonzero,
        "skipped_valid": skipped_valid,
        "fail_clear": fail_clear,
        "fail_nonzero": fail_nonzero,
        "fail_s1ok": fail_s1ok,
        "fail_combo": fail_combo,
    }


def write_npz_single_file(
    out_path: Path,
    s2_band_paths: Dict[str, Path],
    scl_path: Path,
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
        with rasterio.open(s1_path) as s1_ds, rasterio.open(scl_path) as scl_ds:
            for band in S2_BANDS_11:
                s2_handles[band] = rasterio.open(s2_band_paths[band])

            for i, (r0, c0) in enumerate(indices):
                win = Window(c0, r0, tile, tile)

                s1_patch = s1_ds.read([1, 2], window=win).astype(np.float32)
                s1_mm[i] = s1_patch
                for b_idx, band in enumerate(S2_BANDS_11):
                    s2_mm[i, b_idx] = s2_handles[band].read(1, window=win).astype(np.float32)

                scl = scl_ds.read(1, window=win).astype(np.uint8)
                clear = ~np.isin(scl, list(SCL_INVALID))
                nonzero = s2_mm[i, S2_BANDS_11.index("B02")] > 0
                s1_ok = (
                    np.isfinite(s1_patch[0])
                    & np.isfinite(s1_patch[1])
                    & (s1_patch[0] > -80)
                    & (s1_patch[1] > -80)
                )
                valid_mm[i] = (clear & nonzero & s1_ok).astype(np.uint8)
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
    scl_path: Path,
    s1_path: Path,
    indices: List[Tuple[int, int]],
    tile: int,
    meta_base: Dict[str, Any],
    require_aerial: bool = False,
    staging_out_dir: Optional[Path] = None,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    use_staging = require_aerial and staging_out_dir is not None
    if use_staging and staging_out_dir is not None:
        staging_out_dir.mkdir(parents=True, exist_ok=True)
    wrote_paths: List[Path] = []

    if not indices:
        return wrote_paths

    existing_reused = 0
    existing_need_enrich = 0
    existing_pending_reused = 0
    to_write: List[Tuple[int, int, int, Path]] = []
    for i, (r0, c0) in enumerate(indices):
        final_out = out_dir / f"{base_name}_r{r0}_c{c0}.npz"
        if file_is_ready(final_out):
            if require_aerial and not npz_has_aerial(final_out):
                existing_need_enrich += 1
                wrote_paths.append(final_out)
            else:
                existing_reused += 1
            continue

        out = final_out
        if use_staging and staging_out_dir is not None:
            out = staging_out_dir / final_out.name
            if file_is_ready(out):
                existing_pending_reused += 1
                wrote_paths.append(out)
                continue

        to_write.append((i, r0, c0, out))

    if existing_reused > 0 or existing_need_enrich > 0 or existing_pending_reused > 0:
        log(
            "    Patch cache: "
            f"reused={existing_reused} "
            f"needs_aerial_enrich={existing_need_enrich} "
            f"pending_reused={existing_pending_reused} "
            f"to_write={len(to_write)}"
        )

    if not to_write:
        return wrote_paths

    s2_handles: Dict[str, rasterio.DatasetReader] = {}
    try:
        with rasterio.open(s1_path) as s1_ds, rasterio.open(scl_path) as scl_ds:
            for band in S2_BANDS_11:
                s2_handles[band] = rasterio.open(s2_band_paths[band])

            for j, (i, r0, c0, out) in enumerate(to_write):
                win = Window(c0, r0, tile, tile)
                s1 = s1_ds.read([1, 2], window=win).astype(np.float32)

                s2 = np.empty((len(S2_BANDS_11), tile, tile), dtype=np.float32)
                for b_idx, band in enumerate(S2_BANDS_11):
                    s2[b_idx] = s2_handles[band].read(1, window=win).astype(np.float32)

                scl = scl_ds.read(1, window=win).astype(np.uint8)
                clear = ~np.isin(scl, list(SCL_INVALID))
                nonzero = s2[S2_BANDS_11.index("B02")] > 0
                s1_ok = np.isfinite(s1[0]) & np.isfinite(s1[1]) & (s1[0] > -80) & (s1[1] > -80)
                valid = (clear & nonzero & s1_ok).astype(np.uint8)

                meta = dict(meta_base)
                meta["row0"] = int(r0)
                meta["col0"] = int(c0)
                meta["patch_index"] = int(i)

                np.savez_compressed(
                    out,
                    s1=s1,
                    s2=s2,
                    valid=valid,
                    row0=np.array(int(r0), dtype=np.int32),
                    col0=np.array(int(c0), dtype=np.int32),
                    meta=np.array(json.dumps(meta)),
                )
                wrote_paths.append(out)

                if is_progress_tick(j + 1, len(to_write), 500):
                    log(f"    Wrote patch files: {j + 1}/{len(to_write)}")
    finally:
        for ds in s2_handles.values():
            ds.close()

    return wrote_paths


# -----------------------------
# Pipeline core
# -----------------------------
def build_stitched_npz_for_pass(
    job: Job,
    s2_pass_key: str,
    s2_products: List[Dict[str, Any]],
    s1_products: List[Dict[str, Any]],
    token: str,
    s2_target_geom: Optional[BaseGeometry] = None,
    aerial_catalog: Optional[List[CatalogRecord]] = None,
    df_session: Optional[requests.Session] = None,
    df_api_key: str = "",
) -> str:
    # Download all S2 tiles in pass + selected S1 products in parallel.
    s2_zips, s1_zips, token = download_s1_s2_parallel(
        s2_products=s2_products,
        s1_products=s1_products,
        token=token,
    )

    target_crs = utm_epsg_from_bbox(job.bbox_lonlat)
    log(f"  Target CRS: {target_crs}")

    # 1) Stitch S2 AOI mosaics.
    s2_mosaic = build_s2_mosaics(
        job,
        s2_pass_key,
        s2_zips,
        target_crs,
        s2_target_geom=s2_target_geom,
    )

    # 2) Process and align each S1, then merge S1 coverage on S2 grid.
    s1_workers = max(1, min(len(s1_products), S1_SCENE_WORKERS))
    log(
        "  S1 processing tuning: "
        f"workers={s1_workers} (auto={AUTO_S1_WORKERS}) "
        f"reproject_threads={REPROJECT_THREADS}"
    )
    warped_slots: List[Optional[Path]] = [None] * len(s1_products)
    if s1_workers == 1:
        for idx, (p, s1_zip) in enumerate(zip(s1_products, s1_zips)):
            warped_slots[idx] = process_and_warp_single_s1_scene(
                job=job,
                s2_pass_key=s2_pass_key,
                target_crs=target_crs,
                s2_ref_grid_path=s2_mosaic["B02"],
                s1_product=p,
                s1_zip=s1_zip,
            )
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=s1_workers) as pool:
            for idx, (p, s1_zip) in enumerate(zip(s1_products, s1_zips)):
                fut = pool.submit(
                    process_and_warp_single_s1_scene,
                    job,
                    s2_pass_key,
                    target_crs,
                    s2_mosaic["B02"],
                    p,
                    s1_zip,
                )
                futures[fut] = idx

            done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                warped_slots[idx] = fut.result()
                done += 1
                if is_progress_tick(done, len(futures), 2):
                    log(f"  S1 processing progress: {done}/{len(futures)}")

    if any(p is None for p in warped_slots):
        raise RuntimeError("Failed to build all S1 warped sources")
    warped_s1_paths = [p for p in warped_slots if p is not None]

    s1_merge_tag = f"S1multi{len(s1_products)}_{parse_dt(s1_products[0]).strftime('%Y%m%dT%H%M%S')}"
    s1_on_s2 = MOSAIC_DIR / f"{job.name}__{s2_pass_key}__{s1_merge_tag}__S1_on_S2.tif"
    if file_is_ready(s1_on_s2):
        log(f"  Reuse S1 merged cache: {s1_on_s2.name}")
    else:
        merge_warped_s1_scenes(warped_s1_paths, s1_on_s2)

    # 3) Scan valid windows using configured min_valid_frac.
    scan = scan_valid_patch_indices(
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
        combo = scan["fail_combo"]
        log(
            "  skipped_valid combos: "
            f"s1_only={combo['s1_only']} "
            f"nonzero_only={combo['nonzero_only']} "
            f"clear_only={combo['clear_only']} "
            f"nonzero_s1={combo['nonzero_s1']} "
            f"clear_s1={combo['clear_s1']} "
            f"clear_nonzero={combo['clear_nonzero']} "
            f"all_three={combo['clear_nonzero_s1']}"
        )

    indices = scan["indices"]
    with rasterio.open(s2_mosaic["B02"]) as b02_ds:
        mosaic_shape = (b02_ds.height, b02_ds.width)
        mosaic_bounds = tuple(float(x) for x in b02_ds.bounds)
        mosaic_crs = str(b02_ds.crs)

    base_name = f"{job.name}__{s2_pass_key}__{s1_merge_tag}"

    meta = {
        "job": job.name,
        "s2_pass": s2_pass_key,
        "s2_tiles": [p["Name"] for p in s2_products],
        "s1_scenes": [p["Name"] for p in s1_products],
        "tile": int(job.tile),
        "stride": int(job.stride),
        "min_valid_frac": float(job.min_valid_frac),
        "s2_bands": S2_BANDS_11,
        "scl_invalid": sorted(SCL_INVALID),
        "target_crs": mosaic_crs,
        "mosaic_shape_hw": [int(mosaic_shape[0]), int(mosaic_shape[1])],
        "mosaic_bounds": list(mosaic_bounds),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # 4) Write outputs in selected mode.
    if job.output_mode == "per_pass":
        out_path = OUT_DIR / f"{base_name}.npz"
        write_npz_single_file(
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
        patch_paths = write_npz_per_patch(
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
        kept, dropped_aerial = enrich_pass_patch_files_with_aerial(
            job=job,
            s2_pass_key=s2_pass_key,
            s2_b02_path=s2_mosaic["B02"],
            patch_paths=patch_paths,
            catalog=aerial_catalog,
            df_session=df_session,
            df_api_key=df_api_key,
            final_out_dir=OUT_DIR,
        )
        wrote = kept
        shutil.rmtree(staging_patch_dir, ignore_errors=True)
        log(
            f"  Wrote patch NPZ files: {wrote} | "
            f"windows={scan['total_scanned']} skipped_valid={scan['skipped_valid']} "
            f"dropped_aerial={dropped_aerial}"
        )

    return token


def run_batch() -> None:
    token = cdse_token()

    for job in JOBS:
        if job.output_mode != "per_patch":
            raise ValueError(
                f"JOB {job.name}: pipeline5 requires output_mode='per_patch' "
                f"(got '{job.output_mode}')"
            )

        df_api_key = datafordeler_api_key()
        df_session = build_retry_session()
        aerial_catalog = build_aerial_catalog(
            session=df_session,
            api_key=df_api_key,
            datasets=AERIAL_DATASET_PRIORITY,
            rebuild=AERIAL_REBUILD_INDEX,
        )
        if not aerial_catalog:
            raise RuntimeError("Aerial catalog is empty; cannot enrich with aerial")

        log(
            f"\n=== JOB {job.name} | dates={job.date_start}..{job.date_end} "
            f"bbox={job.bbox_lonlat} ==="
        )
        log(
            "Runtime tuning: "
            f"cpu={CPU_COUNT} "
            f"reproject_threads={REPROJECT_THREADS} "
            f"s1_workers={S1_SCENE_WORKERS} "
            f"cdse_download_workers={CDSE_DOWNLOAD_WORKERS} "
            f"cdse_download_attempts={CDSE_DOWNLOAD_ATTEMPTS} "
            f"cdse_retry_base_s={CDSE_DOWNLOAD_RETRY_BASE_SECONDS} "
            f"aerial_download_workers={max(1, int(job.aerial_workers))} "
            f"aerial_chunk_workers={max(1, int(job.aerial_chunk_workers))} "
            f"aerial_download_attempts={AERIAL_DOWNLOAD_ATTEMPTS} "
            f"aerial_retry_base_s={AERIAL_DOWNLOAD_RETRY_BASE_SECONDS}"
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
            s2_target_geom = build_s2_pass_target_geometry(pass_products, job.bbox_lonlat)
            s1_search_top = max(40, job.max_s1_scenes * 12)
            if s1_search_top > ODATA_TOP_MAX:
                log(
                    f"  S1 search $top capped: requested={s1_search_top} "
                    f"capped={ODATA_TOP_MAX}"
                )
            s1_cands, token = odata_search_s1(
                token=token,
                bbox_lonlat=job.bbox_lonlat,
                dt_center_iso=pass_products[0]["ContentDate"]["Start"],
                hours=job.max_time_diff_hours,
                top=s1_search_top,
            )

            s1_selected, s1_stats = pick_s1_scenes_by_overlap_time(
                candidates=s1_cands,
                target_dt=s2_center,
                max_time_diff_hours=job.max_time_diff_hours,
                max_scenes=job.max_s1_scenes,
                target_geom=s2_target_geom,
            )
            if not s1_selected:
                log(f"Skip pass {s2_pass_key}: no S1 candidate within {job.max_time_diff_hours}h")
                continue

            gaps = [float(p.get("_s1_time_gap_hours", abs((parse_dt(p) - s2_center).total_seconds()) / 3600.0)) for p in s1_selected]
            overlaps = [float(p.get("_s1_overlap_ratio", 0.0)) for p in s1_selected]
            gap_min, gap_max = min(gaps), max(gaps)
            ov_min, ov_max = min(overlaps), max(overlaps)

            log(
                f"PASS {s2_pass_key} | S2 tiles={len(pass_products)} | "
                f"S1 scenes={len(s1_selected)} | gap_range={gap_min:.1f}-{gap_max:.1f}h | "
                f"ov_range={ov_min:.2f}-{ov_max:.2f} | "
                f"S1 raw/within/dedup={s1_stats['raw']}/{s1_stats['within_time']}/{s1_stats['deduped']}"
            )

            token = build_stitched_npz_for_pass(
                job=job,
                s2_pass_key=s2_pass_key,
                s2_products=pass_products,
                s1_products=s1_selected,
                token=token,
                s2_target_geom=s2_target_geom,
                aerial_catalog=aerial_catalog,
                df_session=df_session,
                df_api_key=df_api_key,
            )


if __name__ == "__main__":
    run_batch()
