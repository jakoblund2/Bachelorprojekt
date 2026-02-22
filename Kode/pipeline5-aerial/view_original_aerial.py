#!/usr/bin/env python3
"""
Download one original Datafordeler aerial tile and compare it with cached 1m tile.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from dotenv import load_dotenv
from rasterio.enums import Resampling
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DF_BASE = "https://api.datafordeler.dk/FileDownloads"
DF_REGISTER = "GeoDKO"

# -----------------------------
# User config (edit and run)
# -----------------------------
DATASET = "GeoDKO10cm"
FILE_NAME = "2025_1km_6172_487.tif"
OUT_DIR = Path("pipeline5-aerial/data/debug_original")
CACHE_ROOT = Path("pipeline5-aerial/data/aerial_cache_1m")
MAX_DIM = 1400


def log(msg: str) -> None:
    print(msg, flush=True)


def load_api_key() -> str:
    script_env = Path(__file__).with_name(".env")
    if script_env.exists():
        load_dotenv(script_env)
    load_dotenv()
    key = os.environ.get("DATAFORDELER_APIKEY")
    if not key:
        raise RuntimeError("Missing DATAFORDELER_APIKEY in environment or pipeline5-aerial/.env")
    return key


def build_session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def ensure_tif_name(name: str) -> str:
    n = name.strip()
    return n if n.lower().endswith(".tif") else f"{n}.tif"


def download_original(
    session: requests.Session,
    api_key: str,
    dataset: str,
    file_name: str,
    out_path: Path,
) -> Path:
    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        log(f"Reuse existing original: {out_path}")
        return out_path

    params = {
        "apiKey": api_key,
        "Register": DF_REGISTER,
        "DataSetName": dataset,
        "Version": 1,
        "Filename": file_name,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    with session.get(f"{DF_BASE}/GetRasterFile", params=params, stream=True, timeout=(30, 1800)) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(out_path)
    log(f"Downloaded original: {out_path} ({out_path.stat().st_size} bytes)")
    return out_path


def stretch_rgb(arr: np.ndarray) -> np.ndarray:
    # arr: H, W, 3
    m = arr[..., 0] > 0
    if np.any(m):
        p98 = np.percentile(arr[m], 98)
    else:
        p98 = 1.0
    return np.clip(arr / (p98 + 1e-9), 0, 1)


def read_rgb_preview(path: Path, max_dim: int = 1400) -> np.ndarray:
    with rasterio.open(path) as ds:
        if ds.count < 3:
            raise RuntimeError(f"Expected >=3 bands for RGB in {path}, got {ds.count}")

        h, w = ds.height, ds.width
        scale = max(h / max_dim, w / max_dim, 1.0)
        oh = max(1, int(round(h / scale)))
        ow = max(1, int(round(w / scale)))

        rgb = ds.read(
            [1, 2, 3],
            out_shape=(3, oh, ow),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        rgb = np.moveaxis(rgb, 0, -1)
        return stretch_rgb(rgb)


def find_cached_tile(cache_root: Path, dataset: str, file_name: str) -> Optional[Path]:
    stem = Path(file_name).stem
    p = cache_root / dataset / f"{stem}__1p0m.tif"
    if p.exists():
        return p
    return None


def main() -> int:
    dataset = DATASET.strip()
    file_name = ensure_tif_name(FILE_NAME)
    out_dir = OUT_DIR
    cache_root = CACHE_ROOT
    max_dim = max(256, int(MAX_DIM))

    if not dataset:
        raise RuntimeError("DATASET is empty")
    if not file_name:
        raise RuntimeError("FILE_NAME is empty")

    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()
    session = build_session()

    out_orig = out_dir / f"{dataset}__{Path(file_name).name}"
    orig = download_original(session, api_key, dataset, file_name, out_orig)
    cached = find_cached_tile(cache_root, dataset, file_name)

    with rasterio.open(orig) as ds:
        log(
            "Original meta: "
            f"bands={ds.count} dtype={ds.dtypes[0]} size={ds.width}x{ds.height} "
            f"res={ds.res} crs={ds.crs}"
        )
    if cached is not None:
        with rasterio.open(cached) as ds:
            log(
                "Cached meta: "
                f"bands={ds.count} dtype={ds.dtypes[0]} size={ds.width}x{ds.height} "
                f"res={ds.res} crs={ds.crs}"
            )
    else:
        log("Cached tile not found for this filename/dataset.")

    orig_rgb = read_rgb_preview(orig, max_dim=max_dim)
    fig_cols = 2 if cached is not None else 1
    plt.figure(figsize=(7 * fig_cols, 6))
    plt.subplot(1, fig_cols, 1)
    plt.title("Original (Datafordeler)")
    plt.imshow(orig_rgb)
    plt.axis("off")

    if cached is not None:
        cached_rgb = read_rgb_preview(cached, max_dim=max_dim)
        plt.subplot(1, fig_cols, 2)
        plt.title("Cached 1m")
        plt.imshow(cached_rgb)
        plt.axis("off")

    plt.tight_layout()
    preview = out_dir / f"{dataset}__{Path(file_name).stem}__compare.png"
    plt.savefig(preview, dpi=150)
    log(f"Wrote preview: {preview}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
