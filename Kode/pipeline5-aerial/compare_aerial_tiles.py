#!/usr/bin/env python3
"""
Download two original aerial tiles from a Datafordeler page and compare them.
Edit variables below and run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
PAGE = 1
# 1-based positions on the selected page. Must contain exactly two entries.
TILE_NUMBERS = (1, 2)
OUT_DIR = Path("pipeline5-aerial/data/debug_compare")
MAX_DIM = 1600
P_LOW = 2.0
P_HIGH = 98.0
SAVE_FIG = True


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


def _nk(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _nm(d: Dict[str, Any]) -> Dict[str, Any]:
    return {_nk(k): v for k, v in d.items()}


def _ga(item: Dict[str, Any], *keys: str) -> Any:
    nm = _nm(item)
    for k in keys:
        nk = _nk(k)
        if nk in nm:
            return nm[nk]
    return None


def fetch_available_page(
    session: requests.Session,
    api_key: str,
    dataset: str,
    page: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    params = {
        "apiKey": api_key,
        "Register": DF_REGISTER,
        "DataSetName": dataset,
        "Version": 1,
        "FileFormat": "tif",
        "PageNumber": page,
    }
    r = session.get(f"{DF_BASE}/GetAvailableRasterFileDownloads", params=params, timeout=(30, 300))
    r.raise_for_status()
    payload = r.json()
    items = payload.get("availableFileDownloads") or payload.get("AvailableFileDownloads") or []
    meta = payload.get("paginationMetadata") or payload.get("PaginationMetadata") or {}
    return list(items), dict(meta)


def list_file_names(items: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for item in items:
        fn = _ga(item, "fileName")
        if fn:
            out.append(str(fn))
    return out


def ensure_tif_name(name: str) -> str:
    n = name.strip()
    return n if n.lower().endswith(".tif") else f"{n}.tif"


def download_single_file(
    session: requests.Session,
    api_key: str,
    dataset: str,
    file_name: str,
    out_path: Path,
) -> Path:
    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        log(f"Reuse existing: {out_path.name}")
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
    log(f"Downloaded: {out_path.name} ({out_path.stat().st_size} bytes)")
    return out_path


def read_rgb_preview(path: Path, max_dim: int) -> Tuple[np.ndarray, Dict[str, object]]:
    with rasterio.open(path) as ds:
        if ds.count < 3:
            raise RuntimeError(f"{path} has {ds.count} bands, expected >= 3")
        h, w = ds.height, ds.width
        scale = max(h / max_dim, w / max_dim, 1.0)
        oh = max(1, int(round(h / scale)))
        ow = max(1, int(round(w / scale)))
        rgb = ds.read([1, 2, 3], out_shape=(3, oh, ow), resampling=Resampling.bilinear).astype(np.float32)
        rgb = np.moveaxis(rgb, 0, -1)
        meta = {
            "size": f"{w}x{h}",
            "preview": f"{ow}x{oh}",
            "dtype": ds.dtypes[0],
            "res": ds.res,
            "crs": str(ds.crs),
            "count": ds.count,
        }
        return rgb, meta


def valid_pixels(rgb: np.ndarray) -> np.ndarray:
    m = (rgb[..., 0] > 0) | (rgb[..., 1] > 0) | (rgb[..., 2] > 0)
    return rgb[m]


def stretch_with_range(rgb: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    out = (rgb - lo.reshape(1, 1, 3)) / (hi - lo + 1e-9).reshape(1, 1, 3)
    return np.clip(out, 0, 1)


def to_display_no_stretch(rgb: np.ndarray) -> np.ndarray:
    # Keep raw pixel relationships; only map DN range to [0, 1] for display.
    arr = rgb.astype(np.float32)
    vmax = np.max(arr) if arr.size else 1.0
    if vmax <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / max(255.0, float(vmax)), 0, 1)


def percentile_range(vals: np.ndarray, p_low: float, p_high: float) -> Tuple[np.ndarray, np.ndarray]:
    if vals.size == 0:
        lo = np.zeros(3, dtype=np.float32)
        hi = np.ones(3, dtype=np.float32)
        return lo, hi
    lo = np.percentile(vals, p_low, axis=0).astype(np.float32)
    hi = np.percentile(vals, p_high, axis=0).astype(np.float32)
    hi = np.maximum(hi, lo + 1e-6)
    return lo, hi


def main() -> int:
    dataset = DATASET.strip()
    page = int(PAGE)
    numbers = tuple(int(x) for x in TILE_NUMBERS)
    out_dir = OUT_DIR
    max_dim = max(256, int(MAX_DIM))
    p_low = float(P_LOW)
    p_high = float(P_HIGH)

    if not dataset:
        raise RuntimeError("DATASET is empty")
    if len(numbers) != 2:
        raise RuntimeError("TILE_NUMBERS must contain exactly two 1-based indices")
    if not (0 <= p_low < p_high <= 100):
        raise RuntimeError("P_LOW/P_HIGH must satisfy 0 <= P_LOW < P_HIGH <= 100")

    api_key = load_api_key()
    session = build_session()

    items, meta = fetch_available_page(session, api_key, dataset, page)
    names = list_file_names(items)
    if not names:
        raise RuntimeError(f"No files on page {page} for {dataset}")

    log(f"Dataset={dataset} page={page} files_on_page={len(names)}")
    if meta:
        log(f"Pagination metadata keys: {list(meta.keys())}")

    chosen: List[str] = []
    for n in numbers:
        idx = n - 1
        if idx < 0 or idx >= len(names):
            raise IndexError(f"TILE number {n} out of range 1..{len(names)}")
        chosen.append(ensure_tif_name(names[idx]))
    log(f"Selected: #{numbers[0]}={chosen[0]} | #{numbers[1]}={chosen[1]}")

    originals_dir = out_dir / "originals"
    path_a = download_single_file(session, api_key, dataset, chosen[0], originals_dir / f"{dataset}__{chosen[0]}")
    path_b = download_single_file(session, api_key, dataset, chosen[1], originals_dir / f"{dataset}__{chosen[1]}")

    rgb_a, meta_a = read_rgb_preview(path_a, max_dim=max_dim)
    rgb_b, meta_b = read_rgb_preview(path_b, max_dim=max_dim)
    vals_a = valid_pixels(rgb_a)
    vals_b = valid_pixels(rgb_b)
    vals_all = np.vstack([vals_a, vals_b]) if vals_a.size and vals_b.size else (vals_a if vals_a.size else vals_b)

    lo_common, hi_common = percentile_range(vals_all, p_low, p_high)
    lo_a, hi_a = percentile_range(vals_a, p_low, p_high)
    lo_b, hi_b = percentile_range(vals_b, p_low, p_high)

    a_common = stretch_with_range(rgb_a, lo_common, hi_common)
    b_common = stretch_with_range(rgb_b, lo_common, hi_common)
    a_auto = stretch_with_range(rgb_a, lo_a, hi_a)
    b_auto = stretch_with_range(rgb_b, lo_b, hi_b)
    a_raw = to_display_no_stretch(rgb_a)
    b_raw = to_display_no_stretch(rgb_b)

    log(f"A: {path_a.name} meta={meta_a}")
    log(f"B: {path_b.name} meta={meta_b}")
    log(
        f"Common RGB range p{p_low:.1f}-p{p_high:.1f}: "
        f"R={lo_common[0]:.1f}..{hi_common[0]:.1f} "
        f"G={lo_common[1]:.1f}..{hi_common[1]:.1f} "
        f"B={lo_common[2]:.1f}..{hi_common[2]:.1f}"
    )

    plt.figure(figsize=(14, 14))
    plt.subplot(3, 2, 1)
    plt.title(f"A #{numbers[0]} common stretch")
    plt.imshow(a_common)
    plt.axis("off")
    plt.subplot(3, 2, 2)
    plt.title(f"B #{numbers[1]} common stretch")
    plt.imshow(b_common)
    plt.axis("off")
    plt.subplot(3, 2, 3)
    plt.title("A auto stretch")
    plt.imshow(a_auto)
    plt.axis("off")
    plt.subplot(3, 2, 4)
    plt.title("B auto stretch")
    plt.imshow(b_auto)
    plt.axis("off")
    plt.subplot(3, 2, 5)
    plt.title("A no stretch")
    plt.imshow(a_raw)
    plt.axis("off")
    plt.subplot(3, 2, 6)
    plt.title("B no stretch")
    plt.imshow(b_raw)
    plt.axis("off")
    plt.tight_layout()

    plt.figure(figsize=(14, 4))
    labels = ["R", "G", "B"]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        if vals_a.size:
            plt.hist(vals_a[:, i], bins=256, range=(0, 255), alpha=0.5, color="tab:blue", label="A")
        if vals_b.size:
            plt.hist(vals_b[:, i], bins=256, range=(0, 255), alpha=0.5, color="tab:orange", label="B")
        plt.title(f"{labels[i]} histogram")
        plt.xlabel("DN")
        plt.ylabel("Count")
        if i == 0:
            plt.legend()
    plt.tight_layout()

    if SAVE_FIG:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem_a = Path(chosen[0]).stem
        stem_b = Path(chosen[1]).stem
        fig1 = out_dir / f"{dataset}__p{page}__n{numbers[0]}_vs_n{numbers[1]}__{stem_a}__{stem_b}__compare.png"
        fig2 = out_dir / f"{dataset}__p{page}__n{numbers[0]}_vs_n{numbers[1]}__{stem_a}__{stem_b}__hist.png"
        plt.figure(1)
        plt.savefig(fig1, dpi=150)
        plt.figure(2)
        plt.savefig(fig2, dpi=150)
        log(f"Saved: {fig1}")
        log(f"Saved: {fig2}")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
