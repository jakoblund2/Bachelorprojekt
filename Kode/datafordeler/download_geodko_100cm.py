from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import rasterio
import requests
from dotenv import load_dotenv
from rasterio.enums import Resampling
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

BASE_URL = "https://api.datafordeler.dk/FileDownloads"
REGISTER = "GeoDKO"
DATASETS = ("GeoDKO10cm", "GeoDKO12,5cm")
TARGET_RESOLUTION_M = 1.0  # 100 cm


def load_api_key() -> str:
    # Prefer .env next to this script, then fall back to environment/default lookup.
    script_env = Path(__file__).with_name(".env")
    if script_env.exists():
        load_dotenv(script_env)
    load_dotenv()

    api_key = os.getenv("DATAFORDELER_APIKEY")
    if not api_key:
        raise RuntimeError("Missing DATAFORDELER_APIKEY (set it in environment or .env).")
    return api_key


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
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_page(
    session: requests.Session,
    api_key: str,
    dataset_name: str,
    page_number: int,
) -> tuple[list[dict], dict]:
    params = {
        "apiKey": api_key,
        "Register": REGISTER,
        "DataSetName": dataset_name,
        "Version": 1,
        "FileFormat": "tif",
        "PageNumber": page_number,
    }
    response = session.get(
        f"{BASE_URL}/GetAvailableRasterFileDownloads",
        params=params,
        timeout=(30, 300),
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("availableFileDownloads") or payload.get("AvailableFileDownloads") or []
    meta = payload.get("paginationMetadata") or payload.get("PaginationMetadata") or {}
    return items, meta


def iter_tiles(
    session: requests.Session,
    api_key: str,
    dataset_name: str,
    first_items: list[dict],
    total_pages: int,
) -> Iterable[dict]:
    for item in first_items:
        yield item
    for page in range(2, total_pages + 1):
        items, _meta = fetch_page(session, api_key, dataset_name, page)
        for item in items:
            yield item


def download_file(
    session: requests.Session,
    api_key: str,
    dataset_name: str,
    file_name: str,
    tmp_path: Path,
) -> None:
    params = {
        "apiKey": api_key,
        "Register": REGISTER,
        "DataSetName": dataset_name,
        "Version": 1,
        "Filename": file_name,
    }
    with session.get(
        f"{BASE_URL}/GetRasterFile",
        params=params,
        stream=True,
        timeout=(30, 1200),
    ) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("Content-Length", "0") or "0")
        desc = f"Downloading {Path(file_name).name}"
        with tmp_path.open("wb") as handle:
            with tqdm(
                total=(total_bytes if total_bytes > 0 else None),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                leave=False,
                dynamic_ncols=True,
            ) as dl_bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        dl_bar.update(len(chunk))


def is_zip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == b"PK\x03\x04"


def find_tif_in_zip(zip_path: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = [name for name in zf.namelist() if name.lower().endswith(".tif")]
        if not candidates:
            raise RuntimeError(f"No .tif file found in zip: {zip_path}")
        member = candidates[0]
        extracted = out_dir / Path(member).name
        with zf.open(member, "r") as src, extracted.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return extracted


def scale_to_100cm(src_path: Path, dst_path: Path) -> None:
    with rasterio.open(src_path) as src:
        xres = abs(src.transform.a)
        yres = abs(src.transform.e)

        dst_width = max(1, int(round(src.width * xres / TARGET_RESOLUTION_M)))
        dst_height = max(1, int(round(src.height * yres / TARGET_RESOLUTION_M)))
        # Keep exact grid alignment based on source transform and new raster shape.
        dst_transform = src.transform * src.transform.scale(
            src.width / dst_width,
            src.height / dst_height,
        )

        profile = src.profile.copy()
        profile.update(
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        data = src.read(
            out_shape=(src.count, dst_height, dst_width),
            resampling=Resampling.bilinear,
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)
            try:
                dst.colorinterp = src.colorinterp
            except Exception:
                pass


def output_name(tile_meta: dict, default_name: str) -> str:
    source_name = tile_meta.get("fileName") or tile_meta.get("FileName") or default_name
    stem = Path(source_name).stem
    return f"{stem}_100cm.tif"


def process_tile(
    session: requests.Session,
    api_key: str,
    dataset_name: str,
    tile_meta: dict,
    out_root: Path,
) -> None:
    file_name = tile_meta.get("fileName") or tile_meta.get("FileName")
    if not file_name:
        tile_id = tile_meta.get("id") or tile_meta.get("Id") or "unknown"
        raise RuntimeError(f"Tile metadata missing fileName. id={tile_id}")

    dataset_dir = out_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    final_name = output_name(tile_meta, file_name)
    final_path = dataset_dir / final_name
    if final_path.exists():
        return

    tmp_download = dataset_dir / f"{Path(file_name).name}.download"
    source_tif = dataset_dir / Path(file_name).name

    try:
        download_file(session, api_key, dataset_name, file_name, tmp_download)

        if is_zip(tmp_download):
            source_tif = find_tif_in_zip(tmp_download, dataset_dir)
            tmp_download.unlink(missing_ok=True)
        else:
            tmp_download.rename(source_tif)

        scale_to_100cm(source_tif, final_path)
    finally:
        if tmp_download.exists():
            tmp_download.unlink(missing_ok=True)
        if source_tif.exists() and source_tif != final_path:
            source_tif.unlink(missing_ok=True)


def collect_dataset_info(
    session: requests.Session,
    api_key: str,
) -> list[tuple[str, list[dict], int, int]]:
    datasets: list[tuple[str, list[dict], int, int]] = []
    for dataset_name in DATASETS:
        first_items, meta = fetch_page(session, api_key, dataset_name, 1)
        total_pages = int(meta.get("totalPages") or meta.get("TotalPages") or 1)
        total_count = int(meta.get("totalCount") or meta.get("TotalCount") or len(first_items))
        datasets.append((dataset_name, first_items, total_pages, total_count))
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all GeoDKO 10cm + 12,5cm tiles from Datafordeler, "
            "resample each tile to 100cm, delete original, and continue."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datafordeler/geodko_100cm"),
        help="Output root directory for 100cm tiles.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional safety limit: stop after N processed tiles (0 = no limit).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()
    session = build_session()

    datasets = collect_dataset_info(session, api_key)
    total_tiles = sum(info[3] for info in datasets)
    progress_total = min(total_tiles, args.limit) if args.limit else total_tiles

    processed = 0
    failed = 0
    progress = tqdm(total=progress_total, unit="tile", desc="GeoDKO -> 100cm")

    for dataset_name, first_items, total_pages, _total_count in datasets:
        for tile in iter_tiles(session, api_key, dataset_name, first_items, total_pages):
            fname = tile.get("fileName") or tile.get("FileName") or "unknown"
            progress.set_postfix_str(f"{dataset_name} | {fname}", refresh=False)
            try:
                process_tile(session, api_key, dataset_name, tile, out_root)
            except KeyboardInterrupt:
                progress.close()
                print("\nInterrupted by user.")
                return 130
            except Exception as exc:
                failed += 1
                print(f"[ERROR] {dataset_name} {fname}: {exc}", file=sys.stderr)
            finally:
                processed += 1
                progress.update(1)

            if args.limit and processed >= args.limit:
                progress.close()
                print(f"Stopped at --limit={args.limit}. failed={failed}")
                return 0

    progress.close()
    print(f"Done. processed={processed}, failed={failed}, output={out_root}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
