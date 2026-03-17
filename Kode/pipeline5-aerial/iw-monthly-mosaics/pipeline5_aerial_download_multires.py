#!/usr/bin/env python3
"""
Download intersecting Datafordeler aerial tiles and resample them to multiple
output resolutions.

Edit the user config block below, then run:
  python3 pipeline5-aerial/iw-monthly-mosaics/pipeline5_aerial_download_multires.py
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import importlib.util
import json
import re
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin
from rasterio.warp import reproject
# Note: bbox-based selection was intentionally removed for full-catalog runs.


THIS_DIR = Path(__file__).resolve().parent
PARENT_PIPELINE = THIS_DIR.parent / "pipeline5_aerial.py"

_SPEC = importlib.util.spec_from_file_location("pipeline5_aerial_base", PARENT_PIPELINE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load base pipeline from {PARENT_PIPELINE}")
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

base.load_dotenv(THIS_DIR.parent / ".env", override=False)
base.load_dotenv(THIS_DIR / ".env", override=False)

DEFAULT_RESOLUTIONS: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0)


# -----------------------------
# User config
# -----------------------------
# AOI bounding box in EPSG:4326: (min_lon, min_lat, max_lon, max_lat)
BBOX_LONLAT: Tuple[float, float, float, float] = (
    #10.062603417504576, 56.12100397806407, 10.108539774516864, 56.15913417305279
    #10.070327, 56.080836, 10.253217, 56.192769,
    7.808112352639069,
    54.494966893759326,
    12.851698715687899,
    57.8296746572478
)

# Set either REFERENCE_DATE or REFERENCE_YEAR. If both are set, the years must match.
REFERENCE_DATE: Optional[str] = "2025-04-01"
REFERENCE_YEAR: Optional[int] = None

# Point this at your external drive if needed.
OUTPUT_DIR: Path = Path("/run/media/jakob/Seagate/aerial_multires")

# Raw/original downloads and extracted TIFFs are staged here, then deleted after
# all requested resolutions have been written. Set this to a local disk path if
# you do not want the temporary originals on the external drive.
ORIGINALS_DIR: Path = THIS_DIR / "originals_tmp"

# Dataset priority. First entry is preferred when multiple datasets overlap.
DATASETS: Sequence[str] = list(base.AERIAL_DATASET_PRIORITY)

# Target output resolutions in meters.
RESOLUTIONS_M: Sequence[float] = DEFAULT_RESOLUTIONS

DATE_POLICY: str = base.AERIAL_DATE_POLICY
REBUILD_INDEX: bool = False
DOWNLOAD_WORKERS: int = 20
REPROJECT_THREADS_PER_TASK: int = 1

# Optional 1-based inclusive record window for batch processing.
# Example: 1..5000, then 5001..10000.
RECORD_START_INDEX: Optional[int] = None
RECORD_END_INDEX: Optional[int] = None


def log(msg: str) -> None:
    print(msg, flush=True)


def _gtiff_layout(width: int, height: int) -> Dict[str, Any]:
    width = max(1, int(width))
    height = max(1, int(height))

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


def _reproject_aerial(*args: Any, **kwargs: Any) -> Any:
    # We validate/infer georeferencing explicitly before reprojection.
    # Rasterio may still emit internal NotGeoreferencedWarning objects when
    # creating temporary in-memory datasets; ignore those here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        return reproject(*args, **kwargs)


def downsample_aerial_tif(src_tif: Path, out_path: Path, target_res_m: float) -> Path:
    if out_path.exists():
        if base._is_valid_cached_aerial_tif(out_path):
            return out_path
        out_path.unlink(missing_ok=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
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
                _reproject_aerial(
                    source=rasterio.band(src, src_band),
                    destination=rasterio.band(dst, out_band),
                    src_transform=src_transform,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    dst_transform=transform,
                    dst_crs=src_crs,
                    resampling=Resampling.bilinear,
                    init_dest_nodata=False,
                    num_threads=max(1, int(REPROJECT_THREADS_PER_TASK)),
                )

            src_mask = (src.read_masks(1) > 0).astype(np.uint8)
            dst_mask = np.zeros((height, width), dtype=np.uint8)
            _reproject_aerial(
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
                num_threads=max(1, int(REPROJECT_THREADS_PER_TASK)),
            )
            dst.write_mask((dst_mask > 0).astype(np.uint8) * 255)

    tmp.replace(out_path)
    return out_path


def normalize_resolutions(raw: Sequence[float]) -> List[float]:
    out: List[float] = []
    seen: set[float] = set()
    for part in raw:
        value = float(part)
        if value <= 0:
            raise ValueError(f"Resolution must be > 0, got {part}")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if not out:
        return list(DEFAULT_RESOLUTIONS)
    return out


def parse_datasets(raw: Sequence[str]) -> List[str]:
    out = [part.strip() for part in raw if part and part.strip()]
    if not out:
        raise ValueError("At least one dataset must be provided")
    return out


def apply_record_window(
    records: Sequence[base.CatalogRecord],
    start_index: Optional[int],
    end_index: Optional[int],
) -> Tuple[List[base.CatalogRecord], int, int]:
    total = len(records)
    start = 1 if start_index is None else int(start_index)
    end = total if end_index is None else int(end_index)

    if start < 1:
        raise ValueError(f"RECORD_START_INDEX must be >= 1, got {start}")
    if end < start:
        raise ValueError(f"RECORD_END_INDEX ({end}) must be >= RECORD_START_INDEX ({start})")
    if total <= 0:
        return [], start, end
    if start > total:
        return [], start, end

    end = min(end, total)
    sliced = list(records[start - 1 : end])
    return sliced, start, end


def resolve_target_year(date_raw: Optional[str], year_raw: Optional[int]) -> Optional[int]:
    if date_raw:
        dt = datetime.strptime(date_raw, "%Y-%m-%d")
        if year_raw is not None and int(year_raw) != dt.year:
            raise ValueError(f"--date year {dt.year} does not match --year {year_raw}")
        return dt.year
    if year_raw is None:
        return None
    if year_raw < 1900 or year_raw > 2100:
        raise ValueError(f"Invalid year: {year_raw}")
    return int(year_raw)


def configure_output_paths(output_dir: Path, originals_dir: Path) -> Dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    originals_dir = originals_dir.expanduser().resolve()
    catalog_dir = output_dir / "catalog"
    resampled_dir = output_dir / "resampled"
    state_dir = output_dir / ".state"
    manifest_path = output_dir / "manifest.json"

    for path in (output_dir, catalog_dir, originals_dir, resampled_dir, state_dir):
        path.mkdir(parents=True, exist_ok=True)

    base.AERIAL_INDEX_DIR = catalog_dir
    base.DF_CATALOG_JSONL = catalog_dir / "geodko_catalog.jsonl"
    base.DF_CATALOG_PARQUET = catalog_dir / "geodko_catalog.parquet"

    return {
        "root": output_dir,
        "catalog": catalog_dir,
        "originals": originals_dir,
        "resampled": resampled_dir,
        "state": state_dir,
        "manifest": manifest_path,
    }


def sanitize_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", token)


def resolution_label(target_res_m: float) -> str:
    if target_res_m < 1.0:
        centimeters = int(round(target_res_m * 100))
        return f"{centimeters}cm"
    if abs(target_res_m - round(target_res_m)) < 1e-9:
        return f"{int(round(target_res_m))}m"
    return f"{str(target_res_m).replace('.', 'p')}m"


def originals_archive_path(originals_root: Path, dataset: str, file_name: str) -> Path:
    return originals_root / sanitize_token(dataset) / Path(file_name).name


def originals_extract_dir(originals_root: Path, dataset: str, file_name: str) -> Path:
    return originals_root / sanitize_token(dataset) / f"{Path(file_name).stem}__parts"


def resampled_output_path(
    resampled_root: Path,
    dataset: str,
    source_name: str,
    target_res_m: float,
) -> Path:
    return (
        resampled_root
        / resolution_label(target_res_m)
        / sanitize_token(dataset)
        / Path(source_name).name
    )


def list_tifs(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    out: List[Path] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"} and base._is_tiff_file(path):
            out.append(path)
    return out


def record_state_path(state_root: Path, dataset: str, file_name: str) -> Path:
    token = sanitize_token(Path(file_name).stem)
    return state_root / sanitize_token(dataset) / f"{token}.json"


def load_record_state(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        path.unlink(missing_ok=True)
        return None
    if not isinstance(payload, dict):
        path.unlink(missing_ok=True)
        return None
    return payload


def save_record_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def cleanup_original_artifacts(originals_root: Path, dataset: str, file_name: str) -> None:
    archive_path = originals_archive_path(originals_root, dataset, file_name)
    extract_dir = originals_extract_dir(originals_root, dataset, file_name)
    archive_path.unlink(missing_ok=True)
    shutil.rmtree(extract_dir, ignore_errors=True)


def cleanup_originals_on_startup(originals_root: Path) -> None:
    if not originals_root.exists():
        originals_root.mkdir(parents=True, exist_ok=True)
        return
    for child in originals_root.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception as exc:
            log(f"Warning: could not remove startup temp artifact {child}: {exc}")


def record_outputs_if_complete(
    output_root: Path,
    resampled_root: Path,
    dataset: str,
    source_names: Sequence[str],
    resolutions: Sequence[float],
) -> Optional[Dict[str, List[str]]]:
    outputs: Dict[str, List[str]] = {}
    for target_res_m in resolutions:
        label = resolution_label(target_res_m)
        rel_paths: List[str] = []
        for source_name in source_names:
            out_path = resampled_output_path(
                resampled_root=resampled_root,
                dataset=dataset,
                source_name=source_name,
                target_res_m=target_res_m,
            )
            if not base._is_valid_cached_aerial_tif(out_path):
                return None
            rel_paths.append(relpath_or_str(out_path, output_root))
        outputs[label] = rel_paths
    return outputs


def completed_record_outputs(
    rec: base.CatalogRecord,
    output_paths: Dict[str, Path],
    resolutions: Sequence[float],
) -> Optional[Dict[str, Any]]:
    state_path = record_state_path(output_paths["state"], rec.dataset, rec.file_name)
    state = load_record_state(state_path)
    candidates: List[List[str]] = []

    if state is not None:
        raw_names = state.get("source_names") or []
        state_names = [str(name) for name in raw_names if str(name).strip()]
        if state_names:
            candidates.append(state_names)

    default_name = Path(rec.file_name).name
    if [default_name] not in candidates:
        candidates.append([default_name])

    for source_names in candidates:
        outputs = record_outputs_if_complete(
            output_root=output_paths["root"],
            resampled_root=output_paths["resampled"],
            dataset=rec.dataset,
            source_names=source_names,
            resolutions=resolutions,
        )
        if outputs is None:
            continue
        payload = {
            "dataset": rec.dataset,
            "file_name": rec.file_name,
            "year": rec.year,
            "source_names": list(source_names),
            "outputs": outputs,
        }
        save_record_state(state_path, payload)
        cleanup_original_artifacts(output_paths["originals"], rec.dataset, rec.file_name)
        return payload

    return None


def ensure_original_tifs(
    session: Any,
    api_key: str,
    rec: base.CatalogRecord,
    originals_root: Path,
) -> List[Path]:
    archive_path = originals_archive_path(originals_root, rec.dataset, rec.file_name)
    extract_dir = originals_extract_dir(originals_root, rec.dataset, rec.file_name)

    extracted = list_tifs(extract_dir)
    if extracted:
        return extracted

    if archive_path.exists():
        if base._is_tiff_file(archive_path):
            return [archive_path]
        if base._is_zip_file(archive_path):
            return base._extract_tifs(archive_path, extract_dir)
        archive_path.unlink(missing_ok=True)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None
    for attempt in range(1, base.AERIAL_DOWNLOAD_ATTEMPTS + 1):
        tmp = archive_path.with_suffix(archive_path.suffix + ".part")
        tmp.unlink(missing_ok=True)

        prefix = f"Downloading {rec.dataset}/{rec.file_name}"
        if attempt > 1:
            log(f"{prefix} (attempt {attempt}/{base.AERIAL_DOWNLOAD_ATTEMPTS})")
        else:
            log(prefix)

        try:
            base._download_df_single(
                session=session,
                api_key=api_key,
                dataset=rec.dataset,
                file_name=rec.file_name,
                out_path=tmp,
            )
            tmp.replace(archive_path)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < base.AERIAL_DOWNLOAD_ATTEMPTS:
                wait_s = max(1, int(base.AERIAL_DOWNLOAD_RETRY_BASE_SECONDS)) * (2 ** (attempt - 1))
                log(f"  Download failed: {exc}. Retrying in {wait_s}s")
                sleep(float(wait_s))
            else:
                raise

    if last_error is not None:
        raise last_error

    if not (base._is_tiff_file(archive_path) or base._is_zip_file(archive_path)):
        raise RuntimeError(f"Unexpected download format for {archive_path.name}")

    return base._extract_tifs(archive_path, extract_dir)


def relpath_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def process_record(
    rec: base.CatalogRecord,
    index: int,
    total: int,
    output_paths: Dict[str, Path],
    resolutions: Sequence[float],
    api_key: str,
) -> Tuple[Dict[str, Any], int]:
    log(f"[{index}/{total}] {rec.dataset}/{rec.file_name}")
    completed = completed_record_outputs(
        rec=rec,
        output_paths=output_paths,
        resolutions=resolutions,
    )
    if completed is not None:
        log(f"  Reusing existing outputs for {rec.dataset}/{rec.file_name}")
        record_entry = {
            "dataset": rec.dataset,
            "file_name": rec.file_name,
            "year": rec.year,
            "bbox_lonlat": [rec.minx, rec.miny, rec.maxx, rec.maxy],
            "source_names": list(completed.get("source_names", [])),
            "outputs": dict(completed.get("outputs", {})),
        }
        output_count = sum(len(v) for v in record_entry["outputs"].values())
        return record_entry, output_count

    session = base.build_retry_session()
    try:
        source_tifs = ensure_original_tifs(
            session=session,
            api_key=api_key,
            rec=rec,
            originals_root=output_paths["originals"],
        )
    finally:
        session.close()

    source_names = [path.name for path in source_tifs]

    record_entry: Dict[str, Any] = {
        "dataset": rec.dataset,
        "file_name": rec.file_name,
        "year": rec.year,
        "bbox_lonlat": [rec.minx, rec.miny, rec.maxx, rec.maxy],
        "source_names": source_names,
        "outputs": {},
    }

    output_count = 0
    for target_res_m in resolutions:
        label = resolution_label(target_res_m)
        out_paths: List[str] = []
        for src_tif in source_tifs:
            out_path = resampled_output_path(
                resampled_root=output_paths["resampled"],
                dataset=rec.dataset,
                source_name=src_tif.name,
                target_res_m=target_res_m,
            )
            downsample_aerial_tif(src_tif, out_path, target_res_m)
            out_paths.append(relpath_or_str(out_path, output_paths["root"]))
            output_count += 1
        record_entry["outputs"][label] = out_paths

    save_record_state(
        record_state_path(output_paths["state"], rec.dataset, rec.file_name),
        {
            "dataset": rec.dataset,
            "file_name": rec.file_name,
            "year": rec.year,
            "source_names": source_names,
            "outputs": record_entry["outputs"],
        },
    )
    cleanup_original_artifacts(output_paths["originals"], rec.dataset, rec.file_name)
    return record_entry, output_count


def main() -> int:
    output_paths = configure_output_paths(OUTPUT_DIR, ORIGINALS_DIR)
    cleanup_originals_on_startup(output_paths["originals"])
    datasets = parse_datasets(DATASETS)
    resolutions = normalize_resolutions(RESOLUTIONS_M)
    target_year = resolve_target_year(REFERENCE_DATE, REFERENCE_YEAR)

    log(f"Output root: {output_paths['root']}")
    log(f"Original staging dir: {output_paths['originals']}")
    log("Startup cleanup: cleared original staging directory")
    log("Selection mode: full catalog (no AOI bbox filtering)")
    log(f"Datasets: {datasets}")
    log(f"Resolutions: {', '.join(resolution_label(r) for r in resolutions)}")
    log(f"Parallel workers: {DOWNLOAD_WORKERS}")
    if target_year is not None:
        log(f"Reference year: {target_year}")
    else:
        log("Reference year: not set; selector may mix available years")

    api_key = base.datafordeler_api_key()
    session = base.build_retry_session()
    try:
        catalog = base.build_aerial_catalog(
            session=session,
            api_key=api_key,
            datasets=datasets,
            rebuild=bool(REBUILD_INDEX),
        )
    finally:
        session.close()

    # Download all available photos from the selected datasets.
    # Keep dedupe to avoid duplicate dataset/file entries from catalog merge paths.
    selected_all = base.dedupe_catalog_records(catalog)
    selected, range_start, range_end = apply_record_window(
        records=selected_all,
        start_index=RECORD_START_INDEX,
        end_index=RECORD_END_INDEX,
    )
    dataset_label = "+".join(datasets) if datasets else None
    selected_year = None

    if not selected:
        log(
            "No aerial tiles to process for the selected datasets "
            f"and record range {range_start}-{range_end}."
        )
        return 1

    log(f"Record window: {range_start}-{range_end} of {len(selected_all)} total catalog file(s)")
    log(f"Planned downloads: {len(selected)} catalog file(s) (cached outputs may be reused)")

    counts_by_dataset: Dict[str, int] = {}
    for rec in selected:
        counts_by_dataset[rec.dataset] = counts_by_dataset.get(rec.dataset, 0) + 1

    log(f"Selected {len(selected)} tile(s):")
    for dataset, count in sorted(counts_by_dataset.items()):
        log(f"  {dataset}: {count}")

    manifest: Dict[str, Any] = {
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bbox_lonlat": None,
        "date": REFERENCE_DATE,
        "target_year": target_year,
        "date_policy": DATE_POLICY,
        "selected_dataset_label": dataset_label,
        "selected_year": selected_year,
        "record_range": {
            "start_index": range_start,
            "end_index": range_end,
            "total_catalog_records": len(selected_all),
        },
        "datasets": datasets,
        "resolutions_m": resolutions,
        "records": [],
    }

    total_outputs = 0
    workers = max(1, int(DOWNLOAD_WORKERS))
    indexed_records = list(enumerate(selected, start=1))
    ordered_entries: List[Optional[Dict[str, Any]]] = [None] * len(selected)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        record_iter = iter(indexed_records)
        future_map: Dict[Any, Tuple[int, base.CatalogRecord]] = {}

        def submit_next() -> bool:
            try:
                index, rec = next(record_iter)
            except StopIteration:
                return False
            future = pool.submit(
                process_record,
                rec,
                index,
                len(selected),
                output_paths,
                resolutions,
                api_key,
            )
            future_map[future] = (index, rec)
            return True

        for _ in range(min(workers, len(indexed_records))):
            if not submit_next():
                break

        while future_map:
            done, _pending = wait(tuple(future_map.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                index, rec = future_map.pop(future)
                record_entry, output_count = future.result()
                ordered_entries[index - 1] = record_entry
                total_outputs += int(output_count)
                submit_next()

    manifest["records"] = [entry for entry in ordered_entries if entry is not None]

    with output_paths["manifest"].open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    log(f"Wrote manifest: {output_paths['manifest']}")
    log(f"Finished: {len(selected)} source tile(s), {total_outputs} resampled output(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
