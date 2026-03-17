#!/usr/bin/env python3
"""
Estimate how much storage Datafordeler aerial downloads will require for an AOI.

Edit the user config block below, then run:
  python3 pipeline5-aerial/iw-monthly-mosaics/pipeline5_aerial_estimate_storage.py
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from shapely.geometry import box


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
# Keep this script quiet apart from the final total/error message.
base.log = lambda _msg: None


# -----------------------------
# User config
# -----------------------------
# AOI bounding box in EPSG:4326: (min_lon, min_lat, max_lon, max_lat)
# BBOX_LONLAT: Tuple[float, float, float, float] = (
#     10.062603417504576, 56.12100397806407, 10.108539774516864, 56.15913417305279
# )

BBOX_LONLAT: Tuple[float, float, float, float] = (
    7.808112352639069,
    54.494966893759326,
    12.851698715687899,
    57.8296746572478
)


# Set either REFERENCE_DATE or REFERENCE_YEAR. If both are set, the years must match.
REFERENCE_DATE: Optional[str] = "2025-04-01"
REFERENCE_YEAR: Optional[int] = None

# Dataset priority. First entry is preferred when multiple datasets overlap.
DATASETS: Sequence[str] = list(base.AERIAL_DATASET_PRIORITY)

DATE_POLICY: str = base.AERIAL_DATE_POLICY
REBUILD_INDEX: bool = False


def parse_datasets(raw: Sequence[str]) -> List[str]:
    out = [part.strip() for part in raw if part and part.strip()]
    if not out:
        raise ValueError("At least one dataset must be provided")
    return out


def resolve_target_year(date_raw: Optional[str], year_raw: Optional[int]) -> Optional[int]:
    if date_raw:
        dt = datetime.strptime(date_raw, "%Y-%m-%d")
        if year_raw is not None and int(year_raw) != dt.year:
            raise ValueError(f"REFERENCE_DATE year {dt.year} does not match REFERENCE_YEAR {year_raw}")
        return dt.year
    if year_raw is None:
        return None
    if year_raw < 1900 or year_raw > 2100:
        raise ValueError(f"Invalid year: {year_raw}")
    return int(year_raw)


def human_bytes(num_bytes: int) -> str:
    value = float(max(0, int(num_bytes)))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.2f} {units[idx]}"


def main() -> int:
    aoi_bbox = tuple(float(v) for v in BBOX_LONLAT)
    datasets = parse_datasets(DATASETS)
    target_year = resolve_target_year(REFERENCE_DATE, REFERENCE_YEAR)

    if len(aoi_bbox) != 4:
        raise RuntimeError("Expected four bbox values")
    if not (aoi_bbox[0] < aoi_bbox[2] and aoi_bbox[1] < aoi_bbox[3]):
        raise RuntimeError("BBox must be ordered as min_lon min_lat max_lon max_lat")

    api_key = base.datafordeler_api_key()
    session = base.build_retry_session()
    catalog = base.build_aerial_catalog(
        session=session,
        api_key=api_key,
        datasets=datasets,
        rebuild=bool(REBUILD_INDEX),
    )

    selected, _, _ = base.select_aerial_records_for_pass(
        records=catalog,
        dataset_priority=datasets,
        aoi_wgs84=box(*aoi_bbox),
        target_year=target_year,
        date_policy=DATE_POLICY,
    )
    selected = base.dedupe_catalog_records(selected)

    if not selected:
        print("No intersecting aerial tiles found for the provided bbox.", flush=True)
        return 1

    total_known_bytes = 0
    unknown_size_count = 0

    for rec in selected:
        size = int(rec.file_size or 0)
        if size > 0:
            total_known_bytes += size
        else:
            unknown_size_count += 1

    if unknown_size_count > 0:
        print(
            f"{human_bytes(total_known_bytes)} ({total_known_bytes} bytes) known total, "
            f"{unknown_size_count} tile(s) with unknown size",
            flush=True,
        )
    else:
        print(f"{human_bytes(total_known_bytes)} ({total_known_bytes} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
