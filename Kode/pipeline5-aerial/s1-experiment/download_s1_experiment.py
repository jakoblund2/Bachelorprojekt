#!/usr/bin/env python3
"""Small S1 CDSE downloader for polarization experiments.

This script is intentionally standalone so you can test S1 products
(e.g., 1SDV vs 1SDH) without touching the main pipeline.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_ROOT = "https://catalogue.dataspace.copernicus.eu/odata/v1"
ODATA_TOP_MAX = 999

BASE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = BASE_DIR.parent

# ------------------------------------------------------------
# Experiment config (edit these variables)
# ------------------------------------------------------------
ENV_FILE = PIPELINE_DIR / ".env"
OUT_DIR = BASE_DIR / "data" / "downloads"

BBOX_LONLAT: Tuple[float, float, float, float] = (8.8, 55.7, 9.5, 56.2)
DATE_START = "2025-06-12"
DATE_END = "2025-07-13"

# dv=VV/VH, dh=HH/HV, both=both filters, any=no polarization filter
POL_MODE = "dh"  # one of: dv, dh, both, any

SEARCH_TOP = 300
DOWNLOAD_COUNT = 10
LIST_ONLY = True

DOWNLOAD_ATTEMPTS = 5
RETRY_BASE_SECONDS = 4


def log(msg: str) -> None:
    print(msg, flush=True)


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


def cdse_token() -> str:
    user = os.environ.get("CDSE_USER")
    pw = os.environ.get("CDSE_PASS")
    totp = os.environ.get("CDSE_TOTP")
    if not user or not pw:
        raise RuntimeError("Missing CDSE_USER/CDSE_PASS in environment")

    payload: Dict[str, str] = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": user,
        "password": pw,
    }
    if totp:
        payload["totp"] = totp

    resp = requests.post(TOKEN_URL, data=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["access_token"]


def odata_get(
    url: str,
    token: str,
    params: Optional[Dict[str, str]] = None,
    timeout: int = 120,
    max_attempts: int = 5,
    retry_base_s: int = 4,
) -> Tuple[requests.Response, str]:
    cur_token = token
    for attempt in range(1, max_attempts + 1):
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {cur_token}"},
            params=params,
            timeout=timeout,
        )

        if resp.status_code in (401, 403):
            if attempt < max_attempts:
                cur_token = cdse_token()
                continue
            resp.raise_for_status()

        if resp.status_code == 429:
            if attempt < max_attempts:
                wait_s = max(1, retry_base_s) * (2 ** (attempt - 1))
                log(f"  429 from CDSE, retrying in {wait_s}s (attempt {attempt}/{max_attempts})")
                sleep(float(wait_s))
                continue
            resp.raise_for_status()

        resp.raise_for_status()
        return resp, cur_token

    raise RuntimeError("odata_get exhausted retries")


def search_s1(
    token: str,
    bbox_lonlat: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    pol_mode: str,
    top: int,
) -> Tuple[List[Dict[str, Any]], str]:
    wkt = bbox_to_wkt(bbox_lonlat)
    dt0 = f"{date_start}T00:00:00.000Z"
    dt1 = f"{date_end}T23:59:59.999Z"

    filters = [
        "Collection/Name eq 'SENTINEL-1'",
        "contains(Name,'IW_GRDH')",
        f"ContentDate/Start gt {dt0}",
        f"ContentDate/Start lt {dt1}",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
    ]
    if pol_mode == "dv":
        filters.append("contains(Name,'1SDV')")
    elif pol_mode == "dh":
        filters.append("contains(Name,'1SDH')")
    elif pol_mode == "both":
        filters.append("(contains(Name,'1SDV') or contains(Name,'1SDH'))")

    filt = " and ".join(filters)

    target = max(1, int(top))
    products: List[Dict[str, Any]] = []
    skip = 0
    cur_token = token

    while len(products) < target:
        page_top = min(ODATA_TOP_MAX, target - len(products))
        resp, cur_token = odata_get(
            f"{ODATA_ROOT}/Products",
            cur_token,
            params={
                "$filter": filt,
                "$top": str(page_top),
                "$skip": str(skip),
                "$orderby": "ContentDate/Start desc",
            },
            timeout=120,
            max_attempts=5,
            retry_base_s=4,
        )
        vals = resp.json().get("value", [])
        if not vals:
            break
        products.extend(vals)
        skip += len(vals)
        if len(vals) < page_top:
            break

    return products[:target], cur_token


def download_product_zip(
    token: str,
    product: Dict[str, Any],
    out_dir: Path,
    attempts: int,
    retry_base_s: int,
) -> Tuple[Path, str, str]:
    pid = product["Id"]
    name = str(product["Name"])
    out = out_dir / (f"{name}.zip" if name.endswith(".SAFE") else f"{name}.SAFE.zip")

    if out.exists() and out.stat().st_size > 10_000_000:
        return out, token, "cached"

    url = f"{ODATA_ROOT}/Products({pid})/$value"
    tmp = out.with_suffix(out.suffix + ".part")
    cur_token = token

    def _download_once(url_to_get: str, bearer: str) -> None:
        headers = {"Authorization": f"Bearer {bearer}"}
        with requests.get(url_to_get, headers=headers, stream=True, timeout=300, allow_redirects=False) as r1:
            if r1.status_code in (301, 302, 303, 307, 308):
                loc = r1.headers.get("Location")
                if not loc:
                    r1.raise_for_status()
                with requests.get(loc, headers=headers, stream=True, timeout=300) as r2:
                    r2.raise_for_status()
                    with tmp.open("wb") as f:
                        for chunk in r2.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                return

            r1.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r1.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    out_dir.mkdir(parents=True, exist_ok=True)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            tmp.unlink(missing_ok=True)
            _download_once(url, cur_token)
            tmp.replace(out)
            return out, cur_token, "downloaded"
        except requests.HTTPError as exc:
            last_exc = exc
            code = getattr(exc.response, "status_code", None)
            if code in (401, 403) and attempt < attempts:
                cur_token = cdse_token()
                continue
            if code == 429 and attempt < attempts:
                wait_s = max(1, retry_base_s) * (2 ** (attempt - 1))
                sleep(float(wait_s))
                continue
            break
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                wait_s = max(1, retry_base_s) * (2 ** (attempt - 1))
                sleep(float(wait_s))
                continue
            break

    tmp.unlink(missing_ok=True)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Download failed for {name}")


def main() -> None:
    if POL_MODE not in {"dv", "dh", "both", "any"}:
        raise ValueError("POL_MODE must be one of: dv, dh, both, any")

    bbox = tuple(float(x) for x in BBOX_LONLAT)
    if len(bbox) != 4:
        raise ValueError("BBOX_LONLAT must contain exactly 4 values")
    minx, miny, maxx, maxy = bbox
    if not (minx < maxx and miny < maxy):
        raise ValueError("Invalid BBOX_LONLAT ordering")

    load_dotenv(ENV_FILE)
    out_dir = OUT_DIR.resolve()

    log("S1 experiment downloader")
    log(
        f"  bbox={bbox} date={DATE_START}..{DATE_END} "
        f"pol_mode={POL_MODE} search_top={SEARCH_TOP}"
    )

    token = cdse_token()
    products, token = search_s1(
        token=token,
        bbox_lonlat=bbox,
        date_start=DATE_START,
        date_end=DATE_END,
        pol_mode=POL_MODE,
        top=SEARCH_TOP,
    )

    if not products:
        log("No S1 products found for this query.")
        return

    log(f"Found {len(products)} product(s):")
    for i, prod in enumerate(products, start=1):
        dt = str(prod.get("ContentDate", {}).get("Start", "?"))
        name = str(prod.get("Name", ""))
        pol = "DV" if "1SDV" in name else ("DH" if "1SDH" in name else "?")
        log(f"  {i:02d}. [{pol}] {dt}  {name}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bbox": bbox,
        "date_start": DATE_START,
        "date_end": DATE_END,
        "pol_mode": POL_MODE,
        "search_top": int(SEARCH_TOP),
        "products": products,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "last_search_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Saved search manifest: {manifest_path}")

    if LIST_ONLY:
        return

    n = min(max(0, int(DOWNLOAD_COUNT)), len(products))
    if n == 0:
        log("download_count is 0; nothing to download.")
        return

    log(f"Downloading {n} product(s) to {out_dir}")
    ok = 0
    fail = 0
    for i, prod in enumerate(products[:n], start=1):
        name = str(prod.get("Name", ""))
        try:
            path, token, status = download_product_zip(
                token=token,
                product=prod,
                out_dir=out_dir,
                attempts=max(1, int(DOWNLOAD_ATTEMPTS)),
                retry_base_s=max(1, int(RETRY_BASE_SECONDS)),
            )
            ok += 1
            size_gb = path.stat().st_size / (1024**3)
            log(f"  [{i}/{n}] {status:10s} {name} -> {path.name} ({size_gb:.2f} GiB)")
        except Exception as exc:
            fail += 1
            log(f"  [{i}/{n}] FAILED      {name}: {exc}")

    log(f"Done. downloaded_or_cached={ok} failed={fail} out_dir={out_dir}")


if __name__ == "__main__":
    main()
