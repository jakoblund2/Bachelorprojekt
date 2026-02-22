#!/usr/bin/env python3
"""
Small Datafordeler download smoke-test for aerial files.

What it does:
1) Lists available files for one dataset/page.
2) Picks N files.
3) Tries GetRasterMultipleSize + GetRasterMultipleFiles.
4) Falls back to GetRasterFile per file if batch fails.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DF_BASE = "https://api.datafordeler.dk/FileDownloads"
DF_REGISTER = "GeoDKO"


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
        allowed_methods=("GET", "POST"),
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


def download_stream(resp: requests.Response, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def call_multiple_size(
    session: requests.Session,
    api_key: str,
    dataset: str,
    file_names: Sequence[str],
) -> Optional[int]:
    base_params: List[Tuple[str, str]] = [
        ("apiKey", api_key),
        ("Register", DF_REGISTER),
        ("DataSetName", dataset),
        ("Version", "1"),
    ]

    # Try common parameter conventions used by DF services.
    candidates: List[Tuple[str, Any]] = []
    params_filename = list(base_params) + [("Filename", fn) for fn in file_names]
    params_file_name = list(base_params) + [("FileName", fn) for fn in file_names]
    candidates.append(("GET", params_filename))
    candidates.append(("GET", params_file_name))
    candidates.append(("POST", {"apiKey": api_key, "Register": DF_REGISTER, "DataSetName": dataset, "Version": 1, "Filename": list(file_names)}))
    candidates.append(("POST", {"apiKey": api_key, "Register": DF_REGISTER, "DataSetName": dataset, "Version": 1, "FileName": list(file_names)}))

    for method, payload in candidates:
        try:
            if method == "GET":
                r = session.get(f"{DF_BASE}/GetRasterMultipleSize", params=payload, timeout=(30, 180))
            else:
                r = session.post(
                    f"{DF_BASE}/GetRasterMultipleSize",
                    json=payload,
                    timeout=(30, 180),
                    headers={"Content-Type": "application/json"},
                )
            r.raise_for_status()

            ctype = (r.headers.get("content-type") or "").lower()
            if "json" in ctype:
                payload_json = r.json()
                if isinstance(payload_json, dict):
                    for k in ("size", "Size", "totalSize", "TotalSize", "bytes", "Bytes"):
                        if k in payload_json:
                            return int(float(payload_json[k]))
                return int(float(payload_json))

            txt = r.text.strip()
            if txt:
                return int(float(txt))
        except Exception:
            continue
    return None


def download_multiple_files(
    session: requests.Session,
    api_key: str,
    dataset: str,
    file_names: Sequence[str],
    out_path: Path,
) -> Path:
    base_params: List[Tuple[str, str]] = [
        ("apiKey", api_key),
        ("Register", DF_REGISTER),
        ("DataSetName", dataset),
        ("Version", "1"),
    ]

    candidates: List[Tuple[str, Any]] = []
    params_filename = list(base_params) + [("Filename", fn) for fn in file_names]
    params_file_name = list(base_params) + [("FileName", fn) for fn in file_names]
    candidates.append(("GET", params_filename))
    candidates.append(("GET", params_file_name))
    candidates.append(("POST", {"apiKey": api_key, "Register": DF_REGISTER, "DataSetName": dataset, "Version": 1, "Filename": list(file_names)}))
    candidates.append(("POST", {"apiKey": api_key, "Register": DF_REGISTER, "DataSetName": dataset, "Version": 1, "FileName": list(file_names)}))

    errors: List[str] = []

    for idx, (method, payload) in enumerate(candidates, start=1):
        try:
            if method == "GET":
                req = session.get(
                    f"{DF_BASE}/GetRasterMultipleFiles",
                    params=payload,
                    stream=True,
                    timeout=(30, 1800),
                )
            else:
                req = session.post(
                    f"{DF_BASE}/GetRasterMultipleFiles",
                    json=payload,
                    stream=True,
                    timeout=(30, 1800),
                    headers={"Content-Type": "application/json"},
                )

            with req as r:
                r.raise_for_status()
                ctype = (r.headers.get("content-type") or "").lower()
                if "json" in ctype:
                    payload_json = r.json()
                    if isinstance(payload_json, dict):
                        dl_url = payload_json.get("downloadUrl") or payload_json.get("url") or payload_json.get("Url")
                        if dl_url:
                            with session.get(dl_url, stream=True, timeout=(30, 1800)) as r2:
                                r2.raise_for_status()
                                download_stream(r2, out_path)
                            return out_path
                    raise RuntimeError(f"JSON without download URL: {payload_json}")

                download_stream(r, out_path)
                return out_path
        except Exception as exc:
            errors.append(f"attempt{idx}:{method}:{exc}")
            continue

    raise RuntimeError("All batch request formats failed: " + " | ".join(errors))


def download_single_file(
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
        download_stream(r, out_path)
    return out_path


def is_zip_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 4:
        return False
    with path.open("rb") as f:
        return f.read(4) == b"PK\x03\x04"


def list_archive_tifs(path: Path) -> List[str]:
    if not is_zip_file(path):
        return []
    with zipfile.ZipFile(path, "r") as zf:
        return [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test Datafordeler aerial download endpoints.")
    p.add_argument("--dataset", default="GeoDKO10cm", help="Dataset name, e.g. GeoDKO10cm or GeoDKO12,5cm")
    p.add_argument("--page", type=int, default=1, help="Catalog page number")
    p.add_argument("--count", type=int, default=2, help="How many files to test-download")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("pipeline5-aerial/data/test_download"),
        help="Output directory for test files",
    )
    p.add_argument(
        "--single-only",
        action="store_true",
        help="Skip batch endpoints and only test GetRasterFile per selected file",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()
    session = build_session()

    log(f"Dataset={args.dataset} page={args.page} count={args.count}")
    items, meta = fetch_available_page(session, api_key, args.dataset, args.page)
    names = list_file_names(items)
    log(f"Page returned {len(items)} items | metadata keys={list(meta.keys())}")
    if not names:
        log("No filenames found on this page.")
        return 1

    selected = names[: max(1, args.count)]
    log(f"Selected files ({len(selected)}):")
    for fn in selected:
        log(f"  - {fn}")

    if not args.single_only:
        est = call_multiple_size(session, api_key, args.dataset, selected)
        if est is not None:
            log(f"GetRasterMultipleSize: {est / (1024**3):.2f} GiB")
        else:
            log("GetRasterMultipleSize: unavailable")

        multi_path = args.out / f"{args.dataset}_multi_page{args.page}_n{len(selected)}.download"
        try:
            download_multiple_files(session, api_key, args.dataset, selected, multi_path)
            log(f"GetRasterMultipleFiles download saved: {multi_path} ({multi_path.stat().st_size} bytes)")
            if is_zip_file(multi_path):
                tifs = list_archive_tifs(multi_path)
                log(f"Archive contains {len(tifs)} tif files")
        except Exception as exc:
            log(f"Batch download failed: {exc}")

    # Always test single-file as baseline.
    for idx, fn in enumerate(selected, start=1):
        single_path = args.out / f"{args.dataset}_single_{idx}_{Path(fn).name}.download"
        try:
            download_single_file(session, api_key, args.dataset, fn, single_path)
            log(f"GetRasterFile saved: {single_path} ({single_path.stat().st_size} bytes)")
        except Exception as exc:
            log(f"Single download failed for {fn}: {exc}")

    summary = {
        "dataset": args.dataset,
        "page": args.page,
        "selected_files": selected,
        "output_dir": str(args.out),
        "note": "Single-file success means API/auth is working even if batch endpoint formats fail.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote summary: {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
