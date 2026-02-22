#!/usr/bin/env python3
"""
Benchmark parallel single-file aerial downloads from Datafordeler.

Purpose:
- Test how many parallel GetRasterFile downloads are stable for your key/network.
- Report success/fail counts and throughput per worker setting.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
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
) -> List[Dict[str, Any]]:
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
    return payload.get("availableFileDownloads") or payload.get("AvailableFileDownloads") or []


def list_file_names(items: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for item in items:
        fn = _ga(item, "fileName")
        if fn:
            out.append(str(fn))
    return out


def download_single_file(
    session: requests.Session,
    api_key: str,
    dataset: str,
    file_name: str,
    out_path: Path,
) -> int:
    params = {
        "apiKey": api_key,
        "Register": DF_REGISTER,
        "DataSetName": dataset,
        "Version": 1,
        "Filename": file_name,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(f"{DF_BASE}/GetRasterFile", params=params, stream=True, timeout=(30, 1200)) as r:
        r.raise_for_status()
        total = 0
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
    return total


def parse_workers(raw: str) -> List[int]:
    out: List[int] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        v = int(p)
        if v > 0:
            out.append(v)
    if not out:
        raise ValueError("No valid worker counts provided")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark parallel Datafordeler GetRasterFile downloads.")
    p.add_argument("--dataset", default="GeoDKO10cm", help="Dataset name (GeoDKO10cm or GeoDKO12,5cm)")
    p.add_argument("--page", type=int, default=1, help="Catalog page")
    p.add_argument("--count", type=int, default=8, help="How many unique files to use")
    p.add_argument(
        "--workers",
        default="1,2,4,6,8,12",
        help="Comma-separated worker counts to benchmark",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each selected file this many times per benchmark point",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("pipeline5-aerial/data/test_parallel"),
        help="Output root for benchmark files/results",
    )
    p.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep downloaded files (default deletes after each benchmark point)",
    )
    return p.parse_args()


def run_one_worker_setting(
    workers: int,
    session: requests.Session,
    api_key: str,
    dataset: str,
    file_names: Sequence[str],
    repeat: int,
    out_dir: Path,
    keep_files: bool,
) -> Dict[str, Any]:
    tasks: List[Tuple[int, str]] = []
    for rep in range(repeat):
        for fn in file_names:
            tasks.append((rep, fn))

    start = time.perf_counter()
    ok = 0
    fail = 0
    bytes_total = 0
    errors: Dict[str, int] = {}
    paths: List[Path] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_map = {}
        for i, (rep, fn) in enumerate(tasks):
            out_path = out_dir / f"w{workers}_r{rep}_{i}_{Path(fn).name}.download"
            fut = pool.submit(download_single_file, session, api_key, dataset, fn, out_path)
            fut_map[fut] = (fn, out_path)
            paths.append(out_path)

        for fut in as_completed(fut_map):
            fn, out_path = fut_map[fut]
            try:
                nbytes = fut.result()
                bytes_total += int(nbytes)
                ok += 1
            except Exception as exc:
                fail += 1
                key = str(exc).split(":", 1)[0][:120]
                errors[key] = errors.get(key, 0) + 1
                out_path.unlink(missing_ok=True)

    dur = max(1e-9, time.perf_counter() - start)
    mib = bytes_total / (1024**2)
    mib_s = mib / dur

    if not keep_files:
        for p in paths:
            p.unlink(missing_ok=True)

    return {
        "workers": workers,
        "tasks": len(tasks),
        "ok": ok,
        "fail": fail,
        "seconds": dur,
        "mib_downloaded": mib,
        "mib_per_sec": mib_s,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    workers_list = parse_workers(args.workers)

    api_key = load_api_key()
    session = build_session()

    items = fetch_available_page(session, api_key, args.dataset, args.page)
    names = list_file_names(items)
    if not names:
        log("No files found on selected page.")
        return 1

    selected = names[: max(1, args.count)]
    log(f"Dataset={args.dataset} page={args.page} unique_files={len(selected)} repeat={args.repeat}")
    for fn in selected:
        log(f"  - {fn}")

    results: List[Dict[str, Any]] = []
    for w in workers_list:
        log(f"\\nBenchmark workers={w} ...")
        res = run_one_worker_setting(
            workers=w,
            session=session,
            api_key=api_key,
            dataset=args.dataset,
            file_names=selected,
            repeat=max(1, args.repeat),
            out_dir=args.out,
            keep_files=args.keep_files,
        )
        results.append(res)
        log(
            f"  workers={w} ok={res['ok']}/{res['tasks']} fail={res['fail']} "
            f"time={res['seconds']:.1f}s speed={res['mib_per_sec']:.1f} MiB/s"
        )
        if res["errors"]:
            for k, v in sorted(res["errors"].items(), key=lambda kv: kv[1], reverse=True)[:3]:
                log(f"    error x{v}: {k}")

    # pick best stable result (0 failures), else highest success ratio
    stable = [r for r in results if r["fail"] == 0]
    if stable:
        best = max(stable, key=lambda r: r["mib_per_sec"])
        recommendation = {
            "type": "stable_best_speed",
            "workers": best["workers"],
            "speed_mib_per_sec": best["mib_per_sec"],
        }
    else:
        best = max(results, key=lambda r: (r["ok"] / max(1, r["tasks"]), r["mib_per_sec"]))
        recommendation = {
            "type": "best_effort_no_stable_point",
            "workers": best["workers"],
            "success_ratio": best["ok"] / max(1, best["tasks"]),
            "speed_mib_per_sec": best["mib_per_sec"],
        }

    out_json = args.out / "parallel_benchmark_results.json"
    payload = {
        "dataset": args.dataset,
        "page": args.page,
        "selected_files": selected,
        "repeat": args.repeat,
        "workers_tested": workers_list,
        "results": results,
        "recommendation": recommendation,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"\\nSaved results: {out_json}")
    log(f"Recommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

