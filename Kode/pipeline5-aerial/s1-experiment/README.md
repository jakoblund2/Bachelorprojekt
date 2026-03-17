# S1 Experiment Folder

Standalone S1 downloader for trying different polarization modes before changing the main pipeline.

## Script

- `download_s1_experiment.py`

## What it does

1. Reads CDSE credentials from `.env` (defaults to `pipeline5-aerial/.env`).
2. Searches Sentinel-1 IW GRDH scenes for a bbox/date range.
3. Lets you choose polarization mode:
- `dv` -> `1SDV` (VV/VH)
- `dh` -> `1SDH` (HH/HV)
- `both` -> both `1SDV` and `1SDH`
- `any` -> no polarization filter
4. Downloads selected products as SAFE ZIPs into `s1-experiment/data/downloads`.

## Configure and run

Edit variables at the top of `download_s1_experiment.py`:

- `BBOX_LONLAT`
- `DATE_START`, `DATE_END`
- `POL_MODE` (`dv`, `dh`, `both`, `any`)
- `SEARCH_TOP`, `DOWNLOAD_COUNT`
- `LIST_ONLY`
- `OUT_DIR`

Then run:

```bash
python3 pipeline5-aerial/s1-experiment/download_s1_experiment.py
```

A search manifest is written to:
- `.../data/downloads/last_search_manifest.json`
