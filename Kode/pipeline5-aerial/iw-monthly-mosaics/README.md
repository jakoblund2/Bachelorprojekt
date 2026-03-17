# Pipeline5 Aerial: IW Monthly Mosaic Variant

This folder contains a separate `pipeline5-aerial` variant that uses the
Sentinel-1 IW Monthly Mosaics from CDSE `GLOBAL-MOSAICS` instead of
downloading individual Sentinel-1 GRD scenes.

## What changed

- Sentinel-1 search now targets `Collection/Name eq 'GLOBAL-MOSAICS'`.
- Products are filtered to the IW monthly mosaic path:
  `.../Global-Mosaics/Sentinel-1/S1SAR_L3_IW_MCM/...`
- SNAP terrain correction is skipped because the monthly mosaics are already
  terrain-corrected.
- The monthly mosaic rasters are warped directly onto the stitched Sentinel-2 grid
  and converted to dB before patch extraction.
- Output data stays isolated under this subfolder's own `data/`.

## Availability note

The official CDSE documentation page still describes Sentinel-1 IW Monthly
Mosaics with a temporal extent of **January 2023 through December 2023** and
adds "More mosaics coming soon". In practice, Copernicus Browser currently
appears to expose newer dates as well. This variant therefore does **not**
hardcode any year restriction and instead tries the catalog directly for the
month you request.

## Credentials

This variant reuses the same environment variables as the parent pipeline:

- `CDSE_USER`
- `CDSE_PASS`
- `CDSE_TOTP` if needed
- `DATAFORDELER_APIKEY`

It loads `pipeline5-aerial/.env` automatically and also accepts an optional
local `.env` in this folder.

## Run

```bash
python3 Kode/pipeline5-aerial/iw-monthly-mosaics/pipeline5_aerial_iw_monthly_mosaics.py
```

For a full mosaic-only variant using both Sentinel-1 monthly mosaics and
Sentinel-2 quarterly mosaics through Sentinel Hub BYOC:

```bash
python3 Kode/pipeline5-aerial/iw-monthly-mosaics/pipeline5_aerial_s1_s2_mosaics.py
```

## Notes

- The script reuses the parent pipeline's Sentinel-2 stitching, aerial catalog,
  aerial enrichment, patch scanning, and NPZ writing logic.
- Because the monthly mosaic packaging may differ from the single-scene SAFE
  products, this variant tries both product downloads and raster assets, then
  resolves VV/VH rasters from the downloaded files.
- The default job mirrors the original pipeline date (`2025-06-12`) instead of
  forcing a 2023-only test case.

## Full Mosaic Variant

`pipeline5_aerial_s1_s2_mosaics.py` uses Sentinel Hub BYOC for both mosaics.

- It needs Sentinel Hub OAuth client credentials in addition to the existing
  `.env` values:
  - `SH_CLIENT_ID`
  - `SH_CLIENT_SECRET`
- The default test date is `2025-04-01`.
- Its S2 tensor has 4 bands (`B02`, `B03`, `B04`, `B08`) instead of the original
  11-band L2A layout, because that is what the quarterly mosaic collection exposes.
