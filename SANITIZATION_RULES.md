# Sanitization Rules — epf-methodology public extract

**Purpose:** This file documents the sanitization that turns the EPFProject (private) source code into the epf-methodology (public) extract. Future syncs must respect these rules so the public repo stays self-contained and doesn't leak internal infrastructure (database paths, deploy scripts, API keys, hardcoded local paths).

**When to read this:** Before doing any methodology repo sync. Without this document, the naive `cp -f` sync that `scripts/sync_methodology.sh` performs in the EPFProject repo would clobber the public-facing edits and re-introduce the imports that this file says to strip. (See the M0.6 Phase A.3 deferral for the incident that motivated this document.)

---

## What gets extracted

The public methodology repo contains only the 7 files that describe the ML methodology, with their internal infrastructure dependencies stripped. The extracted files are:

```
src/data/feature_engineering.py
src/models/feature_selection.py
src/models/evaluation.py
src/models/trainer.py
src/models/lstm_embedder.py
src/models/direct_trainer.py
src/models/direct_predictor.py
```

The public repo does NOT contain:
- The FastAPI backend (`src/api/`)
- The data ingestion pipelines (`src/data/data_pipeline.py`, `src/data/ree_collector.py`, `src/data/openmeteo_collector.py`, `src/data/intraday_pipeline.py`, `src/data/news_collector.py`, etc.)
- The frontend (`frontend/`, `website/`)
- The deployment infrastructure (`run_deploy.py`, `deploy/`, `scripts/`)
- Database schemas, env vars, secrets

---

## Sanitization patterns

### 1. Top-of-file extraction header

Every file in the public extract starts with this 3-line comment block (above the docstring or first import):

```python
# EPForecast Methodology — <Component Name>
# Extracted from the EPForecast project (github.com/JorgeLopezLan/epf-methodology)
# Full application: epf.productjorge.com | Docs: epforecast.vercel.app
```

For files updated post-M0.6 (2026-04-09), an additional multi-line comment block describes the v11.0 / M0.6 retraction context. See `direct_predictor.py` and `direct_trainer.py` for the canonical version.

### 2. Top-level forbidden imports → comment markers

These imports must NEVER appear at the top of any public-extract file:

```python
from src.data.data_pipeline import DataPipeline
from src.data.ree_collector import REECollector
from src.data.openmeteo_collector import OpenMeteoCollector
from src.data.intraday_pipeline import IntradayPipeline
from src.data.news_collector import NewsCollector
from src.data.commodity_collector import CommodityCollector
from src.data.entsoe_pipeline import EntsoeCollector
from src.api.* import *
```

In EPFProject's source, `direct_predictor.py` has:

```python
from src.data.data_pipeline import DataPipeline
from src.data.ree_collector import REECollector
```

These get **replaced** with comment markers in the public extract:

```python
# DataPipeline: provides historical data access (not included in this extract)
# REECollector: fetches data from REE/ESIOS API (not included in this extract)
```

### 3. Inline imports (inside function bodies) — keep but annotate

Imports that appear inside function bodies (not at module top) are KEPT in the public extract, but a preceding comment explains what they're for. Example from `direct_predictor.py:_fetch_weather_forecast()`:

```python
def _fetch_weather_forecast() -> pd.DataFrame | None:
    """Fetch 7-day hourly weather forecast from Open-Meteo for prediction."""
    try:
        # OpenMeteoCollector: fetches Open-Meteo 7-day hourly weather forecast
        # (population-weighted national average for Spain — not included in this extract)
        from src.data.openmeteo_collector import OpenMeteoCollector
        collector = OpenMeteoCollector()
        ...
```

This is intentional — the inline imports document where the methodology depends on infrastructure code, without breaking the public file's import-time validity (the import only fires if the function is called).

Other internal imports inside method bodies (e.g. `from src.models.conformal import ConformalCalibrator`, `from src.models.feature_selection import FeatureSelector`, `from src.models.bias_corrector import BiasCorrector`) are kept as-is. They reference modules that DO exist in the public repo (`feature_selection.py`) or that the user can implement themselves following the same interface.

### 4. `DirectPredictor.__init__` — pipeline default + remove `self.ree`

In EPFProject:

```python
def __init__(self, trainer: DirectMultiHorizonTrainer | None = None,
             pipeline: DataPipeline | None = None):
    self.pipeline = pipeline or DataPipeline()
    self.trainer = trainer or DirectMultiHorizonTrainer()
    self.ree = REECollector()
    self.model_version = "unknown"
```

In the public extract:

```python
def __init__(self, trainer: DirectMultiHorizonTrainer | None = None,
             pipeline=None):
    # pipeline and ree are infrastructure components for data access and
    # API communication — not included in this methodology extract.
    self.pipeline = pipeline
    self.trainer = trainer or DirectMultiHorizonTrainer()
    self.model_version = "unknown"
```

**Three changes:**
1. `pipeline: DataPipeline | None = None` → `pipeline=None` (drop the type annotation that references the stripped class)
2. `self.pipeline = pipeline or DataPipeline()` → `self.pipeline = pipeline` (don't try to instantiate the stripped class)
3. `self.ree = REECollector()` → REMOVED entirely

### 5. `feature_engineering.py` — inlined `_compute_solar_angle`

EPFProject imports the solar angle helper from `src.data.openmeteo_historical_collector`:

```python
from src.data.openmeteo_historical_collector import compute_solar_angle
df["solar_elevation_deg"] = compute_solar_angle(df.index)
```

The public extract has an inlined version of the function:

```python
# Solar angle (sun elevation) — inlined, no external dependency
def _compute_solar_angle(dt_index, latitude=40.53):
    """Approximate solar elevation angle (degrees) for Spain."""
    day_of_year = dt_index.dayofyear
    hour_utc = dt_index.hour + dt_index.minute / 60.0
    B = 2 * np.pi * (day_of_year - 1) / 365.0
    declination = np.degrees(
        0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
        - 0.006758 * np.cos(2 * B) + 0.000907 * np.sin(2 * B)
    )
    solar_time = hour_utc + 0.23  # longitude correction ~3.5°W
    hour_angle = 15.0 * (solar_time - 12.0)
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)
    sin_elev = (np.sin(lat_rad) * np.sin(dec_rad)
                + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))
    return np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))
df["solar_elevation_deg"] = _compute_solar_angle(df.index)
```

This sanitization is preserved across syncs — when porting feature_engineering.py from EPFProject, replace the import-based version with this inlined version.

---

## Verification checklist (must pass before any sync commit)

After applying the sanitization to a public-extract file, verify:

```bash
# 1. No forbidden top-level imports
cd ~/PROJECT/epf-methodology
grep -nE "^from src\.data\.data_pipeline|^from src\.data\.ree_collector|^from src\.api\.|self\.ree = REECollector" src/models/*.py src/data/*.py
# Expected output: nothing

# 2. AST parses cleanly
python -c "
import ast
for f in ['src/data/feature_engineering.py','src/models/feature_selection.py','src/models/evaluation.py','src/models/trainer.py','src/models/lstm_embedder.py','src/models/direct_trainer.py','src/models/direct_predictor.py']:
    ast.parse(open(f, encoding='utf-8').read())
    print(f'  AST OK: {f}')
"
# Expected output: 7 'AST OK' lines

# 3. Verify the extraction header is present on all 7 files
grep -L "EPForecast Methodology" src/models/*.py src/data/*.py
# Expected output: nothing (every file should have the header)
```

If any of these checks fails, do NOT commit. Investigate and fix the sanitization first.

---

## How to do a methodology sync (the right way)

**Do not** run `bash scripts/sync_methodology.sh` in the EPFProject repo. That script does a `cp -f` of all 6 files which will clobber the public-facing edits described above.

**Instead, do this:**

1. **Identify which files actually need updating.** Run `diff -q EPFProject/src/<f> epf-methodology/src/<f>` for each of the 7 methodology files. Files that differ may need a sync; files that match are already in sync.
2. **For each file that differs**, decide whether the differences are:
   - **Methodology updates** that should propagate (model code changes, bug fixes, new features) → port them
   - **Sanitization** that should NOT propagate (the patterns above) → preserve the public version
3. **For methodology updates that should propagate**, the safest approach is:
   - Copy the EPFProject file to the public repo: `cp EPFProject/src/<f> epf-methodology/src/<f>`
   - Re-apply all sanitization patterns from this document by hand
   - Run the verification checklist above
4. **Write a clear commit message** explaining what's being synced. Example: "Sync v11.0 (M0.6 Phase C) — drop LSTM, ship single-XGBoost + residual_1w + pw3x".
5. **Push to the public repo.** This is a public release; treat it as a real release.

---

## History of broken syncs (so future-you doesn't repeat them)

- **2026-04-08** — Attempted `bash scripts/sync_methodology.sh` during M0.6 Phase A.3. The script tried to propagate ~200 lines of accumulated drift from earlier sprints AND would have re-introduced `from src.data.data_pipeline import DataPipeline` and `from src.data.ree_collector import REECollector` to the public repo. The sync was reverted on the methodology side and the M0.6 Phase A.3 was deferred. This document was created as part of the M0.6 Phase A.3 follow-up to prevent the same incident.
- **2026-04-09** — First successful manual sync since the broken attempt. M0.6 Phase C (v11.0 promotion + v10.x retraction) was the trigger. `direct_predictor.py` and `direct_trainer.py` were updated; the other 5 files were verified unchanged. Commit message: "feat(v11.0): sync to M0.6 Phase C — drop LSTM, ship single-XGBoost + residual_1w + pw3x + d365".
