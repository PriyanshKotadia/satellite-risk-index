# Model Card: Multi-Constellation Collision Risk Score Estimator

## Overview

This model estimates a continuous **Congestion & Proximity Risk Index (R)** for
every cataloged orbital object, using both a static snapshot (`current_catalog`)
and per-object history (`trajectory_timeseries`) to capture how fast an object's
orbital neighbourhood is filling up — not just how crowded it is at a point in time.

**R is a proxy congestion metric, not a true collision probability.** A real
conjunction assessment (e.g. CDM-style Pc) requires state vector covariance, which
is not available in this dataset. R measures *local orbital shell crowding +
closure trend*, useful for triage and visualization, not for maneuver decisions.

---

## Data Provenance

- **Source:** CelesTrak TLE (Two-Line Element) derived tracking data.
  `data_source` column = `celestrak` for 100% of rows.
- **Date range:** 2025-11-04 to 2026-08-15 (approximately 9.5 months).
- **Cadence:** Confirmed approximately daily snapshots. Median gap between
  consecutive snapshots per `norad_id` = 1.0 day; maximum observed gap = 2.0 days
  over a sample of 50 random satellites.
- **Epoch precision:** Microsecond-precision timestamps (e.g.,
  `2026-08-15 09:39:48.566016`) consistent with real TLE-derived orbital
  state vectors, not synthetic or regularly-spaced simulation output.
- **Orbital parameters:** Non-round, physically realistic values across altitude,
  inclination, eccentricity, and mean_motion — consistent with propagated TLE data.
- **Coverage:** 17,073 unique NORAD IDs; 283 distinct snapshot dates;
  4,199,914 total rows after cleaning.

---

## Known Data Quality Issues

### 1. `days_in_orbit_estimate` and `orbit_lifetime_category` — Excluded

**Finding:** `days_in_orbit_estimate` is broken upstream. Approximately 4.17 million
of 4.2 million rows (>99%) have a value of exactly `0.0`. An additional 632 rows
contain physically impossible negative values (min = -4.0 days). These values are
inconsistent with the objects' `launch_year_estimate` and real orbital history.

**Action:** Both `days_in_orbit_estimate` and `orbit_lifetime_category` (which is
derived from `days_in_orbit_estimate`) are **excluded from all model features**.
They remain in the raw and cleaned parquet files for audit purposes but carry no
real signal and would introduce noise or misleading patterns into the model.

See `src/features.py` → `build_feature_matrix()` for the explicit exclusion with
inline comments.

### 2. Starlink Gen 1 Class Imbalance

**Finding:** Starlink Gen 1 represents approximately 67% of all rows in
`trajectory_timeseries` (2,816,622 of 4,199,914 rows) and approximately 66% of
unique satellites by `norad_id` (11,326 of 17,073). This imbalance is genuine —
it reflects the actual composition of the tracked catalog, not an artifact of
snapshot frequency differences between constellations.

**Implication:** Without mitigation, models trained on this data may overfit to
Starlink orbital profiles. **Action at modeling stage:** Train/validation splits
will use stratified splitting by `satellite_constellation` (raw category retained
in feature matrix for this purpose). This will be documented in the training
notebook (`03_model_training.ipynb`).

### 3. Raw `weighted_density` Not Comparable Across Orbital Bands — Corrected

**Finding:** The initial target formulation used `weighted_density` directly in
`R_raw`. However, `weighted_density` is inflated for bands with wider shells:
GEO objects (shell: δh=75 km, δi=15°, area=1125 km·deg) captured far more
neighbours than LEO objects (shell: δh=10 km, δi=3°, area=30 km·deg) even though
GEO is genuinely far less crowded. In the 5-snapshot sanity check, GEO median
`weighted_density` (326) exceeded Starlink-altitude LEO median (206) — a physically
incorrect ordering — purely due to the wider shell capturing more objects.

**Fix applied in `build_target()`:** Normalise by shell area before ranking:

```
density_rate = weighted_density / (dh * di)
R_raw = density_rate + w * max(density_velocity, 0)
```

After this correction, LEO mega-constellation shells rank correctly above GEO in
`density_rate`. Raw `weighted_density` is retained as a diagnostic column in the
base DataFrame but is excluded from the feature matrix and the target.
See `docs/methodology.md §2.4` for full derivation.

---

## Feature Engineering

See `src/features.py` and `docs/methodology.md` for full details.

### Included Features
| Feature | Type | Notes |
|---|---|---|
| `altitude_km` | numeric | Raw |
| `inclination` | numeric | Raw |
| `eccentricity` | numeric | Raw |
| `mean_motion` | numeric | Orbits/day |
| `launch_year_estimate` | numeric | Raw |
| `data_staleness_days` | derived | `snapshot_date - last_seen` in days |
| `d_altitude` | derived | Daily rate of altitude change (§2.3) |
| `d_inclination` | derived | Daily rate of inclination change (§2.3) |
| `d_mean_motion` | derived | Daily rate of mean motion change (§2.3) |
| `has_trajectory_history` | derived | 0 for single-snapshot objects |
| `local_object_count_same_constellation` | derived | Same-constellation neighbours in shell |
| `satellite_constellation_freq` | freq-encoded | Frequency encoding of constellation |
| `country_freq` | freq-encoded | Frequency encoding of country |
| `satellite_constellation` | raw category | Retained for stratified splitting |
| `object_type_clean` | raw category | Normalised to uppercase |
| `band_*` | one-hot | One-hot encoded `orbital_band` |
| `alt_cat_*` | one-hot | One-hot encoded `altitude_category` |

### Excluded Features (with reasons)
| Feature | Reason |
|---|---|
| `days_in_orbit_estimate` | >99% zeros upstream; 632 negative values. No real signal. |
| `orbit_lifetime_category` | Derived from `days_in_orbit_estimate`. Same issue. |
| `weighted_density` | Target component — leakage (methodology §3). |
| `density_velocity` | Target component — leakage (methodology §3). |
| `congestion_risk` | Used for cross-validation sanity check only; not a model input. |

---

## Target Variable

**R**: Percentile-rank normalized Congestion & Proximity Risk Index, range [0, 100].

```
R_raw = weighted_density + w * max(density_velocity, 0)     [w = 1.0]
R     = percentile_rank(R_raw, within snapshot_date) × 100
```

Normalization is per `snapshot_date` so the score is comparable across snapshots
even as the catalog grows over time.

---

## Known Limitations

- R is a relative, snapshot-normalized index — do not present as absolute
  collision probability in UI copy.
- Density is computed only in (altitude, inclination) space, not full 3D
  position. Two objects in the same shell can be on opposite sides of Earth.
  A v2 could incorporate RAAN/argument of latitude if available.
- `launch_year_estimate` being an estimate means age-derived features carry
  inherited uncertainty — small differences should not be over-interpreted.
- Starlink Gen 1 dominance (~67% of catalog) means model performance on
  smaller constellations (OneWeb, Beidou, Galileo, Glonass) may be lower
  and should be evaluated separately per constellation.

---

## Final Model Performance and Justification

### Model Selection
Two models were evaluated:
- **LightGBM Regressor (Primary)**
- **Random Forest Regressor (Baseline)**

**Selected Model: LightGBM**
- **Justification:** Both models achieved comparable performance across 5-fold cross-validation (GroupKFold by `norad_id`) during an initial baseline comparison run on a stratified ~350k-row subsample. However, LightGBM is substantially more computationally efficient. When subsequently training LightGBM on the full dataset of ~4.2 million rows, it completed all 5 CV folds in under 3 minutes total, while preserving high Spearman rank correlation (>0.97). LightGBM also provides native support for categorical features, avoiding the need for manual Ordinal Encoding which can dilute splits.

### Full Dataset Metrics (LightGBM)
The final LightGBM model evaluated on the complete ~4.2M row dataset achieved:
- **Aggregate RMSE:** 6.217
- **Aggregate MAE:** 4.068
- **Aggregate Spearman:** 0.9757

### Per-Constellation Breakdown
To ensure the model is robust despite the massive Starlink Gen 1 imbalance (67% of data), metrics were computed per constellation:
- **Starlink Gen 1:** RMSE = 6.118 | MAE = 4.300 | Spearman = 0.9551
- **OneWeb:** RMSE = 2.176 | MAE = 1.691 | Spearman = 0.8046
- **Other:** RMSE = 6.835 | MAE = 3.888 | Spearman = 0.9101

Performance remains strong across the board, though OneWeb has a slightly lower Spearman correlation, likely due to its highly ordered and distinct orbital structure where variance in `R` is more compressed.

### Feature Importance (SHAP)
Analysis on a stratified sample of 10,000 rows indicates the model primarily relies on:
1. `altitude_km`
2. `mean_motion`
3. `local_object_count_same_constellation`
4. `band_LEO-Polar`
5. `satellite_constellation`

This aligns with physics: altitude defines the orbital shell width and general crowding, mean motion indicates speed (and precise height), and local same-constellation neighbors act as a strong proxy for mega-constellation deployment density.
