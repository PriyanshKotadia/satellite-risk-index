# Methodology: Multi-Constellation Collision Risk Score Estimator

## 1. Problem framing

We estimate a continuous **Congestion & Proximity Risk Index (R)** for every
cataloged object in `current_catalog`, using both a static snapshot
(`current_catalog`) and object-level history (`trajectory_timeseries`) to
capture *how fast* an object's orbital neighborhood is filling up, not just
how crowded it is right now.

**This is a proxy congestion metric, not a true collision probability.**
A real conjunction assessment (e.g. CDM-style Pc) requires state vector
covariance, which we don't have. We say this explicitly in the model card —
R measures *local orbital shell crowding + closure trend*, useful for
triage and visualization, not for maneuver decisions.

---

## 2. Target variable construction

### 2.1 Orbital shell definition

For a snapshot date `d`, define the shell around object `i` as all other
objects `j` (same `d`) satisfying:

```
|altitude_km_j - altitude_km_i| <= δh      (default δh = 25 km)
|inclination_j  - inclination_i| <= δi      (default δi = 5 deg)
```

`δh` and `δi` should scale with `orbital_band` — LEO shells are much more
crowded than MEO/GEO, so use tighter bins in LEO (δh = 10–15 km) and wider
bins above ~2000 km. Store this as a lookup table keyed by `orbital_band`
rather than a single global constant.

### 2.2 Raw density score

```
density_i = count(j in shell_i, j != i)
```

Optionally weight each neighbor by inverse distance in the (altitude,
inclination) plane so closer objects contribute more:

```
weighted_density_i = sum over j in shell_i of  1 / (1 + euclidean_dist(i, j))
```

where `euclidean_dist` is computed on standardized `altitude_km` and
`inclination` (standardize so both dimensions contribute comparably —
altitude is in km, inclination in degrees, very different scales).

### 2.3 Congestion velocity (trajectory-derived)

For objects with multiple snapshots in `trajectory_timeseries`, compute
per-`norad_id`, sorted by `epoch`/`snapshot_date`:

```
d_altitude   = altitude_km(t) - altitude_km(t-1)
d_inclination = inclination(t) - inclination(t-1)
d_mean_motion = mean_motion(t) - mean_motion(t-1)
dt            = snapshot_date(t) - snapshot_date(t-1)   # in days
```

Normalize by `dt` to get daily rates. Also compute the **rate of change of
weighted_density itself** across snapshots — this is the actual "closing
velocity" signal: a shell whose density is rising fast is higher risk than
one that's equally crowded but stable.

```
density_velocity_i = (weighted_density_i(t) - weighted_density_i(t-1)) / dt
```

Objects with only one snapshot get `density_velocity = 0` and a boolean
flag `has_trajectory_history = 0` (impute, don't drop — most of the catalog
may only have one snapshot).

### 2.4 Final target

#### Shell-area normalisation

Raw `weighted_density` is not directly comparable across orbital bands because
the shell half-widths `δh` and `δi` differ by band (see §2.1). A wider shell
captures more neighbours by construction, so a GEO object (δh=75 km, δi=15°)
accumulates higher raw weighted_density than a LEO object (δh=10 km, δi=3°)
even if the true spatial density is lower. Before feeding into the global
percentile rank, we normalise:

```
density_rate_i = weighted_density_i / (δh_band * δi_band)
```

This converts the raw inverse-distance weight sum into a *per-unit-shell-area*
density, making the metric comparable across heterogeneous shell sizes. Raw
`weighted_density` is retained as a diagnostic column but is **not** used in
the target or feature matrix.

#### Target construction

```
R_raw = density_rate + w * max(density_velocity, 0)   [w = 1.0]
```

Only positive velocity (crowding *increasing*) adds risk; a shell that's
clearing out isn't penalised. Start with `w = 1.0`, tune later.

Normalize `R_raw` to `R` in [0, 100] via percentile rank **within each
`snapshot_date`** (not global min-max) so the score is comparable across
snapshots even as the catalog grows over time.

Cross-check: if the existing `congestion_risk` column is categorical
(e.g. low/med/high), compute R's percentile bucket per category and confirm
rank-ordering roughly agrees. Large disagreement is a signal to inspect,
not necessarily a bug — flag and document either way.

---

## 3. Feature engineering (model inputs X)

| Feature | Source | Notes |
|---|---|---|
| `altitude_km` | raw | |
| `inclination` | raw | |
| `eccentricity` | raw | |
| `mean_motion` | raw | orbits/day; proxy for orbital period |
| `orbital_band` | raw | categorical, one-hot |
| `altitude_category` | raw | categorical, one-hot |
| `satellite_constellation` | raw | high-cardinality; target-encode or frequency-encode |
| `country` | raw | high-cardinality; frequency-encode |
| `object_type` | raw | payload / debris / rocket body — likely a strong feature, debris tends to cluster |
| `launch_year_estimate` | raw | numeric |
| `data_staleness_days` | derived | `snapshot_date - last_seen` |
| `d_altitude`, `d_inclination`, `d_mean_motion` | derived | per §2.3, daily rates |
| `has_trajectory_history` | derived | boolean flag |
| `local_object_count_same_constellation` | derived | count of same-constellation objects in shell — separates "your own constellation is dense" from "cross-operator congestion", these have different operational implications |

**Excluded Features**: `days_in_orbit_estimate` and `orbit_lifetime_category` were excluded from the model because diagnostics revealed they are severely corrupted upstream (e.g. over 99% of values are exactly 0.0 or mathematically impossible negative values).

**Do not include `weighted_density` or `density_velocity` as model
features** — they're the components of the target, that's target leakage.
The model should *learn to predict* density/velocity from orbital +
categorical features, not be handed the answer.

**Do not include `congestion_risk`** as a feature for the same reason if
it's derived from density; only use it for cross-validation sanity checks.

---

## 4. Model

- **Primary**: LightGBM Regressor (handles mixed categorical/numeric well,
  fast to iterate, native categorical support via `category` dtype).
- **Baseline for comparison**: Random Forest Regressor (sklearn) — simpler
  to explain, worth reporting both in the model card even if LGBM wins.
- **CV strategy**: GroupKFold grouped by `norad_id`, 5 folds. This is
  important — without grouping, the same satellite's snapshots leak across
  train/val and metrics will look artificially good.
- **Metrics**: RMSE and MAE on `R`, plus Spearman rank correlation between
  predicted and actual `R` per fold (ranking quality matters more than
  point accuracy for a "risk score" used to prioritize attention).
- **Explainability**: SHAP values per prediction, stored alongside each
  prediction so the web UI can show "top 3 contributing factors" on click.

---

## 5. Validation / sanity checks before trusting the model

1. Known high-density regions (Sun-synchronous ~700-800km band, Starlink
   shells ~550km) should score high — spot-check these manually.
2. Debris vs. active payload: debris-heavy shells (e.g. post-fragmentation
   event altitudes, if identifiable from the data) should score high.
3. GEO belt should score low on density (it's sparse in absolute count)
   even though popularly assumed "crowded" — if the model gets this
   backwards, check the shell binning in §2.1 isn't using LEO-scale δh/δi
   for all bands.

---

## 6. Known limitations (state in model card)

- R is a relative, snapshot-normalized index, not an absolute collision
  probability — do not present it as one in the UI copy.
- Density is computed only in (altitude, inclination) space, not full 3D
  position — two objects in the same shell can be on opposite sides of
  Earth at a given instant. This is a known simplification; a v2 could
  incorporate RAAN/argument of latitude if available in the raw data.
- `launch_year_estimate` / `days_in_orbit_estimate` being "estimates"
  (per column naming) means some age-derived features carry inherited
  uncertainty — don't over-interpret small differences.
