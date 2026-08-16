"""
Feature engineering for the Satellite Congestion & Proximity Risk Index (R).

Methodology reference: docs/methodology.md

KNOWN DATA QUALITY EXCLUSIONS (per diagnostics, see docs/model_card.md):
  - days_in_orbit_estimate: ~4.17M of 4.2M rows are exactly 0.0, and 632
    rows contain impossible negative values. The column carries no real signal
    and is excluded from build_feature_matrix().
  - orbit_lifetime_category: derived from days_in_orbit_estimate, so equally
    unreliable. Excluded from build_feature_matrix().
Both columns remain in the raw/cleaned parquet files but must not be used as
model inputs.
"""

import logging
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# §2.1  Per-band shell half-widths (δh in km, δi in degrees)
# ---------------------------------------------------------------------------
BAND_SHELL_PARAMS: dict[str, tuple[float, float]] = {
    "LEO-Polar":       (10.0,  3.0),
    "LEO-Inclined":    (10.0,  3.0),
    "LEO-Equatorial":  (10.0,  3.0),
    "LEO-Retrograde":  (10.0,  3.0),
    "MEO":             (50.0, 10.0),
    "GEO":             (75.0, 15.0),
    "GEO-Inclined":    (75.0, 15.0),
    "HEO":             (100.0, 20.0),
}
DEFAULT_SHELL_PARAMS = (25.0, 5.0)


def _get_shell_params(band: str) -> tuple[float, float]:
    return BAND_SHELL_PARAMS.get(band, DEFAULT_SHELL_PARAMS)


# ---------------------------------------------------------------------------
# Core: fully-vectorised Chebyshev-norm KD-tree kernel
# ---------------------------------------------------------------------------

def _shell_pairs(
    alt: np.ndarray,
    inc: np.ndarray,
    dh: float,
    di: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (row_idx, col_idx) of all pairs (i,j), i≠j, satisfying the
    rectangular shell constraint:
        |alt[j] - alt[i]| <= dh   AND   |inc[j] - inc[i]| <= di

    Implementation:
      1. Normalise coords to (alt/dh, inc/di) so the shell becomes a unit
         Chebyshev ball (p=inf).
      2. Use cKDTree.sparse_distance_matrix(max_distance=1.0, p=inf) which
         returns a scipy sparse matrix entirely in C — no Python pair loop.
      3. Extract (row, col) pairs from sparse.nonzero(); diagonal (i==i)
         is implicitly zero and excluded by nonzero().
    """
    n = len(alt)
    if n <= 1:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    coords = np.column_stack([alt / dh, inc / di])
    tree = cKDTree(coords)

    # sparse_distance_matrix returns a dok_matrix; convert to csr for fast nonzero
    sparse_mat = tree.sparse_distance_matrix(tree, max_distance=1.0, p=np.inf,
                                             output_type='coo_matrix')
    # coo_matrix already has .row / .col arrays — no Python loop needed
    row_idx = sparse_mat.row.astype(np.int32)
    col_idx = sparse_mat.col.astype(np.int32)

    # Remove self-pairs (distance = 0 on diagonal, included when max_distance >= 0)
    mask = row_idx != col_idx
    return row_idx[mask], col_idx[mask]



# ---------------------------------------------------------------------------
# §2.2  Shell density per snapshot
# ---------------------------------------------------------------------------

def _weighted_density_for_band(
    alt: np.ndarray, inc: np.ndarray, dh: float, di: float
) -> np.ndarray:
    """
    Weighted density = Σ_j 1/(1 + euclidean_dist(i,j)) in *standardised*
    (alt, inc) space (§2.2), where pairs are found via Chebyshev-norm KD-tree
    on (alt/dh, inc/di) normalised coords.

    Standardisation for the distance metric: alt/σ_alt, inc/σ_inc so both
    dimensions contribute comparably — independent of the Chebyshev query.
    """
    n = len(alt)
    if n == 0:
        return np.array([], dtype=float)

    row_idx, col_idx = _shell_pairs(alt, inc, dh, di)

    if len(row_idx) == 0:
        return np.zeros(n, dtype=float)

    # Standardise for the distance weight (independent of the shell query)
    alt_std = alt.std() if alt.std() > 0 else 1.0
    inc_std = inc.std() if inc.std() > 0 else 1.0
    alt_z = (alt - alt.mean()) / alt_std
    inc_z = (inc - inc.mean()) / inc_std

    dists = np.sqrt(
        (alt_z[row_idx] - alt_z[col_idx]) ** 2 +
        (inc_z[row_idx] - inc_z[col_idx]) ** 2
    )
    weights = 1.0 / (1.0 + dists)

    return np.bincount(row_idx, weights=weights, minlength=n)


def compute_shell_density(df: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    """
    §2.2: Compute weighted_density for every object in the snapshot.
    Grouped by orbital_band (different shell sizes); fully vectorised
    using Chebyshev-norm cKDTree (no Python per-row loop).
    """
    snap = df[df["snapshot_date"] == snapshot_date]
    if snap.empty:
        return pd.DataFrame(
            columns=["norad_id", "snapshot_date", "weighted_density"])

    parts = []
    for band, group in snap.groupby("orbital_band"):
        dh, di = _get_shell_params(band)
        alt = group["altitude_km"].to_numpy(dtype=float)
        inc = group["inclination"].to_numpy(dtype=float)
        ids = group["norad_id"].to_numpy()
        wd  = _weighted_density_for_band(alt, inc, dh, di)
        parts.append(pd.DataFrame({
            "norad_id":         ids,
            "snapshot_date":    snapshot_date,
            "weighted_density": wd,
        }))

    return pd.concat(parts, ignore_index=True) if parts else \
        pd.DataFrame(columns=["norad_id", "snapshot_date", "weighted_density"])


# ---------------------------------------------------------------------------
# Same-constellation local object count per snapshot
# ---------------------------------------------------------------------------

def compute_local_object_count_same_constellation(
    df: pd.DataFrame, snapshot_date
) -> pd.DataFrame:
    """
    Count same-constellation neighbours in the orbital shell for each object.
    Vectorised via Chebyshev-norm KD-tree per (band, constellation) group.
    """
    snap = df[df["snapshot_date"] == snapshot_date]
    if snap.empty:
        return pd.DataFrame(
            columns=["norad_id", "snapshot_date",
                     "local_object_count_same_constellation"])

    parts = []
    for (band, constel), group in snap.groupby(
            ["orbital_band", "satellite_constellation"]):
        dh, di = _get_shell_params(band)
        alt = group["altitude_km"].to_numpy(dtype=float)
        inc = group["inclination"].to_numpy(dtype=float)
        ids = group["norad_id"].to_numpy()
        n   = len(ids)

        if n <= 1:
            counts = np.zeros(n, dtype=int)
        else:
            row_idx, _ = _shell_pairs(alt, inc, dh, di)
            counts = np.bincount(row_idx, minlength=n).astype(int) \
                if len(row_idx) > 0 else np.zeros(n, dtype=int)

        parts.append(pd.DataFrame({
            "norad_id": ids,
            "snapshot_date": snapshot_date,
            "local_object_count_same_constellation": counts,
        }))

    return pd.concat(parts, ignore_index=True) if parts else \
        pd.DataFrame(
            columns=["norad_id", "snapshot_date",
                     "local_object_count_same_constellation"])


# ---------------------------------------------------------------------------
# §2.3  Congestion velocity (trajectory-derived)
# ---------------------------------------------------------------------------

def compute_trajectory_velocity(
    trajectory_df: pd.DataFrame,
    density_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-norad_id daily rates of change (§2.3):
      d_altitude, d_inclination, d_mean_motion, density_velocity.

    Actual dt computed from snapshot_date differences — handles the occasional
    2-day gaps correctly (confirmed ~daily cadence, max gap = 2 days in sample).

    Single-snapshot and first-row objects: velocities = 0.0,
    has_trajectory_history = 0.
    """
    traj = trajectory_df[
        ["norad_id", "snapshot_date", "altitude_km", "inclination", "mean_motion"]
    ].copy()

    traj = traj.merge(
        density_df[["norad_id", "snapshot_date", "weighted_density"]],
        on=["norad_id", "snapshot_date"],
        how="left",
    )
    traj = traj.sort_values(["norad_id", "snapshot_date"]).reset_index(drop=True)

    traj["dt_days"] = (
        traj.groupby("norad_id")["snapshot_date"]
        .diff()
        .dt.total_seconds()
        / 86400.0
    )

    for col, out in [
        ("altitude_km",      "d_altitude"),
        ("inclination",      "d_inclination"),
        ("mean_motion",      "d_mean_motion"),
        ("weighted_density", "density_velocity"),
    ]:
        delta = traj.groupby("norad_id")[col].diff()
        traj[out] = delta / traj["dt_days"]

    traj["has_trajectory_history"] = traj["dt_days"].notna().astype(int)
    snap_counts = traj.groupby("norad_id")["snapshot_date"].transform("count")
    traj.loc[snap_counts == 1, "has_trajectory_history"] = 0

    for col in ["d_altitude", "d_inclination", "d_mean_motion", "density_velocity"]:
        traj[col] = traj[col].fillna(0.0)

    return traj[
        ["norad_id", "snapshot_date", "d_altitude", "d_inclination",
         "d_mean_motion", "density_velocity", "has_trajectory_history"]
    ]


# ---------------------------------------------------------------------------
# §2.4  Final target variable R
# ---------------------------------------------------------------------------

def build_target(df: pd.DataFrame, w: float = 1.0) -> pd.Series:
    """
    R_raw = density_rate + w * max(density_velocity, 0)
    R     = percentile-rank of R_raw within each snapshot_date x 100 -> [0, 100].

    density_rate = weighted_density / (dh * di)
    -----------------------------------------------------------------------
    Raw weighted_density is inflated for bands with large shells (e.g. GEO
    uses dh=75 km, di=15 deg vs LEO's dh=10 km, di=3 deg) purely because a
    wider search radius captures more neighbours -- not because those shells
    are genuinely more crowded.  Dividing by shell area (dh x di) converts
    the raw inverse-distance weight sum into a density rate comparable across
    heterogeneous shell sizes.  Without this normalisation, global percentile-
    ranking biases GEO objects toward artificially high R.

    Raw weighted_density is retained as a column in the base DataFrame for
    diagnostics (e.g. sanity checks, notebook visualisations).  It remains
    excluded from build_feature_matrix() alongside density_velocity.
    See docs/methodology.md ss2.4 and docs/model_card.md.
    """
    # Build per-row shell-area divisor from the orbital_band lookup table.
    # BAND_SHELL_PARAMS keys must match the orbital_band values in the data.
    shell_area_map = {
        band: dh * di for band, (dh, di) in BAND_SHELL_PARAMS.items()
    }
    default_area = DEFAULT_SHELL_PARAMS[0] * DEFAULT_SHELL_PARAMS[1]

    shell_area = df["orbital_band"].map(shell_area_map).fillna(default_area)

    tmp = df[["snapshot_date", "weighted_density", "density_velocity"]].copy()
    tmp["density_rate"] = tmp["weighted_density"] / shell_area
    tmp["R_raw"] = (
        tmp["density_rate"] + w * tmp["density_velocity"].clip(lower=0)
    )
    R = (
        tmp.groupby("snapshot_date")["R_raw"]
        .rank(pct=True)
        * 100
    )
    return R.rename("R")


# ---------------------------------------------------------------------------
# §3  Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble model input matrix X.

    EXCLUDED features (see docs/model_card.md §Known Data Quality Issues):
      days_in_orbit_estimate    ~4.17M/4.2M rows = 0.0; 632 rows negative.
                                No real signal.
      orbit_lifetime_category   Derived from days_in_orbit_estimate;
                                equally unreliable.
      weighted_density          Target component — leakage (methodology §3).
      density_velocity          Target component — leakage (methodology §3).
      congestion_risk           Cross-validation sanity check only (§3).
    """
    feat = df.copy()

    feat["data_staleness_days"] = (
        (feat["snapshot_date"] - feat["last_seen"]).dt.total_seconds() / 86400.0
    )
    feat["object_type_clean"] = feat["object_type"].str.upper().str.strip()

    for col in ["satellite_constellation", "country"]:
        freq_map = feat[col].value_counts(normalize=True)
        feat[f"{col}_freq"] = feat[col].map(freq_map)

    feat = pd.get_dummies(
        feat,
        columns=["orbital_band", "altitude_category"],
        prefix=["band", "alt_cat"],
        dummy_na=False,
        dtype=float,
    )

    key_cols    = ["norad_id", "snapshot_date"]
    numeric_raw = ["altitude_km", "inclination", "eccentricity", "mean_motion",
                   "launch_year_estimate", "data_staleness_days"]
    derived_num = ["d_altitude", "d_inclination", "d_mean_motion",
                   "local_object_count_same_constellation", "has_trajectory_history"]
    freq_enc    = ["satellite_constellation_freq", "country_freq"]
    raw_cat     = ["satellite_constellation", "object_type_clean"]
    ohe_cols    = [c for c in feat.columns
                   if c.startswith("band_") or c.startswith("alt_cat_")]

    keep = key_cols + numeric_raw + derived_num + freq_enc + raw_cat + ohe_cols
    keep = [c for c in keep if c in feat.columns]
    return feat[keep]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_all_features(
    merged_df: pd.DataFrame,
    sample_snapshots: list | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Full pipeline over all (or sampled) snapshot dates.
    Returns (X_df, R_series, enriched_base_df).
    """
    all_dates = sorted(merged_df["snapshot_date"].unique())
    if sample_snapshots is not None:
        all_dates = [d for d in all_dates if d in sample_snapshots]

    logging.info(f"Processing {len(all_dates)} snapshot dates...")

    density_parts = []
    constel_parts = []

    t0 = time.perf_counter()
    for i, snap_date in enumerate(all_dates):
        density_parts.append(compute_shell_density(merged_df, snap_date))
        constel_parts.append(
            compute_local_object_count_same_constellation(merged_df, snap_date)
        )
        if (i + 1) % 5 == 0 or (i + 1) == len(all_dates):
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta  = (len(all_dates) - i - 1) / rate if rate > 0 else float("inf")
            logging.info(
                f"  Snapshot {i+1}/{len(all_dates)}  "
                f"{elapsed:.1f}s | {rate:.2f} snaps/s | ETA {eta/60:.1f} min"
            )

    density_df = pd.concat(density_parts, ignore_index=True)
    constel_df = pd.concat(constel_parts, ignore_index=True)

    logging.info("Computing trajectory velocity features...")
    traj_subset = merged_df[merged_df["snapshot_date"].isin(all_dates)]
    velocity_df = compute_trajectory_velocity(traj_subset, density_df)

    base = merged_df[merged_df["snapshot_date"].isin(all_dates)].copy()
    base = base.merge(density_df, on=["norad_id", "snapshot_date"], how="left")
    base = base.merge(constel_df, on=["norad_id", "snapshot_date"], how="left")
    base = base.merge(
        velocity_df[["norad_id", "snapshot_date", "d_altitude", "d_inclination",
                     "d_mean_motion", "density_velocity", "has_trajectory_history"]],
        on=["norad_id", "snapshot_date"],
        how="left",
    )

    for col in ["weighted_density", "local_object_count_same_constellation",
                "d_altitude", "d_inclination", "d_mean_motion",
                "density_velocity", "has_trajectory_history"]:
        if col in base.columns:
            base[col] = base[col].fillna(0.0)

    logging.info("Building target R...")
    base["R"] = build_target(base)

    logging.info("Building feature matrix X...")
    X = build_feature_matrix(base)

    total = time.perf_counter() - t0
    logging.info(f"Total pipeline time: {total:.1f}s")

    return X, base["R"], base
