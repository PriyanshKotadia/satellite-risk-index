"""
Re-run the full feature pipeline with the corrected build_target()
(density_rate normalisation), save features_target.parquet, and
report the sanity checks + Spearman correlation.
"""
import sys
sys.path.insert(0, '.')

import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.features import build_all_features, BAND_SHELL_PARAMS

print("Loading merged data...")
df = pd.read_parquet('data/processed/merged_catalog.parquet')
print(f"Shape: {df.shape}")

# -------------------------------------------------------
# Run on 5 snapshots for sanity checks (fast)
# -------------------------------------------------------
all_dates = sorted(df['snapshot_date'].unique())
sample_dates = all_dates[-5:]

print("\nRunning pipeline on 5 snapshots with corrected build_target()...")
t0 = time.perf_counter()
X, R, base = build_all_features(df, sample_snapshots=sample_dates)
print(f"Done in {time.perf_counter()-t0:.1f}s")

# Compute density_rate per row for inspection
shell_area_map = {band: dh * di for band, (dh, di) in BAND_SHELL_PARAMS.items()}
base['shell_area'] = base['orbital_band'].map(shell_area_map).fillna(25.0 * 5.0)
base['density_rate'] = base['weighted_density'] / base['shell_area']
base['R'] = R

print("\n=== Shell area by orbital_band ===")
for band, (dh, di) in sorted(BAND_SHELL_PARAMS.items()):
    print(f"  {band:<20s}  dh={dh:5.0f}  di={di:4.0f}  area={dh*di:7.0f} km*deg")

print("\n=== BEFORE fix: weighted_density by orbital_band ===")
print(base.groupby('orbital_band')['weighted_density']
      .agg(['mean','median','max']).sort_values('median', ascending=False).round(3).to_string())

print("\n=== AFTER fix: density_rate by orbital_band ===")
print(base.groupby('orbital_band')['density_rate']
      .agg(['mean','median','max']).sort_values('median', ascending=False).round(4).to_string())

print("\n=== Spot-checks (density_rate): Sun-sync / Starlink / GEO ===")
sun_sync = base[(base['altitude_km'] >= 700) & (base['altitude_km'] <= 800)]
starlink = base[(base['altitude_km'] >= 530) & (base['altitude_km'] <= 570)]
geo      = base[(base['altitude_km'] >= 35000) & (base['altitude_km'] <= 37000)]
for label, sub in [("Sun-sync 700-800km  (expect HIGH)", sun_sync),
                   ("Starlink ~550km     (expect HIGH)", starlink),
                   ("GEO ~36k km         (expect LOW) ", geo)]:
    print(f"  {label}: n={len(sub):5d}  "
          f"median density_rate={sub['density_rate'].median():.5f}  "
          f"median R={sub['R'].median():.2f}")

print("\n=== R vs congestion_risk Spearman Correlation (corrected R) ===")
risk_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
base['cr_numeric'] = base['congestion_risk'].map(risk_map)
valid = base.dropna(subset=['cr_numeric', 'R'])
rho, pval = spearmanr(valid['R'], valid['cr_numeric'])
print(f"  Spearman rho = {rho:.4f}, p-value = {pval:.4e}")
print("  R by congestion_risk bucket:")
print(base.groupby('congestion_risk')['R'].describe().round(2).to_string())

print("\nSanity checks done. Now running full 283-snapshot pipeline to re-save parquet...")

# -------------------------------------------------------
# Full 283-snapshot run with corrected build_target()
# -------------------------------------------------------
t1 = time.perf_counter()
X_full, R_full, base_full = build_all_features(df)
total_time = time.perf_counter() - t1
print(f"Full pipeline done in {total_time:.0f}s ({total_time/60:.1f} min)")

import os
os.makedirs('data/processed', exist_ok=True)
ft = X_full.copy()
ft['R'] = R_full.values
ft.to_parquet('data/processed/features_target.parquet', index=False)
print(f"Saved features_target.parquet: shape={ft.shape}")
print(f"Columns: {ft.columns.tolist()}")

print("\nExclusion check on saved parquet:")
for col in ['days_in_orbit_estimate','orbit_lifetime_category',
            'weighted_density','density_velocity','congestion_risk']:
    status = 'PRESENT -- ERROR' if col in ft.columns else 'OK - absent'
    print(f"  {col}: {status}")

print("\nDone.")
