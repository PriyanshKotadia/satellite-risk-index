"""
Post-pipeline sanity checks, correlation, and feature matrix validation.
Loads the 5-snapshot pipeline results from memory, or re-runs if needed.
"""
import sys
sys.path.insert(0, '.')

import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.features import build_all_features

print("Loading merged data...")
df = pd.read_parquet('data/processed/merged_catalog.parquet')

all_dates = sorted(df['snapshot_date'].unique())
sample_dates = all_dates[-5:]

print("Running full pipeline on 5 snapshots...")
t_start = time.perf_counter()
X, R, full_df = build_all_features(df, sample_snapshots=sample_dates)
t_end = time.perf_counter()
print(f"5-snapshot pipeline: {t_end-t_start:.1f}s")
print(f"Estimated for 283 snapshots: {(t_end-t_start) * 283/5 / 60:.1f} min")

# -------------------------------------------------------
# Sanity checks -- density by orbital_band
# -------------------------------------------------------
print("\n=== Sanity Check: weighted_density by orbital_band ===")
density_by_band = (
    full_df.groupby('orbital_band')['weighted_density']
    .agg(['mean', 'median', 'max'])
    .sort_values('median', ascending=False)
)
print(density_by_band.to_string())

print("\n=== Sanity Check: Sun-sync (700-800km) vs Starlink (~550km) vs GEO ===")
sun_sync = full_df[(full_df['altitude_km'] >= 700) & (full_df['altitude_km'] <= 800)]
starlink = full_df[(full_df['altitude_km'] >= 530) & (full_df['altitude_km'] <= 570)]
geo      = full_df[(full_df['altitude_km'] >= 35000) & (full_df['altitude_km'] <= 37000)]

for label, sub in [("Sun-sync 700-800km", sun_sync),
                   ("Starlink ~550km",     starlink),
                   ("GEO ~36k km",         geo)]:
    print(f"  {label}: n={len(sub)}, "
          f"median wd={sub['weighted_density'].median():.3f}, "
          f"mean wd={sub['weighted_density'].mean():.3f}")

# -------------------------------------------------------
# R vs congestion_risk Spearman correlation
# -------------------------------------------------------
print("\n=== R vs congestion_risk Spearman Correlation ===")
risk_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
full_df['congestion_risk_numeric'] = full_df['congestion_risk'].map(risk_map)
valid = full_df.dropna(subset=['congestion_risk_numeric', 'R'])
rho, pval = spearmanr(valid['R'], valid['congestion_risk_numeric'])
print(f"  Spearman rho = {rho:.4f}, p-value = {pval:.4e}")
print("  R distribution by congestion_risk bucket:")
print(full_df.groupby('congestion_risk')['R'].describe().round(2).to_string())

# -------------------------------------------------------
# Confirm excluded features are absent from X
# -------------------------------------------------------
print("\n=== Feature matrix columns ===")
print(X.columns.tolist())
excluded = ['days_in_orbit_estimate', 'orbit_lifetime_category',
            'weighted_density', 'density_velocity', 'congestion_risk']
print("\nExclusion check:")
for col in excluded:
    present = col in X.columns
    status = 'FAIL - PRESENT!' if present else 'OK - absent'
    print(f"  {col}: {status}")

print("\n=== R distribution ===")
print(full_df['R'].describe())
print("\n=== density_velocity distribution ===")
print(full_df['density_velocity'].describe())

print("\nDone.")
