"""Quick single-snapshot timing check for the Chebyshev KD-tree implementation."""
import sys
sys.path.insert(0, '.')

import time
import numpy as np
import pandas as pd
from src.features import compute_shell_density, compute_local_object_count_same_constellation

print("Loading data...")
df = pd.read_parquet('data/processed/merged_catalog.parquet')

# Use the most-populated snapshot date
date_counts = df.groupby('snapshot_date').size()
test_date = date_counts.idxmax()
n = date_counts.max()
print(f"Test snapshot: {test_date}  ({n} objects)\n")

print("=== compute_shell_density ===")
t0 = time.perf_counter()
density = compute_shell_density(df, test_date)
t1 = time.perf_counter()
dt_density = t1 - t0
print(f"  Time: {dt_density:.2f}s")
print(f"  Result shape: {density.shape}")
print(f"  weighted_density sample:\n{density['weighted_density'].describe()}")

print("\n=== compute_local_object_count_same_constellation ===")
t2 = time.perf_counter()
constel = compute_local_object_count_same_constellation(df, test_date)
t3 = time.perf_counter()
dt_constel = t3 - t2
print(f"  Time: {dt_constel:.2f}s")
print(f"  Result shape: {constel.shape}")

total_per_snap = dt_density + dt_constel
print(f"\nTotal per snapshot: {total_per_snap:.2f}s")
print(f"Estimated for 283 snapshots: {total_per_snap * 283 / 60:.1f} min")
