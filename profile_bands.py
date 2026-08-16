"""Profile which band is slowest and test a small optimisation."""
import sys; sys.path.insert(0, '.')
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from src.features import _get_shell_params, _shell_pairs

df = pd.read_parquet('data/processed/merged_catalog.parquet')
test_date = df.groupby('snapshot_date').size().idxmax()
snap = df[df['snapshot_date'] == test_date]

print("Band sizes at most-populated snapshot:")
print(snap.groupby('orbital_band').size().sort_values(ascending=False))

print("\nPer-band timing breakdown (compute_shell_density):")
for band, group in snap.groupby('orbital_band'):
    dh, di = _get_shell_params(band)
    alt = group['altitude_km'].to_numpy(dtype=float)
    inc = group['inclination'].to_numpy(dtype=float)
    t0 = time.perf_counter()
    rows, cols = _shell_pairs(alt, inc, dh, di)
    t1 = time.perf_counter()
    print(f"  {band}: {len(group):6d} objects, {len(rows):8d} pairs, {t1-t0:.3f}s")
