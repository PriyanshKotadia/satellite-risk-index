import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/merged_catalog.parquet')

# Check unique snapshot_dates and unique orbital_bands
print("=== Unique orbital_band values ===")
print(df['orbital_band'].value_counts())

print("\n=== Unique snapshot_dates count ===")
print(df['snapshot_date'].nunique())
print("Min/max:", df['snapshot_date'].min(), df['snapshot_date'].max())

print("\n=== Unique altitude_category values ===")
print(df['altitude_category'].value_counts())

print("\n=== congestion_risk values ===")
print(df['congestion_risk'].value_counts())

print("\n=== object_type values ===")
print(df['object_type'].value_counts())

print("\n=== Rows per snapshot_date (sample: first 5) ===")
print(df.groupby('snapshot_date').size().head())

print("\n=== Shape ===")
print(df.shape)

print("\n=== Sample norad_ids snapshot count ===")
print(df.groupby('norad_id').size().describe())
