"""Run full 283-snapshot feature engineering pipeline and save output."""
import sys
sys.path.insert(0, '.')
import os
import pandas as pd
from src.features import build_all_features

print("Loading merged data...")
df = pd.read_parquet('data/processed/merged_catalog.parquet')
print(f"Shape: {df.shape}")

print("Running full pipeline over all 283 snapshots...")
X, R, full_df = build_all_features(df)

os.makedirs('data/processed', exist_ok=True)
features_target = X.copy()
features_target['R'] = R.values
features_target.to_parquet('data/processed/features_target.parquet', index=False)
print(f"Saved features_target.parquet: {features_target.shape}")
print("Column list:")
print(features_target.columns.tolist())
print("Done.")
