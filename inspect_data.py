import pandas as pd
import json

files = ['data/raw/current_catalog.csv', 'data/raw/trajectory_timeseries.csv']
for f in files:
    try:
        df = pd.read_csv(f)
        print(f'\n--- {f} ---')
        print(f'Shape: {df.shape}')
        print(f'dtypes:\n{df.dtypes.to_string()}')
        print(f'Null counts:\n{df.isnull().sum().to_string()}')
        print(f'Unique norad_id: {df["norad_id"].nunique()}')
        
        if 'current' in f:
            counts = df["norad_id"].value_counts()
            multiple = (counts > 1).sum()
            print(f'norad_ids with >1 row in current_catalog: {multiple}')
        if 'trajectory' in f:
            counts = df.groupby("norad_id")["snapshot_date"].nunique()
            avg_snapshots = counts.mean()
            print(f'Avg snapshot_date per norad_id in trajectory_timeseries: {avg_snapshots:.2f}')
    except Exception as e:
        print(f'Error reading {f}: {e}')
