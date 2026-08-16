import pandas as pd

print('Loading merged...')
df = pd.read_parquet('data/processed/merged_catalog.parquet')

print('--- Value Counts ---')
for col in ['object_type', 'orbit_lifetime_category']:
    print(col)
    print(df[col].value_counts(dropna=False))
print('satellite_constellation')
print(df['satellite_constellation'].value_counts(dropna=False).head(15))
print('country')
print(df['country'].value_counts(dropna=False).head(15))

print('\n--- Trajectory Info ---')
traj = df.dropna(subset=['snapshot_date'])
print(f'Total rows: {len(traj)}')
print(f'Date range: {traj["snapshot_date"].min()} to {traj["snapshot_date"].max()}')
print('Snapshots per norad_id description:')
print(traj.groupby('norad_id').size().describe())

print('\n--- Nulls ---')
print(df.isnull().sum())
