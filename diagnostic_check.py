import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet('data/processed/merged_catalog.parquet')

print('=============================================')
print('1. Orbit Lifetime Category & Days in Orbit')
print('=============================================')
print('days_in_orbit_estimate descriptive stats:')
print(df['days_in_orbit_estimate'].describe())
print('\nBinned histogram (10 bins) of days_in_orbit_estimate:')
hist, bins = np.histogram(df['days_in_orbit_estimate'].dropna(), bins=10)
for count, edge in zip(hist, bins):
    print(f"{edge:10.1f} : {count}")
    
print('\n=============================================')
print('2. Starlink Gen 1 Imbalance')
print('=============================================')
print('Row counts per constellation (Top 15):')
print(df['satellite_constellation'].value_counts(dropna=False).head(15))

print('\nUnique norad_id counts per constellation (Top 15):')
unique_counts = df.groupby('satellite_constellation')['norad_id'].nunique().sort_values(ascending=False)
print(unique_counts.head(15))

print('\n=============================================')
print('3. Data Provenance / Realism Check')
print('=============================================')
print('data_source unique values:')
print(df['data_source'].value_counts())

print('\nEpoch sample (first 15):')
print(df['epoch'].head(15).dt.strftime('%Y-%m-%d %H:%M:%S.%f'))
print('\nEpoch seconds/microseconds frequencies (first 1000):')
sample_epoch = df['epoch'].head(1000)
print('Seconds value counts:')
print(sample_epoch.dt.second.value_counts().head(5))
print('Microseconds value counts:')
print(sample_epoch.dt.microsecond.value_counts().head(5))

print('\nOrbital parameters sample (check if suspiciously round):')
print(df[['altitude_km', 'inclination', 'eccentricity', 'mean_motion']].head(15))

print('\n=============================================')
print('4. Snapshot Cadence in trajectory_timeseries')
print('=============================================')
print(f'Actual min snapshot_date: {df["snapshot_date"].min()}')
print(f'Actual max snapshot_date: {df["snapshot_date"].max()}')

np.random.seed(42)
sample_norads = np.random.choice(df['norad_id'].unique(), size=50, replace=False)
sample_df = df[df['norad_id'].isin(sample_norads)].copy()
sample_df = sample_df.sort_values(by=['norad_id', 'snapshot_date'])
sample_df['cadence_days'] = sample_df.groupby('norad_id')['snapshot_date'].diff().dt.days

print('\nDistribution of days between consecutive snapshots (for ~50 random norad_ids):')
print(sample_df['cadence_days'].describe())
print('\nValue counts of cadence_days:')
print(sample_df['cadence_days'].value_counts())
