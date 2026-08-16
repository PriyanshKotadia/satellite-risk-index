import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            'norad_id': 'string',
            'altitude_km': float,
            'inclination': float,
            'eccentricity': float,
            'mean_motion': float,
            'launch_year_estimate': 'Int64',
            'days_in_orbit_estimate': 'Int64'
        }
    )
    df['epoch'] = pd.to_datetime(df['epoch'], format='mixed')
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'], format='mixed')
    df['last_seen'] = pd.to_datetime(df['last_seen'], format='mixed')
    return df

def clean_catalog(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(df)
    
    # Report nulls
    null_pct = df.isnull().sum() / initial_rows * 100
    if null_pct.sum() > 0:
        logging.info(f"Null % per column:\n{null_pct[null_pct > 0]}")
    
    # Handle missing values
    if 'object_type' in df.columns:
        df['object_type'] = df['object_type'].fillna('unknown')
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('unknown')
    
    # Drop exact duplicates
    df = df.drop_duplicates()
    
    # Flag invalid values
    invalid_alt = df['altitude_km'] < 0
    invalid_ecc = (df['eccentricity'] < 0) | (df['eccentricity'] >= 1)
    invalid_inc = (df['inclination'] < 0) | (df['inclination'] > 180)
    invalid_mm = df['mean_motion'] <= 0
    
    invalid_mask = invalid_alt | invalid_ecc | invalid_inc | invalid_mm
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        logging.warning(f"Found {invalid_count} rows with invalid orbital parameters.")
        if invalid_count / initial_rows > 0.01:
            logging.error(f"More than 1% of rows ({invalid_count}) are invalid.")
            raise ValueError(f"More than 1% ({invalid_count}/{initial_rows}) rows have invalid orbital parameters. Please approve dropping.")
        else:
            logging.info(f"Dropping {invalid_count} invalid rows.")
            df = df[~invalid_mask]
            
    final_rows = len(df)
    logging.info(f"Rows before: {initial_rows}, Rows after cleaning: {final_rows}")
    return df

def merge_sources(catalog_df: pd.DataFrame, trajectory_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([catalog_df, trajectory_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=['norad_id', 'snapshot_date'], keep='first')
    return merged

if __name__ == "__main__":
    logging.info("Loading current catalog...")
    current_cat = load_raw("data/raw/current_catalog.csv")
    logging.info("Cleaning current catalog...")
    current_cat = clean_catalog(current_cat)
    
    logging.info("Loading trajectory timeseries...")
    traj_cat = load_raw("data/raw/trajectory_timeseries.csv")
    logging.info("Cleaning trajectory timeseries...")
    traj_cat = clean_catalog(traj_cat)
    
    logging.info("Merging sources...")
    merged_df = merge_sources(current_cat, traj_cat)
    
    os.makedirs("data/processed", exist_ok=True)
    merged_df.to_parquet("data/processed/merged_catalog.parquet", index=False)
    logging.info(f"Saved merged catalog with shape {merged_df.shape} to data/processed/merged_catalog.parquet")
