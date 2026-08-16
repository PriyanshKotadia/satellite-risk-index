import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import joblib
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    logging.info("Loading model and features...")
    model = joblib.load("models/model.pkl")
    
    with open("models/features.json", "r") as f:
        features = json.load(f)
        
    logging.info("Loading data for SHAP analysis...")
    df = pd.read_parquet("data/processed/features_target.parquet")
    
    # We want a 5k-10k row stratified sample.
    # Stratify by satellite_constellation and orbital_band
    df["stratify_col"] = df["satellite_constellation"].astype(str) + "_" + \
                         df[["band_GEO", "band_GEO-Inclined", "band_HEO", "band_LEO-Equatorial",
                             "band_LEO-Inclined", "band_LEO-Polar", "band_LEO-Retrograde",
                             "band_MEO"]].idxmax(axis=1).astype(str)
                             
    counts = df["stratify_col"].value_counts()
    rare = counts[counts < 5].index
    df.loc[df["stratify_col"].isin(rare), "stratify_col"] = "Other"
    
    # Sample 10,000 rows
    sample_size = 10000
    frac = sample_size / len(df)
    
    # If the dataset is smaller than the sample size, use the whole dataset.
    if frac >= 1.0:
        sample_df = df
    else:
        sample_df, _ = train_test_split(
            df, train_size=sample_size, stratify=df["stratify_col"], random_state=42
        )
        
    logging.info(f"Sampled {len(sample_df)} rows for SHAP analysis.")
    
    X_sample = sample_df[features].copy()
    
    # Handle categorical columns for LightGBM explainer if they are category dtype
    cat_features = ["satellite_constellation", "object_type_clean"]
    for c in cat_features:
        if c in X_sample.columns:
            X_sample[c] = X_sample[c].astype("category")

    logging.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    logging.info("Saving SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=300)
    plt.close()
    
    logging.info("Saving per-row SHAP values...")
    # Save the SHAP values back to a parquet with norad_id and snapshot_date
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{f}" for f in features])
    shap_df["norad_id"] = sample_df["norad_id"].values
    shap_df["snapshot_date"] = sample_df["snapshot_date"].values
    
    shap_df.to_parquet("models/shap_sample.parquet", index=False)
    logging.info("Saved models/shap_sample.parquet")

if __name__ == "__main__":
    main()
