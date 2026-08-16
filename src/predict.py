import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    logging.info("Loading model and features...")
    model = joblib.load("models/model.pkl")
    with open("models/features.json", "r") as f:
        feature_names = json.load(f)
        
    logging.info("Loading data...")
    # Load processed features to get the latest snapshot for each satellite
    df_features = pd.read_parquet("data/processed/features_target.parquet")
    latest_features = df_features.sort_values("snapshot_date").groupby("norad_id").tail(1).copy()
    
    # Load current_catalog to get names, object_type, country, etc.
    df_catalog = pd.read_csv("data/raw/current_catalog.csv")
    df_catalog["norad_id"] = df_catalog["norad_id"].astype(np.int64)
    latest_features["norad_id"] = latest_features["norad_id"].astype(np.int64)
    
    # Merge catalog info into features
    df = pd.merge(
        df_catalog[["norad_id", "name", "country", "object_type"]],
        latest_features,
        on="norad_id",
        how="inner"
    )
    
    logging.info(f"Generating predictions for {len(df)} satellites...")
    
    X = df[feature_names].copy()
    
    # Handle categorical features for prediction
    cat_features = ["satellite_constellation", "object_type_clean"]
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].astype("category")
            
    # Predict R
    predicted_r = model.predict(X)
    df["predicted_R"] = np.clip(predicted_r, 0, 100)  # R is [0, 100]
    
    logging.info("Computing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Extract top 3 SHAP features per row
    logging.info("Extracting top 3 SHAP features per satellite...")
    top_3_shap = []
    for i in range(len(df)):
        # Get absolute SHAP values for this row
        row_shap = np.abs(shap_values[i])
        # Get indices of top 3
        top_indices = np.argsort(row_shap)[-3:][::-1]
        top_features = [feature_names[idx] for idx in top_indices]
        top_3_shap.append(top_features)
        
    df["top_3_shap_features"] = top_3_shap
    
    # Generate simulated orbital parameters (RAAN and Mean Anomaly)
    logging.info("Generating simulated orbital parameters...")
    # We want to evenly space out satellites, especially within the same constellation.
    # We can use the rank of the satellite within its constellation to assign RAAN and Mean Anomaly.
    df["_rank"] = df.groupby("satellite_constellation").cumcount()
    df["_count"] = df.groupby("satellite_constellation")["norad_id"].transform("count")
    
    # Spread RAAN evenly [0, 360) and Mean Anomaly evenly [0, 360)
    # Using a prime multiplier to distribute Mean Anomaly differently from RAAN
    df["raan"] = (df["_rank"] / df["_count"]) * 360.0
    df["mean_anomaly"] = ((df["_rank"] * 7) % df["_count"]) / df["_count"] * 360.0
    
    # Prepare final output JSON
    output_cols = [
        "norad_id", "name", "object_type", "satellite_constellation", "country",
        "altitude_km", "inclination", "eccentricity", "mean_motion", 
        "predicted_R", "top_3_shap_features", "raan", "mean_anomaly"
    ]
    
    output_df = df[output_cols].copy()
    
    # Note in a code comment that without true RAAN/anomaly data, orbital paths are a
    # physically-plausible simulation for visualization, not live tracking.
    
    os.makedirs("web/assets", exist_ok=True)
    out_path = "web/assets/predictions.json"
    
    # Save as records
    output_df.to_json(out_path, orient="records")
    logging.info(f"Successfully saved {len(output_df)} predictions to {out_path}")

if __name__ == "__main__":
    main()
