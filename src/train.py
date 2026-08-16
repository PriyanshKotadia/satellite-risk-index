import argparse
import json
import logging
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Risk Index Model")
    parser.add_argument("--subsample", action="store_true", help="Run on a ~350k subsample for fast iteration")
    parser.add_argument("--models", type=str, default="both", choices=["both", "lgbm", "rf"], help="Models to train")
    return parser.parse_args()


def load_data(subsample: bool):
    logging.info("Loading features_target.parquet...")
    df = pd.read_parquet("data/processed/features_target.parquet")

    if subsample:
        logging.info("Subsampling data to ~350k rows by sampling norad_ids...")
        # Stratify by satellite_constellation and orbital_band to sample norad_ids
        latest_df = df.sort_values("snapshot_date").groupby("norad_id").tail(1)
        latest_df["stratify_col"] = latest_df["satellite_constellation"].astype(str) + "_" + \
                                    latest_df[["band_GEO", "band_GEO-Inclined", "band_HEO", "band_LEO-Equatorial",
                                               "band_LEO-Inclined", "band_LEO-Polar", "band_LEO-Retrograde",
                                               "band_MEO"]].idxmax(axis=1).astype(str)
        
        # Combine rare strata into "Other" to avoid ValueError in train_test_split
        counts = latest_df["stratify_col"].value_counts()
        rare = counts[counts < 5].index
        latest_df.loc[latest_df["stratify_col"].isin(rare), "stratify_col"] = "Other"

        from sklearn.model_selection import train_test_split
        sampled_ids, _ = train_test_split(
            latest_df["norad_id"], train_size=0.083, stratify=latest_df["stratify_col"], random_state=42
        )
        df = df[df["norad_id"].isin(sampled_ids)].copy()
        logging.info(f"Subsampled to {len(df)} rows and {len(sampled_ids)} unique satellites.")

    return df


def evaluate(y_true, y_pred, constellation):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # Spearman rank correlation
    try:
        rho, _ = spearmanr(y_true, y_pred)
    except:
        rho = 0.0

    metrics = {"RMSE": rmse, "MAE": mae, "Spearman": rho}

    # Group constellations for reporting: Starlink Gen 1, OneWeb, Other
    report_constellations = ["Starlink Gen 1", "OneWeb"]
    
    per_constel_metrics = {}
    for c in report_constellations + ["Other"]:
        if c == "Other":
            mask = ~constellation.isin(report_constellations)
        else:
            mask = constellation == c
            
        if mask.sum() > 0:
            c_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            c_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            try:
                c_rho, _ = spearmanr(y_true[mask], y_pred[mask])
            except:
                c_rho = 0.0
            per_constel_metrics[c] = {"RMSE": c_rmse, "MAE": c_mae, "Spearman": c_rho}
    
    metrics["per_constellation"] = per_constel_metrics
    return metrics


def train_model():
    args = parse_args()
    df = load_data(args.subsample)

    # Excluded features: 'days_in_orbit_estimate', 'orbit_lifetime_category', 
    # 'weighted_density', 'density_velocity', 'congestion_risk'
    
    target_col = "R"
    group_col = "norad_id"
    # Keep snapshot_date to prevent temporal leakage? The methodology said GroupKFold by norad_id.
    
    features = [c for c in df.columns if c not in [target_col, group_col, "snapshot_date"]]
    
    cat_features = ["satellite_constellation", "object_type_clean"]
    
    # We must properly format categorical features for LightGBM
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype("category")

    X = df[features]
    y = df[target_col]
    groups = df[group_col]
    constellation = df["satellite_constellation"] if "satellite_constellation" in df.columns else None

    # Handle Random Forest requirements (can't natively handle category type with strings)
    # So we'll create a numeric-encoded version of X for RF
    X_rf = X.copy()
    if args.models in ["both", "rf"]:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_rf[cat_features] = encoder.fit_transform(X_rf[cat_features].astype(str))

    gkf = GroupKFold(n_splits=5)

    all_metrics = {"lgbm": [], "rf": []}
    
    best_lgbm_model = None
    best_lgbm_spearman = -1.0
    
    logging.info(f"Training models: {args.models} with GroupKFold(n_splits=5)...")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logging.info(f"--- Fold {fold + 1}/5 ---")
        
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        constel_val = constellation.iloc[val_idx]
        
        # 1. LightGBM
        if args.models in ["both", "lgbm"]:
            t0 = time.time()
            X_train_lgb, X_val_lgb = X.iloc[train_idx], X.iloc[val_idx]
            
            lgb_reg = lgb.LGBMRegressor(
                n_estimators=100,
                n_jobs=-1,
                random_state=42 + fold
            )
            # Pass categorical_feature directly
            lgb_reg.fit(
                X_train_lgb, y_train,
                eval_set=[(X_val_lgb, y_val)],
                categorical_feature=cat_features,
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            
            preds_lgb = lgb_reg.predict(X_val_lgb)
            lgb_mets = evaluate(y_val, preds_lgb, constel_val)
            all_metrics["lgbm"].append(lgb_mets)
            logging.info(f"LGBM Fold {fold+1} | RMSE: {lgb_mets['RMSE']:.2f}, Spearman: {lgb_mets['Spearman']:.4f} | Time: {time.time()-t0:.1f}s")
            
            if lgb_mets["Spearman"] > best_lgbm_spearman:
                best_lgbm_spearman = lgb_mets["Spearman"]
                best_lgbm_model = lgb_reg

        # 2. Random Forest
        if args.models in ["both", "rf"]:
            t0 = time.time()
            X_train_rf, X_val_rf = X_rf.iloc[train_idx], X_rf.iloc[val_idx]
            
            rf_reg = RandomForestRegressor(
                n_estimators=50,
                max_depth=15,
                n_jobs=-1,
                random_state=42 + fold
            )
            rf_reg.fit(X_train_rf, y_train)
            
            preds_rf = rf_reg.predict(X_val_rf)
            rf_mets = evaluate(y_val, preds_rf, constel_val)
            all_metrics["rf"].append(rf_mets)
            logging.info(f"RF   Fold {fold+1} | RMSE: {rf_mets['RMSE']:.2f}, Spearman: {rf_mets['Spearman']:.4f} | Time: {time.time()-t0:.1f}s")

    # Aggregate Metrics
    final_report = {}
    for model_name in ["lgbm", "rf"]:
        if len(all_metrics[model_name]) == 0:
            continue
            
        mean_metrics = {
            "RMSE": np.mean([m["RMSE"] for m in all_metrics[model_name]]),
            "MAE": np.mean([m["MAE"] for m in all_metrics[model_name]]),
            "Spearman": np.mean([m["Spearman"] for m in all_metrics[model_name]])
        }
        
        # Per constellation means
        per_c_means = {}
        for c in ["Starlink Gen 1", "OneWeb", "Other"]:
            c_rmse = np.mean([m["per_constellation"].get(c, {}).get("RMSE", 0) for m in all_metrics[model_name]])
            c_mae = np.mean([m["per_constellation"].get(c, {}).get("MAE", 0) for m in all_metrics[model_name]])
            c_rho = np.mean([m["per_constellation"].get(c, {}).get("Spearman", 0) for m in all_metrics[model_name]])
            per_c_means[c] = {"RMSE": c_rmse, "MAE": c_mae, "Spearman": c_rho}
            
        mean_metrics["per_constellation"] = per_c_means
        final_report[model_name] = {
            "folds": all_metrics[model_name],
            "aggregate": mean_metrics
        }
        
        logging.info(f"\n--- {model_name.upper()} FINAL AGGREGATE ---")
        logging.info(f"RMSE: {mean_metrics['RMSE']:.3f}, Spearman: {mean_metrics['Spearman']:.4f}")
        for c in ["Starlink Gen 1", "OneWeb", "Other"]:
            logging.info(f"  {c:<15}: RMSE = {per_c_means[c]['RMSE']:.3f}, Spearman = {per_c_means[c]['Spearman']:.4f}")
            
    # Save best model and metrics
    os.makedirs("models", exist_ok=True)
    with open("models/metrics.json", "w") as f:
        json.dump(final_report, f, indent=2)
    logging.info("Saved models/metrics.json")
        
    if best_lgbm_model is not None:
        # Save LightGBM model
        joblib.dump(best_lgbm_model, "models/model.pkl")
        logging.info("Saved best LightGBM model to models/model.pkl")
        
        # Save features list for SHAP script
        with open("models/features.json", "w") as f:
            json.dump(features, f)

if __name__ == "__main__":
    train_model()
