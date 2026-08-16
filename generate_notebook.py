import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell("""\
# Model Training and Evaluation
This notebook evaluates the performance of the full LightGBM model trained on the complete dataset.
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
import json
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load Metrics
with open("../models/metrics.json", "r") as f:
    metrics = json.load(f)

print("LightGBM Aggregate Metrics:")
agg = metrics["lgbm"]["aggregate"]
print(f"RMSE: {agg['RMSE']:.4f}")
print(f"MAE: {agg['MAE']:.4f}")
print(f"Spearman: {agg['Spearman']:.4f}\\n")

print("Per-Constellation Metrics:")
for constel, m in agg["per_constellation"].items():
    print(f"--- {constel} ---")
    print(f"  RMSE: {m['RMSE']:.4f}")
    print(f"  MAE: {m['MAE']:.4f}")
    print(f"  Spearman: {m['Spearman']:.4f}")
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""\
## SHAP Analysis
Below is the SHAP summary plot showing feature importance.
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
from IPython.display import Image
Image(filename='../models/shap_summary.png')
"""))

nb.cells.append(nbf.v4.new_code_cell("""\
# Load the SHAP sample to show top features
shap_df = pd.read_parquet("../models/shap_sample.parquet")
feature_cols = [c for c in shap_df.columns if c.startswith("shap_")]
mean_abs_shap = shap_df[feature_cols].abs().mean().sort_values(ascending=False)

print("Top 10 features by mean absolute SHAP value:")
print(mean_abs_shap.head(10))
"""))

with open("notebooks/03_model_training.ipynb", "w") as f:
    nbf.write(nb, f)
