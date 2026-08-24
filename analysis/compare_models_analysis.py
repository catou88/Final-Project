#%%
"""Analysis: compare candidate models (60/20/20 split) and show results.
Run cells in VS Code (Run Cell) or run the whole file with Python.
"""
from __future__ import annotations

#%%
# Lightweight imports and data preparation
import plotly.express as px
import pandas as pd

from pathlib import Path
import sys

# Ensure the project root is on sys.path so `from final_project import ...` works
# when running this script directly from the `analysis/` folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from final_project import prepare_project_data, compare_models

#%%
# Load a manageable sample for interactive analysis (reduce rows for speed)
data = prepare_project_data(row_limit=100_000)
modeling = data["modeling"]

#%%
# Run the comparison (keeps default 60/20/20 chronological split)
results = compare_models(modeling, max_rows=50_000)
metrics = results["metrics"].copy()
print("Model comparison metrics:")
print(metrics.round(4))

#%%
# Bar chart: MAE comparison
mae = metrics[["MAE"]].reset_index()
fig = px.bar(mae, x="Model", y="MAE", title="MAE by model (lower is better)")
fig.update_layout(yaxis_title="MAE (occupancy points)")
fig.show()

#%%
# Save results for review
metrics.to_csv("analysis/model_comparison_metrics.csv")
for name, df in results["predictions"].items():
    safe_name = name.replace(" ", "_").lower()
    df.to_csv(f"analysis/predictions_{safe_name}.csv", index=False)
print("Saved metrics and per-model predictions to analysis/ folder.")

#%%
# Note: classification metrics removed — keep regression metrics only (MAE, RMSE, R2, Within 10 percentage points)
