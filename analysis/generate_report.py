"""Generate an HTML model-comparison report using available CSVs.

Usage:
    python analysis/generate_report.py

This script uses the project's prepared CSVs (no external download). It may take
several minutes depending on dataset size and models.
"""
from __future__ import annotations
from pathlib import Path
import sys

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from final_project import prepare_project_data, compare_models
import pandas as pd
import plotly.express as px
import plotly.io as pio


def main():
    out_dir = Path("analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing project data (using all available rows)...")
    data = prepare_project_data(row_limit=None)
    modeling = data["modeling"]

    print("Running model comparisons on available data (this may take a while) ...")
    results = compare_models(modeling, max_rows=None)
    metrics = results["metrics"].copy()
    metrics.to_csv(out_dir / "model_comparison_metrics.csv")

    # Create MAE bar chart
    mae = metrics[["MAE"]].reset_index()
    mae_fig = px.bar(mae, x="Model", y="MAE", title="MAE by model (lower is better)")
    mae_fig.update_layout(yaxis_title="MAE (occupancy points)")

    # Save per-model predictions
    for name, df in results["predictions"].items():
        safe_name = name.replace(" ", "_").lower()
        df.to_csv(out_dir / f"predictions_{safe_name}.csv", index=False)

    # Build HTML report
    report_parts = []
    report_parts.append("<h1>Model comparison report</h1>")
    report_parts.append("<h2>Regression metrics</h2>")
    report_parts.append(metrics.round(4).to_html(classes="table table-striped"))
    report_parts.append("<h2>MAE comparison</h2>")
    report_parts.append(pio.to_html(mae_fig, full_html=False, include_plotlyjs="cdn"))

    html = "\n".join(report_parts)
    report_path = out_dir / "report_model_comparison.html"
    with open(report_path, "w", encoding="utf8") as out:
        out.write("<!doctype html><html><head><meta charset=\"utf-8\"><title>Model comparison</title></head><body>")
        out.write(html)
        out.write("</body></html>")

    print(f"Saved HTML report: {report_path}")


if __name__ == "__main__":
    main()
