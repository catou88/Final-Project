# SFpark Parking Availability Project

This repository contains the final project code for modeling and visualizing San Francisco parking availability.

## Repository Structure

- `final_project.py` - Main data preparation and model comparison workflow.
- `dashboard.py` - Streamlit dashboard entry point.
- `dashboard_dash.py` - Dash dashboard version.
- `model_api.py` - API-facing model helpers.
- `models/` - Reusable model code.
- `data/raw/` - Source data files used by the project.
- `data/processed/` - Processed modeling sample used by the modeling workflow.
- `analysis/` - Model comparison scripts, generated metrics, prediction outputs, and reports.
- `analysis/eda/` - Exploratory analysis only. These files are not imported by the main app or model pipeline.
- `script/` - Supporting notebooks and visualization scripts from teammates.
- `output/` - Saved figures and visual outputs.

## Main Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main project workflow:

```bash
python final_project.py
```

Run the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

Run the model comparison report:

```bash
python analysis/generate_report.py
```
