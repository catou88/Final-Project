"""Interactive Dash website for the SFpark next-hour occupancy model.

Run from the project directory with:

    python3.12 -m pip install dash plotly pandas numpy scikit-learn catboost certifi
    python3.12 dashboard_dash.py

Then open http://127.0.0.1:8050 in a browser.
"""

from __future__ import annotations

import importlib.util
import json
import math
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html


# ---------------------------------------------------------------------------
# 1. Paths and settings
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "final_project.py"
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
BLOCKFACE_FILE = RAW_DIR / "blockfaces_with_meters.csv"

# Limit the dashboard run so it starts reasonably quickly on a laptop.
DASHBOARD_ROW_LIMIT = 100_000


# ---------------------------------------------------------------------------
# 2. Load the modeling code and train the model once
# ---------------------------------------------------------------------------

def load_project_module():
    """Import final_project.py from the same directory as this dashboard."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_FILE.name}. Put final_project.py in {PROJECT_DIR}."
        )

    specification = importlib.util.spec_from_file_location(
        "sfpark_final_project", MODEL_FILE
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not import {MODEL_FILE}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


project = load_project_module()


@lru_cache(maxsize=1)
def load_dashboard_assets():
    """Load data and train once; callbacks reuse the cached model."""
    project_data = project.prepare_project_data(row_limit=DASHBOARD_ROW_LIMIT)
    model_results = project.train_and_evaluate_model(
        project_data["modeling"],
        max_rows=None,
        include_catboost=False,
    )
    return project_data, model_results


@lru_cache(maxsize=1)
def load_blockfaces() -> pd.DataFrame:
    """Load SFMTA block geometry and calculate each segment's midpoint."""
    if not BLOCKFACE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {BLOCKFACE_FILE}. The interactive map requires this file."
        )

    blockfaces = pd.read_csv(BLOCKFACE_FILE)
    blockfaces.columns = blockfaces.columns.str.strip().str.lower()
    numeric_columns = [
        "blockface_id",
        "block_id",
        "endpt1_longitude",
        "endpt1_latitude",
        "endpt2_longitude",
        "endpt2_latitude",
    ]
    for column in numeric_columns:
        blockfaces[column] = pd.to_numeric(blockfaces[column], errors="coerce")

    blockfaces.dropna(subset=numeric_columns, inplace=True)
    blockfaces["block_id"] = blockfaces["block_id"].astype("int32")
    blockfaces["midpoint_latitude"] = (
        blockfaces["endpt1_latitude"] + blockfaces["endpt2_latitude"]
    ) / 2
    blockfaces["midpoint_longitude"] = (
        blockfaces["endpt1_longitude"] + blockfaces["endpt2_longitude"]
    ) / 2
    return blockfaces


# ---------------------------------------------------------------------------
# 3. Geographic helper functions
# ---------------------------------------------------------------------------

@lru_cache(maxsize=100)
def geocode_destination(destination: str) -> dict[str, object]:
    """Convert a San Francisco address/place name into coordinates."""
    query = destination.strip()
    if not query:
        raise ValueError("Enter a destination address or place name.")
    if "san francisco" not in query.casefold():
        query = f"{query}, San Francisco, CA"

    parameters = urlencode(
        {
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "us",
            "viewbox": "-122.55,37.84,-122.35,37.69",
            "bounded": 1,
            "q": query,
        }
    )
    request = Request(
        f"https://nominatim.openstreetmap.org/search?{parameters}",
        headers={"User-Agent": "SFparkAcademicDashDashboard/1.0"},
    )
    with urlopen(request, timeout=15, context=project.SSL_CONTEXT) as response:
        matches = json.load(response)

    if not matches:
        raise ValueError(
            "Destination not found in San Francisco. Try a complete address."
        )

    return {
        "latitude": float(matches[0]["lat"]),
        "longitude": float(matches[0]["lon"]),
        "display_name": matches[0]["display_name"],
    }


def distances_in_miles(
    latitude: pd.Series,
    longitude: pd.Series,
    destination_latitude: float,
    destination_longitude: float,
) -> np.ndarray:
    """Calculate great-circle distance from the destination to many points."""
    latitudes = np.radians(latitude.astype(float).to_numpy())
    longitudes = np.radians(longitude.astype(float).to_numpy())
    destination_latitude_radians = math.radians(destination_latitude)
    destination_longitude_radians = math.radians(destination_longitude)

    latitude_delta = latitudes - destination_latitude_radians
    longitude_delta = longitudes - destination_longitude_radians
    haversine = (
        np.sin(latitude_delta / 2) ** 2
        + np.cos(destination_latitude_radians)
        * np.cos(latitudes)
        * np.sin(longitude_delta / 2) ** 2
    )
    return 2 * 3958.8 * np.arcsin(np.sqrt(haversine))


def radius_circle(
    latitude: float,
    longitude: float,
    radius_miles: float,
    points: int = 72,
) -> tuple[list[float], list[float]]:
    """Return coordinates for the approximate search-radius circle."""
    angular_distance = radius_miles / 3958.8
    latitude_radians = math.radians(latitude)
    longitude_radians = math.radians(longitude)
    circle_latitudes: list[float] = []
    circle_longitudes: list[float] = []

    for bearing in np.linspace(0, 2 * math.pi, points, endpoint=True):
        circle_latitude = math.asin(
            math.sin(latitude_radians) * math.cos(angular_distance)
            + math.cos(latitude_radians)
            * math.sin(angular_distance)
            * math.cos(bearing)
        )
        circle_longitude = longitude_radians + math.atan2(
            math.sin(bearing)
            * math.sin(angular_distance)
            * math.cos(latitude_radians),
            math.cos(angular_distance)
            - math.sin(latitude_radians) * math.sin(circle_latitude),
        )
        circle_latitudes.append(math.degrees(circle_latitude))
        circle_longitudes.append(math.degrees(circle_longitude))

    return circle_latitudes, circle_longitudes


# ---------------------------------------------------------------------------
# 4. Figure and table builders
# ---------------------------------------------------------------------------

def empty_map(latitude: float = 37.7879, longitude: float = -122.4075):
    """Create an empty San Francisco map used before the first prediction."""
    figure = go.Figure()
    figure.update_layout(
        map={
            "style": "open-street-map",
            "center": {"lat": latitude, "lon": longitude},
            "zoom": 13,
        },
        height=600,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return figure


def build_parking_map(
    nearby_faces: pd.DataFrame,
    block_predictions: pd.DataFrame,
    destination: dict[str, object],
    radius_miles: float,
    search_blocks: int,
) -> go.Figure:
    """Build the interactive map with predicted block availability."""
    destination_latitude = float(destination["latitude"])
    destination_longitude = float(destination["longitude"])

    if nearby_faces.empty:
        figure = empty_map(destination_latitude, destination_longitude)
    else:
        block_distances = (
            nearby_faces.groupby("block_id", as_index=False)["distance_miles"].min()
        )
        if not block_predictions.empty:
            block_predictions = block_predictions.merge(
                block_distances,
                left_on="BLOCK_ID",
                right_on="block_id",
                how="left",
            )
            block_predictions["availability_band"] = np.select(
                [
                    block_predictions["predicted_availability_percent"].ge(40),
                    block_predictions["predicted_availability_percent"].ge(20),
                ],
                ["Higher availability (40%+)", "Moderate availability (20-39%)"],
                default="Limited availability (under 20%)",
            )

        prediction_columns = [
            "BLOCK_ID",
            "STREET_BLOCK",
            "predicted_availability_percent",
            "predicted_occupancy_percent",
            "availability_band",
        ]
        prediction_layer = (
            block_predictions[prediction_columns]
            if not block_predictions.empty
            else pd.DataFrame(columns=prediction_columns)
        )
        mapped_faces = nearby_faces.merge(
            prediction_layer,
            left_on="block_id",
            right_on="BLOCK_ID",
            how="left",
        )

        if {"street_name", "fm_addr_no", "to_addr_no"}.issubset(mapped_faces):
            official_label = (
                mapped_faces["street_name"].astype("string").str.title()
                + " "
                + mapped_faces["fm_addr_no"].astype("Int64").astype("string")
                + "-"
                + mapped_faces["to_addr_no"].astype("Int64").astype("string")
            )
        else:
            official_label = "Block " + mapped_faces["block_id"].astype(str)

        mapped_faces["map_block_name"] = (
            mapped_faces["STREET_BLOCK"].astype("string").fillna(official_label)
        )
        mapped_faces["availability_band"] = (
            mapped_faces["availability_band"]
            .astype("string")
            .fillna("No historical sensor prediction")
        )
        mapped_faces["availability_text"] = mapped_faces[
            "predicted_availability_percent"
        ].map(lambda value: f"{value:.1f}%" if pd.notna(value) else "Not available")
        mapped_faces["occupancy_text"] = mapped_faces[
            "predicted_occupancy_percent"
        ].map(lambda value: f"{value:.1f}%" if pd.notna(value) else "Not available")

        start_points = mapped_faces.assign(
            endpoint_order=1,
            latitude=mapped_faces["endpt1_latitude"],
            longitude=mapped_faces["endpt1_longitude"],
        )
        end_points = mapped_faces.assign(
            endpoint_order=2,
            latitude=mapped_faces["endpt2_latitude"],
            longitude=mapped_faces["endpt2_longitude"],
        )
        map_lines = pd.concat([start_points, end_points], ignore_index=True)
        map_lines.sort_values(["blockface_id", "endpoint_order"], inplace=True)

        figure = px.line_map(
            map_lines,
            lat="latitude",
            lon="longitude",
            color="availability_band",
            line_group="blockface_id",
            hover_name="map_block_name",
            hover_data={
                "availability_text": True,
                "occupancy_text": True,
                "distance_miles": ":.2f",
                "latitude": False,
                "longitude": False,
                "blockface_id": False,
                "availability_band": False,
            },
            labels={
                "availability_text": "Predicted availability",
                "occupancy_text": "Predicted occupancy",
                "distance_miles": "Distance (miles)",
                "availability_band": "Prediction",
            },
            color_discrete_map={
                "Higher availability (40%+)": "#16803C",
                "Moderate availability (20-39%)": "#D97706",
                "Limited availability (under 20%)": "#C2413B",
                "No historical sensor prediction": "#7A7A7A",
            },
            center={"lat": destination_latitude, "lon": destination_longitude},
            zoom=14,
            map_style="open-street-map",
        )
        for trace in figure.data:
            trace.line.width = (
                4 if trace.name == "No historical sensor prediction" else 8
            )

    circle_latitudes, circle_longitudes = radius_circle(
        destination_latitude, destination_longitude, radius_miles
    )
    figure.add_trace(
        go.Scattermap(
            lat=circle_latitudes,
            lon=circle_longitudes,
            mode="lines",
            line={"color": "#2563EB", "width": 2},
            name=f"Approx. {search_blocks}-block range",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scattermap(
            lat=[destination_latitude],
            lon=[destination_longitude],
            mode="markers",
            marker={"size": 16, "color": "#2563EB"},
            text=[destination["display_name"]],
            hovertemplate="Destination<br>%{text}<extra></extra>",
            name="Destination",
        )
    )
    figure.update_layout(
        height=600,
        map={
            "style": "open-street-map",
            "center": {"lat": destination_latitude, "lon": destination_longitude},
            "zoom": 14,
        },
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01},
    )
    return figure


def formatted_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Convert occupancy proportions into percentage-point display values."""
    display = metrics.reset_index().rename(columns={"Model": "model"}).copy()
    display["MAE"] *= 100
    display["RMSE"] *= 100
    display["Within 10 percentage points"] *= 100
    return display


def build_model_figures(results: dict[str, object]):
    """Create model-comparison, actual-vs-predicted, and importance figures."""
    metrics = formatted_metrics(results["metrics"])
    error_comparison = metrics.melt(
        id_vars="model",
        value_vars=["MAE", "RMSE"],
        var_name="metric",
        value_name="occupancy_point_error",
    )
    error_figure = px.bar(
        error_comparison,
        x="metric",
        y="occupancy_point_error",
        color="model",
        barmode="group",
        text_auto=".1f",
        title="Model comparison on later, unseen dates",
        labels={
            "metric": "Error metric (lower is better)",
            "occupancy_point_error": "Occupancy-percentage-point error",
            "model": "Model",
        },
    )

    prediction_sample = results["predictions"].copy()
    if len(prediction_sample) > 3_000:
        prediction_sample = prediction_sample.sample(3_000, random_state=42)
    prediction_sample["Actual occupancy (%)"] = (
        100 * prediction_sample["actual_next_hour"]
    )
    prediction_sample["Predicted occupancy (%)"] = (
        100 * prediction_sample["predicted_next_hour"]
    )
    scatter_figure = px.scatter(
        prediction_sample,
        x="Actual occupancy (%)",
        y="Predicted occupancy (%)",
        opacity=0.35,
        title="Actual vs. predicted next-hour occupancy",
    )
    scatter_figure.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Perfect prediction",
            line={"color": "black", "dash": "dash"},
        )
    )
    scatter_figure.update_xaxes(range=[0, 100])
    scatter_figure.update_yaxes(range=[0, 100])

    importance = results["feature_importance"].head(12).sort_values("importance")
    importance_figure = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        title="Permutation feature importance",
        labels={"importance": "Increase in MAE when shuffled", "feature": "Feature"},
    )
    return metrics, error_figure, scatter_figure, importance_figure


# Load the model once when the server starts. If setup fails, retain the error so
# the webpage can show a useful message rather than terminating without context.
try:
    DATA, RESULTS = load_dashboard_assets()
    MODELING = DATA["modeling"]
    METRICS, ERROR_FIGURE, SCATTER_FIGURE, IMPORTANCE_FIGURE = (
        build_model_figures(RESULTS)
    )
    STARTUP_ERROR = None
except Exception as error:  # Displayed prominently in the layout below.
    DATA = RESULTS = MODELING = METRICS = None
    ERROR_FIGURE = SCATTER_FIGURE = IMPORTANCE_FIGURE = go.Figure()
    STARTUP_ERROR = str(error)


# ---------------------------------------------------------------------------
# 5. Dash HTML layout and DCC interactive components
# ---------------------------------------------------------------------------

app = Dash(__name__)
app.title = "SFpark Parking Finder"

CARD_STYLE = {
    "backgroundColor": "white",
    "border": "1px solid #E5E7EB",
    "borderRadius": "10px",
    "padding": "18px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
}

INPUT_STYLE = {"width": "100%", "minHeight": "38px"}


def metric_card(title: str, element_id: str):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "13px", "color": "#4B5563"}),
            html.Div(
                "Loading...",
                id=element_id,
                style={"fontSize": "23px", "fontWeight": "700", "marginTop": "6px"},
            ),
        ],
        style=CARD_STYLE,
    )


app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("SFpark Historical Parking Finder", style={"marginBottom": "5px"}),
                html.P(
                    "Predict the same block's parking occupancy one hour ahead using "
                    "historical SFpark sensor, time, weather, and event information.",
                    style={"color": "#4B5563", "marginTop": 0},
                ),
            ]
        ),
        html.Div(
            "Historical proof of concept: this dashboard uses 2011-2013 data and "
            "does not confirm that a parking space is currently open.",
            style={
                "backgroundColor": "#FFF7ED",
                "border": "1px solid #FDBA74",
                "borderRadius": "8px",
                "padding": "12px 16px",
                "marginBottom": "20px",
            },
        ),
        html.Div(
            f"Startup error: {STARTUP_ERROR}" if STARTUP_ERROR else "",
            id="startup-error",
            style={
                "display": "block" if STARTUP_ERROR else "none",
                "backgroundColor": "#FEE2E2",
                "color": "#991B1B",
                "padding": "12px",
                "marginBottom": "15px",
            },
        ),
        html.H2("1. Find predicted parking near a destination"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Destination", style={"fontWeight": "600"}),
                        dcc.Input(
                            id="destination-input",
                            type="text",
                            value="Union Square, San Francisco, CA",
                            placeholder="Enter an SF address or landmark",
                            style=INPUT_STYLE,
                        ),
                    ],
                    style={"gridColumn": "span 2"},
                ),
                html.Div(
                    [
                        html.Label("Arrival date", style={"fontWeight": "600"}),
                        dcc.DatePickerSingle(
                            id="arrival-date",
                            date=date.today().isoformat(),
                            display_format="MMM D, YYYY",
                            style=INPUT_STYLE,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Arrival time", style={"fontWeight": "600"}),
                        dcc.Dropdown(
                            id="arrival-hour",
                            options=[
                                {
                                    "label": datetime.combine(
                                        date.today(), time(hour=hour)
                                    ).strftime("%I:00 %p"),
                                    "value": hour,
                                }
                                for hour in range(24)
                            ],
                            value=12,
                            clearable=False,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Approximate search blocks", style={"fontWeight": "600"}),
                        dcc.Slider(
                            id="search-blocks",
                            min=3,
                            max=10,
                            step=1,
                            value=6,
                            marks={value: str(value) for value in range(3, 11)},
                        ),
                    ],
                    style={"gridColumn": "span 2"},
                ),
                html.Button(
                    "Predict nearby parking",
                    id="predict-button",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#2563EB",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "7px",
                        "padding": "11px 18px",
                        "fontWeight": "600",
                        "cursor": "pointer",
                    },
                ),
            ],
            style={
                **CARD_STYLE,
                "display": "grid",
                "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                "gap": "16px",
                "alignItems": "end",
            },
        ),
        dcc.Loading(
            [
                html.Div(id="prediction-message", style={"margin": "16px 0"}),
                dcc.Graph(id="parking-map", figure=empty_map()),
                html.H3("Recommended blocks"),
                dash_table.DataTable(
                    id="recommendation-table",
                    columns=[
                        {"name": "Rank", "id": "Rank"},
                        {"name": "Recommended block", "id": "Recommended block"},
                        {"name": "District", "id": "District"},
                        {
                            "name": "Predicted availability (%)",
                            "id": "Predicted availability (%)",
                            "type": "numeric",
                            "format": {"specifier": ".1f"},
                        },
                        {
                            "name": "Predicted occupancy (%)",
                            "id": "Predicted occupancy (%)",
                            "type": "numeric",
                            "format": {"specifier": ".1f"},
                        },
                        {
                            "name": "Distance (miles)",
                            "id": "Distance (miles)",
                            "type": "numeric",
                            "format": {"specifier": ".2f"},
                        },
                    ],
                    data=[],
                    sort_action="native",
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "10px", "textAlign": "left"},
                    style_header={"fontWeight": "700", "backgroundColor": "#F3F4F6"},
                ),
            ],
            type="circle",
        ),
        html.Hr(style={"margin": "35px 0"}),
        html.H2("2. How accurate is the model?"),
        html.Div(
            [
                metric_card("Average error (MAE) - lower is better", "mae-card"),
                metric_card("RMSE - lower is better", "rmse-card"),
                metric_card("R-squared - higher is better", "r2-card"),
                metric_card("Predictions within 10 points", "within-card"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                "gap": "14px",
            },
        ),
        html.Div(
            [
                dcc.Graph(id="model-error-chart", figure=ERROR_FIGURE),
                dcc.Graph(id="actual-predicted-chart", figure=SCATTER_FIGURE),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                "gap": "12px",
            },
        ),
        dcc.Graph(id="feature-importance-chart", figure=IMPORTANCE_FIGURE),
        html.P(
            "Map geometry: SFMTA/DataSF. Basemap and destination search: "
            "OpenStreetMap/Nominatim.",
            style={"fontSize": "13px", "color": "#6B7280", "marginTop": "25px"},
        ),
    ],
    style={
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "25px",
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#F9FAFB",
        "color": "#111827",
    },
)


# ---------------------------------------------------------------------------
# 6. Dash callbacks: connect interactive inputs to outputs
# ---------------------------------------------------------------------------

@app.callback(
    Output("mae-card", "children"),
    Output("rmse-card", "children"),
    Output("r2-card", "children"),
    Output("within-card", "children"),
    Input("startup-error", "children"),
)
def update_metric_cards(_):
    """Fill the four model metric cards after the model is prepared."""
    if STARTUP_ERROR or METRICS is None or RESULTS is None:
        return "Unavailable", "Unavailable", "Unavailable", "Unavailable"

    selected_name = RESULTS["selected_model_name"]
    row = METRICS.loc[METRICS["model"].eq(selected_name)].iloc[0]
    return (
        f"{row['MAE']:.1f} points",
        f"{row['RMSE']:.1f} points",
        f"{row['R2']:.3f}",
        f"{row['Within 10 percentage points']:.1f}%",
    )


@app.callback(
    Output("parking-map", "figure"),
    Output("recommendation-table", "data"),
    Output("prediction-message", "children"),
    Input("predict-button", "n_clicks"),
    State("destination-input", "value"),
    State("arrival-date", "date"),
    State("arrival-hour", "value"),
    State("search-blocks", "value"),
    prevent_initial_call=True,
)
def update_parking_prediction(
    n_clicks: int,
    destination_query: str,
    arrival_date: str,
    arrival_hour: int,
    search_blocks: int,
):
    """Run the geographic search and update the map, table, and message."""
    del n_clicks
    if STARTUP_ERROR or MODELING is None or RESULTS is None:
        return empty_map(), [], html.Div(
            f"The model is unavailable: {STARTUP_ERROR}",
            style={"color": "#991B1B", "fontWeight": "600"},
        )

    try:
        destination = geocode_destination(destination_query or "")
        radius_miles = int(search_blocks) * 0.08
        prediction_time = pd.Timestamp.combine(
            pd.Timestamp(arrival_date).date(), time(hour=int(arrival_hour))
        )

        blockfaces = load_blockfaces().copy()
        blockfaces["distance_miles"] = distances_in_miles(
            blockfaces["midpoint_latitude"],
            blockfaces["midpoint_longitude"],
            float(destination["latitude"]),
            float(destination["longitude"]),
        )
        nearby_faces = blockfaces.loc[
            blockfaces["distance_miles"].le(radius_miles)
        ].copy()
        nearby_block_ids = nearby_faces["block_id"].drop_duplicates()

        block_predictions, context = project.predict_nearby_blocks(
            modeling=MODELING,
            model_results=RESULTS,
            block_ids=nearby_block_ids,
            prediction_time=prediction_time,
        )
        figure = build_parking_map(
            nearby_faces,
            block_predictions,
            destination,
            radius_miles,
            int(search_blocks),
        )

        if nearby_faces.empty:
            return figure, [], html.Div(
                "No official metered blockfaces were found in this search range.",
                style={"color": "#92400E", "fontWeight": "600"},
            )
        if block_predictions.empty:
            return figure, [], html.Div(
                "Metered blocks were found, but none match the historical sensor data.",
                style={"color": "#92400E", "fontWeight": "600"},
            )

        block_distances = (
            nearby_faces.groupby("block_id", as_index=False)["distance_miles"].min()
        )
        ranked = block_predictions.merge(
            block_distances,
            left_on="BLOCK_ID",
            right_on="block_id",
            how="left",
        )
        ranked = ranked.sort_values(
            ["predicted_availability_percent", "distance_miles"],
            ascending=[False, True],
        ).head(10)
        ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
        ranked.rename(
            columns={
                "STREET_BLOCK": "Recommended block",
                "PM_DISTRICT_NAME": "District",
                "predicted_availability_percent": "Predicted availability (%)",
                "predicted_occupancy_percent": "Predicted occupancy (%)",
                "distance_miles": "Distance (miles)",
            },
            inplace=True,
        )
        table_columns = [
            "Rank",
            "Recommended block",
            "District",
            "Predicted availability (%)",
            "Predicted occupancy (%)",
            "Distance (miles)",
        ]
        table_data = ranked[table_columns].round(2).to_dict("records")
        message = html.Div(
            [
                html.Strong(f"Destination: {destination['display_name']}"),
                html.Br(),
                html.Span(
                    f"Predicted {len(block_predictions):,} historical blocks using "
                    f"{context['matching_rows']:,} observations with "
                    f"{context['reference_level']}."
                ),
            ],
            style={"backgroundColor": "#ECFDF5", "padding": "12px", "borderRadius": "8px"},
        )
        return figure, table_data, message

    except Exception as error:
        return empty_map(), [], html.Div(
            f"Prediction failed: {error}",
            style={"color": "#991B1B", "fontWeight": "600"},
        )


# ---------------------------------------------------------------------------
# 7. Start the local web server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)