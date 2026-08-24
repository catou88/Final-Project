"""Dashboard for the historical SFpark next-hour occupancy model.

Start it from the project environment with either command:

    python "Final Project.py" --dashboard
    streamlit run dashboard.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
from datetime import date, datetime, time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
# Use the refactored main module `final_project.py` instead of the old filename with spaces.
MODEL_FILE = PROJECT_DIR / "final_project.py"
DASHBOARD_ROW_LIMIT = 100_000


def _load_project_module():
    """Import the project file even though its filename contains a space."""
    specification = importlib.util.spec_from_file_location(
        "sfpark_final_project", MODEL_FILE
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not import {MODEL_FILE}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


project = _load_project_module()


@st.cache_resource(show_spinner="Loading historical sensor data and training the model...")
def load_dashboard_assets(source_size: int, source_mtime: float):
    """Load and fit once, then reuse the result across dashboard interactions."""
    del source_size, source_mtime  # Cache invalidation keys.
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        project_data = project.prepare_project_data(
            row_limit=DASHBOARD_ROW_LIMIT
        )
        model_results = project.train_and_evaluate_model(
            project_data["modeling"],
            max_rows=None,
            include_catboost=False,
        )
    return project_data, model_results


def _source_signature() -> tuple[int, float]:
    source = (
        project.PARKING_FILE
        if project.PARKING_FILE.exists()
        else project.PARKING_SAMPLE_FILE
    )
    if not source.exists():
        return 0, 0.0
    status = source.stat()
    return status.st_size, status.st_mtime


def _formatted_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    display = metrics.reset_index().rename(columns={"Model": "model"}).copy()
    display["MAE"] *= 100
    display["RMSE"] *= 100
    display["Within 10 percentage points"] *= 100
    return display


@st.cache_data(show_spinner=False)
def load_blockfaces(source_size: int, source_mtime: float) -> pd.DataFrame:
    """Load official SFMTA block geometry and calculate segment midpoints."""
    del source_size, source_mtime
    blockfaces = pd.read_csv(project.BLOCKFACE_FILE)
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


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def geocode_destination(destination: str) -> dict[str, object]:
    """Resolve one San Francisco destination through OpenStreetMap Nominatim."""
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
        headers={"User-Agent": "SFparkAcademicDashboard/1.0"},
    )
    with urlopen(request, timeout=15, context=project.SSL_CONTEXT) as response:
        matches = json.load(response)
    if not matches:
        raise ValueError(
            "That destination could not be found in San Francisco. Try a full "
            "street address or a well-known place."
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
    """Calculate great-circle distance from a destination to many points."""
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
    """Create a geographic circle used to show the approximate search range."""
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


st.set_page_config(
    page_title="SFpark Historical Parking Finder",
    page_icon="🅿️",
    layout="wide",
)

st.title("SFpark historical parking finder")
st.caption(
    "A proof-of-concept model that predicts parking occupancy one hour ahead "
    "using San Francisco sensor data from the 2011–2013 SFpark pilot."
)

st.header("Start here: what does the model predict?")
explanation_columns = st.columns(2)
with explanation_columns[0]:
    st.markdown(
        """
        #### Prediction

        The model predicts **the percentage of measured parking time that will
        be occupied on the same block one hour later**.

        Example: a prediction of **72% occupancy** means the model expects the
        block's measured spaces to be occupied about 72% of the next hour.
        Estimated availability would be approximately **28%**.
        """
    )
with explanation_columns[1]:
    st.markdown(
        """
        #### Intended use

        This dashboard demonstrates whether historical parking patterns can be
        predicted from current occupancy, location, time, rate, weather, and
        citywide event information.

        It does **not** show live parking because its sensors stopped reporting
        after the historical SFpark pilot.
        """
    )

st.warning(
    "Historical proof of concept: recommendations below describe what the old "
    "sensor data suggests for a similar weekday and hour. They do not confirm "
    "that a parking space is open today."
)

try:
    data, results = load_dashboard_assets(*_source_signature())
except ModuleNotFoundError as error:
    st.error(
        f"A dashboard dependency is missing: {error.name}. Install the project "
        "requirements, then start the dashboard again."
    )
    st.code("python -m pip install -r requirements.txt")
    st.stop()
except Exception as error:
    st.error(f"The dashboard could not prepare the model: {error}")
    st.stop()

modeling = data["modeling"]
metrics = _formatted_metrics(results["metrics"])
selected_model_name = results["selected_model_name"]
model_row = metrics.loc[metrics["model"].eq(selected_model_name)].iloc[0]
baseline_row = metrics.loc[metrics["model"].eq("Current-occupancy baseline")].iloc[0]
mae_improvement = 100 * (baseline_row["MAE"] - model_row["MAE"]) / baseline_row["MAE"]

st.header("1. Find predicted parking near a destination")
st.markdown(
    "Enter a San Francisco address or place and an arrival time. The map shows "
    "historical SFpark blocks within approximately six city blocks, colored by "
    "their predicted available share one hour later."
)

if "parking_destination" not in st.session_state:
    st.session_state.parking_destination = {
        "query": "Union Square, San Francisco, CA",
        "display_name": "Union Square, San Francisco, California",
        "latitude": 37.7879363,
        "longitude": -122.4075174,
    }

with st.form("destination_search"):
    search_columns = st.columns([3, 1.2, 1.2, 1.2])
    destination_query = search_columns[0].text_input(
        "Where do you want to go?",
        value=st.session_state.parking_destination["query"],
        placeholder="Example: 1 Dr Carlton B Goodlett Pl",
        help="Enter a San Francisco address, landmark, or place name.",
    )
    arrival_date = search_columns[1].date_input(
        "Arrival date",
        value=date.today(),
        help="The date determines the weekday pattern; the source is historical.",
    )
    arrival_hour = search_columns[2].selectbox(
        "Arrival time",
        options=list(range(24)),
        index=12,
        format_func=lambda hour: datetime.combine(
            date.today(), time(hour=hour)
        ).strftime("%I:00 %p"),
    )
    search_blocks = search_columns[3].slider(
        "Approx. blocks",
        min_value=3,
        max_value=10,
        value=6,
        help="One block is approximated as 0.08 mile; this is not a walking-route distance.",
    )
    destination_submitted = st.form_submit_button(
        "Predict nearby parking", type="primary"
    )

if destination_submitted:
    try:
        with st.spinner("Finding the destination..."):
            resolved_destination = geocode_destination(destination_query)
        st.session_state.parking_destination = {
            "query": destination_query.strip(),
            **resolved_destination,
        }
    except Exception as error:
        st.error(f"Destination search failed: {error}")

destination = st.session_state.parking_destination
radius_miles = search_blocks * 0.08
prediction_time = pd.Timestamp.combine(arrival_date, time(hour=arrival_hour))

try:
    blockface_status = project.BLOCKFACE_FILE.stat()
    blockfaces = load_blockfaces(
        blockface_status.st_size,
        blockface_status.st_mtime,
    ).copy()
except FileNotFoundError:
    st.error(
        "The official SFMTA block-coordinate file is missing. Restore "
        f"{project.BLOCKFACE_FILE.name} and reload the dashboard."
    )
    st.stop()

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
block_predictions, prediction_context = project.predict_nearby_blocks(
    modeling=modeling,
    model_results=results,
    block_ids=nearby_block_ids,
    prediction_time=prediction_time,
)

circle_latitudes, circle_longitudes = radius_circle(
    float(destination["latitude"]),
    float(destination["longitude"]),
    radius_miles,
)

if nearby_faces.empty:
    parking_map = go.Figure()
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
            ["Higher availability (40%+)", "Moderate availability (20–39%)"],
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
    official_block_label = (
        mapped_faces["street_name"].astype("string").str.title()
        + " "
        + mapped_faces["fm_addr_no"].astype("Int64").astype("string")
        + "–"
        + mapped_faces["to_addr_no"].astype("Int64").astype("string")
    )
    mapped_faces["map_block_name"] = (
        mapped_faces["STREET_BLOCK"].astype("string").fillna(official_block_label)
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
    map_lines = pd.concat([start_points, end_points], ignore_index=True).sort_values(
        ["blockface_id", "endpoint_order"]
    )
    parking_map = px.line_map(
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
            "distance_miles": "Distance from destination (mi)",
            "availability_band": "Prediction",
        },
        color_discrete_map={
            "Higher availability (40%+)": "#16803C",
            "Moderate availability (20–39%)": "#D97706",
            "Limited availability (under 20%)": "#C2413B",
            "No historical sensor prediction": "#7A7A7A",
        },
        category_orders={
            "availability_band": [
                "Higher availability (40%+)",
                "Moderate availability (20–39%)",
                "Limited availability (under 20%)",
                "No historical sensor prediction",
            ]
        },
        center={
            "lat": float(destination["latitude"]),
            "lon": float(destination["longitude"]),
        },
        zoom=14,
        map_style="open-street-map",
    )
    for map_trace in parking_map.data:
        map_trace.line.width = (
            4 if map_trace.name == "No historical sensor prediction" else 8
        )

parking_map.add_trace(
    go.Scattermap(
        lat=circle_latitudes,
        lon=circle_longitudes,
        mode="lines",
        line={"color": "#2563EB", "width": 2},
        name=f"Approx. {search_blocks}-block range",
        hoverinfo="skip",
    )
)
parking_map.add_trace(
    go.Scattermap(
        lat=[float(destination["latitude"])],
        lon=[float(destination["longitude"])],
        mode="markers",
        marker={"size": 16, "color": "#2563EB"},
        text=[destination["display_name"]],
        hovertemplate="Destination<br>%{text}<extra></extra>",
        name="Destination",
    )
)
parking_map.update_layout(
    height=620,
    map={
        "style": "open-street-map",
        "center": {
            "lat": float(destination["latitude"]),
            "lon": float(destination["longitude"]),
        },
        "zoom": 14,
    },
    margin={"l": 0, "r": 0, "t": 10, "b": 0},
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.01},
)
st.plotly_chart(parking_map, use_container_width=True)

if nearby_faces.empty:
    st.warning(
        "No official SFMTA metered blockfaces fall inside this search range. "
        "All streets are still visible on the OpenStreetMap basemap."
    )
elif block_predictions.empty:
    st.warning(
        f"The map shows {nearby_faces['blockface_id'].nunique():,} official metered "
        "blockfaces in gray, but none have matching historical SFpark sensor data "
        "for a prediction. All other streets remain visible on the basemap."
    )
else:
    st.caption(
        f"Showing {nearby_faces['blockface_id'].nunique():,} official metered "
        f"blockfaces within approximately {radius_miles:.2f} mile, including "
        f"{len(block_predictions):,} historically modeled blocks near "
        f"{destination['display_name']}. Gray blockfaces "
        f"have no historical prediction. Predictions use "
        f"{prediction_context['matching_rows']:,} historical observations with "
        f"{prediction_context['reference_level']}."
    )
    best_blocks = (
        block_predictions.sort_values(
            ["predicted_availability_percent", "distance_miles"],
            ascending=[False, True],
        )
        .head(10)
        .copy()
    )
    best_blocks.insert(0, "Rank", np.arange(1, len(best_blocks) + 1))
    best_blocks = best_blocks.rename(
        columns={
            "STREET_BLOCK": "Recommended block",
            "PM_DISTRICT_NAME": "District",
            "predicted_availability_percent": "Predicted availability (%)",
            "predicted_occupancy_percent": "Predicted occupancy (%)",
            "distance_miles": "Distance (miles)",
        }
    )
    st.dataframe(
        best_blocks[
            [
                "Rank",
                "Recommended block",
                "District",
                "Predicted availability (%)",
                "Predicted occupancy (%)",
                "Distance (miles)",
            ]
        ],
        column_config={
            "Predicted availability (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Predicted occupancy (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Distance (miles)": st.column_config.NumberColumn(format="%.2f"),
        },
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    "Map geometry: SFMTA/DataSF. Basemap and destination search: OpenStreetMap. "
    "Every street remains visible on the basemap; the overlay outlines official "
    "metered blockfaces. The radius is straight-line distance, not a walking route."
)

st.divider()
st.header("2. How accurate is the model?")
st.caption(
    f"The interactive website uses {DASHBOARD_ROW_LIMIT:,} parking records so it "
    f"starts quickly. It trained on {results['training_rows']:,} usable "
    f"earlier observations and tested on {results['testing_rows']:,} later, unseen "
    f"observations beginning {results['split_date']:%B %d, %Y}."
)

histogram_row = metrics.loc[
    metrics["model"].eq("Histogram gradient boosting")
].iloc[0]
catboost_rows = metrics.loc[metrics["model"].eq("CatBoost categorical model")]
if selected_model_name == "CatBoost categorical model":
    st.success(
        "CatBoost had the lowest average test error, so it is used for the "
        "street estimates below."
    )
elif not catboost_rows.empty:
    catboost_row = catboost_rows.iloc[0]
    st.info(
        f"CatBoost was tested, but it did not improve accuracy: its average "
        f"error was {catboost_row['MAE']:.2f} occupancy points versus "
        f"{histogram_row['MAE']:.2f} for histogram gradient boosting. "
        f"Therefore, **{selected_model_name}** remains the prediction model."
    )
else:
    st.info(
        "This larger run focuses on histogram gradient boosting because CatBoost "
        "was less accurate in the earlier controlled comparison and substantially "
        "slower as the training data grew."
    )

metric_columns = st.columns(5)
metric_columns[0].metric(
    "Average error (MAE) ↓",
    f"{model_row['MAE']:.1f} occupancy points",
    help="Average difference between predicted and actual occupancy. Lower is better.",
)
metric_columns[1].metric(
    "Large-error score (RMSE) ↓",
    f"{model_row['RMSE']:.1f} occupancy points",
    help="Penalizes large misses more heavily than MAE. Lower is better.",
)
metric_columns[2].metric(
    "Variation explained (R²) ↑",
    f"{model_row['R2']:.3f}",
    help="Share of occupancy variation explained by the model. Closer to 1 is better.",
)
metric_columns[3].metric(
    "Within 10 points ↑",
    f"{model_row['Within 10 percentage points']:.1f}%",
    help="Share of predictions within 10 occupancy percentage points of the actual result.",
)
metric_columns[4].metric(
    "MAE improvement ↑",
    f"{mae_improvement:.1f}%",
    "vs. assuming no change",
)

st.success(
    f"Plain-language result: the prediction is off by about {model_row['MAE']:.1f} "
    "occupancy percentage points on average. For example, if actual occupancy "
    f"is 70%, a typical error of this size is approximately "
    f"{70 - model_row['MAE']:.1f}% to {70 + model_row['MAE']:.1f}%."
)

error_comparison = metrics.melt(
    id_vars="model",
    value_vars=["MAE", "RMSE"],
    var_name="metric",
    value_name="occupancy_point_error",
)
error_comparison["metric"] = error_comparison["metric"].replace(
    {"MAE": "Average error (MAE)", "RMSE": "Large-error score (RMSE)"}
)
error_chart = px.bar(
    error_comparison,
    x="metric",
    y="occupancy_point_error",
    color="model",
    barmode="group",
    text_auto=".1f",
    labels={
        "metric": "Error measure — lower is better",
        "occupancy_point_error": "Occupancy-percentage-point error",
        "model": "Compared method",
    },
    title="Prediction error on later, unseen dates (lower is better)",
)
error_chart.update_layout(legend_title_text="Compared method")
st.plotly_chart(error_chart, use_container_width=True)

prediction_sample = results["predictions"].copy()
if len(prediction_sample) > 3_000:
    prediction_sample = prediction_sample.sample(3_000, random_state=42)
prediction_sample["Actual next-hour occupancy (%)"] = (
    100 * prediction_sample["actual_next_hour"]
)
prediction_sample["Predicted next-hour occupancy (%)"] = (
    100 * prediction_sample["predicted_next_hour"]
)
scatter_chart = px.scatter(
    prediction_sample,
    x="Actual next-hour occupancy (%)",
    y="Predicted next-hour occupancy (%)",
    opacity=0.35,
    title="Actual vs. predicted occupancy (points closer to the diagonal are better)",
)
scatter_chart.add_trace(
    go.Scatter(
        x=[0, 100],
        y=[0, 100],
        mode="lines",
        name="Perfect prediction",
        line={"color": "black", "dash": "dash"},
    )
)
scatter_chart.update_xaxes(range=[0, 100])
scatter_chart.update_yaxes(range=[0, 100])
st.plotly_chart(scatter_chart, use_container_width=True)

error_distribution = 100 * results["predictions"]["absolute_error"]
histogram = px.histogram(
    x=error_distribution,
    nbins=40,
    labels={"x": "Absolute prediction error (occupancy percentage points)"},
    title="Distribution of prediction errors (more observations near zero are better)",
)
histogram.update_layout(yaxis_title="Number of test predictions", showlegend=False)
st.plotly_chart(histogram, use_container_width=True)

with st.expander("Metric definitions"):
    st.markdown(
        """
        - **MAE:** average absolute difference between predicted and actual
          occupancy. Lower is better.
        - **RMSE:** error measure that gives extra weight to large misses. Lower
          is better.
        - **R²:** proportion of occupancy variation explained by the model.
          Closer to 1 is better.
        - **Within 10 points:** percentage of predictions no more than ten
          occupancy percentage points from the observed value. Higher is better.
        - **No-change baseline:** assumes next-hour occupancy will equal current
          occupancy. The trained model should outperform this simple rule.
        - **Histogram gradient boosting:** converts location categories into
          numeric codes and learns nonlinear changes from current occupancy.
        - **CatBoost categorical model:** handles block, district, and weekday
          categories directly. It is included as a fair experiment, but the
          dashboard uses whichever trained model has the lower test MAE.
        """
    )

with st.expander("What information influences the occupancy prediction?"):
    feature_names = {
        "hour": "Time of day",
        "occupancy": "Current occupancy",
        "TOTAL_OCCUPIED_TIME": "Current measured occupied time",
        "TOTAL_VACANT_TIME": "Current measured vacant time",
        "TOTAL_UNKNOWN_TIME": "Current unknown sensor time",
        "RATE": "Parking rate",
        "PM_DISTRICT_NAME": "Parking district",
        "BLOCK_ID": "Parking block",
        "day_of_week": "Day of week",
        "is_weekend": "Weekend indicator",
        "min_temp_f": "Minimum daily temperature",
        "max_temp_f": "Maximum daily temperature",
        "average_temp_f": "Average daily temperature",
        "precipitation_inches": "Daily precipitation",
        "is_raining": "Rain indicator",
        "event_count": "Number of citywide events",
        "has_event": "Citywide event indicator",
    }
    importance = results["feature_importance"].head(12).sort_values("importance").copy()
    importance["feature"] = importance["feature"].replace(feature_names)
    st.caption(
        "Longer bars mean shuffling that input caused a larger loss of test "
        "accuracy. Importance does not show whether the input raises or lowers occupancy."
    )
    importance_chart = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        labels={
            "importance": "Loss of accuracy when shuffled — larger means more important",
            "feature": "Model input",
        },
        title="Inputs that most affect next-hour occupancy accuracy",
    )
    st.plotly_chart(importance_chart, use_container_width=True)

st.divider()
st.header("Important limitations")
st.markdown(
    """
    - Sensor observations are from **2011–2013**, so they may not represent
      current rates, travel patterns, construction, or curb regulations.
    - The model covers only blocks represented in the historical SFpark data.
    - Events are citywide daily indicators because the event file has no coordinates.
    - Some observations contain abrupt occupancy changes that may reflect real
      turnover, sensor resets, or measurement problems.
    - This dashboard should be presented as an academic proof of concept, not a
      live navigation or parking-guarantee product.
    """
)
