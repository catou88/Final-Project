# %% Project overview
"""SFpark parking-availability project: download, review, and prepare data.

Run this file normally for a quick, presentation-friendly data tour:

    python "Final Project.py"

The first run downloads the small events and weather files plus a manageable
sample of the 1.48 GB parking file. To download the complete parking file:

    python "Final Project.py" --download-full

The code is organized as functions so teammates can also import it:

    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("parking_project", "Final Project.py")
    project = module_from_spec(spec)
    spec.loader.exec_module(project)
    data = project.prepare_project_data()
In VS Code, open this file and use **Run Cell** above each ``# %%`` marker.
The last cell contains simple notebook settings for the sample/full dataset.
"""

# %% Imports
from __future__ import annotations

import argparse
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# %% 1. Project settings


def _find_project_directory() -> Path:
    """Find this project folder in scripts and Jupyter-style sessions."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        working_directory = Path.cwd()
        nested_project = working_directory / "Final Project"
        return nested_project if nested_project.is_dir() else working_directory


PROJECT_DIR = _find_project_directory()
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PARKING_URL = (
    "https://safitwebapps.blob.core.windows.net/$web/streets/sfpark/"
    "SFpark_ParkingSensorData_HourlyOccupancy_20112013.csv"
)
EVENTS_URL = "https://www.sfmta.com/media/12019/download?inline="
WEATHER_URL = "https://www.sfmta.com/media/12021/download?inline="

PARKING_FILE = RAW_DIR / "parking_occupancy_2011_2013.csv"
PARKING_SAMPLE_FILE = RAW_DIR / "parking_occupancy_sample.csv"
EVENTS_FILE = RAW_DIR / "events.csv"
WEATHER_FILE = RAW_DIR / "weather.csv"
BLOCKFACE_FILE = RAW_DIR / "blockfaces_with_meters.csv"
MODELING_SAMPLE_FILE = PROCESSED_DIR / "modeling_sample.csv"
DASHBOARD_FILE = PROJECT_DIR / "dashboard.py"

# The complete parking CSV is about 1.48 GB. A byte-range sample lets everyone
# inspect and run the workflow quickly before committing to the full download.
SAMPLE_BYTES = 12 * 1024 * 1024
DEFAULT_ROW_LIMIT = 500_000


def _ssl_context() -> ssl.SSLContext:
    """Use Python's certificates, or macOS's system bundle when needed."""
    certificate_bundle = Path(certifi.where())
    if certificate_bundle.exists():
        return ssl.create_default_context(cafile=str(certificate_bundle))
    default_paths = ssl.get_default_verify_paths()
    if default_paths.cafile and Path(default_paths.cafile).exists():
        return ssl.create_default_context(cafile=default_paths.cafile)
    macos_bundle = Path("/etc/ssl/cert.pem")
    if macos_bundle.exists():
        return ssl.create_default_context(cafile=str(macos_bundle))
    return ssl.create_default_context()


SSL_CONTEXT = _ssl_context()


# %% 2. Download helpers

def _human_size(number_of_bytes: int) -> str:
    """Return a readable file size."""
    size = float(number_of_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} TB"


def download_file(url: str, destination: Path) -> Path:
    """Download a file once and reuse the local copy on later runs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing file: {destination.name} "
              f"({_human_size(destination.stat().st_size)})")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    downloaded = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "INDENG210-SFpark-project/1.0"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"

    print(f"Downloading {destination.name} ...")
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(
            request, timeout=60, context=SSL_CONTEXT
        ) as response:
            # If a server ignores the Range header, start over instead of
            # appending a second complete file to the partial download.
            is_resume = downloaded > 0 and response.status == 206
            mode = "ab" if is_resume else "wb"
            if not is_resume:
                downloaded = 0
            response_size = int(response.headers.get("Content-Length", 0))
            total_size = downloaded + response_size

            with temporary.open(mode) as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = 100 * downloaded / total_size
                        print(
                            f"\r  {_human_size(downloaded)} / "
                            f"{_human_size(total_size)} ({percent:5.1f}%)",
                            end="",
                            flush=True,
                        )
        print()
        temporary.replace(destination)
        return destination
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Could not download {destination.name}. Re-run the script to "
            "resume the download."
        ) from error


def download_parking_sample(destination: Path = PARKING_SAMPLE_FILE) -> Path:
    """Download the beginning of the parking CSV for a fast local demo."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing file: {destination.name} "
              f"({_human_size(destination.stat().st_size)})")
        return destination

    print(f"Downloading {_human_size(SAMPLE_BYTES)} parking sample ...")
    request = urllib.request.Request(
        PARKING_URL,
        headers={
            "Range": f"bytes=0-{SAMPLE_BYTES - 1}",
            "User-Agent": "INDENG210-SFpark-project/1.0",
        },
    )
    with urllib.request.urlopen(
        request, timeout=60, context=SSL_CONTEXT
    ) as response:
        sample = response.read(SAMPLE_BYTES)

    # The byte range usually ends halfway through a CSV row. Remove that row.
    final_newline = sample.rfind(b"\n")
    if final_newline <= 0:
        raise RuntimeError("The parking sample did not contain complete CSV rows.")
    destination.write_bytes(sample[: final_newline + 1])
    print(f"Saved {destination.name} ({_human_size(destination.stat().st_size)})")
    return destination


def _remove_trailing_dos_marker(path: Path) -> Path:
    """Remove the non-data Ctrl-Z byte found after the official CSV's last row."""
    if not path.exists() or path.stat().st_size == 0:
        return path
    with path.open("rb+") as source:
        source.seek(-1, 2)
        if source.read(1) == b"\x1a":
            source.truncate(source.tell() - 1)
            print(f"Removed trailing non-data Ctrl-Z marker from {path.name}.")
    return path


def ensure_data_files(download_full: bool = False) -> Path:
    """Download required source files and return the parking file to load."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    download_file(EVENTS_URL, EVENTS_FILE)
    download_file(WEATHER_URL, WEATHER_FILE)

    if download_full:
        return _remove_trailing_dos_marker(download_file(PARKING_URL, PARKING_FILE))
    if PARKING_FILE.exists() and PARKING_FILE.stat().st_size > 0:
        print("A complete parking file is already available; using it.")
        return _remove_trailing_dos_marker(PARKING_FILE)
    return download_parking_sample()


# %% 3. Load and clean the parking data

def load_parking(path: Path, row_limit: int | None = None) -> pd.DataFrame:
    """Load parking rows, calculate occupancy, and create the future target."""
    columns = [
        "BLOCK_ID",
        "STREET_NAME",
        "STREET_BLOCK",
        "AREA_TYPE",
        "PM_DISTRICT_NAME",
        "RATE",
        "START_TIME_DT",
        "TOTAL_OCCUPIED_TIME",
        "TOTAL_VACANT_TIME",
        "TOTAL_UNKNOWN_TIME",
    ]
    # Explicit compact dtypes keep the complete 7.9-million-row source usable
    # on an ordinary laptop. Repeated location labels are stored as categories
    # instead of millions of separate Python strings.
    parking = pd.read_csv(
        path,
        usecols=columns,
        nrows=row_limit,
        dtype={
            # Read as a nullable numeric type because the official file ends
            # with a DOS control marker after its final real data record.
            "BLOCK_ID": "float32",
            "STREET_NAME": "category",
            "STREET_BLOCK": "category",
            "AREA_TYPE": "category",
            "PM_DISTRICT_NAME": "category",
            "RATE": "float32",
            "TOTAL_OCCUPIED_TIME": "float32",
            "TOTAL_VACANT_TIME": "float32",
            "TOTAL_UNKNOWN_TIME": "float32",
        },
        low_memory=False,
    )
    parking["START_TIME_DT"] = pd.to_datetime(
        parking["START_TIME_DT"], format="%d-%b-%Y %H:%M:%S", errors="coerce"
    )
    parking.dropna(subset=["BLOCK_ID", "START_TIME_DT"], inplace=True)
    parking["BLOCK_ID"] = parking["BLOCK_ID"].astype("int32")

    known_time = (
        parking["TOTAL_OCCUPIED_TIME"] + parking["TOTAL_VACANT_TIME"]
    )
    parking["occupancy"] = np.where(
        known_time > 0,
        parking["TOTAL_OCCUPIED_TIME"] / known_time,
        np.nan,
    )
    parking["date"] = parking["START_TIME_DT"].dt.normalize()
    parking["hour"] = parking["START_TIME_DT"].dt.hour
    parking["day_of_week"] = parking["START_TIME_DT"].dt.day_name()
    parking["is_weekend"] = parking["START_TIME_DT"].dt.dayofweek >= 5

    # Create y = occupancy for the same block exactly one hour later.
    parking.sort_values(["BLOCK_ID", "START_TIME_DT"], inplace=True)
    grouped = parking.groupby("BLOCK_ID", sort=False)
    next_time = grouped["START_TIME_DT"].shift(-1)
    next_occupancy = grouped["occupancy"].shift(-1)
    is_next_hour = next_time.sub(parking["START_TIME_DT"]).eq(pd.Timedelta(hours=1))
    parking["occupancy_next_hour"] = next_occupancy.where(is_next_hour)

    parking.reset_index(drop=True, inplace=True)
    return parking


# %% 4. Load and clean the weather data
def load_weather(path: Path = WEATHER_FILE) -> pd.DataFrame:
    """Load daily San Francisco weather and give columns Python-friendly names."""
    weather = pd.read_csv(path)
    weather.columns = [
        "area_name",
        "date",
        "max_temp_f",
        "min_temp_f",
        "precipitation_inches",
    ]
    weather["date"] = pd.to_datetime(
        weather["date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    weather["area_name"] = weather["area_name"].str.strip()
    weather["average_temp_f"] = (
        weather["max_temp_f"] + weather["min_temp_f"]
    ) / 2
    weather["is_raining"] = weather["precipitation_inches"].fillna(0).gt(0)
    return weather


# %% 5. Load and clean the events data
def load_events(path: Path = EVENTS_FILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load events and expand their date ranges into daily citywide features.

    The official events CSV does not include event coordinates. Therefore these
    are citywide date-level indicators, not distance-to-nearby-event features.
    """
    events = pd.read_csv(path)
    events = events.rename(
        columns={
            "Event ID": "event_id",
            "Event Name": "event_name",
            "Event Description": "event_description",
            "Event Class": "event_class",
            "Effective From Date": "start_date",
            "Effective To Date": "end_date",
            "From Time": "start_time",
            "To Time": "end_time",
        }
    )
    events["start_date"] = pd.to_datetime(events["start_date"], errors="coerce")
    events["end_date"] = pd.to_datetime(events["end_date"], errors="coerce")
    events["end_date"] = events["end_date"].fillna(events["start_date"])

    expanded_rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        if pd.isna(event.start_date) or pd.isna(event.end_date):
            continue
        for date in pd.date_range(event.start_date, event.end_date, freq="D"):
            expanded_rows.append(
                {
                    "date": date,
                    "event_id": event.event_id,
                    "event_class": event.event_class,
                }
            )

    expanded = pd.DataFrame(expanded_rows)
    if expanded.empty:
        daily_events = pd.DataFrame(columns=["date", "event_count", "has_event"])
    else:
        daily_events = (
            expanded.groupby("date", as_index=False)
            .agg(
                event_count=("event_id", "nunique"),
                event_classes=(
                    "event_class",
                    lambda values: ", ".join(sorted(set(values.dropna().astype(str)))),
                ),
            )
        )
        daily_events["has_event"] = daily_events["event_count"].gt(0)

    return events, daily_events


# %% 6. Combine the datasets

def prepare_project_data(
    download_full: bool = False,
    row_limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Download, load, and merge the project data.

    Returns a dictionary so each table is easy to inspect in Python:
    data["parking"], data["weather"], data["events"],
    data["daily_events"], and data["modeling"].
    """
    parking_path = ensure_data_files(download_full=download_full)
    parking = load_parking(parking_path, row_limit=row_limit)
    weather = load_weather()
    events, daily_events = load_events()

    # Date-level joins are implemented as mappings so the full parking table is
    # enriched in place instead of being duplicated by two large merge calls.
    modeling = parking
    weather_by_date = weather.set_index("date")
    for column in [
        "max_temp_f",
        "min_temp_f",
        "precipitation_inches",
        "average_temp_f",
    ]:
        modeling[column] = modeling["date"].map(weather_by_date[column]).astype(
            "float32"
        )
    modeling["is_raining"] = modeling["date"].map(
        weather_by_date["is_raining"]
    ).eq(True)

    events_by_date = daily_events.set_index("date")
    modeling["event_count"] = (
        modeling["date"].map(events_by_date["event_count"]).fillna(0).astype("int16")
    )
    modeling["has_event"] = modeling["event_count"].gt(0)

    return {
        "parking": parking,
        "weather": weather,
        "events": events,
        "daily_events": daily_events,
        "modeling": modeling,
    }


# %% 7. Display and save a concise data tour
def show_data_tour(data: dict[str, pd.DataFrame]) -> None:
    """Print the main facts teammates need to understand the data pipeline."""
    parking = data["parking"]
    weather = data["weather"]
    events = data["events"]
    modeling = data["modeling"]

    print("\n" + "=" * 72)
    print("SFpark PARKING AVAILABILITY PROJECT - DATA TOUR")
    print("=" * 72)
    print("Prediction target: occupancy of the same block one hour later")
    print("Main unit of analysis: one block at one hour")
    print("Train/test plan: earliest 80% of dates vs. latest 20% of dates")

    print("\nDATASET SIZES")
    print(f"  Parking rows loaded: {len(parking):,}")
    print(f"  Weather days:        {len(weather):,}")
    print(f"  Original events:     {len(events):,}")
    print(f"  Modeling rows:       {len(modeling):,}")

    print("\nPARKING COVERAGE")
    print(f"  Dates: {parking['date'].min().date()} to {parking['date'].max().date()}")
    print(f"  Blocks: {parking['BLOCK_ID'].nunique():,}")
    print(f"  Districts: {parking['PM_DISTRICT_NAME'].nunique():,}")
    print(
        "  Rows with one-hour target: "
        f"{modeling['occupancy_next_hour'].notna().sum():,}"
    )

    print("\nMODEL-READY COLUMN EXAMPLES")
    columns_to_show = [
        "BLOCK_ID",
        "STREET_BLOCK",
        "START_TIME_DT",
        "occupancy",
        "RATE",
        "hour",
        "day_of_week",
        "average_temp_f",
        "precipitation_inches",
        "event_count",
        "occupancy_next_hour",
    ]
    print(modeling[columns_to_show].head(8).to_string(index=False))

    print("\nIMPORTANT DATA LIMITATION")
    print(
        "  The events source has dates and times but no coordinates, so the "
        "current event features are citywide rather than 'nearby event' features."
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    modeling.head(50_000).to_csv(MODELING_SAMPLE_FILE, index=False)
    print(f"\nSaved review table: {MODELING_SAMPLE_FILE}")
    print("=" * 72)


# %% 8. Machine-learning setup
# This is a regression problem: the target, occupancy_next_hour, is a number
# between 0 and 1 rather than a category. Gradient-boosted decision trees work
# well here because parking behavior is nonlinear (for example, the effect of
# hour varies by block) and the model can learn interactions automatically.
# Instead of relearning occupancy from scratch, the model predicts the CHANGE
# from current occupancy to next-hour occupancy. That lets it improve on the
# already-strong rule that next hour will resemble the current hour.
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
from models.hist_gradient import build_model as build_hist_gradient


TARGET_COLUMN = "occupancy_next_hour"

NUMERIC_FEATURES = [
    "occupancy",                 # Current occupancy: usually the strongest signal.
    "RATE",                      # Current hourly parking price.
    "hour",                      # Time-of-day demand pattern.
    "is_weekend",
    "TOTAL_OCCUPIED_TIME",
    "TOTAL_VACANT_TIME",
    "TOTAL_UNKNOWN_TIME",
    "max_temp_f",
    "min_temp_f",
    "precipitation_inches",
    "average_temp_f",
    "is_raining",
    "event_count",
    "has_event",
]

CATEGORICAL_FEATURES = [
    "BLOCK_ID",                 # Different blocks have different typical demand.
    "AREA_TYPE",
    "PM_DISTRICT_NAME",
    "day_of_week",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _prepare_catboost_features(features: pd.DataFrame) -> pd.DataFrame:
    """Preserve category meaning and make missing categories safe for CatBoost."""
    prepared = features[MODEL_FEATURES].copy()
    for column in NUMERIC_FEATURES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        prepared[column] = (
            prepared[column].astype("string").fillna("Missing").astype(str)
        )
    return prepared


# %% 9. Build a chronological train/test split
def make_time_split(
    modeling: pd.DataFrame,
    train_fraction: float = 0.80,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Timestamp]:
    """Create an 80/20 past-to-future split for honest model evaluation.

    Rows without a next-hour target cannot teach or test the model, so they are
    removed. ``max_rows`` remains available for quick experiments, but the
    default uses every observation with a valid next-hour target.
    """
    required_columns = MODEL_FEATURES + [TARGET_COLUMN, "date"]
    missing_columns = [
        column for column in required_columns if column not in modeling.columns
    ]
    if missing_columns:
        raise ValueError(f"Modeling data is missing columns: {missing_columns}")

    valid_rows = modeling[TARGET_COLUMN].notna() & modeling["date"].notna()
    model_data = modeling.loc[valid_rows, required_columns].copy()

    if max_rows is not None and len(model_data) > max_rows:
        model_data = model_data.sample(n=max_rows, random_state=42)

    unique_dates = np.sort(model_data["date"].unique())
    if len(unique_dates) < 2:
        raise ValueError("At least two different dates are needed to train and test.")

    split_position = int(len(unique_dates) * train_fraction)
    split_position = min(max(split_position, 1), len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_position])

    train = model_data[model_data["date"] < split_date]
    test = model_data[model_data["date"] >= split_date]

    X_train = train[MODEL_FEATURES]
    y_train = train[TARGET_COLUMN]
    X_test = test[MODEL_FEATURES]
    y_test = test[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test, split_date


def make_time_train_val_test_split(
    modeling: pd.DataFrame,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Timestamp, pd.Timestamp]:
    """Create chronological train / validation / test splits.

    Default splits: 60% train, 20% validation, 20% test by unique dates.
    Returns X_train, X_val, X_test, y_train, y_val, y_test, val_split_date, test_split_date
    """
    required_columns = MODEL_FEATURES + [TARGET_COLUMN, "date"]
    missing_columns = [
        column for column in required_columns if column not in modeling.columns
    ]
    if missing_columns:
        raise ValueError(f"Modeling data is missing columns: {missing_columns}")

    valid_rows = modeling[TARGET_COLUMN].notna() & modeling["date"].notna()
    model_data = modeling.loc[valid_rows, required_columns].copy()
    if max_rows is not None and len(model_data) > max_rows:
        model_data = model_data.sample(n=max_rows, random_state=42)

    unique_dates = np.sort(model_data["date"].unique())
    if len(unique_dates) < 3:
        raise ValueError("At least three different dates are needed for train/val/test split.")

    n = len(unique_dates)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    train_end = min(max(train_end, 1), n - 2)
    val_end = min(max(val_end, train_end + 1), n - 1)

    val_split_date = pd.Timestamp(unique_dates[train_end])
    test_split_date = pd.Timestamp(unique_dates[val_end])

    train = model_data[model_data["date"] < val_split_date]
    val = model_data[(model_data["date"] >= val_split_date) & (model_data["date"] < test_split_date)]
    test = model_data[model_data["date"] >= test_split_date]

    X_train = train[MODEL_FEATURES]
    y_train = train[TARGET_COLUMN]
    X_val = val[MODEL_FEATURES]
    y_val = val[TARGET_COLUMN]
    X_test = test[MODEL_FEATURES]
    y_test = test[TARGET_COLUMN]
    return X_train, X_val, X_test, y_train, y_val, y_test, val_split_date, test_split_date


# %% 10. Train the model and report accuracy metrics
def _regression_metrics(
    actual: pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calculate complementary measures of regression accuracy."""
    errors = np.asarray(actual) - np.asarray(predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": r2_score(actual, predicted),
        "Within 10 percentage points": float(np.mean(np.abs(errors) <= 0.10)),
    }


def train_and_evaluate_model(
    modeling: pd.DataFrame,
    max_rows: int | None = None,
    include_catboost: bool | None = None,
) -> dict[str, object]:
    """Fit boosted trees to occupancy change and compare with a baseline.

    MAE is the average absolute occupancy error. RMSE penalizes large misses
    more heavily. R-squared measures improvement over predicting the test-set
    mean (1 is ideal; 0 matches that naive mean prediction). The final metric
    reports the share of predictions within 0.10 occupancy, or 10 percentage
    points, of the observed next-hour occupancy.
    """
    X_train, X_test, y_train, y_test, split_date = make_time_split(
        modeling, max_rows=max_rows
    )
    if include_catboost is None:
        # The controlled sample already showed CatBoost was less accurate. Its
        # 500-pass full-data fit is prohibitively slow on an 8 GB laptop, so the
        # automatic full run focuses on the winning histogram model.
        include_catboost = len(X_train) <= 250_000

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal_encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    # Build histogram-gradient pipeline using the central model builder
    histogram_model = build_hist_gradient(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    # A useful model should beat this simple rule: next hour equals this hour.
    baseline_train = X_train["occupancy"].fillna(y_train.median()).clip(0, 1)
    baseline_predictions = X_test["occupancy"].fillna(y_train.median()).clip(0, 1)
    occupancy_change = y_train - baseline_train
    histogram_model.fit(X_train, occupancy_change)

    histogram_change = histogram_model.predict(X_test)
    histogram_predictions = np.clip(
        baseline_predictions + histogram_change, 0, 1
    )

    model_predictions = {
        "Histogram gradient boosting": (
            histogram_model,
            histogram_change,
            histogram_predictions,
        ),
    }
    if include_catboost:
        catboost_model = Pipeline(
            steps=[
                (
                    "prepare_features",
                    FunctionTransformer(
                        _prepare_catboost_features,
                        validate=False,
                        feature_names_out="one-to-one",
                    ),
                ),
                (
                    "model",
                    CatBoostRegressor(
                        cat_features=CATEGORICAL_FEATURES,
                        loss_function="MAE",
                        iterations=500,
                        depth=8,
                        learning_rate=0.05,
                        l2_leaf_reg=5.0,
                        random_seed=42,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
        catboost_model.fit(X_train, occupancy_change)
        catboost_change = catboost_model.predict(X_test)
        catboost_predictions = np.clip(
            baseline_predictions + catboost_change, 0, 1
        )
        model_predictions["CatBoost categorical model"] = (
            catboost_model,
            catboost_change,
            catboost_predictions,
        )
    metric_rows = [
        {
            "Model": "Current-occupancy baseline",
            **_regression_metrics(y_test, baseline_predictions),
        }
    ]
    for model_name, (_, _, candidate_predictions) in model_predictions.items():
        metric_rows.append(
            {
                "Model": model_name,
                **_regression_metrics(y_test, candidate_predictions),
            }
        )
    metrics = pd.DataFrame(metric_rows).set_index("Model")

    selected_model_name = metrics.drop(
        index="Current-occupancy baseline"
    )["MAE"].idxmin()
    model, predicted_change, predictions = model_predictions[selected_model_name]

    # Permutation importance measures how much adjustment accuracy worsens when
    # one input is shuffled. Limit the calculation so it stays quick in Jupyter.
    importance_size = min(3_000, len(X_test))
    importance_positions = np.linspace(
        0, len(X_test) - 1, importance_size, dtype=int
    )
    importance_X = X_test.iloc[importance_positions]
    importance_target = (
        y_test - baseline_predictions
    ).iloc[importance_positions]
    importance_result = permutation_importance(
        model,
        importance_X,
        importance_target,
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=42,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance": importance_result.importances_mean,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    prediction_results = X_test.copy()
    prediction_results["actual_next_hour"] = y_test
    prediction_results["baseline_next_hour"] = baseline_predictions
    prediction_results["predicted_change"] = predicted_change
    prediction_results["predicted_next_hour"] = predictions
    prediction_results["absolute_error"] = np.abs(y_test - predictions)

    print("\n" + "=" * 72)
    print("NEXT-HOUR OCCUPANCY MODEL")
    print("=" * 72)
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")
    print(f"Test period begins: {split_date.date()}")
    print("\nACCURACY METRICS")
    print(metrics.round(4).to_string())
    if not include_catboost:
        print("\nCatBoost skipped for the full-data run (sample comparison was weaker).")
    print(f"\nSELECTED MODEL: {selected_model_name} (lowest test MAE)")
    print("\nFEATURE IMPORTANCE FOR THE PREDICTED CHANGE")
    print(feature_importance.head(15).to_string(index=False))
    print("=" * 72)

    return {
        "model": model,
        "selected_model_name": selected_model_name,
        "catboost_evaluated": include_catboost,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "predictions": prediction_results,
        "split_date": split_date,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
    }


def compare_models(modeling: pd.DataFrame, max_rows: int | None = None) -> dict:
    """Train a set of candidate regressors and return comparative metrics.

    Trains each candidate to predict the *change* from current occupancy
    (matching the project's main approach), then reports MAE/RMSE/R2
    for their final clipped next-hour predictions.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, val_split_date, test_split_date = make_time_train_val_test_split(
        modeling, max_rows=max_rows
    )
    # Baseline (persistence) as in `train_and_evaluate_model`
    baseline_train = X_train["occupancy"].fillna(y_train.median()).clip(0, 1)
    baseline_val = X_val["occupancy"].fillna(y_train.median()).clip(0, 1)
    baseline_test = X_test["occupancy"].fillna(y_train.median()).clip(0, 1)
    occupancy_change = y_train - baseline_train

    # Helper to build a lightweight preprocessing pipeline for non-tree models
    def _build_pipeline_for(estimator):
        num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        prep = ColumnTransformer(transformers=[("numeric", num_pipe, NUMERIC_FEATURES), ("categorical", cat_pipe, CATEGORICAL_FEATURES)])
        return Pipeline(steps=[("preprocessing", prep), ("model", estimator)])

    # Keep a small candidate set: linear, histogram GB, and optional CatBoost
    candidates = {
        "Linear regression": _build_pipeline_for(LinearRegression()),
        "Histogram gradient boosting": build_hist_gradient(NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    }

    # Optionally include CatBoost when training size is manageable
    include_catboost = len(X_train) <= 250_000
    if include_catboost:
        catboost_model = Pipeline(
            steps=[
                (
                    "prepare_features",
                    FunctionTransformer(
                        _prepare_catboost_features,
                        validate=False,
                        feature_names_out="one-to-one",
                    ),
                ),
                (
                    "model",
                    CatBoostRegressor(
                        cat_features=CATEGORICAL_FEATURES,
                        loss_function="MAE",
                        iterations=500,
                        depth=8,
                        learning_rate=0.05,
                        l2_leaf_reg=5.0,
                        random_seed=42,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
        candidates["CatBoost categorical model"] = catboost_model

    metric_rows = [
        {"Model": "Current-occupancy baseline", **_regression_metrics(y_test, baseline_test)}
    ]
    predictions_dict: dict[str, pd.DataFrame] = {}

    for name, pipeline in candidates.items():
        # Fit to predict the occupancy change and tune on validation set (no hyperparam search here)
        pipeline.fit(X_train, occupancy_change)
        # Validation predictions (for informational purposes)
        val_change = pipeline.predict(X_val)
        val_preds = np.clip(baseline_val + val_change, 0, 1)
        # Test predictions (final reported numbers)
        test_change = pipeline.predict(X_test)
        preds = np.clip(baseline_test + test_change, 0, 1)
        metric_rows.append({"Model": name, **_regression_metrics(y_test, preds)})

        df = X_test.copy()
        df["actual_next_hour"] = y_test
        df["baseline_next_hour"] = baseline_test
        df["predicted_next_hour"] = preds
        df["absolute_error"] = np.abs(y_test - preds)
        predictions_dict[name] = df

    metrics = pd.DataFrame(metric_rows).set_index("Model")

    return {
        "metrics": metrics,
        "predictions": predictions_dict,
        "val_split_date": val_split_date,
        "test_split_date": test_split_date,
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "testing_rows": len(X_test),
    }


def save_model(model: object, path: Path) -> None:
    """Persist a fitted model pipeline to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> object:
    """Load a persisted model pipeline from disk."""
    return joblib.load(path)


# %% 11. Rank streets by predicted next-hour availability
def predict_nearby_blocks(
    modeling: pd.DataFrame,
    model_results: dict[str, object],
    block_ids: list[int] | np.ndarray | pd.Series,
    prediction_time: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Predict next-hour availability for specified historical SFpark blocks."""
    requested_blocks = pd.Index(block_ids).dropna().astype("int64").unique()
    matching = modeling.loc[modeling["BLOCK_ID"].isin(requested_blocks)].copy()
    if matching.empty:
        return pd.DataFrame(), {
            "reference_level": "no matching historical blocks",
            "matching_rows": 0,
            "matching_blocks": 0,
        }

    prediction_time = pd.Timestamp(prediction_time)
    requested_hour = prediction_time.hour
    requested_day = prediction_time.day_name()
    comparable = matching[
        matching["hour"].eq(requested_hour)
        & matching["day_of_week"].eq(requested_day)
    ]
    reference_level = "same weekday and hour"
    if comparable.empty:
        comparable = matching[matching["hour"].eq(requested_hour)]
        reference_level = "same hour"
    if comparable.empty:
        comparable = matching
        reference_level = "all matching historical observations"

    numeric_columns = [
        "occupancy",
        "RATE",
        "TOTAL_OCCUPIED_TIME",
        "TOTAL_VACANT_TIME",
        "TOTAL_UNKNOWN_TIME",
        "max_temp_f",
        "min_temp_f",
        "precipitation_inches",
        "average_temp_f",
        "event_count",
    ]
    static_columns = [
        "STREET_NAME",
        "STREET_BLOCK",
        "AREA_TYPE",
        "PM_DISTRICT_NAME",
    ]
    aggregation = {column: "median" for column in numeric_columns}
    aggregation.update({column: "last" for column in static_columns})
    candidates = (
        comparable.sort_values("START_TIME_DT")
        .groupby("BLOCK_ID", as_index=False)
        .agg(aggregation)
    )
    candidates["hour"] = requested_hour
    candidates["day_of_week"] = requested_day
    candidates["is_weekend"] = prediction_time.dayofweek >= 5
    candidates["is_raining"] = candidates["precipitation_inches"].fillna(0).gt(0)
    candidates["has_event"] = candidates["event_count"].fillna(0).gt(0)

    fitted_model = model_results["model"]
    predicted_change = fitted_model.predict(candidates[MODEL_FEATURES])
    candidates["predicted_occupancy"] = np.clip(
        candidates["occupancy"].fillna(0.5).to_numpy() + predicted_change,
        0,
        1,
    )
    candidates["predicted_occupancy_percent"] = (
        100 * candidates["predicted_occupancy"]
    )
    candidates["predicted_availability_percent"] = (
        100 - candidates["predicted_occupancy_percent"]
    )
    return candidates, {
        "reference_level": reference_level,
        "matching_rows": len(comparable),
        "matching_blocks": candidates["BLOCK_ID"].nunique(),
        "requested_time": prediction_time,
    }


def predict_available_streets(
    modeling: pd.DataFrame,
    model_results: dict[str, object],
    location: str,
    prediction_time: pd.Timestamp,
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Estimate which streets in a location should have the most availability.

    ``location`` can match a parking district, street name, or street block.
    Because this project contains historical rather than live sensor data, each
    block's inputs are estimated from comparable historical observations at the
    requested hour and weekday. The result is a demonstration forecast, not a
    claim about current real-world parking conditions.
    """
    cleaned_location = location.strip()
    if not cleaned_location:
        raise ValueError("Enter a district, street, or street block.")

    searchable_columns = [
        "PM_DISTRICT_NAME",
        "STREET_NAME",
        "STREET_BLOCK",
    ]
    location_mask = pd.Series(False, index=modeling.index)
    for column in searchable_columns:
        location_mask |= modeling[column].astype("string").fillna("").str.contains(
            cleaned_location, case=False, regex=False
        )
    matching = modeling.loc[location_mask].copy()
    if matching.empty:
        return pd.DataFrame(), {
            "location": cleaned_location,
            "reference_level": "no match",
            "matching_rows": 0,
            "matching_blocks": 0,
        }

    prediction_time = pd.Timestamp(prediction_time)
    requested_hour = prediction_time.hour
    requested_day = prediction_time.day_name()

    comparable = matching[
        matching["hour"].eq(requested_hour)
        & matching["day_of_week"].eq(requested_day)
    ]
    reference_level = "same weekday and hour"
    if comparable.empty:
        comparable = matching[matching["hour"].eq(requested_hour)]
        reference_level = "same hour"
    if comparable.empty:
        comparable = matching
        reference_level = "all matching historical observations"

    numeric_columns = [
        "occupancy",
        "RATE",
        "TOTAL_OCCUPIED_TIME",
        "TOTAL_VACANT_TIME",
        "TOTAL_UNKNOWN_TIME",
        "max_temp_f",
        "min_temp_f",
        "precipitation_inches",
        "average_temp_f",
        "event_count",
    ]
    static_columns = [
        "STREET_NAME",
        "STREET_BLOCK",
        "AREA_TYPE",
        "PM_DISTRICT_NAME",
    ]
    aggregation = {column: "median" for column in numeric_columns}
    aggregation.update({column: "last" for column in static_columns})
    candidates = comparable.sort_values("START_TIME_DT").groupby(
        "BLOCK_ID", as_index=False
    ).agg(aggregation)

    candidates["hour"] = requested_hour
    candidates["day_of_week"] = requested_day
    candidates["is_weekend"] = prediction_time.dayofweek >= 5
    candidates["is_raining"] = candidates["precipitation_inches"].fillna(0).gt(0)
    candidates["has_event"] = candidates["event_count"].fillna(0).gt(0)

    fitted_model = model_results["model"]
    predicted_change = fitted_model.predict(candidates[MODEL_FEATURES])
    predicted_occupancy = np.clip(
        candidates["occupancy"].fillna(0.5).to_numpy() + predicted_change,
        0,
        1,
    )
    candidates["predicted_occupancy"] = predicted_occupancy
    candidates["predicted_availability_percent"] = (
        100 * (1 - candidates["predicted_occupancy"])
    )

    street_ranking = (
        candidates.groupby(
            ["PM_DISTRICT_NAME", "STREET_NAME"],
            as_index=False,
            observed=True,
        )
        .agg(
            predicted_availability_percent=(
                "predicted_availability_percent",
                "mean",
            ),
            predicted_occupancy_percent=("predicted_occupancy", "mean"),
            blocks_evaluated=("BLOCK_ID", "nunique"),
            example_block=("STREET_BLOCK", "first"),
        )
        .sort_values(
            ["predicted_availability_percent", "STREET_NAME"],
            ascending=[False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )
    street_ranking["predicted_occupancy_percent"] *= 100
    street_ranking.insert(0, "rank", np.arange(1, len(street_ranking) + 1))

    context = {
        "location": cleaned_location,
        "reference_level": reference_level,
        "matching_rows": len(comparable),
        "matching_blocks": candidates["BLOCK_ID"].nunique(),
        "requested_time": prediction_time,
    }
    return street_ranking, context


# %% 12. Command-line support
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-full",
        action="store_true",
        help="Download the complete 1.48 GB parking file instead of the sample.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROW_LIMIT,
        help=(
            "Number of parking rows to load (default: 500,000). "
            "Pass 0 to request all rows."
        ),
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Open the interactive model-accuracy and street-ranking dashboard.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.dashboard:
        if not DASHBOARD_FILE.exists():
            print(f"Dashboard file not found: {DASHBOARD_FILE}", file=sys.stderr)
            return 1
        try:
            return subprocess.run(
                [sys.executable, "-m", "streamlit", "run", str(DASHBOARD_FILE)],
                check=False,
            ).returncode
        except KeyboardInterrupt:
            return 0
    try:
        project_data = prepare_project_data(
            download_full=args.download_full,
            row_limit=None if args.rows == 0 else args.rows,
        )
        show_data_tour(project_data)
        model_results = train_and_evaluate_model(project_data["modeling"])
    except Exception as error:
        print(f"\nProject setup failed: {error}", file=sys.stderr)
        return 1
    return 0


# %% 13. Run the project and model
# Change these settings when running this file cell-by-cell.
DOWNLOAD_FULL = False
ROW_LIMIT = DEFAULT_ROW_LIMIT
MODEL_ROW_LIMIT = None

if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        project_data = prepare_project_data(
            download_full=DOWNLOAD_FULL,
            row_limit=ROW_LIMIT,
        )
        show_data_tour(project_data)
        model_results = train_and_evaluate_model(
            project_data["modeling"],
            max_rows=MODEL_ROW_LIMIT,
        )
        # Useful notebook outputs:
        # model_results["metrics"]             -> MAE, RMSE, R-squared, within 10%
        # model_results["predictions"]         -> actual vs. predicted test rows
        # model_results["feature_importance"]  -> inputs that affect adjustments
        # model_results["model"]               -> fitted preprocessing/model pipeline
    else:
        raise SystemExit(main())
