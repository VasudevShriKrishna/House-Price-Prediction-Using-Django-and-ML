import csv
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from django.conf import settings
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)
DATASET_PATH = Path(settings.BASE_DIR) / "108_model_ready.csv"
HISTORY_PATH = Path(settings.BASE_DIR) / "predictions.txt"
TARGET_COLUMN = "Price"

NUMERIC_COLUMNS = [
    "Purchase Budget",
    "House Age",
    "Number of Bedrooms (BHK)",
    "Area's Resident Population",
]

CATEGORICAL_COLUMNS = ["City",
    "Furnishing Status",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
BHK_VALUES = ["1","1.5",
    "2",
    "2.5",
    "3",
    "3.5",
    "4",
    "4.5",
    "5",
]

_MODEL_CACHE: dict[str, Any] = {
    "dataset_modified_time": None,
    "bundle": None,
}

_MODEL_LOCK = threading.Lock()
_HISTORY_LOCK = threading.Lock()
def _empty_form_values() -> dict[str, str]:
    return {
        "J1": "",
        "J2": "",
        "J3": "",
        "J4": "",
        "J5": "",
        "J6": "",
    }
    
def _format_indian_number(value: float | int | str) -> str:
    number = int(round(float(value)))
    sign = "-" if number < 0 else ""
    digits = str(abs(number))

    if len(digits) <= 3:
        return sign + digits

    last_three = digits[-3:]
    remaining = digits[:-3]
    pairs: list[str] = []
    while remaining:
        pairs.append(remaining[-2:])
        remaining = remaining[:-2]
    return sign + ",".join(reversed(pairs)) + "," + last_three

def _display_number(value: float | int | str) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"

def _get_timestamp() -> str:
    current_time = timezone.now()
    if timezone.is_aware(current_time):
        current_time = timezone.localtime(current_time)
    return current_time.strftime("%Y-%m-%d %H:%M:%S")

def _build_prediction_context(
    *,
    model_bundle: dict[str, Any] | None = None,
    form_values: dict[str, str] | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "cities": [],
        "furnishing_statuses": [],
        "bhk_values": BHK_VALUES,
        "input_ranges": {
            "budget_min": 1,
            "budget_max": "",
            "age_min": 0,
            "age_max": 9,
            "population_min": 1,
            "population_max": "",
        },
        "form_values": form_values or _empty_form_values(),
    }

    if model_bundle is not None:
        context["cities"] = model_bundle["cities"]
        context["furnishing_statuses"] = model_bundle[
            "furnishing_statuses"
        ]
        context["bhk_values"] = model_bundle["bhk_values"]
        context["input_ranges"] = model_bundle["input_ranges"]

    if message is not None:
        context["result2"] = message
    return context

def _load_and_clean_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"108_model_ready.csv was not found at: {DATASET_PATH}"
        )
    data = pd.read_csv(DATASET_PATH)
    data.columns = data.columns.astype(str).str.strip()

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required CSV columns are missing: "
            + ", ".join(missing_columns)
        )

    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    for column in CATEGORICAL_COLUMNS:
        data[column] = data[column].astype("string").str.strip()

    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data.drop_duplicates()
    data = data[
        (data["City"] != "")
        & (data["Furnishing Status"] != "")
        & (data["Purchase Budget"] > 0)
        & (data["House Age"] >= 0)
        & (data["Number of Bedrooms (BHK)"] > 0)
        & (data["Area's Resident Population"] > 0)
        & (data["Price"] > 0)
    ].copy()

    if len(data) < 2:
        raise ValueError(
            "108.csv must contain at least two valid records.")
    return data

def _train_model_bundle(data: pd.DataFrame) -> dict[str, Any]:
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()

    def create_model() -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), NUMERIC_COLUMNS),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    CATEGORICAL_COLUMNS,
                ),
            ],
            remainder="drop",
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        )

    # Measure accuracy on unseen rows before fitting the production model on
    # all valid data. This dataset follows a linear target equation, so a linear model preserves small budget changes and monotonic BHK ordering.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    evaluation_model = create_model()
    evaluation_model.fit(X_train, y_train)
    evaluation_predictions = evaluation_model.predict(X_test)
    model = create_model()
    model.fit(X, y)

    numeric_coefficients = model.named_steps["regressor"].coef_[
        : len(NUMERIC_COLUMNS)
    ]
    expected_directions = (1, -1, 1, 1)
    if any(
        coefficient * direction <= 0
        for coefficient, direction in zip(
            numeric_coefficients,
            expected_directions,
        )
    ):
        raise ValueError(
            "Training data does not preserve the expected directions for "
            "budget, age, BHK and population."
        )

    cities = sorted(data["City"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    furnishing_statuses = sorted(
        data["Furnishing Status"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    if not cities:
        raise ValueError("No valid cities were found in 108.csv.")

    if not furnishing_statuses:
        raise ValueError(
            "No valid furnishing statuses were found in 108.csv.")
    bhk_values = sorted(
        data["Number of Bedrooms (BHK)"].astype(float).unique().tolist()
    )
    input_ranges = {
        "budget_min": int(data["Purchase Budget"].min()),
        "budget_max": int(data["Purchase Budget"].max()),
        "age_min": float(data["House Age"].min()),
        "age_max": float(data["House Age"].max()),
        "population_min": int(data["Area's Resident Population"].min()),
        "population_max": int(data["Area's Resident Population"].max()),
    }

    return {
        "model": model,
        "cities": cities,
        "furnishing_statuses": furnishing_statuses,
        "bhk_values": [_display_number(value) for value in bhk_values],
        "bhk_numeric_values": set(bhk_values),
        "input_ranges": input_ranges,
        "training_rows": len(data),
        "metrics": {
            "mae": float(
                mean_absolute_error(y_test, evaluation_predictions)
            ),
            "r2": float(r2_score(y_test, evaluation_predictions)),
        },
    }

def _get_model_bundle() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"108.csv was not found at: {DATASET_PATH}")
    modified_time = DATASET_PATH.stat().st_mtime_ns

    cached_bundle = _MODEL_CACHE["bundle"]
    cached_modified_time = _MODEL_CACHE["dataset_modified_time"]

    if (cached_bundle is not None and cached_modified_time == modified_time):
        return cached_bundle

    with _MODEL_LOCK:
        cached_bundle = _MODEL_CACHE["bundle"]
        cached_modified_time = _MODEL_CACHE["dataset_modified_time"]

        if (cached_bundle is not None and cached_modified_time == modified_time):
            return cached_bundle

        data = _load_and_clean_dataset()
        bundle = _train_model_bundle(data)
        _MODEL_CACHE["dataset_modified_time"] = modified_time
        _MODEL_CACHE["bundle"] = bundle
        return bundle


def _save_prediction(*,
    budget: float,
    house_age: float,
    bhk: float,
    population: float,
    city: str,
    furnishing: str,
    predicted_price: int,
) -> None:
    row = [
        budget,
        house_age,
        bhk,
        population,
        city,
        furnishing,
        predicted_price,
        _get_timestamp(),
    ]

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with _HISTORY_LOCK:
        with HISTORY_PATH.open(
            mode="a",
            encoding="utf-8",
            newline="",
        ) as history_file:
            csv.writer(history_file).writerow(row)

def _parse_form_values(request) -> dict[str, str]:
    return {
        "J1": request.POST.get("J1", "").strip(),
        "J2": request.POST.get("J2", "").strip(),
        "J3": request.POST.get("J3", "").strip(),
        "J4": request.POST.get("J4", "").strip(),
        "J5": request.POST.get("J5", "").strip(),
        "J6": request.POST.get("J6", "").strip(),
    }

def _render_prediction_error(
    request,*,
    model_bundle: dict[str, Any] | None,
    form_values: dict[str, str],
    message: str,
):
    context = _build_prediction_context(model_bundle=model_bundle,
        form_values=form_values,
        message=message,
    )
    return render(request, "predict.html", context)

@require_GET
def home(request):
    return render(request, "home.html")

@require_GET
def predict(request):
    try:
        model_bundle = _get_model_bundle()
        context = _build_prediction_context(model_bundle=model_bundle,)

    except (FileNotFoundError,
        ValueError,
        KeyError,
        OSError,
        pd.errors.ParserError,
    ):
        logger.exception("Unable to load the house-price model")
        context = _build_prediction_context(
            message="The prediction service is temporarily unavailable."
        )
    return render(request, "predict.html", context)

@require_POST
def result(request):
    form_values = _parse_form_values(request)
    try:
        model_bundle = _get_model_bundle()
    except (FileNotFoundError,
        ValueError,
        KeyError,
        OSError,
        pd.errors.ParserError,
    ):
        logger.exception("Unable to load the house-price model")
        return _render_prediction_error(
            request,
            model_bundle=None,
            form_values=form_values,
            message="The prediction service is temporarily unavailable.",
        )

    if not any(form_values.values()):
        return _render_prediction_error(
            request,
            model_bundle=model_bundle,
            form_values=form_values,
            message="Please enter the property information.",
        )

    try:
        budget = float(form_values["J1"])
        house_age = float(form_values["J2"])
        bhk = float(form_values["J3"])
        population = float(form_values["J4"])

    except (TypeError, ValueError):
        return _render_prediction_error(
            request,
            model_bundle=model_bundle,
            form_values=form_values,
            message=(
                "Purchase Budget, House Age, BHK and Resident "
                "Population must contain valid numbers."
            ),
        )

    numeric_values = [budget, house_age, bhk, population]
    if not all(np.isfinite(value) for value in numeric_values):
        return _render_prediction_error(
            request,
            model_bundle=model_bundle,
            form_values=form_values,
            message="Please enter finite numeric values.",
        )
    city = form_values["J5"]
    furnishing = form_values["J6"]

    input_ranges = model_bundle["input_ranges"]

    if not (
        input_ranges["budget_min"]
        <= budget
        <= input_ranges["budget_max"]
    ):
        return _render_prediction_error(request, model_bundle=model_bundle,
            form_values=form_values,
            message=(
                "Purchase Budget must be between "
                f"₹{_format_indian_number(input_ranges['budget_min'])} and "
                f"₹{_format_indian_number(input_ranges['budget_max'])}."
            ),
        )

    if not (
        input_ranges["age_min"]
        <= house_age
        <= input_ranges["age_max"]
    ):
        return _render_prediction_error(request,
            model_bundle=model_bundle,
            form_values=form_values,
            message=(
                "House Age must be between "
                f"{_display_number(input_ranges['age_min'])} and "
                f"{_display_number(input_ranges['age_max'])} years."
            ),
        )

    if not (
        input_ranges["population_min"]
        <= population
        <= input_ranges["population_max"]
    ):
        return _render_prediction_error(
            request, model_bundle=model_bundle,
            form_values=form_values,
            message=(
                "Area's Resident Population must be between "
                f"{_format_indian_number(input_ranges['population_min'])} "
                "and "
                f"{_format_indian_number(input_ranges['population_max'])}."
            ),
        )

    if bhk not in model_bundle["bhk_numeric_values"]:
        return _render_prediction_error(
            request,
            model_bundle=model_bundle, form_values=form_values,
            message="Please select a valid BHK value.",
        )

    if city not in model_bundle["cities"]:
        return _render_prediction_error(request,
            model_bundle=model_bundle,
            form_values=form_values,
            message=(
                "Please select a city that is available in 108.csv."
            ),
        )

    if furnishing not in model_bundle["furnishing_statuses"]:
        return _render_prediction_error(request,
            model_bundle=model_bundle,
            form_values=form_values,
            message="Please select a valid furnishing status.",
        )

    input_data = pd.DataFrame(
        [
            {
                "Purchase Budget": budget,
                "House Age": house_age,
                "Number of Bedrooms (BHK)": bhk,
                "Area's Resident Population": population,
                "City": city,
                "Furnishing Status": furnishing,
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    try:
        prediction = model_bundle["model"].predict(input_data)
        predicted_price = float(prediction[0])

    except (ValueError, TypeError, IndexError):
        logger.exception("The house-price model could not score the request")
        return _render_prediction_error(
            request,
            model_bundle=model_bundle,
            form_values=form_values,
            message="The model could not process these property details.",
        )

    if not np.isfinite(predicted_price):
        return _render_prediction_error(
            request,
            model_bundle=model_bundle,
            form_values=form_values,
            message=(
                "The model could not produce a valid predicted price."
            ),
        )

    predicted_price_int = int(round(predicted_price))

    if predicted_price_int < 0:
        return _render_prediction_error(
            request,
            model_bundle=model_bundle,
            form_values=form_values,
            message=(
                "The entered information produced an invalid "
                "negative prediction."
            ),
        )

    history_warning = ""
    try:
        _save_prediction(
            budget=budget,
            house_age=house_age,
            bhk=bhk,
            population=population,
            city=city,
            furnishing=furnishing,
            predicted_price=predicted_price_int,
        )

    except OSError:
        history_warning = (
            " The prediction succeeded, but history could not be saved."
        )

    context = _build_prediction_context(
        model_bundle=model_bundle,
        form_values=form_values,
        message=(
            "The Predicted Price is "
            f"₹{_format_indian_number(predicted_price_int)}"
            f"{history_warning}"
        ),
    )

    context["predicted_price"] = predicted_price_int
    return render(request, "predict.html", context)

@require_GET
def prediction_history(request):
    history: list[dict[str, str]] = []
    if HISTORY_PATH.exists():
        try:
            with _HISTORY_LOCK:
                with HISTORY_PATH.open(
                    mode="r",
                    encoding="utf-8",
                    newline="",
                ) as history_file:
                    rows = list(csv.reader(history_file))

            for row in reversed(rows):
                if len(history) >= 108:
                    break
                if len(row) != 8:
                    continue
                (
                    budget,
                    house_age,
                    bhk,
                    population,
                    city,
                    furnishing,
                    predicted_price,
                    prediction_date,
                ) = row

                try:
                    display_date = datetime.strptime(
                        prediction_date,
                        "%Y-%m-%d %H:%M:%S",
                    ).strftime("%B %d, %Y at %I:%M %p")
                except ValueError:
                    display_date = prediction_date

                try:
                    display_budget = (
                        f"₹{_format_indian_number(budget)}"
                    )
                except (TypeError, ValueError):
                    display_budget = f"₹{budget}"

                try:
                    display_prediction = (
                        f"₹{_format_indian_number(predicted_price)}"
                    )
                except (TypeError, ValueError):
                    display_prediction = f"₹{predicted_price}"

                try:
                    display_age = _display_number(house_age)
                except (TypeError, ValueError):
                    display_age = house_age

                try:
                    display_bhk = _display_number(bhk)
                except (TypeError, ValueError):
                    display_bhk = bhk
                try:
                    display_population = _format_indian_number(
                        population
                    )
                except (TypeError, ValueError):
                    display_population = population

                history.append(
                    {
                        "budget": display_budget,
                        "age": f"{display_age} years",
                        "bedrooms": display_bhk,
                        "population": display_population,
                        "city": city,
                        "furnishing_status": furnishing,
                        "prediction": display_prediction,
                        "date": display_date,
                    }
                )

        except OSError:
            history = []

    return render(
        request,
        "history.html",
        {"history": history},
    )
