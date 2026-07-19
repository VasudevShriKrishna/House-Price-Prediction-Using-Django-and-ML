import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from django.conf import settings
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_http_methods

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_PATH = Path(settings.BASE_DIR) / "108.csv"

HISTORY_PATH = Path(settings.BASE_DIR) / "predictions.txt"
TARGET_COLUMN = "Price"

NUMERIC_COLUMNS = [
    "Purchase Budget",
    "House Age",
    "Number of Bedrooms (BHK)",
    "Area's Resident Population",
]

CATEGORICAL_COLUMNS = [
    "City",
    "Furnishing Status",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

BHK_MINIMUM_BUDGET = {
    1.0: 1_500_000,
    1.5: 2_200_000,
    2.0: 2_800_000,
    2.5: 3_400_000,
    3.0: 4_000_000,
    3.5: 4_600_000,
    4.0: 5_200_000,
    4.5: 5_800_000,
    5.0: 6_900_000,
}
BHK_VALUES = [
    "1",
    "1.5",
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
        "form_values": form_values or _empty_form_values(),
    }

    if model_bundle is not None:
        context["cities"] = model_bundle["cities"]
        context["furnishing_statuses"] = model_bundle[
            "furnishing_statuses"
        ]

    if message is not None:
        context["result2"] = message
    return context

def _load_and_clean_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"108.csv was not found at: {DATASET_PATH}"
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
        & (data["House Age"] <= 15)
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
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X, y)

    cities = sorted(
        data["City"]
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
    return {
        "model": model,
        "cities": cities,
        "furnishing_statuses": furnishing_statuses,
        "training_rows": len(data),
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
    submitted_data = (request.POST
        if request.method == "POST"
        else request.GET
    )

    return {"J1": submitted_data.get("J1", "").strip(),
        "J2": submitted_data.get("J2", "").strip(),
        "J3": submitted_data.get("J3", "").strip(),
        "J4": submitted_data.get("J4", "").strip(),
        "J5": submitted_data.get("J5", "").strip(),
        "J6": submitted_data.get("J6", "").strip(),
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
    ) as error:
        context = _build_prediction_context(
            message=f"Model configuration error: {error}"
        )
    return render(request, "predict.html", context)

@require_http_methods(["GET", "POST"])
def result(request):
    form_values = _parse_form_values(request)
    try:
        model_bundle = _get_model_bundle()
    except (FileNotFoundError,
        ValueError,
        KeyError,
        OSError,
        pd.errors.ParserError,
    ) as error:
        return _render_prediction_error(
            request,
            model_bundle=None,
            form_values=form_values,
            message=f"Model configuration error: {error}",
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

    if budget <= 0:
        return _render_prediction_error(request, model_bundle=model_bundle,
            form_values=form_values,
            message="Purchase Budget must be greater than zero.",
        )

    if house_age < 0 or house_age > 15:
        return _render_prediction_error(request,
            model_bundle=model_bundle,
            form_values=form_values,
            message="House Age must be between 0 and 15 years.",
        )

    if population <= 0:
        return _render_prediction_error(
            request, model_bundle=model_bundle,
            form_values=form_values,
            message=(
                "Area's Resident Population must be greater than zero."
            ),
        )

    if bhk not in BHK_MINIMUM_BUDGET:
        return _render_prediction_error(
            request,
            model_bundle=model_bundle, form_values=form_values,
            message="Please select a valid BHK value.",
        )

    minimum_budget = BHK_MINIMUM_BUDGET[bhk]
    if budget < minimum_budget:
        return _render_prediction_error(request,
            model_bundle=model_bundle,
            form_values=form_values,
            message=(
                f"The minimum budget for {_display_number(bhk)} BHK "
                f"is ₹{_format_indian_number(minimum_budget)}; please re-enter Budget."
            ),
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

    except (ValueError, TypeError, IndexError) as error:
        return _render_prediction_error(
            request,
            model_bundle=model_bundle,
            form_values=form_values,
            message=f"Prediction error: {error}",
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

            for row in reversed(rows[-108:]):
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
