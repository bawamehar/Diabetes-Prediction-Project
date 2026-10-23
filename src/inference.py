"""Model loading, input validation and prediction, shared by the Streamlit app and the API.

Nothing in this module stores or logs the values a user submits.
"""
import logging
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from preprocess import UI_FEATURES, transform_user_input

logger = logging.getLogger(__name__)

# Resolved relative to this file so it works from any working directory.
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"

# Allowed values are exactly the values present in the training data
# (Data/data/diabetes.csv): two 0/1 flags, BMI between 12 and 98, and four
# whole-number category codes.
BINARY_FEATURES = ("HighBP", "HighChol")
CATEGORY_RANGES = {"GenHlth": (1, 5), "Age": (1, 13), "Income": (1, 8), "Education": (1, 6)}
BMI_RANGE = (12.0, 98.0)


class InputValidationError(ValueError):
    """Submitted values are missing, malformed or out of range. Messages are safe to show users."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


class PredictionError(RuntimeError):
    """A prediction could not be produced. The message is safe to show users."""


class ModelUnavailableError(PredictionError):
    """The trained model file could not be loaded."""


def _to_number(value):
    """Return value as a finite float, or None if it is missing or not a usable number."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_features(features) -> dict:
    """Check `features` against the training schema; return clean floats keyed by feature name."""
    if not isinstance(features, Mapping):
        raise InputValidationError(["No input was provided."])

    problems = []
    unexpected = sorted(str(key) for key in features if key not in UI_FEATURES)
    if unexpected:
        problems.append(f"Unexpected field(s): {', '.join(unexpected)}.")

    clean = {}
    for name in UI_FEATURES:
        number = _to_number(features.get(name))
        if number is None:
            problems.append(f"{name}: a value is required and it must be a number.")
            continue
        if name in BINARY_FEATURES:
            if number not in (0.0, 1.0):
                problems.append(f"{name}: must be 0 (No) or 1 (Yes).")
        elif name == "BMI":
            low, high = BMI_RANGE
            if not low <= number <= high:
                problems.append(f"BMI: must be between {low:g} and {high:g}.")
        else:
            low, high = CATEGORY_RANGES[name]
            if number != int(number) or not low <= number <= high:
                problems.append(f"{name}: must be a whole number from {low} to {high}.")
        clean[name] = number

    if problems:
        raise InputValidationError(problems)
    return clean


@lru_cache(maxsize=1)
def load_model():
    """Load the trained scaler + classifier pipeline once per process."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.exception("Could not load the prediction model")
        raise ModelUnavailableError("The prediction model is currently unavailable.") from exc


def predict_risk_class(features) -> int:
    """Validate `features` and return the model's class: 1 = higher risk, 0 = lower risk.

    The model was trained on survey responses where class 1 means the respondent
    reported prediabetes or diabetes. The class comes from the model's default decision
    rule; no probability is returned because the model is trained with class-balanced
    weights, so its probabilities do not reflect an individual's real-world likelihood.
    """
    clean = validate_features(features)
    model = load_model()
    row = transform_user_input(pd.DataFrame([clean]))
    try:
        return int(model.predict(row)[0])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise PredictionError("The prediction could not be completed.") from exc
