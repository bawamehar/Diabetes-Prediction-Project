"""Model loading, input validation and prediction for the Streamlit app, the API and the MCP server."""
import logging
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

import model_card
from feature_schema import BINARY_FEATURES, BMI_RANGE, CATEGORY_RANGES
from preprocess import UI_FEATURES, transform_user_input

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"


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
    """Load the trained scaler + classifier pipeline once per process.

    The file must be the version named in models/model_card.json and must expect the features in
    training order, otherwise it is refused rather than served.
    """
    try:
        card = model_card.verify(MODEL_PATH)
        model = joblib.load(MODEL_PATH)
        if list(model.feature_names_in_) != list(UI_FEATURES):
            raise ValueError("the model was trained on different features or a different feature order")
    except Exception as exc:
        logger.exception("Could not load the prediction model")
        raise ModelUnavailableError("The prediction model is currently unavailable.") from exc
    if card is None:
        logger.warning("No model card next to the model file, the model version is unknown")
    else:
        logger.info("Serving %s version %s (sha256 %s)", card["registered_model"], card["version"], card["sha256"][:12])
    return model


def predict_risk_class(features) -> int:
    """Validate `features` and return the class: 1 = higher-risk pattern, 0 = lower-risk pattern.

    No probability is returned: the model is trained with class-balanced weights, so its
    scores are not real-world likelihoods.
    """
    clean = validate_features(features)
    model = load_model()
    row = transform_user_input(pd.DataFrame([clean]))
    try:
        return int(model.predict(row)[0])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise PredictionError("The prediction could not be completed.") from exc
