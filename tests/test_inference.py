import logging
import math

import pytest

import inference
from inference import (
    InputValidationError,
    ModelUnavailableError,
    predict_risk_class,
    validate_features,
)

HIGH_RISK = {"HighBP": 1, "HighChol": 1, "BMI": 45.0, "GenHlth": 5, "Age": 13, "Income": 1, "Education": 1}
LOW_RISK = {"HighBP": 0, "HighChol": 0, "BMI": 22.0, "GenHlth": 1, "Age": 3, "Income": 8, "Education": 6}


def with_value(name, value):
    return {**LOW_RISK, name: value}


def without(name):
    return {key: value for key, value in LOW_RISK.items() if key != name}


def test_valid_input_returns_a_class():
    assert predict_risk_class(LOW_RISK) in (0, 1)


def test_model_distinguishes_high_and_low_risk_profiles():
    # Guards against the model collapsing to a single class again.
    assert predict_risk_class(HIGH_RISK) == 1
    assert predict_risk_class(LOW_RISK) == 0


def test_boundary_values_are_accepted():
    for features in (
        {**LOW_RISK, "BMI": 12.0, "Income": 1, "Age": 1, "Education": 1, "GenHlth": 1},
        {**LOW_RISK, "BMI": 98.0, "Income": 8, "Age": 13, "Education": 6, "GenHlth": 5},
    ):
        assert validate_features(features)["BMI"] in (12.0, 98.0)


@pytest.mark.parametrize(
    "features, expected_fragment",
    [
        (without("Age"), "Age"),
        (with_value("BMI", None), "BMI"),
        (with_value("BMI", ""), "BMI"),
        (with_value("BMI", "abc"), "BMI"),
        (with_value("BMI", math.nan), "BMI"),
        (with_value("BMI", math.inf), "BMI"),
        (with_value("BMI", 11.9), "between 12 and 98"),
        (with_value("BMI", 98.1), "between 12 and 98"),
        (with_value("Income", 9), "Income"),
        (with_value("Income", 20), "Income"),
        (with_value("Education", 0), "Education"),
        (with_value("Education", 7), "Education"),
        (with_value("GenHlth", 2.5), "whole number"),
        (with_value("Age", 14), "Age"),
        (with_value("HighBP", 2), "HighBP"),
        (with_value("HighChol", 0.5), "HighChol"),
        ({**LOW_RISK, "Smoker": 1}, "Unexpected"),
        (None, "No input"),
        ([1, 2, 3], "No input"),
    ],
)
def test_invalid_inputs_are_rejected_with_a_clear_message(features, expected_fragment):
    with pytest.raises(InputValidationError) as excinfo:
        predict_risk_class(features)
    assert expected_fragment in str(excinfo.value)


def test_all_problems_are_reported_together():
    bad = {**LOW_RISK, "BMI": 500, "Income": 20, "HighBP": 3}
    with pytest.raises(InputValidationError) as excinfo:
        validate_features(bad)
    assert len(excinfo.value.problems) == 3


def test_error_messages_do_not_echo_submitted_values():
    with pytest.raises(InputValidationError) as excinfo:
        validate_features(with_value("BMI", "secret-looking-text"))
    assert "secret-looking-text" not in str(excinfo.value)


def test_submitted_values_are_never_logged(caplog):
    caplog.set_level(logging.DEBUG)
    predict_risk_class({**LOW_RISK, "BMI": 23.4567})
    with pytest.raises(InputValidationError):
        predict_risk_class({**LOW_RISK, "BMI": 7.6543})
    assert "23.4567" not in caplog.text
    assert "7.6543" not in caplog.text


def test_missing_model_file_gives_a_safe_error(monkeypatch, tmp_path):
    missing = tmp_path / "nope.joblib"
    monkeypatch.setattr(inference, "MODEL_PATH", missing)
    inference.load_model.cache_clear()
    try:
        with pytest.raises(ModelUnavailableError) as excinfo:
            predict_risk_class(LOW_RISK)
        assert str(tmp_path) not in str(excinfo.value)
    finally:
        monkeypatch.undo()
        inference.load_model.cache_clear()
