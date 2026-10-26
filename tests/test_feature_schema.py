import json

import joblib
import pytest

import feature_schema as fs
import inference
from preprocess import UI_FEATURES

LOW_RISK = {"HighBP": 0, "HighChol": 0, "BMI": 22.0, "GenHlth": 1, "Age": 3, "Income": 8, "Education": 6}


def test_schema_features_match_the_trained_model():
    model = joblib.load(inference.MODEL_PATH)
    assert list(model.feature_names_in_) == fs.FEATURES == UI_FEATURES


@pytest.mark.parametrize("name", list(fs.LABELS))
def test_every_labelled_code_is_accepted_and_its_neighbours_are_not(name):
    codes = list(fs.LABELS[name])
    for code in codes:
        inference.validate_features({**LOW_RISK, name: code})
    for bad in (min(codes) - 1, max(codes) + 1):
        with pytest.raises(inference.InputValidationError):
            inference.validate_features({**LOW_RISK, name: bad})


def test_bmi_limits_match_validation():
    low, high = fs.BMI_RANGE
    inference.validate_features({**LOW_RISK, "BMI": low})
    inference.validate_features({**LOW_RISK, "BMI": high})
    for bad in (low - 0.1, high + 0.1):
        with pytest.raises(inference.InputValidationError):
            inference.validate_features({**LOW_RISK, "BMI": bad})


def test_schema_is_plain_json_and_does_not_claim_a_diagnosis():
    schema = json.loads(json.dumps(fs.get_feature_schema()))
    assert [feature["name"] for feature in schema["features"]] == fs.FEATURES
    assert schema["output"]["is_diagnosis"] is False
    assert schema["output"]["probability_available"] is False
    bmi = next(feature for feature in schema["features"] if feature["name"] == "BMI")
    assert (bmi["min"], bmi["max"]) == fs.BMI_RANGE


def test_describe_answers_uses_plain_words():
    described = fs.describe_answers({"HighBP": 1, "BMI": 31.04, "GenHlth": 4, "Age": 9})
    assert described == {
        "High blood pressure": "Yes", "BMI": "31.0 kg/m2", "General health": "Fair", "Age group": "60 to 64",
    }


def test_describe_answers_leaves_out_what_was_not_shared():
    described = fs.describe_answers({"HighBP": 0, "BMI": 25.0})
    assert "Household income" not in described and "Education" not in described
