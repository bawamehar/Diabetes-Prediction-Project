import os

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "AdaBoostClassifier.joblib"
)


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "Model artifact not found"


def test_model_loads_without_error():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_predicts_expected_output_shape():
    model = joblib.load(MODEL_PATH)

    sample_input = pd.DataFrame([{
        "HighBP": 1,
        "HighChol": 1,
        "BMI": 35.0,
        "GenHlth": 5,
        "Age": 13,
        "Income": 1,
        "Education": 1,
    }])

    n_features = getattr(model, "n_features_in_", None)
    values = sample_input.values

    if n_features is not None and values.shape[1] < n_features:
        values = np.pad(values, ((0, 0), (0, n_features - values.shape[1])), "constant")
    elif n_features is not None and values.shape[1] > n_features:
        values = values[:, :n_features]

    prediction = model.predict(values)

    assert prediction.shape[0] == 1
    assert prediction[0] in (0, 1)


def test_model_handles_bad_input_gracefully():
    model = joblib.load(MODEL_PATH)
    bad_input = np.array([[None, None, None, None, None, None, None]])

    try:
        model.predict(bad_input)
        raised = False
    except (ValueError, TypeError):
        raised = True

    assert raised, "Model should raise an error on invalid/None input"