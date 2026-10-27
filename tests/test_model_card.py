import logging
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import inference
import model_card
from feature_schema import FEATURES

REAL_MODEL = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"


@pytest.fixture
def served_file(tmp_path, monkeypatch):
    # inference reads this copy, so each test can change the file or the card freely
    copy = tmp_path / "best_model.joblib"
    shutil.copy(REAL_MODEL, copy)
    shutil.copy(model_card.card_path(REAL_MODEL), model_card.card_path(copy))
    monkeypatch.setattr(inference, "MODEL_PATH", copy)
    inference.load_model.cache_clear()
    yield copy
    inference.load_model.cache_clear()


def test_the_committed_model_is_the_version_its_card_names():
    card = model_card.read_card(REAL_MODEL)
    assert card["registered_model"] == "diabetes-risk-classifier"
    assert card["version"] == 2
    assert card["algorithm"] == "GradientBoostingClassifier"
    assert card["features"] == FEATURES
    assert model_card.verify(REAL_MODEL) == card


def test_the_card_holds_the_recorded_holdout_results():
    metrics = model_card.read_card(REAL_MODEL)["metrics"]
    assert metrics["recall"] == pytest.approx(0.789395, abs=1e-6)
    assert metrics["roc_auc"] == pytest.approx(0.824538, abs=1e-6)
    assert sum(metrics[name] for name in ("true_negatives", "false_positives", "false_negatives", "true_positives")) == 25368


def test_the_loaded_model_keeps_the_training_feature_order():
    assert list(inference.load_model().feature_names_in_) == FEATURES


def test_the_served_version_is_logged_once_at_load(served_file, caplog):
    caplog.set_level(logging.INFO)
    inference.load_model()
    inference.load_model()
    served = [record for record in caplog.records if record.getMessage().startswith("Serving diabetes-risk-classifier")]
    assert len(served) == 1
    assert "version 2" in served[0].getMessage()


def test_a_different_model_in_the_same_file_name_is_refused(served_file, caplog):
    other = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    rng = np.random.default_rng(0)
    other.fit(pd.DataFrame(rng.normal(size=(50, 7)), columns=FEATURES), rng.integers(0, 2, 50))
    joblib.dump(other, served_file)

    with pytest.raises(inference.ModelUnavailableError):
        inference.load_model()
    assert "is not diabetes-risk-classifier version 2" in caplog.text
    assert "registry.py promote" in caplog.text


def test_a_model_trained_on_a_different_feature_order_is_refused(served_file):
    shuffled = [FEATURES[-1], *FEATURES[:-1]]
    rng = np.random.default_rng(0)
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(pd.DataFrame(rng.normal(size=(50, 7)), columns=shuffled), rng.integers(0, 2, 50))
    joblib.dump(model, served_file)
    model_card.card_path(served_file).unlink()

    with pytest.raises(inference.ModelUnavailableError):
        inference.load_model()


def test_a_missing_card_is_allowed_but_warned_about(served_file, caplog):
    model_card.card_path(served_file).unlink()
    assert inference.load_model() is not None
    assert "No model card" in caplog.text


def test_the_error_shown_to_users_does_not_mention_hashes_or_paths(served_file):
    served_file.write_bytes(b"not the model")
    with pytest.raises(inference.ModelUnavailableError) as excinfo:
        inference.load_model()
    assert str(excinfo.value) == "The prediction model is currently unavailable."
