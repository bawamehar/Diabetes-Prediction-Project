import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pytest.importorskip("mlflow")

import mlflow
from mlflow.exceptions import MlflowException

import inference
import model_card
import registry
import tracking
from feature_schema import FEATURES

REAL_MODEL = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"


@pytest.fixture
def store(tmp_path, monkeypatch):
    # a throwaway MLflow store and model folder, so the project's own mlflow_store/ and models/ are never touched
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(tracking, "LOCAL_STORE", tmp_path / "mlflow_store")
    monkeypatch.setattr(model_card, "MODEL_FILE", tmp_path / "models" / "best_model.joblib")
    monkeypatch.setattr(inference, "MODEL_PATH", model_card.MODEL_FILE)
    inference.load_model.cache_clear()
    yield tmp_path
    inference.load_model.cache_clear()


def fitted(seed=0, features=FEATURES):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(200, len(features))), columns=list(features))
    y = (X.iloc[:, 0] + rng.normal(size=200) > 0).astype(int)
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0 + seed))]).fit(X, y)


def log(model, folder, name, model_file=None):
    if model_file is None:
        model_file = folder / f"{name}.joblib"
        joblib.dump(model, model_file)
    return tracking.log_run(
        model, run_name=name, params={"holdout_fraction": "0.1"}, metrics={"recall": 0.5, "accuracy": 0.6},
        tags={"source": "test"}, model_file=model_file, version_tags={"source": "test"}, description="test model",
    )


def champion(client):
    return str(client.get_model_version_by_alias(tracking.REGISTERED_MODEL, tracking.CHAMPION_ALIAS).version)


def test_experiment_is_created_once_and_reused(store, monkeypatch):
    client, first = tracking.configure()
    _, again = tracking.configure()
    assert first == again
    assert client.get_experiment(first).name == tracking.DEFAULT_EXPERIMENT
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "another-experiment")
    _, other = tracking.configure()
    assert other != first


def test_default_store_is_local_and_the_uri_can_be_overridden(store, monkeypatch):
    tracking.configure()
    assert (store / "mlflow_store" / "mlflow.db").exists()
    assert mlflow.get_tracking_uri().startswith("sqlite:///")

    other = store / "elsewhere.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{other.as_posix()}")
    tracking.configure()
    assert mlflow.get_tracking_uri() == f"sqlite:///{other.as_posix()}"
    assert other.exists()


def test_a_run_records_params_metrics_tags_and_artifacts(store):
    model = fitted()
    run_id, version = log(model, store, "first")
    client, _ = tracking.configure()
    run = client.get_run(run_id)

    assert run.data.params["algorithm"] == "LogisticRegression"
    assert run.data.params["clf.C"] == "1.0"
    assert run.data.params["holdout_fraction"] == "0.1"
    assert run.data.metrics == {"recall": 0.5, "accuracy": 0.6}
    assert json.loads(run.data.tags["feature_order"]) == FEATURES
    assert run.data.tags["sha256"] == model_card.file_sha256(store / "first.joblib")
    assert {item.path for item in client.list_artifacts(run_id)} >= {"feature_schema.json", "joblib"}
    assert version == "1"


def test_versions_follow_registration_order_and_nothing_is_promoted(store):
    _, first = log(fitted(1), store, "a")
    _, second = log(fitted(2), store, "b")
    assert (first, second) == ("1", "2")

    client, _ = tracking.configure()
    with pytest.raises(MlflowException):
        champion(client)
    version = client.get_model_version(tracking.REGISTERED_MODEL, "2")
    assert version.description == "test model"
    assert version.tags["sha256"] == model_card.file_sha256(store / "b.joblib")
    assert version.tags["source"] == "test"


def test_promote_serves_exactly_the_chosen_version_and_keeps_the_others(store):
    one, two = fitted(1), fitted(2)
    log(one, store, "a")
    log(two, store, "b")
    client, _ = tracking.configure()

    registry.promote(2)
    assert model_card.file_sha256(model_card.MODEL_FILE) == model_card.file_sha256(store / "b.joblib")
    assert champion(client) == "2"
    card = model_card.read_card()
    assert card["version"] == 2 and card["features"] == FEATURES and card["metrics"]["recall"] == 0.5
    sample = pd.DataFrame([[0, 1, 30.0, 3, 8, 5, 4]], columns=FEATURES)
    assert inference.load_model().predict_proba(sample)[0, 1] == pytest.approx(two.predict_proba(sample)[0, 1])

    registry.promote(1)
    assert model_card.file_sha256(model_card.MODEL_FILE) == model_card.file_sha256(store / "a.joblib")
    assert champion(client) == "1"
    assert model_card.read_card()["version"] == 1
    assert len(client.search_model_versions(f"name='{tracking.REGISTERED_MODEL}'")) == 2


def test_a_file_that_was_swapped_after_promotion_is_refused(store):
    log(fitted(1), store, "a")
    registry.promote(1)
    joblib.dump(fitted(5), model_card.MODEL_FILE)
    inference.load_model.cache_clear()
    with pytest.raises(inference.ModelUnavailableError):
        inference.load_model()


def test_versions_trained_on_other_features_cannot_be_promoted(store):
    log(fitted(3, features=[*FEATURES, "Extra"]), store, "wide")
    with pytest.raises(SystemExit, match="cannot be served"):
        registry.promote(1)
    assert not model_card.MODEL_FILE.exists()
    client, _ = tracking.configure()
    with pytest.raises(MlflowException):
        champion(client)


def test_a_damaged_stored_file_is_not_copied_into_models(store):
    log(fitted(1), store, "a")
    client, _ = tracking.configure()
    client.set_model_version_tag(tracking.REGISTERED_MODEL, "1", "sha256", "0" * 64)
    with pytest.raises(SystemExit, match="recorded hash"):
        registry.promote(1)
    assert not model_card.MODEL_FILE.exists()


def test_show_lists_versions_and_flags_a_mismatching_file(store, capsys):
    log(fitted(1), store, "a")
    registry.promote(1)
    registry.show()
    shown = capsys.readouterr().out
    assert "version 1" in shown and "aliases=['champion']" in shown and "matches it" in shown

    joblib.dump(fitted(9), model_card.MODEL_FILE)
    registry.show()
    assert "does NOT match" in capsys.readouterr().out


def test_the_deployed_model_round_trips_through_the_registry_unchanged(store):
    model = joblib.load(REAL_MODEL)
    log(model, store, "deployed", model_file=REAL_MODEL)
    registry.promote(1)
    assert model_card.file_sha256(model_card.MODEL_FILE) == model_card.file_sha256(REAL_MODEL)

    registered = mlflow.sklearn.load_model(f"models:/{tracking.REGISTERED_MODEL}@{tracking.CHAMPION_ALIAS}")
    rng = np.random.default_rng(7)
    rows = pd.DataFrame({
        "HighBP": rng.integers(0, 2, 150), "HighChol": rng.integers(0, 2, 150),
        "BMI": rng.uniform(12, 98, 150).round(1), "GenHlth": rng.integers(1, 6, 150),
        "Age": rng.integers(1, 14, 150), "Income": rng.integers(1, 9, 150), "Education": rng.integers(1, 7, 150),
    })
    expected = [inference.predict_risk_class(row.to_dict()) for _, row in rows.iterrows()]
    assert list(registered.predict(rows[FEATURES])) == expected
    assert set(expected) == {0, 1}


def test_mlflow_is_not_needed_to_serve_predictions():
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    code = (
        "import sys; sys.path.insert(0, 'src'); import inference; inference.load_model(); "
        "assert 'mlflow' not in sys.modules, 'mlflow was imported'"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=root, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
