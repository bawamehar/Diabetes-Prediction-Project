import pytest

# The patient-logging API is optional and not part of the public deployment, so these
# tests are skipped where its dependencies (see requirements-dev.txt) are not installed.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("psycopg2")

from fastapi.testclient import TestClient

from api import fastapi as api_module

VALID = {"bmi": 33.3, "age": 9, "genhlth": 3, "income": 5, "highbp": 1, "highchol": 0, "education": 4}


@pytest.fixture
def client():
    return TestClient(api_module.app)


def test_valid_record_is_saved(client, monkeypatch):
    saved = []
    monkeypatch.setattr(api_module, "insert_patient", lambda values: saved.append(values) or True)
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert len(saved) == 1
    assert saved[0] == (33.3, 9.0, 3.0, 5.0, 1.0, 0.0, 4.0)


def test_response_does_not_echo_submitted_values(client, monkeypatch):
    monkeypatch.setattr(api_module, "insert_patient", lambda values: True)
    assert "33.3" not in client.post("/predict", json=VALID).text


@pytest.mark.parametrize(
    "override",
    [{"income": 20}, {"bmi": 500}, {"bmi": 3}, {"genhlth": 2.5}, {"highbp": 2}, {"education": 0}],
)
def test_out_of_range_values_are_rejected_and_not_saved(client, monkeypatch, override):
    monkeypatch.setattr(api_module, "insert_patient", lambda values: pytest.fail("must not save"))
    response = client.post("/predict", json={**VALID, **override})
    assert response.status_code == 422


def test_missing_field_is_rejected(client, monkeypatch):
    monkeypatch.setattr(api_module, "insert_patient", lambda values: pytest.fail("must not save"))
    incomplete = {key: value for key, value in VALID.items() if key != "age"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_database_failure_is_reported_not_hidden(client, monkeypatch):
    monkeypatch.setattr(api_module, "insert_patient", lambda values: False)
    response = client.post("/predict", json=VALID)
    assert response.status_code == 503
    assert "could not be saved" in response.json()["detail"]
