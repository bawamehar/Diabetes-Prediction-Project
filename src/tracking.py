"""MLflow experiment tracking and model registry helpers.

Only training and src/registry.py use this. The app never imports MLflow: it serves
models/best_model.joblib and checks it against models/model_card.json (model_card.py).
"""
import datetime
import importlib.metadata
import json
import os
from pathlib import Path

from feature_schema import FEATURES

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_STORE = BASE_DIR / "mlflow_store"
REGISTERED_MODEL = "diabetes-risk-classifier"
CHAMPION_ALIAS = "champion"
DEFAULT_EXPERIMENT = "diabetes-risk"
MODEL_ARTIFACT = "joblib"


class MlflowNotInstalled(RuntimeError):
    pass


def _mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise MlflowNotInstalled("MLflow is not installed. Run: pip install mlflow") from exc
    return mlflow


def configure():
    """Point MLflow at MLFLOW_TRACKING_URI, or at a local SQLite store in ./mlflow_store when it is not set.

    Returns the client and the id of the experiment (MLFLOW_EXPERIMENT_NAME, created if missing).
    """
    mlflow = _mlflow()
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    artifact_location = None
    if not uri:
        LOCAL_STORE.mkdir(exist_ok=True)
        uri = f"sqlite:///{(LOCAL_STORE / 'mlflow.db').as_posix()}"
        artifact_location = (LOCAL_STORE / "artifacts").as_uri()
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "").strip() or DEFAULT_EXPERIMENT
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        name, artifact_location=artifact_location
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    return client, experiment_id


def model_params(model):
    """Algorithm and hyperparameters of a classifier or of the last step of a pipeline."""
    classifier = model.steps[-1][1] if hasattr(model, "steps") else model
    params = {"algorithm": type(classifier).__name__}
    params.update({f"clf.{key}": str(value) for key, value in classifier.get_params().items()})
    return params


def holdout_metrics(y_true, y_pred, y_proba):
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "true_negatives": int(tn), "false_positives": int(fp), "false_negatives": int(fn), "true_positives": int(tp),
    }


def dataset_tags(data_file, rows):
    """Identify the training data by file name, size and hash. No records are logged."""
    from model_card import file_sha256

    return {
        "dataset": "CDC BRFSS 2015 diabetes health indicators",
        "dataset_file": Path(data_file).name,
        "dataset_rows": str(rows),
        "dataset_sha256": file_sha256(data_file),
    }


def _signature(model):
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, Schema

    names = list(getattr(model, "feature_names_in_", FEATURES))
    return ModelSignature(
        inputs=Schema([ColSpec("double", name) for name in names]), outputs=Schema([ColSpec("long")])
    )


def _requirements():
    return [f"{package}=={importlib.metadata.version(package)}" for package in ("scikit-learn", "numpy", "pandas", "joblib")]


def log_run(model, *, run_name, params, metrics, tags, model_file, version_tags, description, extras=None):
    """Log one run with its parameters, metrics, the model and the original model file.

    extras maps an artifact file name to a JSON-able object to store with the run.

    The model is registered as a new version of REGISTERED_MODEL. The champion alias is never
    touched here: serving a different version is a separate step (registry.py promote).
    Returns (run_id, version).
    """
    from model_card import file_sha256

    mlflow = _mlflow()
    client, experiment_id = configure()
    names = list(getattr(model, "feature_names_in_", FEATURES))
    sha256 = file_sha256(model_file)
    version_tags = {**version_tags, "sha256": sha256}
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_params({**model_params(model), **params})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({**tags, "feature_order": json.dumps(names), "sha256": sha256, "model_file": Path(model_file).name})
        mlflow.log_dict({"features": names, "preprocessing": "StandardScaler inside the saved pipeline"}, "feature_schema.json")
        for name, payload in (extras or {}).items():
            mlflow.log_dict(payload, name)
        mlflow.log_artifact(str(model_file), MODEL_ARTIFACT)
        info = mlflow.sklearn.log_model(model, name="model", signature=_signature(model), pip_requirements=_requirements())
        version = str(mlflow.register_model(info.model_uri, REGISTERED_MODEL, tags=version_tags).version)
    client.update_model_version(REGISTERED_MODEL, version, description=description)
    return run.info.run_id, version


def model_card_for(client, version):
    """The model card for a registered version, built from its run."""
    model_version = client.get_model_version(REGISTERED_MODEL, str(version))
    run = client.get_run(model_version.run_id)
    return {
        "registered_model": REGISTERED_MODEL,
        "version": int(model_version.version),
        "alias": CHAMPION_ALIAS,
        "sha256": model_version.tags["sha256"],
        "algorithm": run.data.params["algorithm"],
        "features": json.loads(run.data.tags["feature_order"]),
        **{key: int(run.data.params[key]) for key in ("training_rows", "holdout_rows") if key in run.data.params},
        "metrics": {
            name: int(value) if name.startswith(("true_", "false_")) else round(value, 6)
            for name, value in run.data.metrics.items()
        },
        "mlflow_run_id": run.info.run_id,
        "recorded_on": datetime.date.today().isoformat(),
    }
