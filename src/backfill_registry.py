"""One-off: register the two models this project has served, in the order they were served.

Neither has an MLflow run to reuse. The original AdaBoost model only ever existed as a file in git,
and the current model's runs were in a local store that no longer exists. So each is logged now as a
run that says so (nothing is retrained or invented), and the registry numbers them in registration
order: the original model is version 1 and the current model is version 2.

    python src/backfill_registry.py            # show what would be registered
    python src/backfill_registry.py --apply
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib

import model_card
import tracking
from dataload import DATA_FILE, read_data
from preprocess import preprocess_data

ORIGINAL = {
    "commit": "11ed81e",
    "path": "models/AdaBoostClassifier.joblib",
    "sha256": "3dfdbf87dd3e9db240adf1c216cd16295994dba8a005f18d64ce540b07f87df6",
}
SPLIT = {"holdout_fraction": 0.1, "split_random_state": 25, "stratified": True}
TRAINING_NOTE = (
    "Trained on 190,265 rows by the champion/challenger retraining workflow that has since been removed "
    "(the initial 75% pool plus the first 5 rows of its 15% stream pool). The current src/train.py trains "
    "on all 228,312 non-holdout rows, so running it gives a different model."
)


def original_model(folder):
    git = subprocess.run(
        ["git", "show", f"{ORIGINAL['commit']}:{ORIGINAL['path']}"], cwd=tracking.BASE_DIR, capture_output=True, check=False,
    )
    if git.returncode != 0:
        sys.exit(f"Could not read {ORIGINAL['path']} at commit {ORIGINAL['commit']} from git history.")
    file = Path(folder) / Path(ORIGINAL["path"]).name
    file.write_bytes(git.stdout)
    if model_card.file_sha256(file) != ORIGINAL["sha256"]:
        sys.exit("The file in git history is not the original model that was expected. Nothing was registered.")
    return file


def register_original(folder):
    file = original_model(folder)
    return tracking.log_run(
        joblib.load(file),
        run_name="backfill: original AdaBoost model (git history)",
        params={},
        metrics={},
        tags={
            "source": f"recovered from git commit {ORIGINAL['commit']}",
            "mlflow_tracked_at_training_time": "false",
            "metrics_source": "none: the original run results were not kept and the model was not re-evaluated here",
        },
        model_file=file,
        version_tags={"source": "recovered_from_git", "servable_by_current_app": "false"},
        description=(
            "Original AdaBoost model served by the first version of the app (git commit "
            f"{ORIGINAL['commit']}). Recovered from git and registered later, never tracked by MLflow and not "
            "evaluated here. It takes 21 unscaled features, so the current app cannot serve it."
        ),
    )


def register_current():
    model_file = model_card.MODEL_FILE
    model = joblib.load(model_file)
    df = read_data()
    X, y = preprocess_data(df)
    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=SPLIT["holdout_fraction"], random_state=SPLIT["split_random_state"], stratify=y
    )
    metrics = tracking.holdout_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
    return tracking.log_run(
        model,
        run_name="backfill: current model (models/best_model.joblib)",
        params={
            **{key: str(value) for key, value in SPLIT.items()},
            "class_weighting": "balanced",
            "training_rows": int(model.named_steps["scaler"].n_samples_seen_),
            "holdout_rows": len(X_test),
        },
        metrics=metrics,
        tags={
            "source": "existing artifact models/best_model.joblib",
            "mlflow_tracked_at_training_time": "false",
            "metrics_source": "recomputed from the saved model on the holdout split, which it was not trained on",
            "training_note": TRAINING_NOTE,
            **tracking.dataset_tags(DATA_FILE, len(df)),
        },
        model_file=model_file,
        version_tags={"source": "existing_artifact", "servable_by_current_app": "true"},
        description=(
            "Gradient boosting pipeline (StandardScaler + classifier, 7 features, class-balanced training) "
            "that the app serves. The existing file, registered later; its metrics were recomputed on the holdout split."
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="register the models (default: only describe)")
    args = parser.parse_args()

    print(f"1. original AdaBoost model, git commit {ORIGINAL['commit']}, sha256 {ORIGINAL['sha256'][:12]}")
    print(f"2. current model, {model_card.MODEL_FILE.name}, sha256 {model_card.file_sha256(model_card.MODEL_FILE)[:12]}")
    if not args.apply:
        print("Nothing registered. Run again with --apply.")
        sys.exit(0)

    try:
        client, _ = tracking.configure()
        existing = client.search_model_versions(f"name='{tracking.REGISTERED_MODEL}'")
        if existing:
            sys.exit(f"{tracking.REGISTERED_MODEL} already has {len(existing)} version(s). Version numbers follow "
                     "registration order, so nothing was added. Use `python src/registry.py show`.")
        with tempfile.TemporaryDirectory() as folder:
            run_id, original_version = register_original(folder)
        print(f"registered version {original_version} (run {run_id})")
        run_id, current_version = register_current()
        print(f"registered version {current_version} (run {run_id})")
        client.set_registered_model_alias(tracking.REGISTERED_MODEL, tracking.CHAMPION_ALIAS, current_version)
        model_card.write_card(tracking.model_card_for(client, current_version))
        print(f"alias {tracking.CHAMPION_ALIAS} -> version {current_version}; wrote models/{model_card.CARD_NAME}")
    except tracking.MlflowNotInstalled as exc:
        sys.exit(str(exc))
