"""Choose which registered model version the app serves.

    python src/registry.py show
    python src/registry.py promote 2

promote sets the champion alias, puts that version's model file at models/best_model.joblib and
rewrites models/model_card.json. Training registers new versions but never promotes them.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import model_card
import tracking
from feature_schema import FEATURES


def show():
    client, _ = tracking.configure()
    versions = client.search_model_versions(f"name='{tracking.REGISTERED_MODEL}'")
    aliases = client.get_registered_model(tracking.REGISTERED_MODEL).aliases if versions else {}
    print(f"Registered model {tracking.REGISTERED_MODEL} ({client.tracking_uri})")
    for model_version in sorted(versions, key=lambda item: int(item.version)):
        run = client.get_run(model_version.run_id)
        recall = run.data.metrics.get("recall")
        names = sorted(alias for alias, number in aliases.items() if str(number) == str(model_version.version))
        print(
            f"  version {model_version.version}  aliases={names or '-'}  "
            f"{run.data.params['algorithm']}  source={model_version.tags.get('source', '?')}  "
            f"sha256={model_version.tags['sha256'][:12]}  recall={'n/a' if recall is None else round(recall, 4)}"
        )
    if not versions:
        print("  no versions yet")

    card = model_card.read_card()
    print("Model file served by the app:")
    if card is None:
        print("  no models/model_card.json")
    else:
        matches = model_card.file_sha256(model_card.MODEL_FILE) == card["sha256"]
        print(f"  model card says version {card['version']}; {model_card.MODEL_FILE.name} "
              f"{'matches it' if matches else 'does NOT match it, the app will refuse to load it'}")


def promote(version):
    client, _ = tracking.configure()
    model_version = client.get_model_version(tracking.REGISTERED_MODEL, str(version))
    run = client.get_run(model_version.run_id)
    features = json.loads(run.data.tags["feature_order"])
    if features != list(FEATURES):
        sys.exit(
            f"Version {version} was trained on {len(features)} features that are not this app's inputs, "
            "so it cannot be served here. Nothing was changed."
        )

    expected = model_version.tags["sha256"]
    target = model_card.MODEL_FILE
    if target.exists() and model_card.file_sha256(target) == expected:
        print(f"{target.name} already is version {version}")
    else:
        with tempfile.TemporaryDirectory() as folder:
            stored = client.download_artifacts(run.info.run_id, f"{tracking.MODEL_ARTIFACT}/{run.data.tags['model_file']}", folder)
            if model_card.file_sha256(stored) != expected:
                sys.exit("The stored model file does not match its recorded hash. Nothing was changed.")
            staged = target.with_suffix(".tmp")
            staged.write_bytes(Path(stored).read_bytes())
            os.replace(staged, target)
        print(f"{target.name} replaced with version {version}")

    client.set_registered_model_alias(tracking.REGISTERED_MODEL, tracking.CHAMPION_ALIAS, str(version))
    model_card.write_card(tracking.model_card_for(client, version))
    print(f"Version {version} is now the {tracking.CHAMPION_ALIAS}. Run the tests, then commit models/ to deploy it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show", help="list registered versions and check the model file against its card")
    promote_parser = commands.add_parser("promote", help="serve a registered version")
    promote_parser.add_argument("version", type=int)
    args = parser.parse_args()
    try:
        show() if args.command == "show" else promote(args.version)
    except tracking.MlflowNotInstalled as exc:
        sys.exit(str(exc))
