"""models/model_card.json records which registered model version models/best_model.joblib is.

Standard library only, so the app can check it without MLflow installed.
"""
import hashlib
import json
from pathlib import Path

MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
CARD_NAME = "model_card.json"


class ModelCardMismatch(RuntimeError):
    """The model file is not the version its model card names."""


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def card_path(model_path=None):
    return Path(model_path or MODEL_FILE).with_name(CARD_NAME)


def read_card(model_path=None):
    path = card_path(model_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_card(card, model_path=None):
    card_path(model_path).write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8", newline="\n")


def verify(model_path=None):
    """Return the card for this file, or None if there is no card. Raise if the file is another model."""
    model_path = Path(model_path or MODEL_FILE)
    card = read_card(model_path)
    if card is None:
        return None
    actual = file_sha256(model_path)
    if actual != card["sha256"]:
        raise ModelCardMismatch(
            f"{model_path.name} (sha256 {actual[:12]}) is not {card['registered_model']} version "
            f"{card['version']} (sha256 {card['sha256'][:12]}). "
            "Run `python src/registry.py promote <version>` to choose the version to serve."
        )
    return card
