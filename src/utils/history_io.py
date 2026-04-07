"""
History I/O — save, load, and build training history dicts.
"""

import json
from pathlib import Path


def save_history(history: dict, path: str) -> None:
    """Serialize a Keras history dict (or plain dict) to JSON."""
    clean = {k: [float(v) for v in vals] for k, vals in history.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"  History saved → {path}")


def load_history(path: str) -> dict:
    """Load a history dict from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    with open(path) as f:
        return json.load(f)


def build_history(rows: list) -> dict:
    """
    Convert a list of (train_acc, val_acc, train_loss, val_loss) tuples
    into a history dict (for manually reconstructing logs from terminal output).
    """
    return {
        "accuracy":     [row[0] for row in rows],
        "val_accuracy": [row[1] for row in rows],
        "loss":         [row[2] for row in rows],
        "val_loss":     [row[3] for row in rows],
    }
