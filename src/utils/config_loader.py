"""
Config loader — reads a YAML experiment config, deep-merges it with
defaults.yaml, and validates required fields.
"""

import copy
from pathlib import Path

import yaml

from src.config.schema import validate_config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config(config_path: str) -> dict:
    """
    Load an experiment YAML config, merge it with defaults.yaml,
    validate required fields, and return the final merged dict.

    Args:
        config_path: Path to the experiment YAML (e.g. 'configs/polyu_vgg16_ead.yaml')

    Returns:
        Fully merged and validated config dict.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load defaults (optional — if missing, start from empty)
    defaults_path = config_path.parent / "defaults.yaml"
    defaults = {}
    if defaults_path.exists():
        with open(defaults_path) as f:
            defaults = yaml.safe_load(f) or {}

    # Load experiment config
    with open(config_path) as f:
        experiment_cfg = yaml.safe_load(f) or {}

    # Merge: defaults first, experiment overrides
    merged = _deep_merge(defaults, experiment_cfg)

    # Validate
    validate_config(merged)

    return merged
