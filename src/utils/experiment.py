"""
Experiment directory management.
Creates the standard folder tree for a run and saves a config snapshot.
"""

import shutil
from pathlib import Path

import yaml


def setup_experiment_dir(cfg: dict) -> dict:
    """
    Create the experiment output directory tree and save a config snapshot.

    Structure created:
        experiments/<name>/
            checkpoints/
            histories/
            plots/
            config.yaml     ← snapshot of the merged config used

    Args:
        cfg: Fully merged config dict (from load_config).

    Returns:
        Dict of resolved paths:
            {
                "root":         Path("experiments/polyu_vgg16_ead"),
                "checkpoints":  Path(...)/checkpoints,
                "histories":    Path(...)/histories,
                "plots":        Path(...)/plots,
            }
    """
    exp_name = cfg["experiment"]["name"]
    root = Path(cfg.get("output", {}).get("experiments_root", "experiments")) / exp_name

    paths = {
        "root":        root,
        "checkpoints": root / "checkpoints",
        "histories":   root / "histories",
        "plots":       root / "plots",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # Save config snapshot for reproducibility
    snapshot_path = root / "config.yaml"
    with open(snapshot_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Experiment dir : {root}")
    print(f"  Config snapshot: {snapshot_path}")

    return {k: str(v) for k, v in paths.items()}
