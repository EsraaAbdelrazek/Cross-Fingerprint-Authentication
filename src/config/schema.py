"""
Config schema — defines and validates all YAML fields.
Every experiment YAML is validated against this schema before training starts.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ── Sub-schemas ───────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name: str


@dataclass
class DatasetConfig:
    name: str
    source_dir: str
    num_subjects: int = 300
    img_size: List[int] = field(default_factory=lambda: [224, 224])
    val_split: float = 0.2


@dataclass
class BackboneConfig:
    name: str = "vgg16"
    dense_units: int = 1024
    dropout: float = 0.5


@dataclass
class PhaseConfig:
    epochs: int = 50
    lr: float = 3e-4
    unfreeze_layer: Optional[str] = None


@dataclass
class Stage1Config:
    batch_size: int = 32
    phase_a: PhaseConfig = field(default_factory=lambda: PhaseConfig(epochs=50, lr=3e-4))
    phase_b: PhaseConfig = field(default_factory=lambda: PhaseConfig(epochs=30, lr=5e-5, unfreeze_layer="block5_conv3"))


@dataclass
class Stage2Config:
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-4
    num_pairs: int = 6000
    test_pairs: int = 2000


@dataclass
class TrainingConfig:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)


@dataclass
class SiameseHeadConfig:
    name: str = "ead"
    dense_1: int = 512
    dense_2: int = 256
    dropout: float = 0.3


@dataclass
class OutputConfig:
    experiments_root: str = "experiments"


@dataclass
class FullConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    siamese_head: SiameseHeadConfig = field(default_factory=SiameseHeadConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ── Validation ────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = {
    "experiment": ["name"],
    "dataset": ["name", "source_dir"],
}

VALID_BACKBONES   = {"vgg16", "resnet50", "mobilenetv2", "efficientnet", "efficientnet_b1", "vit"}
VALID_DATASETS    = {"polyu"}
VALID_SIAMESE_HEADS = {"ead", "triplet"}


def validate_config(cfg: dict) -> None:
    """Raise ValueError with a clear message if any required field is missing or invalid."""
    for section, keys in REQUIRED_FIELDS.items():
        if section not in cfg:
            raise ValueError(f"Config missing required section: '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Config missing required field: '{section}.{key}'")

    backbone_name = cfg.get("backbone", {}).get("name", "vgg16")
    if backbone_name not in VALID_BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Available: {sorted(VALID_BACKBONES)}"
        )

    dataset_name = cfg["dataset"]["name"]
    if dataset_name not in VALID_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {sorted(VALID_DATASETS)}"
        )

    head_name = cfg.get("siamese_head", {}).get("name", "ead")
    if head_name not in VALID_SIAMESE_HEADS:
        raise ValueError(
            f"Unknown siamese head '{head_name}'. "
            f"Available: {sorted(VALID_SIAMESE_HEADS)}"
        )
