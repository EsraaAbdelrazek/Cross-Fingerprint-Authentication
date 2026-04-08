"""
Backbone registry — maps config name strings to builder functions.

To add a new backbone:
    1. Create src/models/backbones/<name>.py with build_<name>_fingerprint()
    2. Import it here and add it to BACKBONE_REGISTRY
    3. Add its name to src/config/schema.py VALID_BACKBONES
"""

from src.models.backbones.vgg16 import build_vgg16_fingerprint
from src.models.backbones.resnet50 import build_resnet50_fingerprint
from src.models.backbones.mobilenetv2 import build_mobilenetv2_fingerprint
from src.models.backbones.efficientnet import build_efficientnet_fingerprint
from src.models.backbones.efficientnet_b1 import build_efficientnet_b1_fingerprint
from src.models.backbones.vit import build_vit_fingerprint

BACKBONE_REGISTRY = {
    "vgg16":            build_vgg16_fingerprint,
    "resnet50":         build_resnet50_fingerprint,
    "mobilenetv2":      build_mobilenetv2_fingerprint,
    "efficientnet":     build_efficientnet_fingerprint,
    "efficientnet_b1":  build_efficientnet_b1_fingerprint,
    "vit":              build_vit_fingerprint,
}

# Map each backbone to the layer name used as the encoder output
BACKBONE_ENCODER_LAYERS = {
    "vgg16":            "feature_dense",
    "resnet50":         "feature_dense",
    "mobilenetv2":      "feature_dense",
    "efficientnet":     "feature_dense",
    "efficientnet_b1":  "feature_dense",
    "vit":              "feature_dense",
}


def get_backbone(name: str):
    """Return the builder function for the given backbone name."""
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available: {sorted(BACKBONE_REGISTRY)}"
        )
    return BACKBONE_REGISTRY[name]


def get_encoder_layer(name: str) -> str:
    """Return the encoder layer name for the given backbone."""
    return BACKBONE_ENCODER_LAYERS.get(name, "feature_dense")
