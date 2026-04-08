"""
ResNet50-Fingerprint backbone.
"""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

ENCODER_LAYER = "feature_dense"

# ResNet50 backbone layer names for freeze/unfreeze
RESNET50_LAYER_NAMES = {
    l.name for l in ResNet50(weights=None, include_top=False).layers
} if False else set()  # Populated lazily to avoid loading weights at import time


def build_resnet50_fingerprint(num_classes: int,
                                dense_units: int = 1024,
                                dropout: float = 0.5,
                                img_size: tuple = (224, 224)) -> Model:
    """
    ResNet50 (ImageNet) → fingerprint classification head.

    Architecture:
        ResNet50 conv base
        → GlobalAveragePooling2D
        → Dense(dense_units, relu)  [ENCODER_LAYER]
        → BatchNorm → Dropout
        → Dense(512, relu) → Dropout
        → Dense(num_classes, softmax)
    """
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(*img_size, 3))

    x = base.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dense(dense_units, activation="relu",
               kernel_regularizer=l2(1e-4),
               name=ENCODER_LAYER)(x)
    x = BatchNormalization(name="feature_bn")(x)
    x = Dropout(dropout, name="feature_dropout")(x)
    x = Dense(512, activation="relu",
               kernel_regularizer=l2(1e-4),
               name="feature_dense2")(x)
    x = Dropout(dropout * 0.8, name="feature_dropout2")(x)
    out = Dense(num_classes, activation="softmax", name="classifier")(x)

    return Model(inputs=base.input, outputs=out, name="ResNet50_Fingerprint")
