"""
EfficientNetB1-Fingerprint backbone.

Improvements over B0:
  - 7.8M params vs 5.3M — more capacity for 300-class fingerprint task
  - Native resolution 240px (we use 260px for extra ridge detail)
  - Same BN-sensitive architecture — keep Phase B LR at 1e-5 or lower

Note: EfficientNet includes its own rescaling layer — do NOT apply
rescale=1/255 in the data generator when using this backbone.
"""

from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

ENCODER_LAYER = "feature_dense"


def build_efficientnet_b1_fingerprint(num_classes: int,
                                       dense_units: int = 1024,
                                       dropout: float = 0.3,
                                       img_size: tuple = (260, 260)) -> Model:
    """
    EfficientNetB1 (ImageNet) → fingerprint classification head.

    Args:
        num_classes: Number of subject classes.
        dense_units: Size of the embedding Dense layer.
        dropout:     Dropout rate (0.3 recommended — B0 experiments showed underfitting at 0.5).
        img_size:    Input image size. Default 260×260 for better ridge detail.
    """
    base = EfficientNetB1(weights="imagenet", include_top=False,
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

    return Model(inputs=base.input, outputs=out, name="EfficientNetB1_Fingerprint")
