"""
MobileNetV2-Fingerprint backbone — lightweight, fast for ablation studies.
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

ENCODER_LAYER = "feature_dense"


def build_mobilenetv2_fingerprint(num_classes: int,
                                   dense_units: int = 1024,
                                   dropout: float = 0.5,
                                   img_size: tuple = (224, 224)) -> Model:
    """
    MobileNetV2 (ImageNet) → fingerprint classification head.
    Lightweight alternative to VGG16/ResNet50 for quick ablation runs.
    """
    base = MobileNetV2(weights="imagenet", include_top=False,
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

    return Model(inputs=base.input, outputs=out, name="MobileNetV2_Fingerprint")
