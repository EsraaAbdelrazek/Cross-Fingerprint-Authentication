"""
VGG16-Fingerprint backbone.
Pretrained on ImageNet, fine-tuned for fingerprint subject classification.
"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# Layer to use as the encoder output for Stage 2
ENCODER_LAYER = "feature_dense"

# All VGG16 backbone layer names (used for freeze/unfreeze logic)
VGG16_LAYER_NAMES = {
    "block1_conv1", "block1_conv2", "block1_pool",
    "block2_conv1", "block2_conv2", "block2_pool",
    "block3_conv1", "block3_conv2", "block3_conv3", "block3_pool",
    "block4_conv1", "block4_conv2", "block4_conv3", "block4_pool",
    "block5_conv1", "block5_conv2", "block5_conv3", "block5_pool",
}


def build_vgg16_fingerprint(num_classes: int,
                             dense_units: int = 1024,
                             dropout: float = 0.5,
                             img_size: tuple = (224, 224)) -> Model:
    """
    VGG16 (ImageNet) → fingerprint classification head.

    Architecture:
        VGG16 conv base
        → GlobalAveragePooling2D
        → Dense(dense_units, relu)  [ENCODER_LAYER — used by Stage 2]
        → BatchNorm → Dropout(0.6)
        → Dense(512, relu) → Dropout(0.4)
        → Dense(num_classes, softmax)

    Args:
        num_classes: Number of subjects (classes).
        dense_units: Size of the feature embedding (encoder output).
        dropout:     Base dropout rate (head uses 0.6 and 0.4 around it).
        img_size:    Input image spatial size (H, W).
    """
    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(*img_size, 3))

    x = base.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dense(dense_units, activation="relu",
               kernel_regularizer=l2(1e-4),
               name=ENCODER_LAYER)(x)
    x = BatchNormalization(name="feature_bn")(x)
    x = Dropout(0.6, name="feature_dropout")(x)
    x = Dense(512, activation="relu",
               kernel_regularizer=l2(1e-4),
               name="feature_dense2")(x)
    x = Dropout(0.4, name="feature_dropout2")(x)
    out = Dense(num_classes, activation="softmax", name="classifier")(x)

    return Model(inputs=base.input, outputs=out, name="VGG16_Fingerprint")
