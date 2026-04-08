"""
EAD Siamese Head — Euclidean distance + Element-wise Absolute Difference.

Based on: Ezz et al., CSSE 2023
    Two shared encoder branches
    → Euclidean distance (scalar)
    → EAD — Element-wise Absolute Difference (encoder_dim)
    Concatenate → Dense → Dense → Sigmoid

Label convention:  0 = Similar,  1 = Not Similar
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    Lambda, Concatenate, Subtract,
)
from tensorflow.keras.models import Model


def euclidean_distance(tensors):
    """L2 distance between two feature vectors, shape (batch, 1)."""
    a, b = tensors
    return K.sqrt(K.maximum(
        K.sum(K.square(a - b), axis=1, keepdims=True),
        K.epsilon()
    ))


def build_ead_head(encoder: Model, cfg: dict) -> Model:
    """
    Build the EAD Siamese network on top of a frozen encoder.

    Args:
        encoder: Trained feature extractor (output: embedding vector).
        cfg:     Full experiment config dict.

    Returns:
        Compiled-ready Siamese model with two inputs (anchor, pair).
    """
    head_cfg = cfg.get("siamese_head", {})
    dense_1  = head_cfg.get("dense_1",  512)
    dense_2  = head_cfg.get("dense_2",  256)
    dropout  = head_cfg.get("dropout",  0.3)
    img_size = tuple(cfg["dataset"].get("img_size", [224, 224]))

    encoder.trainable = False

    anchor_input = Input(shape=(*img_size, 3), name="anchor")
    pair_input   = Input(shape=(*img_size, 3), name="pair")

    feat_a = encoder(anchor_input)
    feat_p = encoder(pair_input)

    # Euclidean distance (scalar similarity)
    euclidean = Lambda(euclidean_distance, name="euclidean_distance")([feat_a, feat_p])

    # Element-wise Absolute Difference
    diff = Subtract(name="subtract")([feat_a, feat_p])
    ead  = Lambda(lambda x: tf.abs(x), name="ead")(diff)

    # Concatenate both measures and classify
    merged = Concatenate(name="concat")([euclidean, ead])

    x = Dense(dense_1, activation="relu", name="dense_1")(merged)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(dropout, name="drop_1")(x)
    x = Dense(dense_2, activation="relu", name="dense_2")(x)
    x = Dropout(dropout, name="drop_2")(x)
    output = Dense(1, activation="sigmoid", name="classification")(x)

    return Model(
        inputs=[anchor_input, pair_input],
        outputs=output,
        name="Siamese_EAD",
    )
