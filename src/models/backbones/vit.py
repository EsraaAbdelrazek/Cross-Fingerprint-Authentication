"""
Vision Transformer (ViT) Fingerprint backbone.

Uses keras_cv or vit-keras if available, otherwise falls back to a
lightweight patch-based ViT implemented directly in Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

ENCODER_LAYER = "feature_dense"


# ── Lightweight ViT building blocks ──────────────────────────────────────────

def _mlp(x, hidden_units: list, dropout_rate: float):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def _transformer_block(x, num_heads: int, projection_dim: int,
                        mlp_head_units: list, dropout: float):
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=dropout
    )(x1, x1)
    x2 = layers.Add()([attn, x])
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = _mlp(x3, mlp_head_units, dropout)
    return layers.Add()([x3, x2])


def build_vit_fingerprint(num_classes: int,
                           dense_units: int = 1024,
                           dropout: float = 0.5,
                           img_size: tuple = (224, 224),
                           patch_size: int = 16,
                           projection_dim: int = 64,
                           num_heads: int = 4,
                           transformer_layers: int = 8) -> Model:
    """
    Lightweight Vision Transformer for fingerprint classification.

    Args:
        num_classes:        Number of subjects.
        dense_units:        Embedding dim for the encoder output (ENCODER_LAYER).
        dropout:            Dropout rate.
        img_size:           Input image spatial size (H, W).
        patch_size:         Size of each image patch (default 16 → 14×14 patches for 224px).
        projection_dim:     ViT token embedding dimension.
        num_heads:          Number of attention heads.
        transformer_layers: Number of transformer blocks.
    """
    h, w = img_size
    num_patches = (h // patch_size) * (w // patch_size)
    mlp_head_units = [projection_dim * 2, projection_dim]

    inputs = layers.Input(shape=(*img_size, 3), name="input_image")

    # ── Patch extraction + linear projection ─────────────────────────────────
    patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patch_embed",
    )(inputs)
    patches = layers.Reshape((num_patches, projection_dim), name="patch_reshape")(patches)

    # ── Positional encoding ───────────────────────────────────────────────────
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = layers.Embedding(input_dim=num_patches,
                                  output_dim=projection_dim,
                                  name="pos_embed")(positions)
    x = patches + pos_embed

    # ── Transformer blocks ────────────────────────────────────────────────────
    for _ in range(transformer_layers):
        x = _transformer_block(x, num_heads, projection_dim,
                                mlp_head_units, dropout * 0.5)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu",
                      name=ENCODER_LAYER)(x)
    x = layers.Dropout(dropout * 0.8)(x)
    out = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return Model(inputs=inputs, outputs=out, name="ViT_Fingerprint")
