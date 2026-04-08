"""
Triplet Loss Siamese Head — stub for future implementation.

Triplet networks take (anchor, positive, negative) triplets and use
a margin-based loss to push same-subject embeddings closer together
while pushing different-subject embeddings further apart.
"""

from tensorflow.keras.models import Model


def build_triplet_head(encoder: Model, cfg: dict) -> Model:
    """
    Stub — not yet implemented.
    Will build a triplet-loss network on top of the encoder.
    """
    raise NotImplementedError(
        "Triplet head is not yet implemented. "
        "Use siamese_head.name: ead for now."
    )
