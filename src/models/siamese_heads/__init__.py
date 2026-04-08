from .ead_head import build_ead_head
from .triplet_head import build_triplet_head

SIAMESE_HEAD_REGISTRY = {
    "ead":     build_ead_head,
    "triplet": build_triplet_head,
}


def get_siamese_head(name: str):
    """Return the builder function for the given siamese head name."""
    if name not in SIAMESE_HEAD_REGISTRY:
        raise ValueError(
            f"Unknown siamese head '{name}'. "
            f"Available: {sorted(SIAMESE_HEAD_REGISTRY)}"
        )
    return SIAMESE_HEAD_REGISTRY[name]
