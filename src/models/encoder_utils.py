"""
Encoder extraction and loading utilities.
Works with any backbone that declares ENCODER_LAYER.
"""

from pathlib import Path

from tensorflow.keras.models import Model, load_model


def extract_encoder(model: Model, encoder_path: str,
                    encoder_layer: str = "feature_dense") -> Model:
    """
    Strip the classification head from a trained backbone model,
    returning only the feature extractor up to encoder_layer.

    Args:
        model:         Trained full classification model.
        encoder_path:  Where to save the encoder .keras file.
        encoder_layer: Name of the layer whose output is the embedding.

    Returns:
        The encoder model (saved to encoder_path).
    """
    encoder = Model(
        inputs=model.input,
        outputs=model.get_layer(encoder_layer).output,
        name=f"{model.name}_Encoder",
    )
    Path(encoder_path).parent.mkdir(parents=True, exist_ok=True)
    encoder.save(encoder_path)
    print(f"  Encoder ({encoder.output_shape[-1]}-dim) saved → {encoder_path}")
    return encoder


def load_encoder(encoder_path: str) -> Model:
    """Load a saved encoder model from disk."""
    path = Path(encoder_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Encoder not found at '{encoder_path}'.\n"
            "Run Stage 1 training first (--stage 1 or --stage both)."
        )
    encoder = load_model(str(path))
    print(f"  Encoder loaded: {encoder_path}  (output: {encoder.output_shape})")
    return encoder
