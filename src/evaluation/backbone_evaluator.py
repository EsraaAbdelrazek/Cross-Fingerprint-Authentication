"""
Stage 1 backbone evaluation — classification accuracy on the validation set.
"""

from tensorflow.keras.models import Model


def evaluate(model: Model, val_gen) -> float:
    """
    Evaluate classification accuracy on the validation generator.

    Args:
        model:   Trained classification model.
        val_gen: Keras validation generator (reset before eval).

    Returns:
        Validation accuracy as a float in [0, 1].
    """
    print("\n" + "═" * 55)
    print("  STAGE 1 EVALUATION")
    print("═" * 55)

    val_gen.reset()
    loss, acc = model.evaluate(val_gen, verbose=1)
    print(f"\n  Validation Loss    : {loss:.4f}")
    print(f"  Validation Accuracy: {acc * 100:.2f}%")
    return acc
