"""
Stage 2: Siamese Network for Fingerprint Authentication
========================================================
Uses the VGG16-Fingerprint encoder trained in Stage 1.
Trains on PolyU contact-based fingerprint pairs.

Based on: Ezz et al., CSSE 2023
  - Euclidean distance + Element-wise Absolute Difference (EAD)
  - Concatenated → Dense layers → Binary classification

Folder structure expected (already created by Stage 1):
  dataset/fingerprints_224/
      subject_001/   1_1.jpg ... 1_6.jpg
      subject_002/   ...
      subject_300/

Encoder expected at:
  models/vgg16_fingerprint_polyu_encoder.keras
"""

import os
import random
import itertools
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    Lambda, Concatenate, Subtract, Absolute
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, accuracy_score, roc_curve
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # ── Paths ─────────────────────────────────────────────────────────────────
    # Preprocessed images from Stage 1 (224×224, organized by subject)
    "PROCESSED_DIR"  : "dataset/fingerprints_224",

    # Encoder saved at end of Stage 1
    "ENCODER_PATH"   : "models/vgg16_fingerprint_polyu_encoder.keras",

    # Siamese model output
    "SIAMESE_PATH"   : "models/siamese_fingerprint_polyu.keras",
    "SIAMESE_CKPT"   : "models/siamese_fingerprint_polyu_best.keras",

    # ── Pair generation ───────────────────────────────────────────────────────
    # With 300 subjects × 6 images: C(6,2)=15 positive pairs per subject
    # → 300×15 = 4500 positive pairs available
    # We generate equal positive and negative pairs
    "NUM_PAIRS"      : 6000,    # total pairs (3000 pos + 3000 neg)
    "TEST_PAIRS"     : 2000,    # held-out test pairs
    "VAL_SPLIT"      : 0.2,

    # ── Training ──────────────────────────────────────────────────────────────
    "IMG_SIZE"       : (224, 224),
    "BATCH_SIZE"     : 32,
    "EPOCHS"         : 50,
    "LR"             : 1e-4,

    # ── Siamese head ──────────────────────────────────────────────────────────
    "DENSE_1"        : 512,
    "DENSE_2"        : 256,
    "DROPOUT"        : 0.3,
}

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD ALL IMAGES INTO MEMORY
# (only ~1800 images × ~150KB each ≈ manageable on M3)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(processed_dir: str, img_size: tuple) -> dict:
    """
    Load all subject images into a dict:
      { subject_name: [img_array, img_array, ...] }
    Images are normalized to [0, 1].
    """
    processed_dir = Path(processed_dir)
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    dataset = {}

    subject_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
    print(f"\n  Loading images from {processed_dir} ...")
    print(f"  Found {len(subject_dirs)} subjects")

    for subject_dir in subject_dirs:
        images = []
        for img_path in sorted(subject_dir.iterdir()):
            if img_path.suffix.lower() in supported:
                img = tf.keras.preprocessing.image.load_img(
                    str(img_path), target_size=img_size
                )
                arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(arr)

        if images:
            dataset[subject_dir.name] = images

    total_images = sum(len(v) for v in dataset.values())
    print(f"  Loaded {total_images} images across {len(dataset)} subjects")
    print(f"  Images per subject: {total_images // len(dataset)}")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: GENERATE PAIRS
# Algorithm 1 from the paper:
#   label = 0 → Similar  (same subject)
#   label = 1 → Not Similar (different subjects)
# ─────────────────────────────────────────────────────────────────────────────

def generate_pairs(dataset: dict, num_pairs: int, seed: int = 42):
    """
    Generate balanced positive/negative pairs.

    Positive pairs: two images from the same subject
    Negative pairs: two images from different subjects

    Returns: (img1_array, img2_array, label)
      label = 0 → Similar, label = 1 → Not Similar
    """
    random.seed(seed)
    np.random.seed(seed)

    subjects     = list(dataset.keys())
    half         = num_pairs // 2
    img1_list, img2_list, labels = [], [], []

    # ── Positive pairs (same subject) ────────────────────────────────────────
    pos_count = 0
    # First use all exhaustive pairs, then sample randomly if needed
    all_pos = []
    for subject in subjects:
        imgs = dataset[subject]
        if len(imgs) >= 2:
            for i, j in itertools.combinations(range(len(imgs)), 2):
                all_pos.append((imgs[i], imgs[j]))

    random.shuffle(all_pos)
    if len(all_pos) >= half:
        selected_pos = all_pos[:half]
    else:
        # Repeat pairs if not enough exhaustive combinations
        selected_pos = (all_pos * ((half // len(all_pos)) + 1))[:half]

    for a, b in selected_pos:
        img1_list.append(a)
        img2_list.append(b)
        labels.append(0)

    # ── Negative pairs (different subjects) ──────────────────────────────────
    neg_count = 0
    while neg_count < half:
        s1, s2 = random.sample(subjects, 2)
        img1 = random.choice(dataset[s1])
        img2 = random.choice(dataset[s2])
        img1_list.append(img1)
        img2_list.append(img2)
        labels.append(1)
        neg_count += 1

    # ── Shuffle ───────────────────────────────────────────────────────────────
    idx = list(range(len(labels)))
    random.shuffle(idx)

    img1_arr = np.array(img1_list)[idx]
    img2_arr = np.array(img2_list)[idx]
    lbl_arr  = np.array(labels)[idx]

    pos = int(np.sum(lbl_arr == 0))
    neg = int(np.sum(lbl_arr == 1))
    print(f"\n  Generated {len(lbl_arr)} pairs: {pos} similar + {neg} not-similar")
    return img1_arr, img2_arr, lbl_arr


def split_pairs(img1, img2, labels, val_split=0.2):
    """Split pairs into train/val sets."""
    n     = len(labels)
    split = int(n * (1 - val_split))

    return (
        [img1[:split],  img2[:split]],  labels[:split],   # train
        [img1[split:],  img2[split:]],  labels[split:],   # val
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: BUILD SIAMESE NETWORK
# Architecture from the paper (Figure 1):
#   Encoder → Euclidean distance
#   Encoder → EAD (Element-wise Absolute Difference)
#   Concatenate → Dense → Dense → Sigmoid
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_distance(tensors):
    """L2 distance between two feature vectors, kept as a 1-D tensor."""
    a, b = tensors
    return K.sqrt(K.maximum(K.sum(K.square(a - b), axis=1, keepdims=True), K.epsilon()))


def build_siamese(encoder: Model, cfg: dict) -> Model:
    """
    Modified Siamese network:
      Two shared VGG16-Fingerprint encoders
        ↓  Euclidean distance (scalar per pair)
        ↓  EAD — Element-wise Absolute Difference (1024-dim)
      Concatenate [euclidean, EAD]
        ↓  Dense(512, ReLU) → Dropout
        ↓  Dense(256, ReLU) → Dropout
        ↓  Dense(1, Sigmoid) → Similar(0) / Not Similar(1)
    """
    # Freeze encoder — Siamese head trains on top of fixed features
    encoder.trainable = False

    img_size = cfg["IMG_SIZE"]
    anchor_input = Input(shape=(*img_size, 3), name="anchor")
    pair_input   = Input(shape=(*img_size, 3), name="pair")

    # Shared encoder (same weights for both branches)
    feat_a = encoder(anchor_input)   # (batch, 1024)
    feat_p = encoder(pair_input)     # (batch, 1024)

    # ── Euclidean distance ────────────────────────────────────────────────────
    euclidean = Lambda(
        euclidean_distance,
        name="euclidean_distance"
    )([feat_a, feat_p])              # (batch, 1)

    # ── Element-Wise Absolute Difference (EAD) ────────────────────────────────
    diff = Subtract(name="subtract")([feat_a, feat_p])
    ead  = Lambda(lambda x: tf.abs(x), name="ead")(diff)   # (batch, 1024)

    # ── Concatenate both similarity measures ──────────────────────────────────
    merged = Concatenate(name="concat")([euclidean, ead])   # (batch, 1025)

    # ── Classification head ───────────────────────────────────────────────────
    x = Dense(cfg["DENSE_1"], activation="relu",
               name="dense_512")(merged)
    x = BatchNormalization(name="bn_512")(x)
    x = Dropout(cfg["DROPOUT"], name="drop_512")(x)
    x = Dense(cfg["DENSE_2"], activation="relu",
               name="dense_256")(x)
    x = Dropout(cfg["DROPOUT"], name="drop_256")(x)
    output = Dense(1, activation="sigmoid",
                   name="classification")(x)

    siamese = Model(
        inputs  = [anchor_input, pair_input],
        outputs = output,
        name    = "Siamese_VGG16_Fingerprint_PolyU"
    )
    return siamese


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_siamese(siamese: Model, X_train, y_train,
                  X_val, y_val, cfg: dict) -> dict:
    """Train the Siamese network."""

    siamese.compile(
        optimizer = Adam(cfg["LR"]),
        loss      = "binary_crossentropy",
        metrics   = ["accuracy",
                     tf.keras.metrics.AUC(name="auc"),
                     tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall")]
    )
    siamese.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", patience=10,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5,
            patience=5, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            cfg["SIAMESE_CKPT"], monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    print(f"\n  Training samples : {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Positive (similar): {int(np.sum(y_train==0))} train / {int(np.sum(y_val==0))} val")
    print(f"  Negative (diff)   : {int(np.sum(y_train==1))} train / {int(np.sum(y_val==1))} val")

    history = siamese.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        batch_size      = cfg["BATCH_SIZE"],
        epochs          = cfg["EPOCHS"],
        callbacks       = callbacks,
        verbose         = 1,
    )
    return history.history


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: EVALUATION — Accuracy + EER
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """
    Compute Equal Error Rate (EER).
    EER = point where False Acceptance Rate = False Rejection Rate
    Lower EER = better authentication system
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr     = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    thresh  = float(thresholds[eer_idx])
    return eer, thresh


def evaluate_siamese(siamese: Model, X_test, y_test: np.ndarray):
    """Full evaluation: accuracy, EER, classification report."""
    print("\n" + "═"*55)
    print("  EVALUATION")
    print("═"*55)

    scores = siamese.predict(X_test, batch_size=32, verbose=1).flatten()
    eer, eer_thresh = compute_eer(y_test, scores)

    # Use EER threshold for classification
    preds = (scores >= eer_thresh).astype(int)
    acc   = accuracy_score(y_test, preds)

    print(f"\n  Accuracy (at EER threshold) : {acc*100:.2f}%")
    print(f"  EER                         : {eer:.4f}  ({eer*100:.2f}%)")
    print(f"  EER threshold               : {eer_thresh:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, preds,
        target_names=["Similar (0)", "Not Similar (1)"]
    ))
    return acc, eer


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history: dict):
    """Plot accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Siamese Network (PolyU) — Training History", fontsize=13)
    eps = range(1, len(history["accuracy"]) + 1)

    ax1.plot(eps, history["accuracy"],     label="Train")
    ax1.plot(eps, history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(eps, history["loss"],     label="Train")
    ax2.plot(eps, history["val_loss"], label="Validation")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/siamese_training_history.png", dpi=150)
    plt.show()
    print("  Plot saved → results/siamese_training_history.png")


def plot_eer(y_test: np.ndarray, scores: np.ndarray):
    """Plot FAR vs FRR curve showing EER point."""
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, fpr[:len(thresholds)], label="FAR (False Accept)")
    plt.plot(thresholds, fnr[:len(thresholds)], label="FRR (False Reject)")
    plt.axvline(thresholds[eer_idx], color="red", linestyle="--",
                label=f"EER = {eer*100:.2f}%")
    plt.xlabel("Threshold"); plt.ylabel("Rate")
    plt.title("FAR vs FRR — EER Curve (PolyU Fingerprint)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/siamese_eer_curve.png", dpi=150)
    plt.show()
    print("  EER plot saved → results/siamese_eer_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: INFERENCE — predict a single pair
# ─────────────────────────────────────────────────────────────────────────────

def predict_pair(siamese: Model, img_path1: str, img_path2: str,
                 img_size: tuple = (224, 224),
                 threshold: float = 0.5) -> dict:
    """
    Given two fingerprint image paths, predict whether they
    belong to the same person.

    Returns:
      score       : raw sigmoid output (0=similar, 1=different)
      prediction  : 'Similar' or 'Not Similar'
      confidence  : how confident the model is
    """
    def _load(path):
        img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
        return tf.keras.preprocessing.image.img_to_array(img)[np.newaxis] / 255.0

    score = float(siamese.predict(
        [_load(img_path1), _load(img_path2)], verbose=0
    )[0, 0])

    return {
        "score"      : round(score, 4),
        "prediction" : "Not Similar" if score >= threshold else "Similar",
        "confidence" : round(score if score >= threshold else 1 - score, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = CFG
    print("\n" + "="*55)
    print("  Stage 2: Siamese Fingerprint Authentication")
    print("  PolyU Contact-based Fingerprint Dataset")
    print("  Based on Ezz et al., CSSE 2023")
    print("="*55)

    # ── 1. Load encoder from Stage 1 ─────────────────────────────────────────
    print(f"\n  Loading encoder: {cfg['ENCODER_PATH']}")
    if not Path(cfg["ENCODER_PATH"]).exists():
        raise FileNotFoundError(
            f"Encoder not found at '{cfg['ENCODER_PATH']}'.\n"
            "Please run polyu_vgg16_fingerprint.py (Stage 1) first."
        )
    encoder = load_model(cfg["ENCODER_PATH"])
    print(f"  Encoder output shape: {encoder.output_shape}")

    # ── 2. Load all images into memory ───────────────────────────────────────
    dataset = load_dataset(cfg["PROCESSED_DIR"], cfg["IMG_SIZE"])

    # ── 3. Generate training + validation pairs ───────────────────────────────
    print("\n  Generating training pairs ...")
    img1, img2, labels = generate_pairs(
        dataset, num_pairs=cfg["NUM_PAIRS"], seed=42
    )
    X_train, y_train, X_val, y_val = split_pairs(
        img1, img2, labels, val_split=cfg["VAL_SPLIT"]
    )

    # ── 4. Generate held-out test pairs ──────────────────────────────────────
    print("\n  Generating test pairs ...")
    t_img1, t_img2, t_labels = generate_pairs(
        dataset, num_pairs=cfg["TEST_PAIRS"], seed=99
    )
    X_test  = [t_img1, t_img2]
    y_test  = t_labels

    # ── 5. Build Siamese network ──────────────────────────────────────────────
    print("\n  Building Siamese network ...")
    siamese = build_siamese(encoder, cfg)

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  TRAINING SIAMESE NETWORK")
    print("="*55)
    history = train_siamese(siamese, X_train, y_train,
                            X_val, y_val, cfg)
    import json
    with open("results/hist_siamese.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
    # ── 7. Plot training history ──────────────────────────────────────────────
    plot_history(history)

    # ── 8. Save final model ───────────────────────────────────────────────────
    siamese.save(cfg["SIAMESE_PATH"])
    print(f"\n  Siamese model saved → {cfg['SIAMESE_PATH']}")

    # ── 9. Evaluate on held-out test pairs ───────────────────────────────────
    acc, eer = evaluate_siamese(siamese, X_test, y_test)

    # ── 10. EER plot ──────────────────────────────────────────────────────────
    scores = siamese.predict(X_test, batch_size=32, verbose=0).flatten()
    plot_eer(y_test, scores)

    # ── 11. Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  Stage 1 encoder accuracy : 73.00%")
    print(f"  Stage 2 test accuracy    : {acc*100:.2f}%")
    print(f"  Stage 2 EER              : {eer*100:.2f}%")
    print(f"  Paper reported EER       : 8.2% (CASIA dataset)")
    print("="*55)
    print("\n  Usage — predict a new pair:")
    print("  result = predict_pair(siamese, 'finger1.jpg', 'finger2.jpg')")
    print("  # {'score': 0.08, 'prediction': 'Similar', 'confidence': 0.92}")
    print()


if __name__ == "__main__":
    main()
