"""
PolyU Contactless 2D to Contact-based Fingerprint Dataset
==========================================================
Reorganizes the dataset into subject subfolders and runs VGG16 transfer learning.

Dataset naming convention:
  contact-based fingerprints/  X_Y.jpg
    X = finger/user ID (1..336)
    Y = sample number (1..6)

After reorganization:
  dataset/fingerprints/
    subject_001/   1_1.jpg  1_2.jpg  ...  1_6.jpg
    subject_002/   2_1.jpg  ...
    ...
    subject_300/   (first 300 only, matching the paper's Stage 1)
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — update SOURCE paths to match where you extracted the dataset
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # ── Input: raw dataset paths (pick ONE source) ──────────────────────────
    # Option A: contact-based fingerprints (RECOMMENDED)
    "SOURCE_DIR": "contact-based_fingerprints/first_session",

    # Option B: already-downsampled contactless (uncomment to use instead)
    # "SOURCE_DIR"    : "PolyU_dataset/processed_contactless_2d_fingerprint_images",

    # ── Output: reorganized + preprocessed ──────────────────────────────────
    "ORGANIZED_DIR"   : "dataset/fingerprints_organized",   # after reorganization
    "PROCESSED_DIR"   : "dataset/fingerprints_224",         # after 224×224 resize

    # ── Training ─────────────────────────────────────────────────────────────
    "NUM_SUBJECTS"    : 300,          # use first 300 clients (paper's Stage 1)
    "IMG_SIZE"        : (224, 224),
    "BATCH_SIZE"      : 32,
    "EPOCHS_A"        : 50,           # Phase A: frozen VGG16 backbone
    "EPOCHS_B"        : 30,           # Phase B: unfreeze block5_conv3
    "LR_A"            : 3e-4,    # was 1e-3, lower to reduce overfitting
    "LR_B"            : 5e-5,    # was 1e-4 
    "VAL_SPLIT"       : 0.2,
    "DENSE_UNITS"     : 1024,
    "DROPOUT"         : 0.5,

    # ── Saved models ─────────────────────────────────────────────────────────
    "MODEL_PATH"      : "models/vgg16_fingerprint_polyu.keras",
    "BEST_CKPT"       : "models/vgg16_fingerprint_polyu_best.keras",
    "ENCODER_PATH"    : "models/vgg16_fingerprint_polyu_encoder.keras",
}

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DETECT DATASET FORMAT & REORGANIZE INTO SUBJECT SUBFOLDERS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_format(source_dir: str) -> str:
    """
    Detect whether images are flat (X_Y.jpg) or already in subfolders (pX/pY).
    Returns: 'flat' or 'nested'
    """
    source = Path(source_dir)
    # Check for flat files like "1_1.jpg"
    flat_files = list(source.glob("*.jpg")) + list(source.glob("*.bmp"))
    if flat_files:
        return "flat"
    # Check for nested like pX/pY
    subdirs = [d for d in source.iterdir() if d.is_dir()]
    if subdirs:
        return "nested"
    return "unknown"


def reorganize_flat(source_dir: str, out_dir: str,
                    num_subjects: int = 300) -> dict:
    """
    Reorganize flat files (X_Y.jpg format) into subject subfolders.

    contact-based fingerprints/
        1_1.jpg  1_2.jpg ... 1_6.jpg      (subject 1)
        2_1.jpg  2_2.jpg ... 2_6.jpg      (subject 2)
        ...

    →  dataset/fingerprints_organized/
           subject_001/  1_1.jpg ... 1_6.jpg
           subject_002/  ...
    """
    source = Path(source_dir)
    out    = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Group files by subject ID (X in X_Y.ext)
    subject_files = defaultdict(list)
    for f in sorted(source.iterdir()):
        if f.suffix.lower() in {".jpg", ".jpeg", ".bmp", ".png"}:
            parts = f.stem.split("_")
            if len(parts) >= 2:
                subject_id = int(parts[0])
                subject_files[subject_id].append(f)

    subject_ids = sorted(subject_files.keys())[:num_subjects]
    print(f"\n{'─'*55}")
    print(f"  Reorganizing dataset")
    print(f"  Source     : {source_dir}")
    print(f"  Output     : {out_dir}")
    print(f"  Format     : flat (X_Y.jpg)")
    print(f"  Subjects   : {len(subject_ids)} (of {len(subject_files)} found)")
    print(f"{'─'*55}")

    stats = {}
    for sid in subject_ids:
        folder_name = f"subject_{sid:03d}"
        subject_out = out / folder_name
        subject_out.mkdir(exist_ok=True)

        for src_file in subject_files[sid]:
            dst = subject_out / src_file.name
            if not dst.exists():
                shutil.copy2(src_file, dst)

        stats[folder_name] = len(subject_files[sid])
        print(f"  ✓ {folder_name}  →  {len(subject_files[sid])} images")

    total = sum(stats.values())
    print(f"\n  Done. {total} images across {len(stats)} subjects.\n")
    return stats


def reorganize_nested(source_dir: str, out_dir: str,
                      num_subjects: int = 300) -> dict:
    """
    Reorganize nested pX/pY structure (contactless downsampled folder)
    into flat subject subfolders.

    processed_contactless/
        p001/  p1.bmp  p2.bmp ...
        p002/  ...

    →  dataset/fingerprints_organized/
           subject_001/  p1.bmp ...
    """
    source = Path(source_dir)
    out    = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([d for d in source.iterdir() if d.is_dir()])[:num_subjects]

    print(f"\n{'─'*55}")
    print(f"  Reorganizing nested dataset (pX/pY format)")
    print(f"  Subjects : {len(subject_dirs)}")
    print(f"{'─'*55}")

    stats = {}
    for i, sdir in enumerate(subject_dirs, 1):
        folder_name = f"subject_{i:03d}"
        subject_out = out / folder_name
        subject_out.mkdir(exist_ok=True)

        images = list(sdir.glob("*"))
        for img in images:
            dst = subject_out / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

        stats[folder_name] = len(images)
        print(f"  ✓ {sdir.name} → {folder_name}  ({len(images)} images)")

    total = sum(stats.values())
    print(f"\n  Done. {total} images across {len(stats)} subjects.\n")
    return stats


def reorganize_dataset(source_dir: str, out_dir: str, num_subjects: int = 300):
    """Auto-detect format and reorganize."""
    if Path(out_dir).exists() and any(Path(out_dir).iterdir()):
        print(f"[Reorganize] Already done: {out_dir}")
        return

    fmt = detect_format(source_dir)
    print(f"[Reorganize] Detected format: {fmt}")

    if fmt == "flat":
        reorganize_flat(source_dir, out_dir, num_subjects)
    elif fmt == "nested":
        reorganize_nested(source_dir, out_dir, num_subjects)
    else:
        raise ValueError(
            f"Could not detect dataset format in '{source_dir}'.\n"
            "Expected either flat files (1_1.jpg) or nested folders (p001/p1.bmp)."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: PREPROCESS — RESIZE TO 224×224 USING OPENCV
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_to_224(src_dir: str, dst_dir: str,
                      img_size: tuple = (224, 224)):
    """
    Resize all images to 224×224 using OpenCV (as specified in the paper).
    Converts grayscale images to RGB (VGG16 expects 3 channels).
    """
    if Path(dst_dir).exists() and any(Path(dst_dir).iterdir()):
        print(f"[Preprocess] Already done: {dst_dir}")
        return

    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    total = 0

    print(f"\n{'─'*55}")
    print(f"  Preprocessing: {src_dir} → {dst_dir}")
    print(f"  Resize to    : {img_size[0]}×{img_size[1]} (OpenCV)")
    print(f"{'─'*55}")

    subject_dirs = sorted([d for d in src.iterdir() if d.is_dir()])

    for subject_dir in subject_dirs:
        out_subject = dst / subject_dir.name
        out_subject.mkdir(exist_ok=True)

        images = [f for f in subject_dir.iterdir()
                  if f.suffix.lower() in supported]
        count = 0

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [WARN] Skipping unreadable: {img_path}")
                continue

            # If grayscale, convert to BGR for VGG16 compatibility
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Resize using INTER_AREA (best for downscaling, as per OpenCV docs)
            img_resized = cv2.resize(img, img_size,
                                     interpolation=cv2.INTER_AREA)

            # Save as JPEG
            out_path = out_subject / (img_path.stem + ".jpg")
            cv2.imwrite(str(out_path), img_resized,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1
            total += 1

        print(f"  ✓ {subject_dir.name:<20}  {count} images → {img_size[0]}×{img_size[1]}")

    print(f"\n  Total images preprocessed: {total}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def create_generators(processed_dir: str, img_size: tuple,
                      batch_size: int, val_split: float):
    """Train/validation generators with strong augmentation (only 6 imgs/subject)."""

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        shear_range=0.1,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
    )

    train_gen = train_datagen.flow_from_directory(
        processed_dir, target_size=img_size,
        batch_size=batch_size, class_mode="categorical",
        subset="training", shuffle=True, seed=42,
    )
    val_gen = val_datagen.flow_from_directory(
        processed_dir, target_size=img_size,
        batch_size=batch_size, class_mode="categorical",
        subset="validation", shuffle=False, seed=42,
    )

    return train_gen, val_gen

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: VGG16-FINGERPRINT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_vgg16_fingerprint(num_classes: int,
                             dense_units: int = 1024,
                             dropout_rate: float = 0.5) -> Model:
    """
    VGG16-ImageNet → VGG16-Fingerprint with stronger regularization.
    Needed because dataset has only 6 images per subject.
    """
    from tensorflow.keras.regularizers import l2

    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(224, 224, 3))

    x = base.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dense(dense_units, activation="relu",
               kernel_regularizer=l2(1e-4),
               name="feature_dense")(x)
    x = BatchNormalization(name="feature_bn")(x)
    x = Dropout(0.6, name="feature_dropout")(x)
    x = Dense(512, activation="relu",
               kernel_regularizer=l2(1e-4),
               name="feature_dense2")(x)
    x = Dropout(0.4, name="feature_dropout2")(x)
    out = Dense(num_classes, activation="softmax", name="classifier")(x)

    return Model(inputs=base.input, outputs=out, name="VGG16_Fingerprint_PolyU")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def make_callbacks(ckpt_path: str) -> list:
    return [
        EarlyStopping(monitor="val_accuracy", patience=8,   # was 12
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                          patience=4, min_lr=1e-7, verbose=1),  # was 6
        ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

# Names of the VGG16 backbone layers (everything before the custom head)
VGG16_LAYER_NAMES = {
    'block1_conv1', 'block1_conv2', 'block1_pool',
    'block2_conv1', 'block2_conv2', 'block2_pool',
    'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool',
    'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
    'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool',
}

def train(model: Model, train_gen, val_gen, cfg: dict):
    """Two-phase training strategy (Table 3 of the paper)."""

    # ── Phase A: Freeze all VGG16 layers, train head only ─────────────────
    print("\n" + "═"*55)
    print("  PHASE A — Freeze all VGG16, train head only")
    print("  Equivalent to experiment #19 in Table 3 (best result: ~98.5%)")
    print("═"*55)
    for layer in model.layers:
        layer.trainable = layer.name not in VGG16_LAYER_NAMES
    model.compile(optimizer=Adam(cfg["LR_A"]),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    hist_a = model.fit(train_gen, validation_data=val_gen,
                       epochs=cfg["EPOCHS_A"],
                       callbacks=make_callbacks(cfg["BEST_CKPT"]),
                       verbose=1).history

    # ── Phase B: Unfreeze block5_conv3 only ────────────────────────────────
    print("\n" + "═"*55)
    print("  PHASE B — Unfreeze block5_conv3 + feature_dense")
    print("  Equivalent to experiment #17 in Table 3 (97.52% train)")
    print("═"*55)
    for layer in model.layers:
        if layer.name in VGG16_LAYER_NAMES:
            layer.trainable = (layer.name == "block5_conv3")
    model.compile(optimizer=Adam(cfg["LR_B"]),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    hist_b = model.fit(train_gen, validation_data=val_gen,
                       epochs=cfg["EPOCHS_B"],
                       callbacks=make_callbacks(cfg["BEST_CKPT"]),
                       verbose=1).history

    return hist_a, hist_b


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: PLOTS + EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_history(hist_a: dict, hist_b: dict):
    """Accuracy and loss curves across both phases."""
    acc  = hist_a["accuracy"]  + hist_b["accuracy"]
    val  = hist_a["val_accuracy"] + hist_b["val_accuracy"]
    loss = hist_a["loss"]      + hist_b["loss"]
    vloss= hist_a["val_loss"]  + hist_b["val_loss"]
    eps  = range(1, len(acc) + 1)
    split = len(hist_a["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VGG16-Fingerprint (PolyU) — Training History", fontsize=13)

    for ax, y, vy, title, ylabel in [
        (ax1, acc, val, "Accuracy", "Accuracy"),
        (ax2, loss, vloss, "Loss", "Loss"),
    ]:
        ax.plot(eps, y,  label="Train")
        ax.plot(eps, vy, label="Validation")
        ax.axvline(split, color="gray", linestyle="--",
                   label=f"Phase B (ep {split})")
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/polyu_training_history.png", dpi=150)
    plt.show()
    print("  Plot saved → results/polyu_training_history.png")


def evaluate(model: Model, val_gen) -> float:
    """Print accuracy on the validation set."""
    print("\n" + "═"*55)
    print("  EVALUATION")
    print("═"*55)
    val_gen.reset()
    loss, acc = model.evaluate(val_gen, verbose=1)
    print(f"\n  Validation Loss    : {loss:.4f}")
    print(f"  Validation Accuracy: {acc*100:.2f}%")
    return acc


def extract_encoder(model: Model, encoder_path: str) -> Model:
    """Strip classifier → 1024-dim encoder for Siamese Stage 2."""
    encoder = Model(
        inputs  = model.input,
        outputs = model.get_layer("feature_dense").output,
        name    = "VGG16_Fingerprint_Encoder_PolyU"
    )
    encoder.save(encoder_path)
    print(f"\n  Encoder (1024-dim) saved → {encoder_path}")
    return encoder


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = CFG
    print("\n🖐  VGG16 Transfer Learning — PolyU Fingerprint Dataset")
    print("    Following Ezz et al. (CSSE 2023) methodology\n")

    # 1. Reorganize raw dataset into subject subfolders
    reorganize_dataset(cfg["SOURCE_DIR"], cfg["ORGANIZED_DIR"],
                       num_subjects=cfg["NUM_SUBJECTS"])

    # 2. Resize all images to 224×224 using OpenCV
    preprocess_to_224(cfg["ORGANIZED_DIR"], cfg["PROCESSED_DIR"],
                      img_size=cfg["IMG_SIZE"])

    # 3. Create train / val generators
    train_gen, val_gen = create_generators(
        cfg["PROCESSED_DIR"], cfg["IMG_SIZE"],
        cfg["BATCH_SIZE"], cfg["VAL_SPLIT"]
    )
    num_classes = train_gen.num_classes
    print(f"\n  Subjects (classes)  : {num_classes}")
    print(f"  Training samples    : {train_gen.samples}")
    print(f"  Validation samples  : {val_gen.samples}")

    # 4. Build VGG16-Fingerprint model
    model = build_vgg16_fingerprint(
        num_classes  = num_classes,
        dense_units  = cfg["DENSE_UNITS"],
        dropout_rate = cfg["DROPOUT"],
    )
    model.summary()

    # 5. Train (Phase A + Phase B)
    hist_a, hist_b = train(model, train_gen, val_gen, cfg)
    import json

    import json
    with open("results/hist_a.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist_a.items()}, f)
    with open("results/hist_b.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist_b.items()}, f)
    # 6. Plot training curves
    plot_history(hist_a, hist_b)

    # 7. Save final model
    model.save(cfg["MODEL_PATH"])
    print(f"\n  Final model saved → {cfg['MODEL_PATH']}")

    # 8. Evaluate
    evaluate(model, val_gen)

    # 9. Extract encoder for Siamese Stage 2
    extract_encoder(model, cfg["ENCODER_PATH"])

    print("\n" + "═"*55)
    print("  ✅  Stage 1 complete!")
    print("  The encoder is ready for the Siamese network (Stage 2)")
    print("═"*55 + "\n")


if __name__ == "__main__":
    main()
