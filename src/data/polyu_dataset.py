"""
PolyU Contactless 2D to Contact-based Fingerprint Dataset.

Supports two raw layouts:
  flat   — X_Y.jpg files directly in the folder (contact-based)
  nested — pX/pY.bmp subfolder structure (processed contactless)

After reorganize():  dataset/fingerprints_organized/subject_001/ ...
After preprocess():  dataset/fingerprints_224/subject_001/ ...
"""

import shutil
import cv2
from collections import defaultdict
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.data.base_dataset import BaseDataset


class PolyUDataset(BaseDataset):

    # ── Format detection ──────────────────────────────────────────────────────

    def detect_format(self, source_dir: str) -> str:
        """Returns 'flat', 'nested', or raises ValueError."""
        source = Path(source_dir)
        if list(source.glob("*.jpg")) or list(source.glob("*.bmp")):
            return "flat"
        if any(d.is_dir() for d in source.iterdir()):
            return "nested"
        raise ValueError(
            f"Cannot detect dataset format in '{source_dir}'. "
            "Expected flat files (1_1.jpg) or nested folders (p001/p1.bmp)."
        )

    # ── Reorganize ────────────────────────────────────────────────────────────

    def reorganize(self, source_dir: str, out_dir: str, num_subjects: int) -> None:
        """Auto-detect format and reorganize into subject subfolders."""
        if Path(out_dir).exists() and any(Path(out_dir).iterdir()):
            print(f"  [Reorganize] Already done: {out_dir}")
            return
        fmt = self.detect_format(source_dir)
        print(f"  [Reorganize] Detected format: {fmt}")
        if fmt == "flat":
            self._reorganize_flat(source_dir, out_dir, num_subjects)
        else:
            self._reorganize_nested(source_dir, out_dir, num_subjects)

    def _reorganize_flat(self, source_dir: str, out_dir: str,
                         num_subjects: int) -> None:
        """Flat X_Y.jpg → subject_XXX/X_Y.jpg"""
        source = Path(source_dir)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        subject_files = defaultdict(list)
        for f in sorted(source.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".bmp", ".png"}:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    subject_files[int(parts[0])].append(f)

        subject_ids = sorted(subject_files)[:num_subjects]
        print(f"  Reorganizing {len(subject_ids)} subjects (flat format) ...")

        for sid in subject_ids:
            subject_out = out / f"subject_{sid:03d}"
            subject_out.mkdir(exist_ok=True)
            for src_file in subject_files[sid]:
                dst = subject_out / src_file.name
                if not dst.exists():
                    shutil.copy2(src_file, dst)

        total = sum(len(subject_files[sid]) for sid in subject_ids)
        print(f"  Done. {total} images across {len(subject_ids)} subjects.")

    def _reorganize_nested(self, source_dir: str, out_dir: str,
                            num_subjects: int) -> None:
        """Nested pX/pY.bmp → subject_XXX/pY.bmp"""
        source = Path(source_dir)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        subject_dirs = sorted(d for d in source.iterdir() if d.is_dir())[:num_subjects]
        print(f"  Reorganizing {len(subject_dirs)} subjects (nested format) ...")

        for i, sdir in enumerate(subject_dirs, 1):
            subject_out = out / f"subject_{i:03d}"
            subject_out.mkdir(exist_ok=True)
            for img in sdir.glob("*"):
                dst = subject_out / img.name
                if not dst.exists():
                    shutil.copy2(img, dst)

        print(f"  Done. {len(subject_dirs)} subjects reorganized.")

    # ── Preprocess ────────────────────────────────────────────────────────────

    def preprocess(self, src_dir: str, dst_dir: str,
                   img_size: tuple = (224, 224)) -> None:
        """Resize all images to img_size using OpenCV INTER_AREA."""
        if Path(dst_dir).exists() and any(Path(dst_dir).iterdir()):
            print(f"  [Preprocess] Already done: {dst_dir}")
            return

        src = Path(src_dir)
        dst = Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        total = 0

        print(f"  Preprocessing: {src_dir} → {dst_dir}  ({img_size[0]}×{img_size[1]})")

        for subject_dir in sorted(d for d in src.iterdir() if d.is_dir()):
            out_subject = dst / subject_dir.name
            out_subject.mkdir(exist_ok=True)

            for img_path in subject_dir.iterdir():
                if img_path.suffix.lower() not in supported:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  [WARN] Skipping unreadable: {img_path}")
                    continue
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
                out_path = out_subject / (img_path.stem + ".jpg")
                cv2.imwrite(str(out_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                total += 1

        print(f"  Total preprocessed: {total} images.")

    # ── Load images into memory ───────────────────────────────────────────────

    def load_images(self, processed_dir: str, img_size: tuple) -> dict:
        """
        Load all images into a dict: { subject_name: [img_array, ...] }
        Arrays are normalized to [0, 1].
        """
        processed_dir = Path(processed_dir)
        supported = {".jpg", ".jpeg", ".png", ".bmp"}
        dataset = {}

        subject_dirs = sorted(d for d in processed_dir.iterdir() if d.is_dir())
        print(f"\n  Loading {len(subject_dirs)} subjects from {processed_dir} ...")

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

        total = sum(len(v) for v in dataset.values())
        print(f"  Loaded {total} images across {len(dataset)} subjects "
              f"({total // len(dataset)} per subject).")
        return dataset

    # ── Keras generators (Stage 1) ────────────────────────────────────────────

    def create_generators(self, processed_dir: str, img_size: tuple,
                          batch_size: int, val_split: float):
        """
        Create augmented train generator + clean val generator for Stage 1.
        Returns: (train_gen, val_gen)
        """
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
