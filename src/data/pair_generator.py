"""
Pair generation for Siamese network training.

Labels:
    0 → Similar   (same subject)
    1 → Not Similar (different subjects)
"""

import itertools
import random

import numpy as np


def generate_pairs(dataset: dict, num_pairs: int, seed: int = 42):
    """
    Generate balanced positive/negative image pairs.

    Args:
        dataset:   { subject_name: [img_array, ...] }
        num_pairs: Total pairs to generate (half positive, half negative).
        seed:      Random seed for reproducibility.

    Returns:
        img1_arr, img2_arr, lbl_arr  (numpy arrays)
        label 0 = similar, label 1 = not similar
    """
    random.seed(seed)
    np.random.seed(seed)

    subjects = list(dataset.keys())
    half = num_pairs // 2
    img1_list, img2_list, labels = [], [], []

    # ── Positive pairs (same subject) ────────────────────────────────────────
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
        selected_pos = (all_pos * ((half // len(all_pos)) + 1))[:half]

    for a, b in selected_pos:
        img1_list.append(a)
        img2_list.append(b)
        labels.append(0)

    # ── Negative pairs (different subjects) ──────────────────────────────────
    neg_count = 0
    while neg_count < half:
        s1, s2 = random.sample(subjects, 2)
        img1_list.append(random.choice(dataset[s1]))
        img2_list.append(random.choice(dataset[s2]))
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
    print(f"  Generated {len(lbl_arr)} pairs: {pos} similar + {neg} not-similar")
    return img1_arr, img2_arr, lbl_arr


def split_pairs(img1: np.ndarray, img2: np.ndarray,
                labels: np.ndarray, val_split: float = 0.2):
    """
    Split pairs into train and validation sets.

    Returns:
        X_train, y_train, X_val, y_val
        where X_train = [img1_train, img2_train]
    """
    n = len(labels)
    split = int(n * (1 - val_split))

    X_train = [img1[:split],  img2[:split]]
    y_train = labels[:split]
    X_val   = [img1[split:],  img2[split:]]
    y_val   = labels[split:]

    return X_train, y_train, X_val, y_val
