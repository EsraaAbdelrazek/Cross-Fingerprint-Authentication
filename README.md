# Cross-Fingerprint Authentication

A two-stage deep learning pipeline for fingerprint authentication applied to the **PolyU Cross-Fingerprint Database**.  
Based on [Ezz et al., CSSE 2023](https://doi.org/10.32604/csse.2023.036567).

---

## Overview

The pipeline has two stages:

- **Stage 1 — Backbone Training:** A pre-trained CNN (VGG16, ResNet50, or EfficientNetB0) is fine-tuned on fingerprint classification using a two-phase progressive unfreezing strategy. The trained model is then cut at the embedding layer to produce an encoder.
- **Stage 2 — Siamese Authentication:** The frozen encoder feeds two parallel branches. Euclidean distance and Element-wise Absolute Difference (EAD) are concatenated and passed through a classifier head to predict whether two fingerprints belong to the same subject.

---

## Dataset

**PolyU Contactless 2D to Contact-based Fingerprint Database** (Lin & Kumar, 2018)

| Property | Details |
|---|---|
| Subjects | 336 total, 300 used for single-source experiments |
| Sessions | 2 sessions, 6 impressions per subject per modality |
| Modalities | Contact-based (328×356 px) + Contactless 2D (1400×900 px) |
| Total images | 5,952 across both sessions and modalities |
| Input size | Resized to 224×224 for all models |

> The dataset is not included in this repository. It is available for research purposes from [The Hong Kong Polytechnic University](https://www4.comp.polyu.edu.hk/~csajaykr/myhome/database_request/hkpu/index.html).

---

## Results

### Stage 1 — Backbone Training (val accuracy)

| Backbone | Phase A | Phase B | Unfreeze Layer |
|---|---|---|---|
| VGG16 | 67.0% | **75.3%** | `block5_conv3` |
| EfficientNetB0 | 71.3% | **72.7%** | `top_conv` |
| ResNet50 | 13.0% | 14.7% | `conv5_block3_out` |

### Stage 2 — Siamese Network (VGG16 encoder)

| Metric | Value |
|---|---|
| Test Accuracy | **95.95%** |
| Equal Error Rate (EER) | **4.05%** |
| Precision / Recall | 96% / 96% |
| F1-score | 0.96 |
| AUC | 0.9887 |

---

## Project Structure

```
.
├── configs/                        # Experiment configuration files
│   ├── defaults.yaml               # Default hyperparameters
│   ├── polyu_vgg16_ead.yaml
│   ├── polyu_resnet50_ead.yaml
│   ├── polyu_efficientnet_ead.yaml
│   ├── polyu_all_sources_resnet50_ead.yaml   # Multi-source + 5-fold CV
│   └── polyu_vgg16_triplet.yaml
│
├── scripts/
│   ├── train.py                    # Main training entry point
│   └── evaluate.py                 # Evaluation entry point
│
├── src/
│   ├── config/                     # Config schema and loader
│   ├── data/
│   │   ├── base_dataset.py
│   │   ├── polyu_dataset.py        # PolyU dataset handler (single + multi-source)
│   │   └── pair_generator.py       # Siamese pair generation
│   ├── models/
│   │   ├── backbones/
│   │   │   ├── vgg16.py
│   │   │   ├── resnet50.py
│   │   │   ├── efficientnet.py
│   │   │   ├── mobilenetv2.py
│   │   │   └── vit.py
│   │   ├── siamese_heads/
│   │   │   ├── ead_head.py         # Euclidean + EAD head
│   │   │   └── triplet_head.py     # Triplet loss head (stub)
│   │   ├── backbone_registry.py
│   │   └── encoder_utils.py
│   ├── training/
│   │   ├── backbone_trainer.py     # Phase A + Phase B training logic
│   │   ├── siamese_trainer.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── backbone_evaluator.py
│   │   ├── siamese_evaluator.py
│   │   └── metrics.py              # EER, AUC, FAR/FRR
│   ├── visualization/
│   │   ├── backbone_plots.py
│   │   ├── siamese_plots.py
│   │   └── pipeline_plots.py
│   └── utils/
│       ├── config_loader.py
│       ├── experiment.py
│       └── history_io.py
│
└── experiments/                    # Auto-generated (gitignored)
    └── <experiment_name>/
        ├── checkpoints/            # .keras model files
        ├── histories/              # Training JSON logs
        ├── plots/                  # Training curve PNGs
        └── config.yaml             # Config snapshot
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/EsraaAbdelrazek/Cross-Fingerprint-Authentication.git
cd Cross-Fingerprint-Authentication

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, TensorFlow 2.13+, OpenCV, NumPy, scikit-learn, Matplotlib, PyYAML

---

## Usage

### Train Stage 1 (backbone)

```bash
python scripts/train.py --config configs/polyu_vgg16_ead.yaml --stage 1
python scripts/train.py --config configs/polyu_efficientnet_ead.yaml --stage 1
python scripts/train.py --config configs/polyu_resnet50_ead.yaml --stage 1
```

### Train Stage 2 (Siamese head)

```bash
python scripts/train.py --config configs/polyu_vgg16_ead.yaml --stage 2
```

### Train both stages

```bash
python scripts/train.py --config configs/polyu_vgg16_ead.yaml
```

### Evaluate

```bash
python scripts/evaluate.py --config configs/polyu_vgg16_ead.yaml
```

---

## Configuration

Each experiment is defined by a YAML config that overrides `configs/defaults.yaml`. Example:

```yaml
experiment:
  name: polyu_vgg16_ead

dataset:
  name: polyu
  source_dir: contact-based_fingerprints/first_session
  num_subjects: 300

backbone:
  name: vgg16
  dense_units: 1024
  dropout: 0.5

training:
  stage1:
    phase_a:
      epochs: 50
      lr: 3.0e-4
    phase_b:
      epochs: 30
      lr: 5.0e-5
      unfreeze_layer: block5_conv3
  stage2:
    epochs: 50
    num_pairs: 6000

siamese_head:
  name: ead
```

---

## Methodology

### Stage 1 — Two-Phase Progressive Unfreezing

- **Phase A:** All backbone layers frozen. Only the classification head is trained.
- **Phase B:** The final convolutional block is unfrozen for domain-specific fine-tuning at a lower learning rate.
- EarlyStopping (patience=8) and ReduceLROnPlateau are applied throughout.

### Stage 2 — EAD Siamese Head

Two encoder branches (shared weights, frozen) process an anchor and a pair image:
- **Euclidean distance** — scalar similarity measure
- **Element-wise Absolute Difference (EAD)** — full-dimensional difference vector

Both are concatenated and passed through `Dense(512) → Dropout → Dense(256) → Dropout → Sigmoid`.  
Training uses 6,000 balanced pairs (3,000 same-subject, 3,000 different-subject).

---

## References

- Ezz et al. (2023). *Improved Siamese Palmprint Authentication Using Pre-Trained VGG16-Palmprint and EAD.* CSSE, 46(2). [DOI: 10.32604/csse.2023.036567](https://doi.org/10.32604/csse.2023.036567)
- Lin, C. & Kumar, A. (2018). *Matching Contactless and Contact-based Fingerprint Images.* IEEE Trans. Image Processing, 27(4), 2008–2021.
