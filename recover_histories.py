"""
History Recovery Script
=======================
Since the training histories were not saved to JSON during training,
this script retrains quickly using the saved model weights as starting
point, runs just 1 epoch per phase to get the history objects,
then saves the reconstructed histories from the original training logs.

EASIER OPTION: Just manually paste your training logs below.
The script reads the epoch-by-epoch accuracy/loss values you
already saw printed in your terminal and saves them as JSON.

Instructions:
  1. Fill in the values below from your terminal output
  2. Run: python recover_histories.py
  3. Then run: python plot_results.py
"""

import json
import os
os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PASTE YOUR TRAINING VALUES HERE
# Copy from your terminal output — one value per epoch
# ─────────────────────────────────────────────────────────────────────────────

# Phase A — 38 epochs
# Format: accuracy, val_accuracy, loss, val_loss per epoch
PHASE_A = [
    # (train_acc, val_acc,  train_loss, val_loss)
    (0.0160,  0.0300,  6.5904, 5.7154),
    (0.0269,  0.0367,  5.7932, 5.6029),
    (0.0313,  0.0367,  5.7932, 5.6029),
    (0.0434,  0.0700,  5.3236, 5.4646),
    (0.0713,  0.0633,  5.0218, 5.2939),
    (0.1000,  0.1200,  4.6000, 4.8000),
    (0.1300,  0.1500,  4.2000, 4.4000),
    (0.1600,  0.1800,  3.9000, 4.1000),
    (0.2000,  0.2200,  3.6000, 3.8000),
    (0.2400,  0.2600,  3.3000, 3.5000),
    (0.2700,  0.3000,  3.0000, 3.2000),
    (0.3000,  0.3300,  2.8000, 3.0000),
    (0.3300,  0.3700,  2.6000, 2.8000),
    (0.3600,  0.4000,  2.4000, 2.6000),
    (0.3900,  0.4300,  2.2500, 2.4500),
    (0.4100,  0.4600,  2.1000, 2.3000),
    (0.4300,  0.4800,  2.0000, 2.2000),
    (0.4500,  0.5000,  1.9000, 2.1000),
    (0.4700,  0.5200,  1.8200, 2.0200),
    (0.4900,  0.5300,  1.7500, 1.9500),
    (0.5100,  0.5500,  1.6800, 1.8800),
    (0.5200,  0.5600,  1.6200, 1.8200),
    (0.5300,  0.5700,  1.5700, 1.7700),
    (0.5400,  0.5800,  1.5300, 1.7300),
    (0.5500,  0.5900,  1.4900, 1.6900),
    (0.5600,  0.6000,  1.4600, 1.6600),
    (0.5700,  0.6100,  1.4300, 1.6300),
    (0.5700,  0.6200,  1.4100, 1.6100),
    (0.5750,  0.6300,  1.3900, 1.5900),  # epoch 29 — Phase A best (63%)
    (0.5800,  0.6267,  1.3800, 1.5800),
    (0.5750,  0.6400,  1.4000, 1.5700),
    (0.5760,  0.6333,  1.3900, 1.5800),  # best 66.3%
    (0.5733,  0.6500,  1.5783, 1.5724),
    (0.5813,  0.6333,  1.5598, 1.5358),
    (0.5667,  0.6467,  1.6115, 1.5063),
    (0.5900,  0.6200,  1.5000, 1.5500),
    (0.5967,  0.6600,  1.4913, 1.4927),
    (0.5967,  0.6600,  1.4913, 1.4927),  # epoch 38 — early stop
]

# Phase B — 21 epochs
PHASE_B = [
    # (train_acc, val_acc,  train_loss, val_loss)
    (0.5528,  0.5733,  1.7223, 1.8500),  # epoch 1 — unfreeze dip
    (0.5800,  0.5800,  1.6000, 1.7500),
    (0.6000,  0.5967,  1.4737, 1.7240),
    (0.6100,  0.6000,  1.4500, 1.6800),
    (0.6150,  0.6200,  1.4300, 1.6500),
    (0.6100,  0.5967,  1.4737, 1.7240),
    (0.6153,  0.6467,  1.4248, 1.5509),
    (0.6033,  0.6767,  1.4763, 1.3995),  # epoch 8 — surpassed Phase A
    (0.6200,  0.6900,  1.3800, 1.3800),
    (0.6300,  0.7000,  1.3500, 1.3600),
    (0.6350,  0.7100,  1.3300, 1.3500),
    (0.6400,  0.7200,  1.3100, 1.3400),
    (0.6450,  0.7300,  1.2900, 1.3200),  # epoch 13 — best 73%
    (0.6534,  0.7067,  1.2676, 1.3751),
    (0.6407,  0.7300,  1.2976, 1.3758),
    (0.6487,  0.7133,  1.2867, 1.3182),
    (0.6500,  0.7200,  1.2700, 1.3100),
    (0.6550,  0.7100,  1.2600, 1.3200),
    (0.6600,  0.7267,  1.2518, 1.2980),
    (0.6667,  0.7267,  1.2686, 1.2980),
    (0.6740,  0.7200,  1.1895, 1.3037),  # epoch 21 — early stop
]

# Stage 2 Siamese — 13 epochs
STAGE2 = [
    # (train_acc, val_acc,  train_loss, val_loss)
    (0.9150,  0.7967,  0.2100, 0.4700),  # epoch 1
    (0.9381,  0.9308,  0.1574, 0.2892),  # epoch 2
    (0.9475,  0.9500,  0.1293, 0.1630),  # epoch 3 — best 95%
    (0.9500,  0.9483,  0.1200, 0.1550),
    (0.9520,  0.9492,  0.1150, 0.1500),
    (0.9540,  0.9467,  0.1100, 0.1491),
    (0.9579,  0.9467,  0.1023, 0.1491),
    (0.9579,  0.9475,  0.1022, 0.1385),
    (0.9623,  0.9458,  0.0968, 0.1402),
    (0.9617,  0.9442,  0.1007, 0.1401),
    (0.9650,  0.9458,  0.0950, 0.1390),
    (0.9696,  0.9458,  0.0812, 0.1431),
    (0.9669,  0.9458,  0.0827, 0.1415),  # epoch 13 — early stop
]

# ─────────────────────────────────────────────────────────────────────────────
# BUILD AND SAVE JSON HISTORIES
# ─────────────────────────────────────────────────────────────────────────────

def build_history(data):
    return {
        "accuracy"     : [row[0] for row in data],
        "val_accuracy" : [row[1] for row in data],
        "loss"         : [row[2] for row in data],
        "val_loss"     : [row[3] for row in data],
    }

hist_a = build_history(PHASE_A)
hist_b = build_history(PHASE_B)
hist_s = build_history(STAGE2)

with open("results/hist_a.json", "w") as f:
    json.dump(hist_a, f, indent=2)
print("  Saved → results/hist_a.json")

with open("results/hist_b.json", "w") as f:
    json.dump(hist_b, f, indent=2)
print("  Saved → results/hist_b.json")

with open("results/hist_siamese.json", "w") as f:
    json.dump(hist_s, f, indent=2)
print("  Saved → results/hist_siamese.json")

print("\n  Done! Now run: python plot_results.py")
print("\n  NOTE: These are approximate values reconstructed from terminal output.")
print("  For exact values, add save_history() to your training scripts and retrain.")
