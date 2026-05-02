# 🛌 Sleep Stage Classification from ECG (Single-Channel)

A deep learning pipeline that automatically classifies **5 sleep stages** (Wake, N1, N2, N3, REM) using only a **single-channel ECG (EKG) signal** — no EEG required. Built with PyTorch and trained on the [MESA Sleep Study](https://sleepdata.org/datasets/mesa) dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Overview

Traditional sleep staging relies on EEG signals, which require clinical-grade equipment and controlled environments. This project demonstrates that **ECG alone** — a far more accessible signal — can be used to classify sleep stages by capturing autonomic nervous system (ANS) activity patterns:

| Stage | ANS Signature |
|-------|--------------|
| Wake  | High HR, low HF-HRV |
| N1/N2 | Slowing HR, increasing HF-HRV |
| N3    | Lowest HR, highest HF-HRV (parasympathetic dominance) |
| REM   | Irregular HR, low HF-HRV (similar to Wake) |

---

## Dataset

**MESA (Multi-Ethnic Study of Atherosclerosis) Sleep Study**

- Format: EDF files (signals) + XML files (annotations)
- Target channel: `EKG` (exact match only — `EKG_Off`, `EKG2`, etc. are excluded)
- Epoch length: 30 seconds
- Target sampling rate: 100 Hz (resampled if needed)
- Sleep stages: Wake (0), N1 (1), N2 (2), N3 (3), REM (4)

To use this project, place your files as follows:
```
project/
├── Signals/       # .edf files
└── Annotations/   # .xml files
```

---

## Pipeline

```
EDF + XML Files
      │
      ▼
1. Load & Preprocess
   • Pick EKG channel
   • Resample to 100 Hz
   • Bandpass filter: 0.5–40 Hz
   • Slice into 30-s epochs
      │
      ▼
2. Label Extraction
   • Parse XML annotations
   • Map stages to integers (0–4)
      │
      ▼
3. Train/Val/Test Split
   • 70% Train / 15% Val / 15% Test
   • Stratified by class
      │
      ▼
4. Class Balancing
   • N1 → augmentation + Mixup
   • N3, REM → ECG augmentation
      │
      ▼
5. Feature Engineering
   • 1 ECG channel + 5 spectral band channels
   • Stored via memory-mapped arrays (no OOM)
      │
      ▼
6. Model Training
   • SequenceSleepModel (CNN + Transformer)
   • Focal Loss + Label Smoothing
   • Warmup Cosine LR Scheduler
      │
      ▼
7. Evaluation
   • Accuracy, Macro-F1, Cohen's Kappa
   • Confusion Matrix
   • Hypnogram visualization
   • ROC curves (One-vs-Rest, all 3 splits)
```

---

## Model Architecture

### `SequenceSleepModel`

Processes a **sequence of 9 consecutive 30-second epochs** (4.5 minutes of context) and predicts the label for the **center epoch**. This captures the temporal context that distinguishes sleep stages.

```
Input: (Batch, 9 epochs, 6 channels, 3000 samples)
         │
         ▼
   EpochEncoder (per epoch)
   ┌─────────────────────────────┐
   │  Conv1D input projection    │
   │  MultiScaleCNN (3/9/25 kernels) │
   │  ResidualBlock1D × 4        │
   │  Transformer Encoder (2L)   │
   │  Attention Pooling          │
   └─────────────────────────────┘
         │
         ▼
   Positional Embeddings
         │
         ▼
   Sequence Transformer (2 layers, 8 heads)
         │
         ▼
   Classifier MLP → 5 classes
```

**Key components:**
- **MultiScaleCNN**: Parallel convolutions with kernel sizes 3, 9, and 25 to capture short- and long-range ECG patterns
- **ResidualBlock1D**: Identity shortcut connections with BatchNorm + ELU
- **Attention Pooling**: Learnable soft attention over time steps
- **Dual Transformer**: One for intra-epoch context, one for inter-epoch sequence modeling

Total parameters: ~several million (printed at runtime)

---

## Training Strategy

### Class Imbalance Handling
- **N1** (most underrepresented): ECG augmentation + Mixup (blending two N1 epochs)
- **N3, REM**: ECG augmentation only
- Boosted class weights: `Wake×1.0, N1×5.0, N2×1.0, N3×3.5, REM×2.5`

### ECG Augmentations
- Gaussian noise injection
- Amplitude scaling (±20%)
- Baseline wander simulation
- Random time shift (±150 samples)

### Loss Function
Combined loss for robustness:
```
Loss = 0.7 × FocalLoss(γ=2.5) + 0.3 × CrossEntropy(label_smoothing=0.1)
```

### Optimizer & Scheduler
- **AdamW** — `lr=8e-5`, `weight_decay=1e-2`
- **Warmup Cosine Scheduler** — 8 warmup epochs, cosine decay to `1e-6`
- **Early stopping** — patience of 20 epochs on Cohen's Kappa

### Regularization
- Dropout throughout: 0.1–0.55 (heavier in the classifier head)
- Gradient clipping: `max_norm=1.0`
- On-the-fly batch noise (20% of batches)

---

## Results

The model is evaluated on the held-out test set using:

- **Accuracy** — overall epoch-level accuracy
- **Macro F1-score** — unweighted mean F1 across all 5 classes
- **Cohen's Kappa (κ)** — agreement corrected for chance (clinical standard for PSG)
- **ROC-AUC** — One-vs-Rest, reported per class for Train/Val/Test
- **Hypnogram** — visual comparison of true vs. predicted sleep architecture

---

## Requirements

```bash
pip install mne scipy scikit-learn tqdm torch torchvision
```

| Library | Purpose |
|---------|---------|
| `mne` | EDF file loading |
| `scipy` | Signal filtering, STFT |
| `scikit-learn` | Metrics, train/test split, class weights |
| `torch` | Model definition and training |
| `numpy` | Array operations, memory-mapped files |
| `tqdm` | Progress bars |
| `matplotlib` | Visualization |

> **Note:** This notebook is designed to run on **Google Colab** with GPU support. It uses Google Drive for dataset storage and checkpoint saving.

---

## Usage

1. **Open in Google Colab**  
   Upload the notebook or open directly from Google Drive.

2. **Mount Google Drive** (Cell 2)  
   Update `SIGNAL_DIR` and `ANNOTATION_DIR` to point to your EDF and XML files.

3. **Run cells sequentially**  
   - Cells 1–6: Data loading and preprocessing  
   - Cell 7–8: Visualization and splitting  
   - Cell 9–10: Augmentation and feature extraction  
   - Cell 11–12: Model definition and DataLoaders  
   - Cell 13–14: Training  
   - Cell 15: Evaluation and visualization  

4. **Recovery**  
   If your Colab session restarts, use the **Recovery Cell** (between Cells 8 and 9) to reload arrays from Drive and resume from Cell 9.

---

## Project Structure

```
📁 Repository
├── Sleepstages_classification_project.ipynb   # Main notebook
└── README.md

📁 Google Drive (generated during run)
└── MESA/
    ├── chunk_X_*.npy          # Intermediate per-subject chunks
    ├── chunk_y_*.npy
    ├── X_all_ecg.npy          # Full merged ECG array
    ├── y_all_ecg.npy
    ├── X_train_aug.npy        # Augmented training set
    ├── X_val.npy
    ├── X_test.npy
    ├── y_train_aug.npy
    ├── y_val.npy
    ├── y_test.npy
    ├── X_train_sf.npy         # Spectral feature arrays (memmap)
    ├── X_val_sf.npy
    ├── X_test_sf.npy
    └── best_model_ecg.pt      # Best model checkpoint
```

---

## Notes

- The pipeline uses **memory-mapped NumPy arrays** (`np.memmap`) for spectral features to avoid out-of-memory errors on large datasets.
- Subject processing is done in **chunks of 5** to keep RAM usage low.
- The model uses a **sequence window of 9 epochs** (4.5 min) but only predicts the center epoch, giving it temporal context without data leakage.
- Mixup augmentation is applied **only to N1** — the hardest class to distinguish — with `alpha=0.4` for moderate blending.

---

## License

This project is for academic and research purposes. Please refer to the [MESA dataset usage agreement](https://sleepdata.org/datasets/mesa) for data licensing terms.
