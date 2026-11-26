# Automated Skin Disease Classification Using Deep Learning

A deep learning project developed for the **5th Semester Curriculum** focusing on multi-class dermatoscopic lesion diagnosis using modern convolutional vision backbones, robust validation strategies, and test-time augmentation.

---

## 👨‍🎓 Authors & Academic Details

- **Anurag Saha** — Roll No: `24101106037`
- **Ayan Karak** — Roll No: `24101106013`

---

## 📊 Project Overview

Dermatological condition diagnosis from skin lesion imagery often suffers from high visual ambiguity and severe inter-class imbalance. This project implements an end-to-end computer vision pipeline trained on the **HAM10000** dataset to categorize skin lesions into seven distinct diagnostic classes.

### Target Classes:
1. **nv** — Melanocytic nevi
2. **mel** — Melanoma
3. **bkl** — Benign keratosis-like lesions
4. **bcc** — Basal cell carcinoma
5. **akiec** — Actinic keratoses and intraepithelial carcinoma
6. **vasc** — Vascular lesions
7. **df** — Dermatofibroma

---

## 🧠 Key Technical Highlights

- **Leak-Free Data Splitting:** Implemented `StratifiedGroupKFold` grouped by patient lesion ID (`lesion_id`). This guarantees that multiple images of the same lesion never cross between the training, validation, and test partitions.
- **Architecture:** Utilized `convnext_small.fb_in22k_ft_in1k` pretrained via `timm` with custom dropout (`p=0.3`) for robust feature representation.
- **Data Augmentation:** Configured an Albumentations pipeline including horizontal/vertical flips, 90-degree rotations, `ShiftScaleRotate`, and `ColorJitter`.
- **Optimization & Regularization:**
  - Label Smoothing Cross-Entropy Loss (`smoothing=0.1`) to prevent overconfident predictions.
  - AdamW optimizer combined with a `CosineAnnealingLR` scheduler.
  - Mixed-precision training via PyTorch AMP (`torch.amp.autocast` & `GradScaler`) for efficient GPU utilization.
- **Inference Enhancement:** Implemented Test-Time Augmentation (TTA) using horizontal and vertical flips during final evaluation.

---

## 📈 Experimental Results

### Overall Test Performance

| Metric | Score |
| :--- | :--- |
| **Overall Test Accuracy** | **87.98%** |
| **Balanced Accuracy** | **74.76%** |
| **Macro Average F1-Score** | **0.7534** |
| **Weighted Average F1-Score** | **0.8761** |

### Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **nv** | 0.9483 | 0.9649 | 0.9565 | 684 |
| **mel** | 0.7160 | 0.5524 | 0.6237 | 105 |
| **bkl** | 0.7518 | 0.7923 | 0.7715 | 130 |
| **bcc** | 0.6786 | 0.9744 | 0.8000 | 39 |
| **akiec** | 0.6667 | 0.5000 | 0.5714 | 28 |
| **vasc** | 0.7333 | 0.6875 | 0.7097 | 16 |
| **df** | 0.9412 | 0.7619 | 0.8421 | 21 |

### Performance Charts

#### Aggregate Metrics

![Aggregate performance metrics](Performance_Charts/chart_aggregate_metrics.png)

#### Per-Class Metrics

![Per-class precision, recall, F1-score, and support](Performance_Charts/chart_per_class_metrics.png)

#### Class Imbalance vs. F1-Score

![Relationship between class support and F1-score](Performance_Charts/chart_imbalance_vs_f1.png)

> Additional report and chart details are available in [`04_Performance_Chart_Analysis_Skin_Disease_Classification_JGEC.docx`](04_Performance_Chart_Analysis_Skin_Disease_Classification_JGEC.docx).

---

## 🛠️ Tech Stack & Requirements

- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch, Torchvision
- **Model Hub:** `timm` (PyTorch Image Models)
- **Image Processing & Augmentation:** Albumentations, OpenCV
- **Data Handling & Metrics:** NumPy, Pandas, Scikit-learn
- **Hardware:** NVIDIA GPU with CUDA support recommended for training

## 📂 Repository Structure

```text
├── skin.ipynb
├── 04_Performance_Chart_Analysis_Skin_Disease_Classification_JGEC.docx
├── Performance_Charts/
│   ├── chart_aggregate_metrics.png
│   ├── chart_per_class_metrics.png
│   └── chart_imbalance_vs_f1.png
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/sahaanurag207-oss/SKIN-DISEASE-CLASSIFICATION-FROM-IMAGES.git](https://github.com/sahaanurag207-oss/SKIN-DISEASE-CLASSIFICATION-FROM-IMAGES.git)
cd SKIN-DISEASE-CLASSIFICATION-FROM-IMAGES
```

### 2. Create a Virtual Environment (Recommended)

Using `venv`:
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux / macOS:
source venv/bin/activate
```

Or using `conda`:
```bash
conda create -n skin-disease python=3.10 -y
conda activate skin-disease
```

### 3. Install PyTorch (with CUDA Support)

Install PyTorch matching your system's CUDA version from the [official PyTorch website](https://pytorch.org/get-started/locally/). For example, for CUDA 12.1:

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

*(For CPU-only execution, run `pip install torch torchvision`.)*

### 4. Install Project Dependencies

```bash
pip install timm albumentations opencv-python pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 5. Dataset Preparation

1. Download the **HAM10000** dataset (`HAM10000_metadata.csv` and image folders `HAM10000_images_part_1` / `HAM10000_images_part_2`) from the [project dataset link](https://share.google/BKuGoG9tJ6k4HAIBd).
2. Extract the archive into a local `data/` directory inside the repository root (or update the dataset path variable in the first configuration cell of `skin.ipynb`).

### 6. Run the Notebook

Launch Jupyter Notebook or JupyterLab and execute the cells sequentially:

```bash
jupyter notebook skin.ipynb
```
