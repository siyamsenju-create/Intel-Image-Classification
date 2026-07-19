# 🖼️ Intel Image Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59-red?logo=streamlit&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.15-darkred?logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-blue?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**An end-to-end deep learning system for classifying natural scenes using Transfer Learning (MobileNetV2), with an interactive Streamlit web application.**

[🚀 Run Locally](#-running-locally) • [📊 Results](#-model-performance) • [🗂️ Project Structure](#️-project-structure) • [📖 How to Use](#-how-to-use)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Model Performance](#-model-performance)
- [Project Structure](#️-project-structure)
- [Installation](#-installation)
- [Running Locally](#-running-locally)
- [How to Use](#-how-to-use)
- [Training the Model](#-training-the-model)
- [Evaluating the Model](#-evaluating-the-model)
- [Running Inference](#-running-inference)
- [Streamlit App Features](#-streamlit-app-features)
- [Results](#-results)
- [Technologies Used](#-technologies-used)

---

## 🌟 Overview

This project builds a complete image classification pipeline for Intel's Natural Scene Classification dataset. It classifies images of natural scenes into **6 categories** using a pretrained **MobileNetV2** model via Transfer Learning.

The project includes:
- ✅ A fully automated **ML training & evaluation pipeline**
- ✅ A **premium Streamlit web application** for interactive predictions
- ✅ Detailed **performance metrics** and visualizations (confusion matrix, training curves, classification report)
- ✅ Clean, modular codebase organized by responsibility

---

## 🗃️ Dataset

The dataset used is the **[Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)** from Kaggle, originally released for a data science challenge by Intel.

| Split | Images | Description |
|---|---|---|
| Train | ~14,000 | Labelled images for model training |
| Test | 3,000 | Labelled images for evaluation |
| Pred | ~7,300 | Unlabelled images for prediction |

### Classes

| Label | Scene |
|---|---|
| 🏢 `buildings` | Urban buildings and architecture |
| 🌲 `forest` | Dense tree forests |
| 🧊 `glacier` | Icy glacial landscapes |
| ⛰️ `mountain` | Mountain ranges and peaks |
| 🌊 `sea` | Ocean and coastal scenes |
| 🛣️ `street` | Roads, streets and city views |

All images are **150 × 150 pixels** in RGB format.

> **Note:** The `data/` directory is excluded from this repository (listed in `.gitignore`) because of its size (~350 MB). Follow the [Installation](#-installation) steps to download it automatically.

---

## 🧠 Model Architecture

We use **Transfer Learning** with **MobileNetV2** pretrained on ImageNet as a frozen feature extractor, with a custom classification head.

```
Input (150×150×3)
    ↓
Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
    ↓
Rescaling [-1, 1]
    ↓
MobileNetV2 Base (frozen, 2,257,984 params)
    ↓
Global Average Pooling 2D
    ↓
Dropout (0.3)
    ↓
Dense (6 units, Softmax)
```

| Parameter | Value |
|---|---|
| Base Model | MobileNetV2 (ImageNet weights, frozen) |
| Input Size | 150 × 150 × 3 |
| Trainable Params | 7,686 |
| Total Params | 2,281,044 |
| Optimizer | Adam (lr=1e-3) |
| Loss | Categorical Crossentropy |

---

## 📊 Model Performance

The model was trained for **3 epochs** (early stopping saved the best checkpoint at **epoch 3**).

| Metric | Value |
|---|---|
| ✅ **Test Accuracy** | **88.27%** |
| 📉 Test Loss | 0.3247 |
| 🎯 Validation Accuracy | 88.52% |

### Per-Class F1-Scores

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 🌲 Forest | 0.99 | 0.98 | **0.98** | 474 |
| 🛣️ Street | 0.95 | 0.86 | **0.90** | 501 |
| 🌊 Sea | 0.85 | 0.96 | **0.91** | 510 |
| 🏢 Buildings | 0.86 | 0.94 | **0.89** | 437 |
| ⛰️ Mountain | 0.82 | 0.82 | **0.82** | 525 |
| 🧊 Glacier | 0.85 | 0.76 | **0.80** | 553 |
| **Weighted Avg** | **0.88** | **0.88** | **0.88** | **3000** |

> Glacier had the lowest recall (76%) as it shares visual features with sea and mountain scenes. Fine-tuning deeper layers would improve this further.

---

## 🗂️ Project Structure

```
Intel-Image-Classification/
│
├── app.py                    # 🌐 Streamlit web application (interactive UI)
│
├── src/                      # 🧩 Core ML pipeline
│   ├── preprocess.py         # Data loading & augmentation (tf.data)
│   ├── model.py              # MobileNetV2 transfer learning model definition
│   ├── train.py              # Training loop with callbacks & history plots
│   ├── evaluate.py           # Test set evaluation, metrics, confusion matrix
│   └── predict.py            # Single image inference utility
│
├── data/                     # 📁 Dataset (excluded from git, download via Kaggle)
│   ├── train/                # Training images (6 class subdirectories)
│   ├── test/                 # Test images (6 class subdirectories)
│   └── pred/                 # Unlabelled prediction images
│
├── models/                   # 💾 Saved model checkpoints (excluded from git)
│   └── best_model.keras      # Best model saved during training
│
├── results/                  # 📈 Evaluation outputs (committed to git)
│   ├── class_names.json      # Class label mapping
│   ├── metrics.json          # Test accuracy, loss & classification report
│   ├── classification_report.txt
│   ├── confusion_matrix.png  # Heatmap of predictions vs actuals
│   └── training_curves.png   # Accuracy & loss curves over epochs
│
├── .vscode/settings.json     # VS Code interpreter config
├── pyrightconfig.json        # Pyright/Pyrefly type checker config
├── pyproject.toml            # Pyrefly linter config
├── requirements.txt          # Full dependency list
└── .gitignore
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- A Kaggle account (for downloading the dataset)

### 1. Clone the Repository

```bash
git clone https://github.com/siyamsenju-create/Intel-Image-Classification.git
cd Intel-Image-Classification
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Set up your Kaggle API token:

```bash
# Option A: Save token to file
mkdir -p ~/.kaggle && echo YOUR_KAGGLE_TOKEN > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

# Option B: Environment variable
export KAGGLE_API_TOKEN=YOUR_KAGGLE_TOKEN
```

Then download and extract the dataset:

```bash
kaggle datasets download -d puneet6060/intel-image-classification -p data/ --unzip
```

The `data/` folder should contain `train/`, `test/`, and `pred/` after extraction.

---

## 🚀 Running Locally

Start the Streamlit web application:

```bash
PYTHONPATH=. streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

---

## 📖 How to Use

### 🌐 Streamlit App

The app has 3 pages accessible from the sidebar:

#### 🏠 Dashboard & Predictor
- **Upload** any image (JPG, JPEG, PNG) from your computer
- Or **select a random sample** from the unlabelled `data/pred` folder
- See the **predicted scene class** with confidence percentage
- Visual **confidence score bars** for all 6 classes

#### 📊 Model Insights
- **Training curves** — Accuracy and loss over epochs
- **Test accuracy & loss** metrics cards
- **Per-class classification report** table (precision, recall, F1)
- **Confusion matrix heatmap** — see which classes get confused

#### ℹ️ Project Info
- Full dataset and methodology explanation

---

## 🏋️ Training the Model

To re-train the model from scratch:

```bash
PYTHONPATH=. python src/train.py --epochs 10 --batch_size 32
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `10` | Number of training epochs |
| `--batch_size` | `32` | Batch size for training |
| `--data_dir` | `data` | Path to dataset directory |
| `--model_path` | `models/best_model.keras` | Where to save best model |
| `--results_dir` | `results` | Directory for outputs |

Training features:
- **EarlyStopping** (patience=3, monitors val_accuracy)
- **ModelCheckpoint** (saves best model automatically)
- Generates **training curves** PNG to `results/`
- Saves **training history** JSON to `results/`

---

## 📐 Evaluating the Model

Run evaluation on the test set to generate metrics:

```bash
PYTHONPATH=. python src/evaluate.py
```

This generates:
- `results/metrics.json` — Test accuracy, loss, full classification report
- `results/classification_report.txt` — Per-class precision/recall/F1
- `results/confusion_matrix.png` — Heatmap visualization

---

## 🔍 Running Inference

Classify a single image from the command line:

```bash
PYTHONPATH=. python src/predict.py --image_path path/to/image.jpg
```

**Output example:**
```
Prediction Results:
------------------------------
Predicted Class: mountain
Confidence:      94.23%

Class Probabilities:
  mountain     : 94.23%
  glacier      : 3.71%
  sea          : 1.15%
  buildings    : 0.54%
  forest       : 0.24%
  street       : 0.13%
------------------------------
```

---

## ✨ Streamlit App Features

- **Premium Light Theme** — Clean off-white background with glassmorphism-inspired white cards
- **Dark Navy Sidebar** with color-coded navigation
- **Custom CSS** — Inter font, gradient metric cards, animated confidence bars
- **Model cached with `@st.cache_resource`** — loads only once per session for fast predictions
- **Hot-reload** — Code changes apply instantly without restarting

---

## 📈 Results

### Training Curves (3 Epochs)

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|---|---|---|---|---|
| 1 | 72.68% | 85.00% | 0.7500 | 0.4300 |
| 2 | 82.01% | 88.13% | 0.5100 | 0.3400 |
| 3 | **84.51%** | **88.52%** ✅ | 0.4358 | 0.3092 |

The model converged very quickly thanks to pretrained ImageNet weights in MobileNetV2.

---

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.21, Keras 3.15 |
| Base Model | MobileNetV2 (ImageNet pretrained) |
| Data Processing | NumPy, Pillow (PIL), OpenCV |
| Metrics | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit 1.59 |
| Dataset API | Kaggle Python SDK |

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">
Made with ❤️ using TensorFlow and Streamlit
</div>
