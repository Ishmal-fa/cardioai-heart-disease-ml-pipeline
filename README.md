# 🫀 Heartbeat to Heatmap

> Unsupervised Learning, Ensemble Methods, and Neural Networks on Heart Disease & Handwritten Digit Data

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Setup & Installation](#setup--installation)
- [Running the Notebook](#running-the-notebook)
- [Running the Dashboard](#running-the-dashboard)
- [Results Summary](#results-summary)
- [Assignment Coverage](#assignment-coverage)
- [References](#references)

---

## Project Overview

This project was built for **CardioAI Labs** — a fictional health-tech startup building clinical decision-support tools for community hospitals. The pipeline covers four complementary areas applied to real cardiology data:

| Part | Topic | Dataset |
|------|-------|---------|
| Pre  | Data Cleaning, Encoding, EDA | UCI Cleveland Heart Disease |
| A    | Unsupervised Learning (K-Means, Hierarchical, PCA, t-SNE) | UCI Cleveland |
| B    | Bagging & Boosting (Random Forest, XGBoost + SHAP) | UCI Cleveland |
| C    | Neural Networks (SLP, MLP, Ablation Study) | UCI Cleveland |
| D    | CNN on Handwritten Digits | MNIST (12k subset) |
| E    | Interactive Streamlit Dashboard | UCI Cleveland |

---

## Repository Structure

```
DS3002-Assignment4/
│
├── notebooks/
│   └── DS3002_Assignment4.ipynb      # Main notebook — all parts A–E
│
├── app/
│   ├── app.py                        # Streamlit dashboard (Part E)
│   ├── model.pkl                     # Saved XGBoost model
│   ├── scaler.pkl                    # Fitted StandardScaler
│   └── train_cols.pkl                # Feature column names
│
├── report/
│   └── DS3002_Assignment4_Report.docx
│
├── data/
│   └── processed.cleveland.data      # UCI Cleveland dataset (place here)
│
├── requirements.txt
└── README.md
```

---

## Datasets

### Dataset 1 — UCI Heart Disease (Cleveland)
Used for **Parts Pre, A, B, C, E**

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **File:** `processed.cleveland.data`
- **Size:** 303 rows × 14 columns (13 features + 1 target)
- **Task:** Binary classification — heart disease present (1) vs absent (0)

**Download steps:**
1. Visit https://archive.ics.uci.edu/dataset/45/heart+disease
2. Download `processed.cleveland.data`
3. Place it in the `data/` folder **or** update `DATA_PATH` in the notebook (Cell 4)

### Dataset 2 — MNIST Handwritten Digits
Used for **Part D (CNN only)**

No download needed — loaded automatically via Keras:
```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
Only the first **12,000 training** and **2,000 test** images are used.

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you are on a system where pip is externally managed (e.g. Ubuntu 24+), use:
> ```bash
> pip install -r requirements.txt --break-system-packages
> ```

### Verify installation
```python
import sklearn, xgboost, shap, tensorflow, streamlit
print("All packages OK")
```

---

## Running the Notebook

1. Clone or download this repository
2. Place `processed.cleveland.data` in the `data/` folder
3. Open the notebook:

```bash
# Jupyter Notebook
jupyter notebook notebooks/DS3002_Assignment4.ipynb

# OR Jupyter Lab
jupyter lab notebooks/DS3002_Assignment4.ipynb

# OR Google Colab
# Upload the .ipynb and the data file, then update DATA_PATH in Cell 4
```

4. **Run all cells top-to-bottom** (`Kernel → Restart & Run All`)

> ⏱️ **Estimated runtime:** Under 15 minutes on a standard laptop (4 GB RAM, no GPU required).
> The CNN section (Part D) takes 2–3 minutes on CPU.

### Random Seeds
All random seeds are fixed at **42** throughout:
```python
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```
Results are fully reproducible across runs.

---

## Running the Dashboard

The Part E Streamlit dashboard provides an interactive heart disease risk prediction interface.

### Step 1 — Generate model files
Run the full notebook at least once. This saves:
- `app/model.pkl` — trained XGBoost model
- `app/scaler.pkl` — fitted StandardScaler  
- `app/train_cols.pkl` — feature column names

### Step 2 — Launch the dashboard

```bash
streamlit run app/app.py
```

Then open your browser at **http://localhost:8501**

### Dashboard features
- 13 labelled clinical input fields with valid-range hints
- Pre-populated with a real test patient for instant demo
- Colour-coded risk result (🔴 High Risk / 🟢 Low Risk) with probability gauge
- SHAP waterfall chart showing top prediction drivers
- Plain-English clinical note for nursing staff

---

## Results Summary

### Heart Disease Classification (UCI Cleveland, n=297, 80/20 split)

| Model | Accuracy | Macro F1 | AUC-ROC | Disease Recall |
|-------|----------|----------|---------|----------------|
| SLP (linear) | 0.8167 | 0.8141 | 0.9531 | 0.750 |
| MLP (64→32) | 0.7000 | 0.6914 | 0.8906 | **0.929** |
| Random Forest | 0.8167 | 0.8141 | **0.9208** | 0.750 |
| **XGBoost** ✅ | **0.8500** | **0.8479** | 0.9107 | 0.786 |

> **Deployed model: XGBoost** — best accuracy and F1; SHAP-interpretable; runs in <1s on CPU.

### MNIST Digit Recognition (12k train / 2k test subset)

| Model | Test Accuracy | Macro F1 |
|-------|---------------|----------|
| MLP Baseline (Flatten→Dense64→Dense10) | 0.9110 | — |
| **Lightweight CNN** ✅ | **0.9805** | **0.9803** |

> CNN surpasses MLP baseline at **Epoch 1** (val_acc = 0.9155).

### Unsupervised Learning

| Method | Chosen k | ARI vs True Labels |
|--------|----------|--------------------|
| K-Means | 3 | 0.2690 |
| Hierarchical (Ward) | 3 | 0.1362 |

> 10 PCA components explain 90% of variance. t-SNE shows partial but imperfect class separatio

---

## References

- Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- LeCun, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 86(11).
- van der Maaten, L. & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*, 9, 2579–2605.

---

> *Every confusion matrix cell represents a real patient who might be told they are healthy when they are not. Build models you can explain, run experiments you can reproduce, and write interpretations a cardiologist could trust.*
