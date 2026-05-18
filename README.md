# Breast Cancer Wisconsin — Binary Classification Pipeline

A full supervised machine learning pipeline for classifying breast tumors as malignant or benign, built on the [Breast Cancer Wisconsin Diagnostic (WDBC)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) dataset from the UCI Machine Learning Repository.

**Course:** Applied Machine Learning Basic  
**Author:** Mahan Balooei

---

## Overview

Each sample in the WDBC dataset represents measurements extracted from a digitized fine needle aspirate (FNA) image of a breast mass. The task is binary classification: predict **Malignant (M = 1)** vs **Benign (B = 0)**.

This notebook treats **malignant as the positive class**, because a false negative — a missed malignant case — is the highest-cost error in a diagnostic-support setting. The primary optimization metric is therefore **malignant-class F1-score**, with explicit reporting of false-negative counts.

---

## Dataset

| Property | Value |
|---|---|
| Source | UCI ML Repository / `sklearn.datasets.load_breast_cancer()` |
| Samples | 569 (357 benign, 212 malignant) |
| Features | 30 continuous numeric (10 nuclear properties × mean / SE / worst) |
| Missing values | None |
| Class imbalance | Moderate (62.7% / 37.3%) |

The notebook automatically fetches the raw UCI file at runtime and falls back to scikit-learn's bundled copy if the URL is unavailable, so no manual data download is required.

---

## Pipeline Structure

| Section | Content |
|---|---|
| 1. Setup | Imports, `RANDOM_STATE`, plot style |
| 2. Loading | UCI fetch with sklearn fallback; duplicate detection; ID drop; target encoding |
| 3. Class distribution | Counts, pie/bar charts |
| 4. EDA | Descriptive stats, skewness/kurtosis, feature-group comparison, correlation heatmap, target-correlation bar chart |
| 5. Train/test split | Stratified 80/20 split |
| 6. Baseline | `DummyClassifier` |
| 7. Model comparison | `LogisticRegression`, `KNN`, `SVM`, `RandomForest`, `LogReg+SelectKBest` evaluated with stratified 5-fold CV |
| 8. Hyperparameter tuning | `GridSearchCV` on LR, SVM, RF; confusion matrices; summary comparison table; PR and ROC curves |
| 9. Model interpretation | RF feature importances; LR standardized coefficients |
| 10. Error analysis | False-negative profiling on top-5 RF features |
| 11. Conclusion | Evidence synthesis, limitations, future work |

---

## Results

| Model | Accuracy | Precision (M) | Recall (M) | F1 (M) | ROC-AUC | FN |
|---|---|---|---|---|---|---|
| **LogReg (tuned)** ✓ | **0.9737** | **0.9756** | **0.9524** | **0.9639** | **0.9960** | **2** |
| SVM (tuned) | 0.9737 | 0.9750 | 0.9286 | 0.9512 | 0.9979 | 3 |
| RF (tuned) | 0.9737 | 0.9512 | 0.9286 | 0.9396 | 0.9973 | 3 |
| KNN (default) | 0.9561 | 0.9268 | 0.9048 | 0.9156 | 0.9691 | 4 |
| LogReg+KBest10 | 0.9561 | 0.9268 | 0.9048 | 0.9156 | 0.9915 | 4 |
| Dummy (baseline) | 0.6228 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 42 |

**Final model: Tuned Logistic Regression.** Tuned SVM has a marginally higher CV F1 (0.9674 vs 0.9640, Δ = 0.0034), but Logistic Regression is simpler, more interpretable, directly regularized for the high feature multicollinearity observed in EDA, and achieves the same test accuracy with **one fewer false negative**.

---

## Key Findings

- The strongest discriminative signal comes from **size and shape features**, particularly `_worst` measurements (`concave points_worst`, `perimeter_worst`, `radius_worst`, `area_worst`) and `concave points_mean`.
- `_worst` features have the highest average absolute target correlation (0.596) vs `_mean` (0.537) and `_se` (0.279).
- Strong multicollinearity exists between radius, perimeter, and area features (ρ ≥ 0.98), supporting L2 regularization.
- The 2 false negatives missed by the final model share a profile of **less extreme size/shape measurements**, falling closer to the benign morphology range on multiple features simultaneously.

---

## Limitations

1. **Internal validation only** — test set is a held-out split from the same cohort; no external validation.
2. **Small dataset** — 569 samples; each additional false negative changes recall noticeably.
3. **Default decision threshold** — threshold optimization for a target recall constraint is left as future work.
4. **Not for clinical use** — this is an educational ML project.

---

## Setup

```bash
git clone https://github.com/<your-username>/breast-cancer-wdbc.git
cd breast-cancer-wdbc
pip install -r requirements.txt
jupyter notebook notebooks/breast_cancer_classification.ipynb
```

No data download needed — the notebook fetches the dataset automatically.

---

## Requirements

See [`requirements.txt`](requirements.txt). Core dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`.

---

## License

[MIT](LICENSE)
