
# 🧠 Stroke Prediction — Logistic Regression Optimization & Model Comparison
- By : ABeer Al-Zebda | Machine Learning Engineer
  
**Can machine learning reliably flag patients at risk of stroke before it happens?**  
This project explores that question using a real-world healthcare dataset, progressing from a failing baseline model to a clinically meaningful predictor — and uncovering why accuracy alone is a dangerous metric in medicine.

---

## Overview

Stroke is the second leading cause of death globally and a major cause of long-term disability. Early identification of high-risk patients gives clinicians a critical window for intervention. This project builds and iteratively improves a stroke prediction model using logistic regression, systematic hyperparameter tuning, class balancing strategies, and a gradient boosting comparison — all applied to a labeled dataset of 5,110 patient records.

The goal was not simply to achieve high accuracy, but to build a model that meaningfully detects stroke cases — a much harder and more important challenge given how rare strokes are in the data.

---

## Dataset

| Detail | Value |
|---|---|
| **Source** | Healthcare Stroke Dataset (Kaggle) |
| **Records** | 5,110 patients |
| **Features** | 12 — including age, BMI, glucose level, hypertension, heart disease, smoking status, and work type |
| **Target** | Binary — Stroke (1) / No Stroke (0) |
| **Missing Values** | BMI column (~201 entries) |
| **Class Balance** | ~95% No Stroke / ~5% Stroke |

---

## Insight 1 — The Imbalance Problem

The dataset is severely skewed: only 249 of 5,110 patients experienced a stroke. A model can achieve 95% accuracy simply by predicting "No Stroke" for every single patient — which is exactly what the baseline model did.

The chart below shows the class distribution and how different balancing strategies affected the model's ability to actually detect stroke cases (recall for class 1).

![Insight 1 — Class Imbalance & Sampling Strategy Comparison](insight1_class_imbalance.png)

**Key finding:** Without addressing the imbalance, all hyperparameter tuning — regularization strength, solver algorithm, penalty type, iteration count — had zero effect on stroke detection. The model learned to ignore the minority class entirely. Only once sampling strategies were introduced did the model begin identifying real stroke patients.

---

## Insight 2 — Model Architecture Matters More Than Tuning

Once the imbalance was addressed, two model architectures were compared: Logistic Regression (a linear classifier) and CatBoost (a gradient-boosted decision tree ensemble). The best configuration from each was evaluated on stroke recall and weighted F1-score.

![Insight 2 — Final Model Comparison & Best Model Confusion Matrix](insight2_model_results.png)

**Key finding:** SMOTE combined with CatBoost's default parameters produced the highest stroke recall (81%) with the best balance between sensitivity and overall accuracy. The model successfully identified 51 of 63 actual stroke cases in the test set — compared to zero detections in the baseline.

---

## Modeling Approach

The project followed a structured, iterative pipeline:

1. **Preprocessing** — Median imputation for missing BMI values, standard scaling for numerical features, one-hot encoding for categorical variables. All preprocessing was encapsulated in a `ColumnTransformer` pipeline to prevent data leakage.

2. **Baseline Model** — Default Logistic Regression, establishing the performance floor.

3. **Hyperparameter Tuning** — GridSearchCV tested regularization strength (C), solver algorithm, penalty type (L1/L2), max iterations, class weight, and L1 ratio. The most impactful parameter was `C`; `max_iter` and `l1_ratio` had no measurable effect on this dataset.

4. **Sampling Strategies** — Random Under Sampling, Random Over Sampling, and SMOTE (Synthetic Minority Oversampling Technique) were each applied and evaluated.

5. **Advanced Model** — CatBoost Classifier tested with and without SMOTE, with multiple regularization configurations.

---

## Results Summary

| Model | Test Accuracy | Stroke Recall | Weighted F1 | Overfitting |
|---|---|---|---|---|
| Baseline Logistic Regression | 95.1% | 0% | 0.93 | None |
| LR — Full GridSearch Tuned | 95.1% | 0% | 0.93 | None |
| LR + Random Under Sampling | 68.5% | 70% | 0.49 | None |
| LR + Random Over Sampling | 88.9% | 44% | 0.76 | Slight |
| **LR + SMOTE (Best LR)** | **86.7%** | **73%** | **0.77** | **Low** |
| CatBoost Default | 95.9% | 17% | 0.94 | High |
| CatBoost + Balanced Weights | 86.1% | 78% | 0.77 | Low |
| **SMOTE + CatBoost ★ Best** | **85.7%** | **81%** | **0.78** | **Low** |

### Best Model — SMOTE + CatBoost (Default Parameters)

```
              precision    recall    f1-score
  No Stroke      0.99       0.86       0.92
     Stroke      0.23       0.81       0.36

  Accuracy                             85.7%
  Weighted avg   0.94       0.86       0.89
```

**Confusion Matrix:**
- True Positives (Stroke correctly detected): **51**
- False Negatives (Missed strokes): **12**
- True Negatives: **1,045**
- False Positives: **170**

In a clinical screening context, this trade-off is intentional — it is far more acceptable to flag 170 patients for further review than to miss 51 real stroke cases.

---

## Key Takeaways

- **Accuracy is misleading on imbalanced medical data.** A model that detects zero strokes can still report 95% accuracy. Always evaluate with recall and F1-score for the minority class.
- **Hyperparameter tuning is not a substitute for data-level intervention.** No combination of solver, penalty, or regularization could overcome class imbalance without a sampling strategy.
- **SMOTE outperformed random sampling methods** because it generates diverse synthetic examples rather than simply duplicating or discarding data.
- **CatBoost's ensemble architecture** captured non-linear relationships in the feature space that logistic regression — a linear boundary model — could not.
- **The most effective parameter** in logistic regression was regularization strength (`C`), followed by class weight handling.

---

## Repository Structure

```
stroke-prediction/
│
├── stroke.ipynb                      # Full analysis notebook
├── README.md                         # This file
├── insight1_class_imbalance.png      # Visualization — class distribution & sampling
└── insight2_model_results.png        # Visualization — model comparison & confusion matrix
```

---

## Tools & Libraries

`Python` · `scikit-learn` · `imbalanced-learn` · `CatBoost` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`

---

*For full technical details, preprocessing steps, and model code — see the notebook.*
