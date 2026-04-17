# EasyVisa — Visa Approval Prediction

**Tools:** Python · scikit-learn · XGBoost · imbalanced-learn · pandas · matplotlib · seaborn  
**Type:** Supervised Learning / Binary Classification

---

## Overview

The U.S. Office of Foreign Labor Certification (OFLC) processes hundreds of thousands of visa applications annually — a volume that makes manual review increasingly difficult. This project builds a machine learning solution to predict visa certification outcomes, helping prioritize applications with higher approval likelihood and supporting more efficient review workflows.

**Dataset:** ~25 features covering applicant education, job experience, employer size, prevailing wage, continent of origin, and more.

---

## Key Results

| Model | Training F1 | Validation F1 | Notes |
|---|---|---|---|
| Decision Tree | 1.000 | — | Overfit |
| Random Forest | 1.000 | 0.805 | Overfit |
| Bagging | 0.990 | 0.778 | High variance |
| AdaBoost | 0.820 | 0.818 | Stable |
| Gradient Boosting | 0.829 | 0.827 | Strong |
| **XGBoost (tuned)** | **0.834** | **0.828** | **✅ Final model** |

> **F1-score** was selected as the primary metric due to moderate class imbalance between certified and denied cases.

**Final model:** Tuned XGBoost classifier on original (non-oversampled) data  
- Training F1: **0.834** | Validation F1: **0.828**  
- Minimal train/validation gap indicates good generalization with limited overfitting

---

## Key Findings

**Top predictors of visa approval (feature importance):**
1. `education_of_employee` — High School level negatively associated with approval
2. `has_job_experience` — Prior experience is a strong positive predictor
3. `education_of_employee` — Master's degree positively associated with approval

**Business insights:**
- Education level and job experience are the most influential factors in approval decisions
- Company size (number of employees) has only marginal impact on approval rate
- Oversampling (SMOTE) did not meaningfully improve model performance, indicating the class imbalance was manageable with the original data

---

## Business Recommendations

1. **Prioritize high-education, experienced applicants** — Applications from candidates with a Master's degree or higher and prior job experience show the highest approval likelihood; these can be fast-tracked for review.
2. **Flag high-school-only applicants for additional review** — Not to deny automatically, but to ensure supporting documentation is thorough.
3. **Deploy model for pre-screening** — The XGBoost model (F1: 0.828) can reduce manual review burden while maintaining accuracy.
4. **Retrain periodically** — Labor market conditions change; model should be updated with new application data regularly.

---

## ML Techniques Demonstrated

- Exploratory Data Analysis (univariate & bivariate)
- Feature encoding and preprocessing
- Class imbalance handling (SMOTE via imbalanced-learn)
- Model training and comparison: Decision Tree, Random Forest, Bagging, AdaBoost, Gradient Boosting, XGBoost
- Hyperparameter tuning
- Evaluation with precision, recall, F1-score, confusion matrix
- Feature importance analysis

---

## Files

| File | Description |
|---|---|
| `EasyVisa_Analysis.ipynb` | Full notebook with code, outputs, and commentary |
