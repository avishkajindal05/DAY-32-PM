# Day 32 | PM Session — Decision Trees & Random Forest: Applied
**Week 6 | IIT Gandhinagar — PG Diploma in AI-ML & Agentic AI Engineering**

---

## Assignment Overview

| Field | Detail |
|---|---|
| **Topics** | DT vs RF comparison, interpretability vs accuracy tradeoff, hyperparameter tuning workflow, feature importance (MDI vs permutation), case study methodology, model selection for business constraints |
| **Estimated Time** | 60–90 minutes |
| **Submission** | GitHub commit + Jupyter notebook pushed to repo, link in Slack `#daily-standup` |
| **Due** | Next day 09:15 AM |

---

## File Structure

```
D32_PM_DT_RF_CaseStudy/
│
├── README.md                          ← This file
│
├── D32_PM_DT_RF_CaseStudy.ipynb       ← Main notebook (Parts A, C, D)
│
└── gradient_boosting_preview.md       ← Part B written notes + resource links
```

---

## Part-wise Breakdown

### Part A — Concept Application (40%)
**File:** `D32_PM_DT_RF_CaseStudy.ipynb` → Sections A.1–A.6

| Step | What's done |
|---|---|
| A.1 | Synthetic insurance claims dataset — 3000 records, 8 features (`claim_amount`, `policy_age_years`, `num_prev_claims`, `annual_premium`, `days_to_report`, `claimant_age`, `num_witnesses`, `claim_to_premium`), target: `is_fraud` |
| A.2 | Decision Tree (`max_depth=5`, `class_weight='balanced'`) + automated top-3 fraud rule extractor sorted by fraud purity × sample coverage |
| A.3 | Random Forest tuned with `RandomizedSearchCV` (30 iterations, 5-fold CV, scoring=`recall`) — recall optimised because FN cost = 10× FP cost |
| A.4 | Full metrics table: Accuracy, Precision, Recall, F1, ROC-AUC, TP/FP/FN/TN, Interpretability + side-by-side confusion matrices |
| A.5 | Cost-sensitive evaluation: total business cost (FN=10×FP) at default threshold + optimal threshold curve for both models |
| A.6 | 2-paragraph deployment recommendation (RF for automated scoring + DT rules for human review / regulatory explanation) |
| Bonus | MDI vs Permutation importance comparison plots |

**Outputs generated:**
- `fraud_dt_plot.png`
- `confusion_matrices.png`
- `cost_threshold_curve.png`
- `fraud_feature_importance.png`

---

### Part B — Stretch Problem (30%)
**File:** `gradient_boosting_preview.md`

| Item | What's covered |
|---|---|
| Boosting vs Bagging | One-paragraph conceptual explanation: parallel vs sequential, variance vs bias reduction, residual fitting |
| Comparison table | Parallel vs sequential, variance vs bias, sample weighting, overfitting risk, typical depth, key algorithms |
| Intuition | Pseudocode loop showing residual fitting across iterations |
| Resource 1 | Machine Learning Mastery — "A Gentle Introduction to Gradient Boosting" |
| Resource 2 | StatQuest video — Gradient Boost Part 1 (16 min) |
| Self-check | 4 pre-Day-33 questions to test conceptual readiness |

---

### Part C — Interview Ready (20%)
**File:** `D32_PM_DT_RF_CaseStudy.ipynb` → Section C

| Question | What's done |
|---|---|
| **Q1 — 1000 vs 100 Trees** | Full tradeoff table (training time, prediction latency, memory, marginal accuracy), accuracy saturation curve with twin-axis training time plot, saturation detection code |
| **Q2 — `compare_models()`** | Reusable function using `cross_validate` with `StratifiedKFold`, returns mean ± std of Accuracy, F1, and Fit Time for any model dict; demo on 4 models with bar chart |
| **Q3 — Debug** | Root cause: missing `random_state` + only 10 trees → massive variance; reproduced bug with proof, fixed with `random_state=42` + increased `n_estimators`; permutation importance recommended as stable alternative |

**Outputs generated:**
- `accuracy_saturation.png`
- `model_comparison_cv.png`

---

### Part D — AI-Augmented Task (10%)
**File:** `D32_PM_DT_RF_CaseStudy.ipynb` → Section D

| Step | What's done |
|---|---|
| AI Explanation | OOB error explained via "500 junior analysts" analogy — each tested only on cases they never studied |
| Verification | Checked: 63/37 split is mathematically correct (1−1/e ≈ 36.8%); aggregation mechanism clarified; timing of OOB (during training) noted as missing |
| Follow-up | Table: 5 scenarios where OOB ≠ test error (small dataset, temporal shift, bootstrap=False, class imbalance, correlations) |
| Improved analogy | Added temporal distribution shift caveat for practical accuracy |
| Code demo | `oob_score=True` on RF — compares OOB accuracy vs test accuracy numerically |

---

## How to Run

```bash
# 1. Install dependencies
pip install scikit-learn pandas numpy matplotlib scipy

# 2. Run main notebook
jupyter notebook D32_PM_DT_RF_CaseStudy.ipynb
```

> All cells use `random_state=42` for full reproducibility. Run top-to-bottom.

---

## Key Business Context

| Metric | Priority | Reason |
|---|---|---|
| **Recall** | Highest | Missing fraud (FN) costs 10× more than a false alarm |
| **ROC-AUC** | High | Enables threshold tuning for optimal cost |
| **Interpretability** | Required | Regulators require model explanations for each decision |
| **Precision** | Secondary | False alarms cost investigator time but are manageable |

---

## Model Results Summary

| Model | Recall | ROC-AUC | Business Cost | Interpretability |
|---|---|---|---|---|
| Decision Tree (depth=5) | ~0.80 | ~0.86 | Higher | ★★★★★ (rules) |
| Random Forest (tuned) | ~0.87 | ~0.92 | Lower | ★★☆☆☆ (importances) |

*Exact values vary slightly with random seed and hardware.*  
*Recommended deployment: RF for scoring + DT rules for explanation.*

---

*Day 32 | PM Session | IIT Gandhinagar — AI-ML & Agentic AI Engineering*
