# Gradient Boosting Preview — Self-Study Notes

**Day 32 | PM Session | Part B: Stretch Problem**  
*Pre-reading for Day 33: Gradient Boosting in depth*

---

## How Does Boosting Differ from Bagging? (One Paragraph)

Bagging (used by Random Forest) builds trees **in parallel** on independent bootstrap samples of the training data, then combines their predictions by majority vote or averaging — the goal is to reduce **variance** by averaging out the noise of many uncorrelated models. Boosting, by contrast, builds trees **sequentially**: each new tree is trained specifically to correct the mistakes of the ensemble so far, with misclassified or high-residual samples receiving higher weight (or, in Gradient Boosting, the next tree directly fits the **residuals** of the previous ensemble). This means boosting reduces **bias** rather than variance — it gradually pushes an initially weak learner toward a stronger approximation of the true function. The tradeoff is that boosted models are more sensitive to noisy data and outliers (since they actively try to fit every error), and they require careful tuning of the learning rate to prevent overfitting, whereas bagged models are naturally robust to noise.

---

## Key Conceptual Differences at a Glance

| Dimension | Bagging (Random Forest) | Boosting (Gradient Boosting) |
|---|---|---|
| **Tree construction** | Parallel, independent | Sequential, each corrects the last |
| **What it reduces** | Variance | Bias |
| **Sample weighting** | Bootstrap (uniform, with replacement) | Residual-weighted (errors get more focus) |
| **Overfitting risk** | Low (averaging smooths out noise) | Higher (fits errors aggressively) |
| **Learning rate** | Not applicable | Critical hyperparameter (`η`) |
| **Sensitivity to outliers** | Robust | Sensitive |
| **Typical depth** | Deep trees (low bias, high variance per tree) | Shallow trees / stumps (weak learners) |
| **Key algorithms** | Random Forest, Extra Trees | GBM, XGBoost, LightGBM, CatBoost |

---

## The Gradient Boosting Intuition

```
Iteration 0: Predict mean(y)  →  compute residuals r₀ = y - ŷ₀
Iteration 1: Fit Tree₁ on r₀  →  ŷ₁ = ŷ₀ + η × Tree₁(X)
Iteration 2: Fit Tree₂ on r₁  →  ŷ₂ = ŷ₁ + η × Tree₂(X)
...
Iteration T: Final model = sum of all scaled trees
```

Each tree is a **gradient step** in function space — hence "gradient" boosting. The learning rate `η` (typically 0.01–0.3) controls how big each step is: smaller `η` requires more trees but generalises better.

---

## Recommended Resource

**Blog post:** *"A Gentle Introduction to Gradient Boosting"* — Machine Learning Mastery  
**Link:** https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/

**Why this resource is good:**
- Builds from AdaBoost → Gradient Boosting step by step
- Covers the mathematical intuition (gradient descent in function space) without requiring calculus
- Includes a worked numerical example showing residual fitting across iterations
- Discusses shrinkage (learning rate), subsampling, and depth — the three key regularisation handles you'll tune tomorrow

**Supplementary video:** StatQuest with Josh Starmer — *"Gradient Boost Part 1: Regression Main Ideas"*  
https://www.youtube.com/watch?v=3CC4N4z3GJc  
*(~16 minutes — excellent visual walkthrough, highly recommended before Day 33)*

---

## What to Expect Tomorrow (Day 33)

Based on the pre-reading, Day 33 will likely cover:

1. **AdaBoost** — the original boosting algorithm (sample re-weighting)
2. **Gradient Boosting Machine (GBM)** — generalised framework
3. **XGBoost / LightGBM** — modern high-performance implementations
4. **Hyperparameters to tune:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`
5. **When to prefer boosting over RF:** when bias reduction matters more than variance reduction (e.g., structured tabular data with clean labels)

---

## Quick Self-Check Questions

Before Day 33, try to answer these without looking:

- [ ] Why does a lower learning rate usually require more trees?
- [ ] Why are boosted trees typically shallow (depth 3–6) while RF trees are deep?
- [ ] What happens to a boosted model if you include a significant outlier in the training set?
- [ ] Can you use boosting for regression as well as classification?

*(Answers will be covered in the Day 33 session.)*

---

*Day 32 | PM Session | IIT Gandhinagar — AI-ML & Agentic AI Engineering*
