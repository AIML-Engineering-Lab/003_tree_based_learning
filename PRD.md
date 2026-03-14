# Product Requirements Document
## Project 003: Tree-Based Learning — Decision Trees & Random Forests

**Series:** AI Engineering Lab | **Era:** 1 — Classic Machine Learning | **Post:** 003

---

## 1. Objective

Teach Decision Trees and Random Forests through two intuitive, real-world prediction problems. The reader should understand: (a) how a tree makes decisions, (b) why a single tree overfits, (c) how Random Forests fix this through ensemble averaging, and (d) how to handle class imbalance in rare-event prediction.

---

## 2. Core Concepts Covered

| Concept | Depth |
|---|---|
| Decision Tree: entropy, Gini impurity, information gain | Full derivation |
| Tree hyperparameters: max_depth, min_samples_leaf, max_features | Tuning demo |
| Overfitting vs underfitting in trees | Before/after comparison |
| Pruning: pre-pruning via hyperparameters | Practical |
| Random Forest: bagging, feature randomness, OOB error | Full explanation |
| Feature importance: mean decrease impurity | Visualization |
| Class imbalance: class_weight, stratified split | Applied |
| Evaluation: ROC-AUC, Precision-Recall, Confusion Matrix | All metrics |
| Cross-validation: StratifiedKFold | Applied |

---

## 3. Datasets

### Dataset A: EV Battery Thermal Runaway
- **File:** `data/ev_battery_thermal_runaway.csv`
- **Rows:** 8,000 | **Features:** 5 | **Target:** `thermal_runaway` (binary)
- **Class ratio:** 95% Safe, 5% Fire
- **Key insight:** Fire requires a combination of conditions — no single threshold explains it

### Dataset B: Wafer Edge Yield Drop-off
- **File:** `data/wafer_edge_yield.csv`
- **Rows:** 10,000 | **Features:** 5 | **Target:** `yield_pass` (binary)
- **Class ratio:** ~90% Pass, ~10% Fail
- **Key insight:** Non-linear edge effect — yield is stable then drops sharply beyond 120mm

---

## 4. Block Diagram

```
Raw Data (CSV)
     │
     ▼
EDA & Visualization
(distributions, wafer map, class balance)
     │
     ▼
Preprocessing
(stratified train/test split, class_weight)
     │
     ├──────────────────────────────────┐
     ▼                                  ▼
Decision Tree                    Random Forest
(max_depth, min_samples_leaf)    (n_estimators, max_features)
     │                                  │
     ▼                                  ▼
Evaluation                       Evaluation
(ROC-AUC, PR-AUC, CM)           (ROC-AUC, PR-AUC, CM)
     │                                  │
     └──────────────┬───────────────────┘
                    ▼
         Model Comparison & Feature Importance
                    │
                    ▼
         Predicted Wafer Map (Posiva)
```

---

## 5. Visualizations

| Figure | Type | Key Message |
|---|---|---|
| `fig1` | Tree diagram (3 levels) | IF-THEN rules are human-readable |
| `fig2` | Wafer map: actual vs predicted | Model learned the spatial edge pattern |
| `fig3` | Feature importance bar chart | Which physical conditions drive failure |
| `fig4` | ROC curves (DT vs RF, both datasets) | RF consistently outperforms single tree |
| `fig5` | 1D ensemble variance demo | Why averaging trees reduces overfitting |

---

## 6. Tech Stack

Python 3.11 | scikit-learn | pandas | numpy | matplotlib | seaborn | Jupyter

---

## 7. Implementation Prompt

Use this prompt with any LLM-assisted coding tool (Claude Code, Cursor, GitHub Copilot):

```
You are implementing a machine learning project on tree-based models.
The project has two datasets already generated in data/:
  - ev_battery_thermal_runaway.csv (8000 rows, 5% fire rate)
  - wafer_edge_yield.csv (10000 rows, 10% fail rate)

For each dataset, implement a complete Jupyter notebook that:
1. Loads the data and performs EDA (distributions, class balance, correlation heatmap)
2. Performs stratified train/test split (80/20)
3. Trains a shallow Decision Tree (depth=2, no class weighting) as the BEFORE baseline
4. Trains a tuned Decision Tree (depth=5, min_samples_leaf=10, class_weight='balanced') as AFTER
5. Prints the text representation of the tree rules using export_text
6. Trains a Random Forest (200 trees, depth=8, class_weight='balanced')
7. Compares all three models with: classification_report, ROC-AUC, ROC curve, PR curve, confusion matrix
8. Plots feature importances as a horizontal bar chart
9. For the wafer dataset: converts polar to Cartesian coordinates and plots the wafer map,
   then generates a prediction probability grid and plots actual vs predicted wafer maps side by side
10. Includes detailed markdown cells explaining each concept before the code

All plots should be saved to assets/ as PNG files.
Use class_weight='balanced' to handle imbalance. Never use accuracy as the primary metric.
```
