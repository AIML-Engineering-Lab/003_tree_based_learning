# Post 003 — Tree-Based Learning: Decision Trees & Random Forests

**AI Engineering Lab Series** | Era 1: Classic Machine Learning

---

## Overview

This project demonstrates **Decision Trees** and **Random Forests** applied to two real-world prediction problems. The goal is to show how tree-based models learn non-linear, multi-condition rules — and why averaging many trees (ensemble learning) dramatically outperforms a single tree.

| Concept | Description |
|---|---|
| **Decision Tree** | Learns human-readable IF-THEN rules by recursively splitting data on feature thresholds |
| **Random Forest** | Builds hundreds of trees on random data/feature subsets; majority vote reduces variance |
| **Feature Importance** | Measures how much each feature reduces impurity across all trees |
| **Class Imbalance** | Handled via `class_weight='balanced'` and evaluated with ROC-AUC, not accuracy |
| **Ensemble Principle** | Many weak learners combined outperform one strong learner; errors cancel out |

---

## Datasets

### Dataset A: EV Battery Thermal Runaway Prediction
Predicts whether a lithium-ion battery will experience thermal runaway (fire) based on charging and environmental conditions.

| Feature | Description |
|---|---|
| `charge_rate_kw` | Power delivered during charging (kW) |
| `ambient_temp_c` | Environment temperature (°C) |
| `cooling_flow_rate` | Coolant flow through the battery pack (L/min) |
| `cell_voltage_variance` | Voltage spread across cells (imbalance indicator) |
| `age_cycles` | Number of charge/discharge cycles completed |
| **`thermal_runaway`** | **Target: 1 = Fire, 0 = Safe** |

- **Rows:** 8,000 | **Fire rate:** ~5% (severe class imbalance)
- **Why novel:** Teaches rare-event prediction with a universally understood physical scenario

### Dataset B: Wafer Edge Yield Drop-off (Post-Silicon Validation)
Predicts whether a die on a silicon wafer will pass or fail yield testing, based on its spatial position and process parameters.

| Feature | Description |
|---|---|
| `distance_from_center_mm` | Radial distance from wafer center (mm) |
| `angle_degrees` | Angular position on the wafer (degrees) |
| `etch_gas_flow` | Gas flow rate during etch step (sccm) |
| `spin_coat_rpm` | Spin speed during photoresist coating (RPM) |
| `litho_overlay_error_nm` | Misalignment between lithography layers (nm) |
| **`yield_pass`** | **Target: 1 = Pass, 0 = Fail** |

- **Rows:** 10,000 | **Pass rate:** ~90% (edge dies fail at higher rate)
- **Why novel:** The non-linear edge drop-off pattern is a canonical semiconductor manufacturing problem

---

## Project Structure

```
003_tree_based_learning/
├── data/
│   ├── ev_battery_thermal_runaway.csv
│   └── wafer_edge_yield.csv
├── notebooks/
│   ├── 01_tree_based_ev_battery.ipynb
│   └── 02_tree_based_wafer_yield.ipynb
├── src/
│   ├── data_generator.py
│   └── generate_visuals.py
├── assets/
│   ├── fig1_decision_tree_structure.png
│   ├── fig2_wafer_map_actual_vs_predicted.png
│   ├── fig3_feature_importance_both.png
│   ├── fig4_roc_comparison.png
│   └── fig5_ensemble_intuition.png
├── PRD.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Key Visualizations

| Figure | Description |
|---|---|
| `fig1` | Decision Tree structure (first 3 levels) — human-readable IF-THEN rules |
| `fig2` | Wafer map: actual yield vs Random Forest predicted yield probability |
| `fig3` | Feature importance for both datasets — what physically drives failures |
| `fig4` | ROC curves comparing Decision Tree vs Random Forest on both datasets |
| `fig5` | Ensemble intuition — why averaging many trees reduces variance |

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| scikit-learn | ≥1.3 | Decision Tree, Random Forest, metrics |
| pandas | ≥2.0 | Data manipulation |
| numpy | ≥1.24 | Numerical operations |
| matplotlib | ≥3.7 | All visualizations |
| seaborn | ≥0.12 | Statistical plots |

---

## Quick Start

```bash
git clone https://github.com/AIML-Engineering-Lab/003_tree_based_learning.git
cd 003_tree_based_learning
pip install -r requirements.txt
python src/data_generator.py
jupyter notebook notebooks/
```

---

## Series Navigation

| Post | Topic | Repo |
|---|---|---|
| 001 | Linear Regression Engine | [001_linear_regression_engine](https://github.com/AIML-Engineering-Lab/001_linear_regression_engine) |
| 002 | Classification Engine | [002_classification_engine](https://github.com/AIML-Engineering-Lab/002_classification_engine) |
| **003** | **Tree-Based Learning** | **This repo** |
| 004 | The Boosting Revolution | Coming soon |

---

*Part of the [AI Engineering Lab](https://github.com/AIML-Engineering-Lab) series — a progressive curriculum from Classic ML to Agentic AI, grounded in real engineering problems.*
