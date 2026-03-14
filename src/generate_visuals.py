"""
Standalone Visualization Generator for Project 003: Tree-Based Learning
Generates publication-quality figures for the AI Engineering Lab post.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ASSETS = Path(__file__).parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)
DATA = Path(__file__).parent.parent / "data"

plt.rcParams.update({'figure.dpi': 150, 'font.size': 11, 'font.family': 'DejaVu Sans'})

# ── Load data ──────────────────────────────────────────────────────────────────
df_ev = pd.read_csv(DATA / "ev_battery_thermal_runaway.csv")
df_wf = pd.read_csv(DATA / "wafer_edge_yield.csv")

features_ev = ['charge_rate_kw', 'ambient_temp_c', 'cooling_flow_rate',
               'cell_voltage_variance', 'age_cycles']
features_wf = ['distance_from_center_mm', 'angle_degrees', 'etch_gas_flow',
               'spin_coat_rpm', 'litho_overlay_error_nm']

X_ev = df_ev[features_ev]; y_ev = df_ev['thermal_runaway']
X_wf = df_wf[features_wf]; y_wf = df_wf['yield_pass']

Xtr_ev, Xte_ev, ytr_ev, yte_ev = train_test_split(X_ev, y_ev, test_size=0.2, random_state=42, stratify=y_ev)
Xtr_wf, Xte_wf, ytr_wf, yte_wf = train_test_split(X_wf, y_wf, test_size=0.2, random_state=42, stratify=y_wf)

dt_ev = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42)
rf_ev = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', n_jobs=-1, random_state=42)
dt_wf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight='balanced', random_state=42)
rf_wf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42)

for m, X, y in [(dt_ev, Xtr_ev, ytr_ev), (rf_ev, Xtr_ev, ytr_ev),
                (dt_wf, Xtr_wf, ytr_wf), (rf_wf, Xtr_wf, ytr_wf)]:
    m.fit(X, y)

# ── Figure 1: Decision Tree Structure (EV Battery) ─────────────────────────────
fig, ax = plt.subplots(figsize=(20, 9))
plot_tree(dt_ev, feature_names=features_ev, class_names=['Safe', 'Fire'],
          filled=True, rounded=True, max_depth=3, ax=ax,
          proportion=False, fontsize=9, impurity=True)
plt.title('Decision Tree: EV Battery Thermal Runaway\n(First 3 levels of 5 — Human-Readable IF-THEN Rules)',
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(ASSETS / "fig1_decision_tree_structure.png", bbox_inches='tight')
plt.close()
print("fig1 saved")

# ── Figure 2: Wafer Map — Actual vs Predicted ──────────────────────────────────
df_wf['x_mm'] = df_wf['distance_from_center_mm'] * np.cos(np.radians(df_wf['angle_degrees']))
df_wf['y_mm'] = df_wf['distance_from_center_mm'] * np.sin(np.radians(df_wf['angle_degrees']))

# Build prediction grid
grid_size = 120
x_grid = np.linspace(-150, 150, grid_size)
y_grid = np.linspace(-150, 150, grid_size)
xx, yy = np.meshgrid(x_grid, y_grid)
dist_grid = np.sqrt(xx**2 + yy**2).flatten()
angle_grid = np.degrees(np.arctan2(yy, xx)).flatten() % 360
inside = dist_grid <= 150
grid_df = pd.DataFrame({
    'distance_from_center_mm': dist_grid,
    'angle_degrees': angle_grid,
    'etch_gas_flow': df_wf['etch_gas_flow'].median(),
    'spin_coat_rpm': df_wf['spin_coat_rpm'].median(),
    'litho_overlay_error_nm': df_wf['litho_overlay_error_nm'].median()
})
pred_proba = np.full(len(dist_grid), np.nan)
pred_proba[inside] = rf_wf.predict_proba(grid_df[inside])[:, 1]
pred_grid = pred_proba.reshape(grid_size, grid_size)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
sc = ax.scatter(df_wf['x_mm'], df_wf['y_mm'], c=df_wf['yield_pass'],
                cmap='RdYlGn', alpha=0.35, s=6)
plt.colorbar(sc, ax=ax, label='Actual Yield (1=Pass)')
circle = plt.Circle((0,0), 150, fill=False, color='navy', lw=2, linestyle='--')
ax.add_patch(circle)
circle2 = plt.Circle((0,0), 120, fill=False, color='orange', lw=1.5, linestyle=':')
ax.add_patch(circle2)
ax.set_aspect('equal'); ax.set_xlim(-165,165); ax.set_ylim(-165,165)
ax.set_title('Actual Wafer Yield Map\n(orange = 120mm edge zone)', fontweight='bold')
ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')

ax = axes[1]
im = ax.imshow(pred_grid, extent=[-150,150,-150,150], origin='lower',
               cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
plt.colorbar(im, ax=ax, label='Predicted Pass Probability')
ax.set_title('Random Forest: Predicted Yield Map\n(Model learned the edge drop-off)', fontweight='bold')
ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')

plt.suptitle('Wafer Edge Yield: Actual vs Model Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig2_wafer_map_actual_vs_predicted.png", bbox_inches='tight')
plt.close()
print("fig2 saved")

# ── Figure 3: Feature Importance Comparison (Both Datasets) ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, rf_model, features, title, color in [
    (axes[0], rf_ev, features_ev, 'EV Battery: What Causes Fires?', '#F44336'),
    (axes[1], rf_wf, features_wf, 'Wafer Yield: What Drives Failures?', '#1565C0')
]:
    imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
    colors = [color if v == imp.max() else '#90CAF9' if color == '#1565C0' else '#FFCDD2'
              for v in imp.values]
    bars = ax.barh(imp.index, imp.values, color=colors, edgecolor='white')
    ax.set_xlabel('Feature Importance (Mean Decrease Impurity)')
    ax.set_title(title, fontweight='bold')
    for bar, val in zip(bars, imp.values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)

plt.suptitle('Random Forest Feature Importance: Both Domains', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig3_feature_importance_both.png", bbox_inches='tight')
plt.close()
print("fig3 saved")

# ── Figure 4: ROC Curves — DT vs RF on Both Datasets ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, dt_m, rf_m, Xte, yte, title in [
    (axes[0], dt_ev, rf_ev, Xte_ev, yte_ev, 'EV Battery Thermal Runaway'),
    (axes[1], dt_wf, rf_wf, Xte_wf, yte_wf, 'Wafer Edge Yield')
]:
    for name, model, color, ls in [
        ('Decision Tree', dt_m, '#1565C0', '--'),
        ('Random Forest', rf_m, '#E65100', '-')
    ]:
        proba = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, proba)
        auc = roc_auc_score(yte, proba)
        ax.plot(fpr, tpr, color=color, lw=2.5, linestyle=ls, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1],'k--', alpha=0.3, label='Random (0.500)')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve: {title}', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle('Decision Tree vs Random Forest: ROC Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig4_roc_comparison.png", bbox_inches='tight')
plt.close()
print("fig4 saved")

# ── Figure 5: Ensemble Intuition — Single Tree vs Forest Variance ─────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Generate a simple 1D regression problem to show variance
np.random.seed(42)
X_1d = np.sort(np.random.uniform(0, 10, 200))
y_1d = np.sin(X_1d) + np.random.normal(0, 0.3, 200)
X_plot = np.linspace(0, 10, 500).reshape(-1, 1)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

ax = axes[0]
ax.scatter(X_1d, y_1d, alpha=0.4, s=20, color='steelblue', label='Training data')
for seed in range(5):
    idx = np.random.choice(len(X_1d), len(X_1d), replace=True)
    dt_s = DecisionTreeRegressor(max_depth=5, random_state=seed)
    dt_s.fit(X_1d[idx].reshape(-1,1), y_1d[idx])
    ax.plot(X_plot, dt_s.predict(X_plot), alpha=0.5, lw=1.5,
            label=f'Tree {seed+1}' if seed < 3 else '_nolegend_')
ax.set_title('5 Individual Trees\n(Each overfits differently)', fontweight='bold')
ax.set_xlabel('Feature'); ax.set_ylabel('Prediction')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(X_1d, y_1d, alpha=0.4, s=20, color='steelblue', label='Training data')
rf_1d = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
rf_1d.fit(X_1d.reshape(-1,1), y_1d)
ax.plot(X_plot, rf_1d.predict(X_plot), color='#E65100', lw=3, label='Random Forest (200 trees)')
ax.plot(X_plot, np.sin(X_plot), color='green', lw=2, linestyle='--', label='True signal')
ax.set_title('Random Forest Average\n(Errors cancel → smooth, accurate)', fontweight='bold')
ax.set_xlabel('Feature'); ax.set_ylabel('Prediction')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle('Why Ensembles Work: Variance Reduction Through Averaging', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(ASSETS / "fig5_ensemble_intuition.png", bbox_inches='tight')
plt.close()
print("fig5 saved")

print("\nAll 5 figures generated successfully.")
