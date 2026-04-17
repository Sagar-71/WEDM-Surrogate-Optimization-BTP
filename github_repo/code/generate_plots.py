#!/usr/bin/env python3
"""
BTP-II REPORT PLOTS — 3 Surrogates (RF, SVR, XGBoost), Knee-point, No ANN/TOPSIS
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, joblib, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score
from matplotlib.patches import Patch

SAVE_DIR = "./wedm_results"
PLOT_DIR = "./wedm_plots_v2"
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight', 'font.family': 'serif'
})

# === DATA ===
l9 = pd.DataFrame({
    'Ip': [20,20,20,25,25,25,30,30,30],
    'Ton': [110,115,120,110,115,120,110,115,120],
    'Toff': [50,55,60,55,60,50,60,50,55],
    'Vs': [220,230,240,240,220,230,230,240,220],
    'MRR': [7.87,7.98,8.08,8.22,8.35,8.56,8.47,8.78,8.29],
    'SR': [2.14,2.19,2.23,2.47,2.66,2.84,2.56,3.01,2.79]
})

def rsm_mrr(Ip,Ton,Toff,Vs):
    return (-78.77+0.3170*Ip+1.132*Ton-0.8390*Toff+0.3392*Vs-0.005267*Ip**2-0.004867*Ton**2+0.007533*Toff**2-0.000717*Vs**2)
def rsm_sr(Ip,Ton,Toff,Vs):
    return (-44.57+0.4000*Ip+1.081*Ton-0.4140*Toff-0.09000*Vs-0.006800*Ip**2-0.004600*Ton**2+0.003600*Toff**2+0.000200*Vs**2)

# Load artifacts
scaler_X = joblib.load(f"{SAVE_DIR}/scaler_X.joblib")
scaler_mrr = joblib.load(f"{SAVE_DIR}/scaler_mrr.joblib")
scaler_sr = joblib.load(f"{SAVE_DIR}/scaler_sr.joblib")
expanded = pd.read_csv(f"{SAVE_DIR}/expanded_dataset_1100.csv")
cv_df = pd.read_csv(f"{SAVE_DIR}/cv_results.csv")
val_df = pd.read_csv(f"{SAVE_DIR}/validation_results.csv")
hv_df = pd.read_csv(f"{SAVE_DIR}/hypervolume_comparison.csv")

# Filter to 3 models only
MODELS = ['RF', 'SVR', 'XGBoost']
MODEL_NAMES_CV = ['Random Forest', 'SVR', 'XGBoost']
cv3 = cv_df[cv_df['Model'].isin(MODEL_NAMES_CV)].reset_index(drop=True)
# Rename for consistency
cv3['Model'] = cv3['Model'].replace('Random Forest', 'RF')
val3 = val_df[val_df['Model'].isin(['RF', 'SVR', 'XGBoost', 'RSM (Baseline)'])].reset_index(drop=True)
hv3 = hv_df[hv_df['Surrogate'].isin(['RF', 'SVR', 'XGBoost'])].reset_index(drop=True)

pareto = {}
for name in MODELS:
    fp = f"{SAVE_DIR}/pareto_{name}.csv"
    if os.path.exists(fp):
        pareto[name] = pd.read_csv(fp)

# Load models and compute L9 predictions
X_val = l9[['Ip','Ton','Toff','Vs']].values
X_val_sc = scaler_X.transform(X_val)

rf_mrr = joblib.load(f"{SAVE_DIR}/rf_mrr.joblib")
rf_sr = joblib.load(f"{SAVE_DIR}/rf_sr.joblib")
svr_mrr = joblib.load(f"{SAVE_DIR}/svr_mrr.joblib")
svr_sr = joblib.load(f"{SAVE_DIR}/svr_sr.joblib")
xgb_mrr = joblib.load(f"{SAVE_DIR}/xgb_mrr.joblib")
xgb_sr = joblib.load(f"{SAVE_DIR}/xgb_sr.joblib")

preds = {
    'RF': (rf_mrr.predict(X_val_sc), rf_sr.predict(X_val_sc)),
    'SVR': (scaler_mrr.inverse_transform(svr_mrr.predict(X_val_sc).reshape(-1,1)).ravel(),
            scaler_sr.inverse_transform(svr_sr.predict(X_val_sc).reshape(-1,1)).ravel()),
    'XGBoost': (xgb_mrr.predict(X_val_sc), xgb_sr.predict(X_val_sc)),
}

COLORS = {'RF': '#2196F3', 'SVR': '#FF5722', 'XGBoost': '#4CAF50'}
MARKERS = {'RF': 'o', 'SVR': 's', 'XGBoost': '^'}

def knee_point(pdf):
    m = (pdf['MRR'].values - pdf['MRR'].min()) / (pdf['MRR'].max() - pdf['MRR'].min() + 1e-10)
    s = (pdf['SR'].values - pdf['SR'].min()) / (pdf['SR'].max() - pdf['SR'].min() + 1e-10)
    i1, i2 = np.argmax(m), np.argmin(s)
    p1, p2 = np.array([m[i1], s[i1]]), np.array([m[i2], s[i2]])
    lv = p2 - p1; ll = np.linalg.norm(lv)
    if ll < 1e-10: return 0
    return np.argmax([abs(np.cross(lv, p1 - np.array([m[i], s[i]]))) / ll for i in range(len(m))])

print("Generating plots...\n")

# =========================================================================
# PLOT 1: METHODOLOGY (updated - no ANN)
# =========================================================================
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 15); ax.set_ylim(0, 4); ax.axis('off')
ax.set_title('Methodology — RSM-Augmented ML Surrogate Framework', fontsize=15, fontweight='bold', pad=15)

steps = [
    (1.5, 2, 'L9 Experimental\nData (9 pts)', '#E8F5E9'),
    (4.5, 2, 'RSM Equations\n(Minitab)', '#E3F2FD'),
    (7.5, 2, 'Expanded Dataset\n(1100 integer pts)', '#E3F2FD'),
    (10.5, 2, 'ML Surrogates\nRF / SVR / XGBoost', '#FFF3E0'),
    (13.5, 2, 'NSGA-II → Knee-point\nOptimal Parameters', '#FCE4EC'),
]
for x, y, text, color in steps:
    ax.add_patch(plt.Rectangle((x-1.2, y-0.6), 2.4, 1.2, facecolor=color,
                                edgecolor='black', linewidth=1.5, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)
for i in range(len(steps)-1):
    ax.annotate('', xy=(steps[i+1][0]-1.2, steps[i+1][1]),
                xytext=(steps[i][0]+1.2, steps[i][1]),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.5))
plt.savefig(f"{PLOT_DIR}/01_methodology.png"); plt.close()
print("✓ 01 Methodology")

# =========================================================================
# PLOT 2: L9 DATA (reuse as-is, but regenerate for consistency)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
sc = ax.scatter(l9.MRR, l9.SR, c=l9.Ip, cmap='viridis', s=140, edgecolors='black', linewidths=1.2, zorder=3)
for _, r in l9.iterrows():
    ax.annotate(f"  Ip={int(r.Ip)}", (r.MRR, r.SR), fontsize=8)
ax.set_xlabel('MRR (mm³/min)'); ax.set_ylabel('SR (µm)'); ax.set_title('L9 Experimental Data')
plt.colorbar(sc, ax=ax, label='Peak Current (A)'); ax.grid(alpha=0.3)

ax = axes[1]
params = ['Ip','Ton','Toff','Vs']; x = np.arange(9); w = 0.2
for i, p in enumerate(params):
    norm = (l9[p] - l9[p].min()) / (l9[p].max() - l9[p].min())
    ax.bar(x + i*w, norm, w, label=p, alpha=0.8)
ax.set_xticks(x + 1.5*w); ax.set_xticklabels([f'E{i+1}' for i in range(9)])
ax.set_ylabel('Normalized Value'); ax.set_title('L9 Parameter Variation')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/02_experimental_data.png"); plt.close()
print("✓ 02 Experimental data")

# =========================================================================
# PLOT 3: PARAMETER EFFECTS
# =========================================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
params_info = [('Ip', np.arange(20,31), 0), ('Ton', np.arange(110,121), 1),
               ('Toff', np.arange(50,61), 2), ('Vs', np.arange(220,241), 3)]
base = [25, 115, 55, 230]
for col, (pname, prange, pidx) in enumerate(params_info):
    vals = np.tile(base, (len(prange), 1)).astype(float); vals[:, pidx] = prange
    m = rsm_mrr(vals[:,0], vals[:,1], vals[:,2], vals[:,3])
    s = rsm_sr(vals[:,0], vals[:,1], vals[:,2], vals[:,3])
    axes[0,col].plot(prange, m, 'b-o', markersize=4); axes[0,col].set_xlabel(pname)
    axes[0,col].set_ylabel('MRR'); axes[0,col].set_title(f'{pname} → MRR'); axes[0,col].grid(alpha=0.3)
    axes[1,col].plot(prange, s, 'r-s', markersize=4); axes[1,col].set_xlabel(pname)
    axes[1,col].set_ylabel('SR (µm)'); axes[1,col].set_title(f'{pname} → SR'); axes[1,col].grid(alpha=0.3)
plt.suptitle('RSM — Effect of Each Parameter (others at midpoint)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/03_parameter_effects.png"); plt.close()
print("✓ 03 Parameter effects")

# =========================================================================
# PLOT 4: EXPANDED DATASET
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(expanded.MRR, expanded.SR, s=5, alpha=0.3, c='steelblue', label='Expanded (1100)')
axes[0].scatter(l9.MRR, l9.SR, s=100, c='red', marker='*', zorder=3, edgecolors='k', label='L9 (9)')
axes[0].set_xlabel('MRR'); axes[0].set_ylabel('SR (µm)'); axes[0].set_title('RSM-Expanded Dataset')
axes[0].legend(); axes[0].grid(alpha=0.3)
for col, color in zip(['Ip','Ton','Toff','Vs'], ['#1f77b4','#ff7f0e','#2ca02c','#d62728']):
    axes[1].hist(expanded[col], bins=15, alpha=0.5, label=col, color=color, edgecolor='black')
axes[1].set_xlabel('Parameter Value'); axes[1].set_ylabel('Count')
axes[1].set_title('Parameter Distribution'); axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/04_expanded_dataset.png"); plt.close()
print("✓ 04 Expanded dataset")

# =========================================================================
# PLOT 5: CV COMPARISON (3 models only)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
models = cv3['Model'].tolist(); x = np.arange(len(models)); w = 0.35

ax = axes[0]
b1 = ax.bar(x-w/2, cv3['MRR_R2_mean'], w, yerr=cv3['MRR_R2_std'], label='MRR', color='#2196F3', edgecolor='black', capsize=5)
b2 = ax.bar(x+w/2, cv3['SR_R2_mean'], w, yerr=cv3['SR_R2_std'], label='SR', color='#FF5722', edgecolor='black', capsize=5)
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel('R²'); ax.set_title('5-Fold CV — R²')
ax.set_ylim(0.95, 1.005); ax.legend(); ax.grid(axis='y', alpha=0.3)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f'{b.get_height():.4f}', ha='center', fontsize=9)

ax = axes[1]
ax.bar(x-w/2, cv3['MRR_RMSE_mean'], w, yerr=cv3['MRR_RMSE_std'], label='MRR', color='#2196F3', edgecolor='black', capsize=5)
ax.bar(x+w/2, cv3['SR_RMSE_mean'], w, yerr=cv3['SR_RMSE_std'], label='SR', color='#FF5722', edgecolor='black', capsize=5)
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel('RMSE'); ax.set_title('5-Fold CV — RMSE')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/05_cv_comparison.png"); plt.close()
print("✓ 05 CV comparison")

# =========================================================================
# PLOT 6: PREDICTED vs ACTUAL (3 models)
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for idx, (name, (pm, ps)) in enumerate(preds.items()):
    ax = axes[0, idx]
    ax.scatter(l9.MRR, pm, c=COLORS[name], edgecolors='k', s=70, zorder=3)
    lims = [min(l9.MRR.min(), pm.min())-0.05, max(l9.MRR.max(), pm.max())+0.05]
    ax.plot(lims, lims, 'k--', alpha=0.5); ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Experimental'); ax.set_ylabel('Predicted')
    ax.set_title(f'{name} — MRR (R²={r2_score(l9.MRR, pm):.4f})'); ax.grid(alpha=0.3)

    ax = axes[1, idx]
    ax.scatter(l9.SR, ps, c=COLORS[name], edgecolors='k', s=70, zorder=3)
    lims = [min(l9.SR.min(), ps.min())-0.05, max(l9.SR.max(), ps.max())+0.05]
    ax.plot(lims, lims, 'k--', alpha=0.5); ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Experimental'); ax.set_ylabel('Predicted')
    ax.set_title(f'{name} — SR (R²={r2_score(l9.SR, ps):.4f})'); ax.grid(alpha=0.3)
plt.suptitle('L9 Validation — Predicted vs Experimental', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/06_predicted_vs_actual.png"); plt.close()
print("✓ 06 Predicted vs actual")

# =========================================================================
# PLOT 7: VALIDATION TABLE (3 models + RSM)
# =========================================================================
fig, ax = plt.subplots(figsize=(12, 3.5)); ax.axis('off')
td = []
for _, r in val3.iterrows():
    td.append([r['Model'], f"{r['MRR_R2']:.4f}", f"{r['MRR_RMSE']:.4f}", f"{r['MRR_MAE']:.4f}",
               f"{r['SR_R2']:.4f}", f"{r['SR_RMSE']:.4f}", f"{r['SR_MAE']:.4f}"])
cols = ['Model', 'MRR R²', 'MRR RMSE', 'MRR MAE', 'SR R²', 'SR RMSE', 'SR MAE']
tbl = ax.table(cellText=td, colLabels=cols, loc='center', cellLoc='center',
               colWidths=[0.16]+[0.12]*6)
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.8)
for j in range(len(cols)):
    tbl[0, j].set_facecolor('#37474F'); tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(len(td)):
    for j in range(len(cols)): tbl[i+1, j].set_facecolor('#E8F5E9' if i < 3 else '#FFF9C4')
ax.set_title('External Validation on L9 Data', fontsize=14, fontweight='bold', pad=20)
plt.savefig(f"{PLOT_DIR}/07_validation_table.png"); plt.close()
print("✓ 07 Validation table")

# =========================================================================
# PLOT 8: PARETO FRONTS — 3 SURROGATES + KNEE POINTS (Combined)
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 7))
for name, pdf in pareto.items():
    ki = knee_point(pdf)
    ax.scatter(pdf.MRR, pdf.SR, c=COLORS[name], marker=MARKERS[name],
               s=40, alpha=0.6, edgecolors='k', linewidths=0.3,
               label=f'{name} ({len(pdf)} pts)')
    kp = pdf.iloc[ki]
    ax.scatter(kp.MRR, kp.SR, c=COLORS[name], s=200, marker='*', zorder=5,
               edgecolors='black', linewidths=1.5)
    ax.annotate(f'  {name} knee\n  MRR={kp.MRR:.2f}, SR={kp.SR:.2f}',
                (kp.MRR, kp.SR), fontsize=8, fontweight='bold')

ax.scatter(l9.MRR, l9.SR, c='black', s=100, marker='X', zorder=5, linewidths=1.5, label='L9 experimental')
ax.set_xlabel('Material Removal Rate (MRR, mm³/min)', fontsize=13)
ax.set_ylabel('Surface Roughness (SR, µm)', fontsize=13)
ax.set_title('NSGA-II Pareto Fronts with Knee-Point Solutions', fontsize=14)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.annotate('← Higher MRR preferred', xy=(0.65, 0.02), xycoords='axes fraction', fontsize=9, color='gray')
ax.annotate('↓ Lower SR preferred', xy=(0.02, 0.12), xycoords='axes fraction', fontsize=9, color='gray', rotation=90)
plt.savefig(f"{PLOT_DIR}/08_pareto_with_kneepoints.png"); plt.close()
print("✓ 08 Pareto fronts with knee-points")

# =========================================================================
# PLOT 9: HYPERVOLUME (3 models only)
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 4))
hv_s = hv3.sort_values('Hypervolume', ascending=True)
colors_hv = [COLORS[n] for n in hv_s['Surrogate']]
bars = ax.barh(range(len(hv_s)), hv_s['Hypervolume'], color=colors_hv, edgecolor='black', height=0.5)
ax.set_yticks(range(len(hv_s))); ax.set_yticklabels(hv_s['Surrogate'])
ax.set_xlabel('Hypervolume (larger = better)'); ax.set_title('Pareto Front Quality — Hypervolume', fontsize=14)
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, hv_s['Hypervolume']):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/09_hypervolume.png"); plt.close()
print("✓ 09 Hypervolume")

# =========================================================================
# PLOT 10: KNEE-POINT SOLUTIONS TABLE
# =========================================================================
fig, ax = plt.subplots(figsize=(12, 3)); ax.axis('off')
td = []
for name, pdf in pareto.items():
    ki = knee_point(pdf); kp = pdf.iloc[ki]
    td.append([name, f"{int(kp.Ip)}", f"{int(kp.Ton)}", f"{int(kp.Toff)}", f"{int(kp.Vs)}",
               f"{kp.MRR:.4f}", f"{kp.SR:.4f}"])
cols = ['Surrogate', 'Ip (A)', 'Ton (µs)', 'Toff (µs)', 'Vs (V)', 'MRR', 'SR (µm)']
tbl = ax.table(cellText=td, colLabels=cols, loc='center', cellLoc='center',
               colWidths=[0.16, 0.1, 0.1, 0.1, 0.1, 0.12, 0.12])
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.8)
for j in range(len(cols)):
    tbl[0, j].set_facecolor('#37474F'); tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(len(td)):
    for j in range(len(cols)): tbl[i+1, j].set_facecolor('#E3F2FD')
ax.set_title('Knee-Point Optimal Solutions — All Surrogates', fontsize=13, fontweight='bold', pad=20)
plt.savefig(f"{PLOT_DIR}/10_kneepoint_table.png"); plt.close()
print("✓ 10 Knee-point table")

# =========================================================================
# PLOT 11: MRR-SR CORRELATION
# =========================================================================
from itertools import product as iter_product
grid = np.array(list(iter_product([20,25,30], [110,115,120], [50,55,60], [220,230,240])))
mrr_g = rsm_mrr(grid[:,0], grid[:,1], grid[:,2], grid[:,3])
sr_g = rsm_sr(grid[:,0], grid[:,1], grid[:,2], grid[:,3])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(mrr_g, sr_g, c='steelblue', s=20, alpha=0.5, label='RSM predictions (81 pts)')
ax.scatter(l9.MRR, l9.SR, c='red', s=100, marker='*', zorder=3, edgecolors='k', label='L9 experimental')
corr = np.corrcoef(mrr_g, sr_g)[0,1]
z = np.polyfit(mrr_g, sr_g, 1); p = np.poly1d(z)
x_fit = np.linspace(mrr_g.min(), mrr_g.max(), 100)
ax.plot(x_fit, p(x_fit), 'k--', alpha=0.5, label=f'Trend (r={corr:.3f})')
ax.set_xlabel('MRR (mm³/min)'); ax.set_ylabel('SR (µm)')
ax.set_title('MRR–SR Trade-off in Design Space', fontsize=14)
ax.legend(); ax.grid(alpha=0.3)
plt.savefig(f"{PLOT_DIR}/11_mrr_sr_correlation.png"); plt.close()
print("✓ 11 MRR-SR correlation")

# =========================================================================
# PLOT 12: EXPERIMENTAL VALIDATION
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# MRR
ax = axes[0]
vals = [7.98, 7.50]
bars = ax.bar(['Predicted\n(SVR)', 'Experimental'], vals, color=['#2196F3', '#4CAF50'],
              edgecolor='black', width=0.5)
ax.set_ylabel('MRR (mm³/min)', fontsize=12)
ax.set_title('MRR — Predicted vs Experimental', fontsize=13)
ax.set_ylim(0, 9)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.15, f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
# Error annotation
ax.annotate(f'Error: 6.0%', xy=(0.5, 0.85), xycoords='axes fraction',
            fontsize=11, ha='center', color='#1565C0', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1565C0'))
ax.grid(axis='y', alpha=0.3)

# SR
ax = axes[1]
vals = [1.97, 1.74]
bars = ax.bar(['Predicted\n(SVR)', 'Experimental'], vals, color=['#FF5722', '#4CAF50'],
              edgecolor='black', width=0.5)
ax.set_ylabel('SR (µm)', fontsize=12)
ax.set_title('SR — Predicted vs Experimental', fontsize=13)
ax.set_ylim(0, 2.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
ax.annotate(f'Error: 11.7%\n(favorable direction)', xy=(0.5, 0.85), xycoords='axes fraction',
            fontsize=11, ha='center', color='#BF360C', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FBE9E7', edgecolor='#BF360C'))
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Experimental Validation — Balanced Optimum (Ip=20, Ton=110, Toff=60, Vs=233)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/12_experimental_validation.png"); plt.close()
print("✓ 12 Experimental validation")

# =========================================================================
# DONE
# =========================================================================
print(f"\n{'='*60}")
print(f"ALL 12 PLOTS SAVED TO: {PLOT_DIR}/")
print(f"{'='*60}")
print("""
  01_methodology.png             — Pipeline (RF/SVR/XGBoost only)
  02_experimental_data.png       — L9 data overview
  03_parameter_effects.png       — RSM parameter effects
  04_expanded_dataset.png        — 1100-point expansion
  05_cv_comparison.png           — 5-Fold CV (3 models)
  06_predicted_vs_actual.png     — Parity plots (3 models)
  07_validation_table.png        — L9 metrics table
  08_pareto_with_kneepoints.png  — Pareto fronts + knee-points
  09_hypervolume.png             — Hypervolume (3 models)
  10_kneepoint_table.png         — Knee-point solutions table
  11_mrr_sr_correlation.png      — MRR-SR trade-off
  12_experimental_validation.png — Predicted vs measured
""")
