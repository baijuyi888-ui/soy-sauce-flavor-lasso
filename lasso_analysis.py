# -*- coding: utf-8 -*-
"""
题目一：酱油风味物质 Lasso 降维
======================================================================
流程：读取 Z-score 标准化数据 → LassoCV（10 折交叉验证）
      → 图1 系数路径 | 图2 交叉验证误差曲线 | 图3 非零系数条形图

技术要求：sklearn.linear_model.LassoCV；random_state=42；
          网格 200 个 λ、max_iter=20000；图 300 dpi、英文标注、Times New Roman；
          代码一键运行、相对路径。
======================================================================
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, lasso_path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 10

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
C_PATH = "#4C72B0"   # 系数路径颜色
C_LMIN = "black"
C_L1SE = "#C44E52"

# 指标名称中英文对照（绘图用英文，避免中文乱码）
NAME_MAP = {
    '丙酮酸': 'Pyruvic acid', '酒石酸': 'Tartaric acid', 'L-苹果酸': 'Malic acid',
    '焦谷氨酸': 'Pyroglutamic acid', '乳酸': 'Lactic acid', '富马酸': 'Fumaric acid',
    '琥珀酸': 'Succinic acid', '草酸': 'Oxalic acid', '柠檬酸': 'Citric acid',
    '维生素C': 'Ascorbic acid',
    "5'-AMP": "5'-AMP", "5'-GMP": "5'-GMP", "5'-CMP": "5'-CMP",
    "5'-UMP": "5'-UMP", "5'-IMP": "5'-IMP",
    'Ala丙氨酸': 'Ala', 'Arg精氨酸': 'Arg', 'Asp天冬氨酸': 'Asp', 'Glu谷氨酸': 'Glu',
    'Gly甘氨酸': 'Gly', 'His组氨酸': 'His', 'Ile异亮氨酸': 'Ile', 'Leu亮氨酸': 'Leu',
    'Lys赖氨酸': 'Lys', 'Met蛋氨酸': 'Met', 'Phe苯丙氨酸': 'Phe', 'Pro脯氨酸': 'Pro',
    'Ser丝氨酸': 'Ser', 'Thr苏氨酸': 'Thr', 'Trp色氨酸': 'Trp', 'Tyr酪氨酸': 'Tyr',
    'Val缬氨酸': 'Val', 'cys胱氨酸': 'Cys',
    'Na+': 'Na+', 'K+': 'K+', 'Ca2+': 'Ca2+', 'Mg2+': 'Mg2+', 'Cl-': 'Cl-',
    'NSSS': 'NSSS', 'NaCl': 'NaCl',
}

# ======================================================================
# 1. 读取数据（已 Z-score 标准化）
# ======================================================================
df = pd.read_csv(DATA / "soy_sauce_flavor_standardized.csv", encoding="utf-8")
X = df.drop(columns=["ID", "咸味评分"])
y = df["咸味评分"].to_numpy()
features_cn = list(X.columns)
features_en = [NAME_MAP[c] for c in features_cn]
print("数据: X %s, y %s" % (X.shape, y.shape))

# 顺带核对: 用原始数据自己算 Z-score 与提供的标准化文件是否一致
# (该数据集用总体标准差 ddof=0 标准化)
raw = pd.read_csv(DATA / "soy_sauce_flavor_raw.csv", encoding="utf-8")
X_raw = raw.drop(columns=["ID", "咸味评分"]).to_numpy(dtype=float)
X_self = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)   # ddof=0
print("自查: 手工 Z-score 与标准化文件的最大绝对差 = %.2e (应为 0 或极小值)"
      % np.max(np.abs(X_self - X.to_numpy(dtype=float))))

# ======================================================================
# 2. LassoCV：10 折交叉验证
# ======================================================================
model = LassoCV(cv=N_FOLDS, alphas=200, max_iter=20000,
                random_state=RANDOM_STATE).fit(X, y)
alphas = model.alphas_                            # sklearn 按降序排列
desc = bool(alphas[0] > alphas[-1])
log_alphas = np.log10(alphas)
mse_mean = model.mse_path_.mean(axis=1)           # 每个 λ 的 CV 均方误差均值
mse_se = model.mse_path_.std(axis=1, ddof=1) / np.sqrt(N_FOLDS)   # 标准误

i_min = int(np.argmin(mse_mean))
alpha_min = float(alphas[i_min])
# 一倍标准误准则: 误差不超过 (最小值 + 最小值的标准误) 的"最简约"模型(最大 λ)
threshold = mse_mean[i_min] + mse_se[i_min]
cand = np.where(mse_mean <= threshold)[0]
i_1se = int(cand.min()) if desc else int(cand.max())
alpha_1se = float(alphas[i_1se])

coefs = model.coef_
n_nonzero = int(np.count_nonzero(coefs))
zero_idx = [i for i, c in enumerate(coefs) if c == 0]
zero_names_cn = [features_cn[i] for i in zero_idx]
zero_names_en = [features_en[i] for i in zero_idx]

print("-" * 64)
print("λmin  = %.6g   log10(λmin)  = %.4f   (CV MSE 最小)" % (alpha_min, np.log10(alpha_min)))
print("λ1se  = %.6g   log10(λ1se)  = %.4f   (一倍标准误准则)" % (alpha_1se, np.log10(alpha_1se)))
print("λmin 下非零系数个数: %d / 40" % n_nonzero)
print("被压缩为 0 的指标 (%d 个): %s" % (len(zero_idx),
      ", ".join("%s(%s)" % (en, cn) for en, cn in zip(zero_names_en, zero_names_cn))))

# ======================================================================
# 3. 图1：系数路径图
# ======================================================================
path_alphas, path_coefs, _ = lasso_path(X, y, alphas=alphas, max_iter=20000)
path_log = np.log10(path_alphas)
order = np.argsort(path_log)                      # 横轴按 log(λ) 从小到大
path_log_sorted = path_log[order]
path_coefs_sorted = path_coefs[:, order]

# 被淘汰的 4 个指标（λmin 处系数为 0）与需要标注的关键指标
ELIMINATED_CN = ["Gly甘氨酸", "Leu亮氨酸", "Thr苏氨酸", "Val缬氨酸"]
LABEL_CN = ["Mg2+", "Na+", "5'-AMP", "Lys赖氨酸", "K+"]
elim_idx = {features_cn.index(c) for c in ELIMINATED_CN}
label_idx = {features_cn.index(c): NAME_MAP[c] for c in LABEL_CN}

fig1, ax = plt.subplots(figsize=(10.5, 6.5))
for j in range(len(features_cn)):
    if j in elim_idx:      # 被淘汰的指标: 灰色加粗, 一眼可辨
        ax.plot(path_log_sorted, path_coefs_sorted[j], color="#B3B3B3",
                lw=1.8, alpha=0.95, zorder=4)
    else:
        ax.plot(path_log_sorted, path_coefs_sorted[j], color=C_PATH,
                lw=1.1, alpha=0.6, zorder=3)
ax.axhline(0, color="#999999", lw=0.8, zorder=2)

# 在曲线左端给关键指标加带引线的标签（自动防重叠）
x0 = float(path_log_sorted[0])
items = sorted(((j, float(path_coefs_sorted[j][0])) for j in label_idx),
               key=lambda t: -t[1])
ys = [y for _, y in items]
min_gap = 0.045
for k in range(1, len(ys)):
    if ys[k - 1] - ys[k] < min_gap:
        ys[k] = ys[k - 1] - min_gap
for (j, y_curve), y_text in zip(items, ys):
    ax.annotate(label_idx[j], xy=(x0 + 0.02, y_curve),
                xytext=(x0 + 0.16, y_text), fontsize=9.5, va="center",
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CCCCCC", lw=0.6),
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7))

ax.axvline(np.log10(alpha_min), color=C_LMIN, ls="--", lw=1.6)
ax.axvline(np.log10(alpha_1se), color=C_L1SE, ls="-.", lw=1.6)

handles = [
    Line2D([0], [0], color=C_PATH, lw=1.5, alpha=0.8,
           label="Retained at λmin (36 indicators)"),
    Line2D([0], [0], color="#B3B3B3", lw=2.2,
           label="Eliminated at λmin: Gly, Leu, Thr, Val"),
    Line2D([0], [0], color=C_LMIN, ls="--", lw=1.6,
           label="λmin (log10 λ = %.2f)" % np.log10(alpha_min)),
    Line2D([0], [0], color=C_L1SE, ls="-.", lw=1.6,
           label="λ1se (log10 λ = %.2f)" % np.log10(alpha_1se)),
]
ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95)
ax.set_xlabel("log10(λ)")
ax.set_ylabel("Coefficient")
ax.set_title("Lasso Coefficient Profiles (40 Flavor Indicators)")
fig1.tight_layout()
fig1.savefig(FIG / "fig1_lasso_paths.png")
plt.close(fig1)

# ======================================================================
# 4. 图2：交叉验证误差曲线（MSE，误差棒为标准误）
# ======================================================================
fig2, ax = plt.subplots(figsize=(9, 5.5))
ax.errorbar(path_log_sorted, mse_mean[order], yerr=mse_se[order],
            fmt="o", ms=3.5, lw=1.2, color=C_PATH, ecolor="#888888",
            elinewidth=0.9, capsize=2, alpha=0.9,
            label="10-fold CV MSE (mean ± SE)")
ax.axvline(np.log10(alpha_min), color=C_LMIN, ls="--", lw=1.6)
ax.axvline(np.log10(alpha_1se), color=C_L1SE, ls="-.", lw=1.6)
txt = ("λmin  = %.6g  (log10 = %.2f)\n"
       "λ1se  = %.6g  (log10 = %.2f)" % (alpha_min, np.log10(alpha_min),
                                          alpha_1se, np.log10(alpha_1se)))
ax.text(0.02, 0.06, txt, transform=ax.transAxes, fontsize=9.5,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#999999", alpha=0.95))
ax.set_xlabel("log10(λ)")
ax.set_ylabel("Cross-Validation MSE")
ax.set_title("Lasso 10-fold Cross-Validation Error Curve")
ax.legend(loc="upper center")
fig2.tight_layout()
fig2.savefig(FIG / "fig2_cv_mse.png")
plt.close(fig2)

# ======================================================================
# 5. 图3：λmin 下非零系数条形图
# ======================================================================
nz_idx = [i for i, c in enumerate(coefs) if c != 0]
nz = pd.Series(coefs[nz_idx], index=[features_en[i] for i in nz_idx])
nz = nz.sort_values()                             # 从负到正, barh 顶部为最大正值

fig3, ax = plt.subplots(figsize=(8.5, 9.5))
ax.barh(nz.index, nz.values, color=C_PATH, alpha=0.9, height=0.62)
for yi, v in enumerate(nz.values):
    ax.text(v + (0.004 if v >= 0 else -0.004), yi, "%.4f" % v,
            va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
ax.axvline(0, color="#666666", lw=0.9)
margin = max(abs(nz.values)) * 0.22
ax.set_xlim(nz.values.min() - margin, nz.values.max() + margin)
ax.set_xlabel("Lasso Coefficient (at λmin)")
ax.set_title("Non-zero Coefficients at λmin (%d indicators)" % len(nz))
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", labelsize=9)
fig3.tight_layout()
fig3.savefig(FIG / "fig3_coefficients.png")
plt.close(fig3)

print("-" * 64)
print("非零系数（按大小排序）:")
for name, v in nz.sort_values(ascending=False).items():
    print("  %-22s %.4f" % (name, v))
print("三张图已保存到:", FIG)
