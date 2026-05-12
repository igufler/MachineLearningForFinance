import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    train_test_split, LeaveOneOut, KFold, RepeatedKFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# -----------------------------
# Load + create next-month return
# -----------------------------
data = pd.read_stata("Stocks_Monthly.dta")
data = data.sort_values(["permno", "date"]).copy()

# Make sure date is datetime (robust filtering)
data["date"] = pd.to_datetime(data["date"])

# next month return (t+1)
data["ret_t+1"] = data.groupby("permno")["ret"].shift(-1)

# One pure cross-section (example month)
cs_date = pd.Timestamp("2022-06-30")
cols = ["permno", "date", "Gat", "profit", "LNme", "LNbe", "accruals", "ret_t+1"]
data = data.loc[data["date"].eq(cs_date), cols].dropna().copy()

features = ["Gat", "profit", "LNme", "LNbe", "accruals"]
X = data[features].to_numpy()
y = data["ret_t+1"].to_numpy()

lr = LinearRegression()

K = 100
# -----------------------------
# Helpers
# -----------------------------
def cv_mse(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    fold_mse = -scores
    return fold_mse.mean(), fold_mse.std(ddof=1), fold_mse

# -----------------------------
# 1) Validation-set approach (ISL) + show variability across random splits
# -----------------------------
n_reps = 100
val_mse = []
for seed in range(n_reps):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.8, random_state=seed)
    lr.fit(X_tr, y_tr)
    pred = lr.predict(X_te)
    val_mse.append(np.mean((pred - y_te) ** 2))
val_mse = np.array(val_mse)

# One “single split” number (like ISL)
mse_valset = val_mse.mean()


plt.figure(figsize=(8, 5))

plt.hist(val_mse, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(mse_valset, linestyle='--', linewidth=2, label=f"Mean MSE = {mse_valset:.4f}")

plt.title("Distribution of Validation MSE across Random Splits")
plt.xlabel("Validation MSE")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("Ch4_MSE.png", dpi=300, bbox_inches='tight')
# -----------------------------
# 2) LOOCV (ISL)
# -----------------------------
loo_scores = cross_val_score(lr, X, y, cv=LeaveOneOut(), scoring="neg_mean_squared_error")
mse_loocv = -loo_scores
mean_mse_loocv = mse_loocv.mean()


plt.figure(figsize=(8, 5))

plt.hist(-loo_scores, bins=1000, edgecolor='black', alpha=0.7)
plt.axvline(mse_loocv, linestyle='--', linewidth=2, label=f"Mean MSE = {mse_loocv:.4f}")

plt.title("Distribution of LOOCV MSE across Random Splits")
plt.xlabel("LOOCV MSE")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("Ch4_LOOCV_MSE.png", dpi=300, bbox_inches='tight')
# -----------------------------
# 3) k-fold CV (ISL): show k=5,10,20, 50, 100
# -----------------------------
mse_5fold, sd_5fold, fold_mse_5 = cv_mse(lr, X, y, KFold(n_splits=5, shuffle=True, random_state=0))
mse_10fold, sd_10fold, fold_mse_10 = cv_mse(lr, X, y, KFold(n_splits=10, shuffle=True, random_state=0))
mse_20fold, sd_20fold, fold_mse_20 = cv_mse(lr, X, y, KFold(n_splits=20, shuffle=True, random_state=0))
mse_50fold, sd_50fold, fold_mse_50 = cv_mse(lr, X, y, KFold(n_splits=50, shuffle=True, random_state=0))
mse_100fold, sd_100fold, fold_mse_100 = cv_mse(lr, X, y, KFold(n_splits=100, shuffle=True, random_state=0))

# -----------------------------
# Histograms for K-Fold CV MSEs
# -----------------------------
fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=False)

configs = [
    (fold_mse_5, mse_5fold, "5-Fold CV"),
    (fold_mse_10, mse_10fold, "10-Fold CV"),
    (fold_mse_20, mse_20fold, "20-Fold CV"),
    (fold_mse_50, mse_50fold, "50-Fold CV"),
    (fold_mse_100, mse_50fold, "100-Fold CV")
]

for ax, (fold_mse, mean_mse, title) in zip(axes, configs):
    ax.hist(fold_mse, bins=10, edgecolor='black', alpha=0.7, density=False)
    ax.axvline(mean_mse, linestyle='--', linewidth=2, label=f"Mean = {mean_mse:.4f}")

    ax.set_title(title)
    ax.set_xlabel("Fold MSE")
    ax.grid(alpha=0.3)
    ax.legend()

axes[0].set_ylabel("Density")

plt.tight_layout()
plt.savefig("Ch4_KFOLD_MSE.png", dpi=300, bbox_inches='tight')
