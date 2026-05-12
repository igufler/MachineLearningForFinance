# CAPM with Recession Interaction — Block Bootstrap
# Block bootstrap preserves temporal dependence in daily returns.

import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import numpy as np

# --------------------
# Inputs
# --------------------
ticker = "IBM"
market = "^GSPC"
start = "1980-01-01"
end = "2025-01-01"

fred_rf = "DGS3MO"
fred_rec = "USRECD"
trading_days = 252

# --------------------
# 1) Daily returns (stock + market)
# --------------------
px = yf.download([ticker, market], start=start, end=end, auto_adjust=True, progress=False)["Close"]
px.columns = [c.upper() for c in px.columns]
ret = px.pct_change().dropna()

# --------------------
# 2) Daily risk-free
# --------------------
rf = pdr.DataReader(fred_rf, "fred", start, end).dropna()
rf["RF"] = rf[fred_rf] / 100.0 / trading_days
rf = rf[["RF"]]

# --------------------
# 3) Recession dummy (NBER)
# --------------------
rec = pdr.DataReader(fred_rec, "fred", start, end).dropna()
rec = rec.rename(columns={fred_rec: "REC"})

# --------------------
# 4) Build dataset
# --------------------
df = ret.join(rf, how="inner").join(rec, how="inner").dropna()
df["STOCK_EXCESS"] = df[ticker.upper()] - df["RF"]
df["MKT_EXCESS"] = df[market.upper()] - df["RF"]
df["REC"] = df["REC"].astype(int)
df["MKT_X_REC"] = df["MKT_EXCESS"] * df["REC"]

# --------------------
# 5) Original OLS with HAC
# --------------------
y = df["STOCK_EXCESS"]
X = sm.add_constant(df[["MKT_EXCESS", "REC", "MKT_X_REC"]])
T = len(df)
maxlags = 5

res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
print(res.summary())

# --------------------
# 6) Block bootstrap (preserves temporal dependence)
# --------------------
B = 5000
rng = np.random.default_rng(0)

# Block length: rule of thumb T^(1/3)
block_len = int(np.ceil(T ** (1 / 3)))
n_blocks = int(np.ceil(T / block_len))

print(f"\nBlock bootstrap: T={T}, block_length={block_len}, "
      f"blocks_per_resample={n_blocks}")

y_arr = y.to_numpy()
X_arr = X.to_numpy()

boot_block = np.zeros((B, X.shape[1]))
for b in range(B):
    starts = rng.integers(0, T - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:T]
    fit_b = sm.OLS(y_arr[idx], X_arr[idx]).fit()
    boot_block[b, :] = fit_b.params

boot_df = pd.DataFrame(boot_block, columns=X.columns)

# --------------------
# 7) Comparison table: HAC vs Block Bootstrap
# --------------------
alpha = 0.05
summary = pd.DataFrame({
    "coef": res.params,
    "HAC_se": res.bse,
    "block_se": boot_df.std(ddof=1),
    "block_ci_lo": boot_df.quantile(alpha / 2),
    "block_ci_hi": boot_df.quantile(1 - alpha / 2),
})

print(f"\n{'=' * 70}")
print("Comparison: HAC vs Block Bootstrap")
print(f"{'=' * 70}")
print(summary.round(6).to_string())

# --------------------
# 8) Visualisation
# --------------------
params = list(X.columns)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, p in enumerate(params):
    ax = axes[i]
    se = boot_df[p].std(ddof=1)
    ci_lo = boot_df[p].quantile(alpha / 2)
    ci_hi = boot_df[p].quantile(1 - alpha / 2)

    ax.hist(boot_df[p], bins=60, color="#E8A87C", edgecolor="white",
            linewidth=0.3, alpha=0.85, density=True)
    ax.axvline(res.params[p], color="#D94E3F", linewidth=2, linestyle="--",
               label=f"OLS = {res.params[p]:.4f}")
    ax.axvline(ci_lo, color="#2C5F8A", linewidth=1.5, linestyle="--",
               label=f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    ax.axvline(ci_hi, color="#2C5F8A", linewidth=1.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=1, linestyle=":", alpha=0.5)

    ax.set_title(f"{p}  (SE={se:.5f})", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel("Density")

fig.suptitle(f"Block Bootstrap Distributions (B={B:,}, block_length={block_len})",
             fontsize=13, fontweight="bold")
fig.tight_layout()
plt.savefig("Ch4_BlockBootstrap.png", dpi=300, bbox_inches="tight")
