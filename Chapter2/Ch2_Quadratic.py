# pip install yfinance pandas statsmodels matplotlib

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

start = "1990-01-01"
end = "2025-01-01"

# 1) Download daily S&P 500 and VIX levels
px = yf.download(["^GSPC", "^VIX"], start=start, end=end, auto_adjust=True, progress=False)["Close"].dropna()
px.columns = ["SPX", "VIX"]

# 2) Daily log returns for S&P 500
ret = np.log(px["SPX"]).diff()

# 3) Forward realized vol over next ~1 month (22 trading days), annualized
h = 22
rv_fwd = ret.shift(-1).rolling(h).std() * np.sqrt(252)
rv_fwd.name = "RV_FWD"

# 4) Dataset (use VIX as decimal: 20 -> 0.20)
df = pd.concat([px["VIX"], rv_fwd], axis=1).dropna()
df["VIX_DEC"] = df["VIX"] / 100.0

# Create polynomial terms up to degree 5
for k in range(2, 6):
    df[f"VIX{k}"] = df["VIX_DEC"] ** k

maxlags = 5  # HAC lags (daily)

# 5) Degree 1 (linear)
X1 = sm.add_constant(df[["VIX_DEC"]])
m1 = sm.OLS(df["RV_FWD"], X1).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
print("\n=== Degree 1: RV_FWD ~ VIX ===")
print(m1.summary())

# 6) Degree 2 (quadratic)
X2 = sm.add_constant(df[["VIX_DEC", "VIX2"]])
m2 = sm.OLS(df["RV_FWD"], X2).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
print("\n=== Degree 2: RV_FWD ~ VIX + VIX^2 ===")
print(m2.summary())

# 7) Degree 5
X5 = sm.add_constant(df[["VIX_DEC", "VIX2", "VIX3", "VIX4", "VIX5"]])
m5 = sm.OLS(df["RV_FWD"], X5).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
print("\n=== Degree 5: RV_FWD ~ VIX + ... + VIX^5 ===")
print(m5.summary())

# 8) Plot: data + fitted curves (degrees 1, 2, 5)
v_grid = np.linspace(df["VIX_DEC"].min(), df["VIX_DEC"].max(), 300)

Xg1 = sm.add_constant(pd.DataFrame({"VIX_DEC": v_grid}))
y1 = m1.predict(Xg1)

Xg2 = sm.add_constant(pd.DataFrame({"VIX_DEC": v_grid, "VIX2": v_grid**2}))
y2 = m2.predict(Xg2)

Xg5 = pd.DataFrame({"VIX_DEC": v_grid})
for k in range(2, 6):
    Xg5[f"VIX{k}"] = v_grid**k
Xg5 = sm.add_constant(Xg5)
y5 = m5.predict(Xg5)

plt.figure(figsize=(8, 5))
plt.scatter(df["VIX_DEC"] * 100, df["RV_FWD"] * 100, alpha=0.50, s=5, label="data")
plt.plot(v_grid * 100, y1 * 100, linewidth=2, label="degree 1", color='magenta')
plt.plot(v_grid * 100, y2 * 100, linewidth=2, label="degree 2", color='orange')
plt.plot(v_grid * 100, y5 * 100, linewidth=2, label="degree 5", color='red')
plt.xlabel("VIX level (%)")
plt.ylabel("Next-month realized volatility of S&P 500 (annualized, %)")
plt.title("Future realized volatility vs VIX: polynomial fits")
plt.legend()
plt.tight_layout()
plt.savefig("Quadratic_data.png", dpi=300, bbox_inches='tight')

