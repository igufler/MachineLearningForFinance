# pip install yfinance pandas numpy statsmodels matplotlib

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -----------------------------
# 1) Download data (SPY has reliable volume)
# -----------------------------
start = "1993-01-01"
end = "2025-01-01"

px = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)[["Close", "Volume"]].dropna()
px.columns = ["P", "VOL"]

ret = np.log(px["P"]).diff()

# -----------------------------
# 2) Build regime labels (3 unordered classes)
#    - HIGHVOL: top 20% realized vol
#    - UP/DOWN: split remaining by median trend (balanced)
# -----------------------------
trend = px["P"].rolling(20).mean() / px["P"].rolling(60).mean() - 1.0
vol20 = ret.rolling(20).std() * np.sqrt(252)     # annualized realized vol
dlogvol = np.log(px["VOL"]).diff()

tmp = pd.DataFrame({"trend": trend, "vol20": vol20, "dlogvol": dlogvol, "ret": ret}).dropna()

hv_thr = tmp["vol20"].quantile(0.80)
is_hv = tmp["vol20"] >= hv_thr

# balanced UP/DOWN within non-highvol days
trend_med = tmp.loc[~is_hv, "trend"].median()

regime = np.where(is_hv, "HIGHVOL", np.where(tmp["trend"] > trend_med, "UP", "DOWN"))
tmp["REGIME"] = regime

# encode for MNLogit (0 baseline)
# (baseline chosen as DOWN)
map_reg = {"DOWN": 0, "UP": 1, "HIGHVOL": 2}
tmp["Y"] = tmp["REGIME"].map(map_reg)

print("Class shares (0=DOWN,1=UP,2=HIGHVOL):")
print(tmp["Y"].value_counts(normalize=True).sort_index().round(3))

# -----------------------------
# 3) Predictors (lagged; no look-ahead)
# -----------------------------
df = pd.DataFrame(index=tmp.index)
df["Y"] = tmp["Y"]

df["r1"] = tmp["ret"].shift(5)
df["r5"] = tmp["ret"].rolling(5).sum().shift(5)
df["trend1"] = tmp["trend"].shift(5)
df["vol20_1"] = tmp["vol20"].shift(5)
df["dlogvol1"] = tmp["dlogvol"].shift(5)

df = df.dropna()

# -----------------------------
# 4) Train/test split (time-based) + standardize X (helps MNLogit stability)
# -----------------------------
split = int(len(df) * 0.8)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

Xcols = ["r1", "r5", "trend1", "vol20_1", "dlogvol1"]

mu = train[Xcols].mean()
sd = train[Xcols].std().replace(0, 1.0)

X_tr = (train[Xcols] - mu) / sd
X_te = (test[Xcols] - mu) / sd

X_tr = sm.add_constant(X_tr)
X_te = sm.add_constant(X_te)

y_tr = train["Y"].astype(int)
y_te = test["Y"].astype(int)

# -----------------------------
# 5) Multinomial logit
# -----------------------------
mnl = sm.MNLogit(y_tr, X_tr).fit(disp=0, maxiter=200)
print(mnl.summary())

# -----------------------------
# 6) Confusion matrix on test set
# -----------------------------
p_te = mnl.predict(X_te)
yhat = p_te.values.argmax(axis=1)

cm = pd.crosstab(pd.Series(y_te.values, name="Actual"),
                 pd.Series(yhat, name="Pred"),
                 normalize="index")
print("\nConfusion matrix (row-normalized):")
print(cm.round(3))

# -----------------------------
# 7) Plot predicted probabilities vs ONE regressor (choose the informative one)
#    r1 is often weak; vol20_1 or trend1 gives a clearer picture.
# -----------------------------
xvar = "vol20_1"   # or "trend1"
grid_raw = np.linspace(train[xvar].quantile(0.01), train[xvar].quantile(0.99), 300)

med = train[Xcols].median()
Xg_raw = pd.DataFrame({c: med[c] for c in Xcols}, index=range(len(grid_raw)))
Xg_raw[xvar] = grid_raw

# --- standardize using train moments ---
Xg_std = (Xg_raw - mu) / sd

# --- IMPORTANT: add constant FOR SURE and match training column order ---
Xg = sm.add_constant(Xg_std, has_constant="add")
Xg = Xg[X_tr.columns]   # same column order as in estimation

pg = mnl.predict(Xg)

plt.figure(figsize=(8, 5))
plt.plot(grid_raw, pg.iloc[:, 0], linewidth=2, label="DOWN (0)")
plt.plot(grid_raw, pg.iloc[:, 1], linewidth=2, label="UP (1)")
plt.plot(grid_raw, pg.iloc[:, 2], linewidth=2, label="HIGHVOL (2)")
plt.xlabel(xvar)
plt.ylabel("Predicted probability")
plt.title("Multinomial logit: predicted regime probabilities")
plt.legend()
plt.tight_layout()
plt.show()