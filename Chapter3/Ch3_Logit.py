import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ----------------------------
# Load your Compustat quarterly extract
# Must have: gvkey, datadate, dldte, dlrsn, atq
# Optionally: ltq, niq if you want leverage/roa instead of size
# ----------------------------
df = pd.read_csv("Delisting.csv", parse_dates=["datadate", "dldte"])

# Keep observations that are before deletion date (or never deleted)
df = df[df["dldte"].isna() | (df["datadate"] < df["dldte"])].copy()

# Clean deletion reason codes to 2-digit strings: "1" -> "01"
df["dlrsn"] = df["dlrsn"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)

# ----------------------------
# Define an event that is "distress-like"
# Bankruptcy / liquidation are common choices for a clear classroom example.
# (You can change these codes if your Compustat manual for your extract differs.)
# ----------------------------
FAIL_CODES = {"02", "03"}  # bankruptcy, liquidation

days_to_del = (df["dldte"] - df["datadate"]).dt.days
df["FAIL_1Y"] = ((days_to_del >= 1) & (days_to_del <= 365) & (df["dlrsn"].isin(FAIL_CODES))).astype(int)

# Clean control group: survives at least 5 years from datadate (no deletion soon)
df["SURV_5Y"] = (df["dldte"].isna() | ((df["dldte"] - df["datadate"]).dt.days > 5 * 365)).astype(int)

# ----------------------------
# Choose predictor
# ----------------------------
df = df[df['atq'] > 0]
df["X"] = np.log(df["atq"])  # SIZE


use_all = df[["FAIL_1Y", "SURV_5Y", "X"]].replace([np.inf, -np.inf], np.nan).dropna()

# Trim extreme X to make the plot readable
x_lo, x_hi = use_all["X"].quantile([0.001, 0.999])

use_all = use_all[(use_all["X"] >= x_lo) & (use_all["X"] <= x_hi)].copy()

# ----------------------------
# CHERRYPICK FOR TEACHING: balanced sample
# ----------------------------
events = use_all[use_all["FAIL_1Y"] == 1]
controls = use_all[(use_all["FAIL_1Y"] == 0) & (use_all["SURV_5Y"] == 1)]

n = len(events)
if n == 0:
    raise ValueError("No failure events found under your FAIL_CODES/window. Check dlrsn codes and dldte coverage.")
if len(controls) < n:
    controls_s = controls.sample(n=len(controls), random_state=0)
    events_s = events.sample(n=len(controls), random_state=0)
else:
    controls_s = controls.sample(n=n, random_state=0)
    events_s = events

use = pd.concat([events_s, controls_s], axis=0).sample(frac=1, random_state=1)  # shuffle

y = use["FAIL_1Y"].astype(float)
X = sm.add_constant(use[["X"]])

# OLS linear probability model
ols = sm.OLS(y, X).fit(cov_type="HC1")

# Logit
logit = sm.Logit(y, X).fit(disp=0)

print("\nBalanced sample event rate:", np.round(y.mean(), 3), "N:", len(use))
print("\n=== OLS (LPM) ===")
print(ols.summary())
print("\n=== Logit ===")
print(logit.summary())

# Predictions on a grid
x_grid = np.linspace(use["X"].min(), use["X"].max(), 400)
Xg = sm.add_constant(pd.DataFrame({"X": x_grid}))
p_ols = ols.predict(Xg)
p_log = logit.predict(Xg)

ame = logit.get_margeff(at='overall')  # AME
mem = logit.get_margeff(at='mean')  # MEM

print(ame.summary())
print(mem.summary())


# Show how OLS violates [0,1] on the grid
print("\nOLS p-hat on grid: min =", float(p_ols.min()), "max =", float(p_ols.max()))
print("Share of grid with p<0 or p>1:", float(((p_ols < 0) | (p_ols > 1)).mean()))

# Plot (rug/jitter)
rng = np.random.default_rng(0)
n_show = min(20000, len(use))
idx = rng.choice(use.index.to_numpy(), size=n_show, replace=False)
jitter = rng.uniform(-0.02, 0.02, size=n_show)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ax.scatter(use.loc[idx, "X"], use.loc[idx, "FAIL_1Y"] + jitter, s=2, alpha=0.5, label='data')
ax.plot(x_grid, p_ols, linewidth=2, label='OLS')
ax.plot(x_grid, p_log, linewidth=2, color='orange', label='Logit')

ax.axhline(0, linestyle="--", linewidth=1, color='black')
ax.axhline(1, linestyle="--", linewidth=1, color='black')
ax.set_title("OLS vs. Logit")
ax.set_xlabel("log(assets)")
ax.set_ylabel("Probability of failure-type deletion within 1 year")
ax.set_ylim(-0.1, 1.2)
ax.set_xlim([x_lo, x_hi])
plt.legend()
plt.tight_layout()
plt.savefig("Logit.png", dpi=300, bbox_inches="tight")