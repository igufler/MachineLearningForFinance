import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ----------------------------
# Load Compustat quarterly extract
# Needs: gvkey, datadate, dldte, dlrsn, atq, ltq, niq, cheq
# ----------------------------
df = pd.read_csv("Delisting.csv", parse_dates=["datadate", "dldte"])

# Keep observations that are before deletion date (or never deleted)
df = df[df["dldte"].isna() | (df["datadate"] < df["dldte"])].copy()

# Clean deletion reason codes to 2-digit strings
df["dlrsn"] = df["dlrsn"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)

# ----------------------------
# Outcome: "failure-like deletion" within 1 year
# ----------------------------
FAIL_CODES = {"02", "03"}  # bankruptcy, liquidation (change if your coding differs)

days_to_del = (df["dldte"] - df["datadate"]).dt.days
df["FAIL_1Y"] = ((days_to_del >= 1) & (days_to_del <= 365) & (df["dlrsn"].isin(FAIL_CODES))).astype(int)

# Clean controls: survives at least 5 years
df["SURV_5Y"] = (df["dldte"].isna() | ((df["dldte"] - df["datadate"]).dt.days > 5 * 365)).astype(int)

# ----------------------------
# Multiple regressors (the four)
# ----------------------------
df = df[(df["atq"] > 0) & (df["ltq"] >= 0)].copy()

df["SIZE"] = np.log(df["atq"])
df["LEV"] = df["ltq"] / df["atq"]
df["ROA"] = df["niq"] / df["atq"]
df["CASH"] = df["cheq"] / df["atq"]

use_all = df[["FAIL_1Y", "SURV_5Y", "SIZE", "LEV", "ROA", "CASH"]].replace([np.inf, -np.inf], np.nan).dropna()

# Winsorize predictors (for stability/plot readability)
for c in ["SIZE", "LEV", "ROA", "CASH"]:
    lo, hi = use_all[c].quantile([0.001, 0.999])
    use_all[c] = use_all[c].clip(lo, hi)

# ----------------------------
# CHERRYPICK FOR TEACHING: balanced sample (events vs clean survivors)
# ----------------------------
events = use_all[use_all["FAIL_1Y"] == 1]
controls = use_all[(use_all["FAIL_1Y"] == 0) & (use_all["SURV_5Y"] == 1)]

n = len(events)
if n == 0:
    raise ValueError("No failure events found. Check FAIL_CODES, dldte coverage, and dlrsn coding.")

if len(controls) < n:
    controls_s = controls.sample(n=len(controls), random_state=0)
    events_s = events.sample(n=len(controls), random_state=0)
else:
    controls_s = controls.sample(n=n, random_state=0)
    events_s = events

use = pd.concat([events_s, controls_s], axis=0).sample(frac=1, random_state=1)  # shuffle

y = use["FAIL_1Y"].astype(float)
X = sm.add_constant(use[["SIZE", "LEV", "ROA", "CASH"]])

# Logit (with robust SE in summary)
logit = sm.Logit(y, X).fit(disp=0)

print("\nBalanced sample event rate:", np.round(y.mean(), 3), "N:", len(use))

print("\n=== Logit ===")
print(logit.summary())


