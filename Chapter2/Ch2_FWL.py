import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# -----------------------------
# Settings
# -----------------------------
np.random.seed(42)
n = 500

# x2 = market factor
mkt = np.random.normal(0, 1, n)

# x1 = "peer stock" return, correlated with the market
peer = 3 * mkt + np.random.normal(0, 0.6, n)

# y = target stock return, depends on both peer and market
y = 0.0 + 0.7 * peer + 2 * mkt + np.random.normal(0, 0.8, n)


# -----------------------------
# 1) Full regression
# -----------------------------
X = sm.add_constant(np.column_stack([peer, mkt]))
model = sm.OLS(y, X).fit()

alpha, beta_peer, beta_mkt = model.params


# -----------------------------
# Plot style
# -----------------------------
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6
})


# -----------------------------
# 2) WRONG plot
# -----------------------------
x_raw = peer
y_raw = y

x_grid = np.linspace(x_raw.min(), x_raw.max(), 200)
mkt_mean = mkt.mean()

y_line_wrong = alpha + beta_peer * x_grid

plt.figure()
plt.plot(x_raw, y_raw, ".", alpha=0.5, label="Data")
plt.plot(x_grid, y_line_wrong, linewidth=2, label="Regression line")

plt.xlabel("Peer stock return")
plt.ylabel("Target stock return")
plt.title("Wrong: Raw relationship (market not partialed out)")
plt.legend()
plt.tight_layout()
plt.savefig("Ch2_FWL_wrong.png", dpi=300, bbox_inches='tight')

# -----------------------------
# 3) FWL
# -----------------------------
X_mkt = sm.add_constant(mkt)

y_resid = sm.OLS(y, X_mkt).fit().resid
peer_resid = sm.OLS(peer, X_mkt).fit().resid

fwl_model = sm.OLS(y_resid, sm.add_constant(peer_resid)).fit()


# -----------------------------
# 4) Correct plot
# -----------------------------
x_fwl = peer_resid
y_fwl = y_resid

x_grid_fwl = np.linspace(x_fwl.min(), x_fwl.max(), 200)
y_line_fwl = fwl_model.params[0] + fwl_model.params[1] * x_grid_fwl

plt.figure()
plt.plot(x_fwl, y_fwl, ".", alpha=0.5, label="Residualized data")
plt.plot(x_grid_fwl, y_line_fwl, linewidth=2, label="FWL line")

plt.xlabel("Residualized peer return")
plt.ylabel("Residualized target return")
plt.title("Correct: FWL (market partialed out)")
plt.legend()
plt.tight_layout()
plt.savefig("Ch2_FWL.png", dpi=300, bbox_inches='tight')