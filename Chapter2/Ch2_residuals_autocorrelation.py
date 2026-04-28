import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

np.random.seed(42)

# Simulate data
n = 200
x = np.random.normal(size=n)

# Autocorrelated errors: AR(1)
rho = 0.8
u = np.zeros(n)
eps = np.random.normal(size=n)

for t in range(1, n):
    u[t] = rho * u[t-1] + eps[t]

# Outcome
y = 1 + 2 * x + u

# OLS
X = sm.add_constant(x)
ols = sm.OLS(y, X).fit()

print(ols.summary())

# Autocorrelation plot of residuals
# compute ACF values
acf_vals = acf(ols.resid, nlags=20)

lags = np.arange(len(acf_vals))

conf = 1.96 / np.sqrt(len(ols.resid))


plt.figure()
plt.fill_between(lags, -conf, conf, color="#a9d6e5", alpha=0.5)
plt.plot(lags, acf_vals, marker='o', color="#2a6f97")
plt.axhline(0, linestyle='--', linewidth=1)
plt.xlim(0, 20)
plt.ylim(-1, 1)
plt.xticks(range(21))
plt.grid(True)
plt.title("ACF")
plt.show()


# Same coefficients, HAC / Newey-West standard errors
hac = ols.get_robustcov_results(cov_type="HAC", maxlags=5)

print(hac.summary())
