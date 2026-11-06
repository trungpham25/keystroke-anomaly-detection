"""Simulation check with known truth (Iteration 2, section 7: simulated data).

Usage (from the repository root):
    python src/simulation.py

Part 1: draw normal data in d = 5 from a covariance we choose, estimate mu and Sigma (Q1),
whiten with Sigma^(-1/2) built from the eigendecomposition (Q4), compute Mahalanobis
distances (Q2), compare d^2 with the chi-squared density (Q3), and check the false-alarm
rate of the chi-squared threshold (Q5). Shifted "attacks" measure the detection rate.

Part 2: repeat with heavy-tailed data (multivariate t, 3 degrees of freedom, same
covariance) to show that the false-alarm rate is no longer alpha.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figures")

rng = np.random.default_rng(0)
d, n, alpha, nu = 5, 5000, 0.01, 3

# ----- true model: we choose mu and Sigma, so the right answer is known -----
B = rng.normal(size=(d, d))
Sigma = B @ B.T + d * np.eye(d)  # symmetric positive definite
mu = np.zeros(d)


def fit(X):
    """Q1: maximum-likelihood estimates (note 1/n, not 1/(n-1))."""
    mu_hat = X.mean(axis=0)
    D = X - mu_hat
    return mu_hat, D.T @ D / len(X)


def inv_sqrt(S):
    """Q4: Sigma^(-1/2) = Q Lambda^(-1/2) Q^T from the eigendecomposition S = Q Lambda Q^T."""
    lam, Q = np.linalg.eigh(S)
    return Q @ np.diag(lam ** -0.5) @ Q.T, lam


def mahalanobis_sq(X, mu_hat, S_inv_half):
    """Q2/Q3: d^2 = (x - mu)^T Sigma^-1 (x - mu) = ||Sigma^(-1/2)(x - mu)||^2."""
    Z = (X - mu_hat) @ S_inv_half
    return np.sum(Z ** 2, axis=1), Z


tau = stats.chi2.ppf(1 - alpha, df=d)  # Q5: (1 - alpha) quantile of chi^2_d

# ===== Part 1: normal data =====
X = rng.multivariate_normal(mu, Sigma, size=n)
mu_hat, S_hat = fit(X)
S_inv_half, lam = inv_sqrt(S_hat)
d2, Z = mahalanobis_sq(X, mu_hat, S_inv_half)

# check: d^2 via Sigma^(-1/2) equals d^2 via a direct solve with Sigma^-1
D = X - mu_hat
d2_direct = np.einsum("ij,ij->i", D, np.linalg.solve(S_hat, D.T).T)

attacks = rng.multivariate_normal(mu + 4, Sigma, size=50)
d2_att, _ = mahalanobis_sq(attacks, mu_hat, S_inv_half)

print(f"Model: d = {d}, n = {n}, alpha = {alpha}, seed 0; tau = chi2_{d} quantile at {1 - alpha} = {tau:.3f}\n")
print("=== Part 1: true values vs estimates (normal data) ===")
rows = [
    ("||mu_hat - mu||", 0.0, np.linalg.norm(mu_hat - mu)),
    ("||Sigma_hat - Sigma||_F / ||Sigma||_F", 0.0, np.linalg.norm(S_hat - Sigma) / np.linalg.norm(Sigma)),
    ("smallest eigenvalue of Sigma_hat (> 0: PD)", np.linalg.eigvalsh(Sigma).min(), lam.min()),
    ("||cov(Z) - I||_F  (whitened data)*", 0.0, np.linalg.norm(np.cov(Z.T, bias=True) - np.eye(d))),
    ("min d^2  (Q2: never negative)", ">= 0", d2.min()),
    ("max |d^2 (whitening) - d^2 (solve)|", 0.0, np.abs(d2 - d2_direct).max()),
    ("mean d^2  (chi2_d has mean d)*", d, d2.mean()),
    ("var d^2   (chi2_d has variance 2d)", 2 * d, d2.var()),
    ("false-alarm rate  P(d^2 > tau)  (Q5)", alpha, np.mean(d2 > tau)),
    ("detection rate on 50 shifted attacks", None, np.mean(d2_att > tau)),
]
print(f"{'quantity':<46}{'true / theory':>15}{'simulated':>12}")
for name, true, est in rows:
    t = "-" if true is None else true if isinstance(true, str) else f"{true:.4f}"
    print(f"{name:<46}{t:>15}{est:>12.4f}")
print("* exact by construction: whitening with the same Sigma_hat that was fitted gives cov(Z) = I,")
print("  and then mean d^2 = trace(cov(Z)) = d. The checks that test the theory are the variance")
print("  of d^2, the false-alarm rate and the histogram.")

# ===== Part 2: heavy-tailed data (multivariate t with nu = 3, same covariance) =====
# X = mu + sqrt(nu / W) * G, G ~ N(0, scale), W ~ chi2_nu; Cov(X) = nu/(nu-2) * scale
scale = Sigma * (nu - 2) / nu
G = rng.multivariate_normal(np.zeros(d), scale, size=n)
W = rng.chisquare(nu, size=n)
Xt = mu + G * np.sqrt(nu / W)[:, None]
mu_t, S_t = fit(Xt)
S_t_inv_half, _ = inv_sqrt(S_t)
d2_t, _ = mahalanobis_sq(Xt, mu_t, S_t_inv_half)
print(f"\n=== Part 2: heavy-tailed data (t with {nu} degrees of freedom) ===")
print(f"false-alarm rate P(d^2 > tau) = {np.mean(d2_t > tau):.4f}  vs  promised alpha = {alpha}")
print("-> the chi-squared threshold only gives rate alpha when the data are normal.")

# ===== figure: histogram of d^2 against the chi^2_d density =====
os.makedirs(FIGS, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
grid = np.linspace(0, 30, 400)
for a, vals, title in ((ax[0], d2, "Normal data"), (ax[1], d2_t, f"Heavy-tailed data (t, {nu} d.f.)")):
    a.hist(vals[vals < 30], bins=80, density=True, color="#4C72B0", alpha=0.7, label="simulated $d^2$")
    a.plot(grid, stats.chi2.pdf(grid, d), color="#C44E52", lw=2, label=rf"$\chi^2_{d}$ density")
    a.axvline(tau, color="black", ls="--", lw=1, label=rf"threshold $\tau$ ($\alpha$={alpha})")
    rate = np.mean(vals > tau)
    a.set(title=f"{title}: false-alarm rate {rate:.3f}", xlabel="$d^2$", ylabel="density")
    a.legend(fontsize=8)
fig.tight_layout()
path = os.path.join(FIGS, "simulation_d2_vs_chi2.png")
fig.savefig(path, dpi=150)
print(f"\nSaved figures/simulation_d2_vs_chi2.png")
