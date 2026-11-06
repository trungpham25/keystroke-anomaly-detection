"""
simulation.py -- Simulation with known truth for Project 9 (Anomaly Detection).

Run from the repository root:
    python src/simulation.py

The simulation itself needs no data file. One optional section at the end
applies the same functions to the real keystroke data (downloaded on demand);
if that is not possible, the section is skipped and everything else still runs.

Model: x_1..x_n ~ N(mu, Sigma) in R^d, with mu and Sigma chosen by us.
Everything (MLEs, Sigma^{-1/2}, Mahalanobis distance, threshold) is computed by
hand from numpy primitives; no library routine solves the whole method.

Checks (question numbers refer to the Iteration 2 handout):
    Q1  MLEs  mu_hat = mean,  Sigma_hat = (1/n) sum (x - mu_hat)(x - mu_hat)^T
    Q2  Mahalanobis d^2 >= 0 when Sigma is positive definite
    Q3  d^2 ~ chi^2_d   (histogram vs density, KS test)
    Q4  Sigma^{-1/2} = Q Lambda^{-1/2} Q^T whitens the data (cov(Z) = I)
    Q5  threshold at the (1 - alpha) chi^2 quantile -> false-alarm rate alpha
    +   heavy-tailed (t, 3 d.f.) data: the false-alarm rate is no longer alpha
    +   effect of sample size n on estimation error and false-alarm rate

Outputs:
    figures/sim_chi2_histogram.png
    figures/sim_heavy_tail.png
    figures/sim_error_vs_n.png
    docs/simulation_table.csv, docs/simulation_table.tex, docs/simulation_log.txt
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DOC_DIR = ROOT / "docs"

SEED = 0
D, N, ALPHA = 5, 5000, 0.01
ATTACK_SHIFT, N_ATTACK = 4.0, 50
T_DF = 3


# ----------------------------------------------------------------------------
# Method, written out by hand
# ----------------------------------------------------------------------------
def mle_gaussian(X):
    """Q1: MLEs of N(mu, Sigma). Note the divisor n (not n - 1)."""
    n = X.shape[0]
    mu_hat = X.mean(axis=0)
    Xc = X - mu_hat
    Sigma_hat = Xc.T @ Xc / n
    return mu_hat, Sigma_hat


def inv_sqrt_psd(S, tol=1e-12):
    """Q4: Sigma^{-1/2} = Q Lambda^{-1/2} Q^T from the eigendecomposition.

    Raises if S is not (numerically) positive definite, because then S^{-1}
    does not exist -- this is exactly what happens on the raw keystroke data.
    """
    lam, Q = np.linalg.eigh(S)
    if lam.min() <= tol * lam.max():
        raise np.linalg.LinAlgError(
            f"Matrix not positive definite (smallest eigenvalue {lam.min():.2e})."
        )
    return Q @ np.diag(lam ** -0.5) @ Q.T


def mahalanobis_sq(X, mu_hat, S_inv_half):
    """Q2: squared Mahalanobis distance d^2 = ||Sigma^{-1/2}(x - mu)||^2."""
    Z = (X - mu_hat) @ S_inv_half  # symmetric matrix, so no transpose needed
    return np.sum(Z ** 2, axis=1), Z


def threshold(alpha, d):
    """Q5: the (1 - alpha) quantile of chi^2_d."""
    return stats.chi2.ppf(1.0 - alpha, df=d)


def multivariate_t(rng, mu, Sigma, df, size):
    """Draw multivariate t samples whose *covariance equals Sigma*.

    x = mu + sqrt((df-2)/df) * L z / sqrt(w/df),  z ~ N(0, I), w ~ chi2_df.
    The scale factor makes Cov(x) = Sigma (valid for df > 2), so the t data
    and the normal data have the same mean and covariance and differ only in
    tail weight.
    """
    d = len(mu)
    L = np.linalg.cholesky(Sigma)
    z = rng.standard_normal((size, d))
    w = rng.chisquare(df, size=size)
    scale = np.sqrt((df - 2) / df) / np.sqrt(w / df)
    return mu + (z @ L.T) * scale[:, None]


def fit_and_score(X, alpha, d):
    """Fit on X, return (mu_hat, Sigma_hat, S_inv_half, d2, Z, tau)."""
    mu_hat, S_hat = mle_gaussian(X)
    S_inv_half = inv_sqrt_psd(S_hat)
    d2, Z = mahalanobis_sq(X, mu_hat, S_inv_half)
    return mu_hat, S_hat, S_inv_half, d2, Z, threshold(alpha, d)


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def make_truth(rng, d):
    B = rng.standard_normal((d, d))
    Sigma = B @ B.T + d * np.eye(d)  # positive definite by construction
    mu = np.zeros(d)
    return mu, Sigma


def main():
    FIG_DIR.mkdir(exist_ok=True)
    DOC_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    log = []

    def out(s=""):
        print(s)
        log.append(s)

    mu, Sigma = make_truth(rng, D)
    out(f"Seed={SEED}  d={D}  n={N}  alpha={ALPHA}  attack shift={ATTACK_SHIFT} per coord")
    out(f"True mu = {mu}")
    out(f"True Sigma =\n{np.round(Sigma, 3)}")
    out(f"Eigenvalues of true Sigma: {np.round(np.linalg.eigvalsh(Sigma), 3)} "
        f"(all > 0, so Sigma is positive definite)")

    # ---- Experiment 1: Gaussian data ------------------------------------
    X = rng.multivariate_normal(mu, Sigma, size=N)
    mu_hat, S_hat, S_inv_half, d2, Z, tau = fit_and_score(X, ALPHA, D)

    out("\n--- Q1: MLEs ---")
    out(f"mu_hat = {np.round(mu_hat, 4)}")
    out(f"max |mu_hat - mu|         = {np.abs(mu_hat - mu).max():.4f}")
    out(f"max |Sigma_hat - Sigma|   = {np.abs(S_hat - Sigma).max():.4f}")
    out(f"relative Frobenius error  = "
        f"{np.linalg.norm(S_hat - Sigma) / np.linalg.norm(Sigma):.4f}")
    # Sanity comparison only (not part of the method): numpy's biased cov.
    out(f"agrees with np.cov(bias=True): "
        f"{np.allclose(S_hat, np.cov(X.T, bias=True))}")

    out("\n--- Q2: Mahalanobis distance ---")
    out(f"min d^2 = {d2.min():.6f}  (>= 0: {bool((d2 >= 0).all())})")

    out("\n--- Q3: d^2 ~ chi^2_d ---")
    out(f"mean d^2 = {d2.mean():.4f}  (chi^2_{D} mean = {D})")
    out(f"var  d^2 = {d2.var(ddof=1):.4f}  (chi^2_{D} var  = {2 * D})")
    ks = stats.kstest(d2, "chi2", args=(D,))
    out(f"KS test vs chi^2_{D}: statistic = {ks.statistic:.4f}, p = {ks.pvalue:.3f}")

    out("\n--- Q4: whitening ---")
    covZ = np.cov(Z.T, bias=True)
    out(f"max |cov(Z) - I|         = {np.abs(covZ - np.eye(D)).max():.2e}")
    out(f"max |mean(Z)|            = {np.abs(Z.mean(axis=0)).max():.2e}")
    out(f"Sigma_hat^-1/2 symmetric = {np.allclose(S_inv_half, S_inv_half.T)}")
    out(f"(S^-1/2)^2 S = I         = "
        f"{np.allclose(S_inv_half @ S_inv_half @ S_hat, np.eye(D))}")

    out("\n--- Q5: threshold and false-alarm rate ---")
    far = float(np.mean(d2 > tau))
    se_far = np.sqrt(ALPHA * (1 - ALPHA) / N)
    out(f"tau = chi2.ppf(1 - {ALPHA}, {D}) = {tau:.4f}")
    out(f"false-alarm rate = {far:.4f}  (target {ALPHA}; binomial s.e. = {se_far:.4f})")

    attacks = rng.multivariate_normal(mu + ATTACK_SHIFT, Sigma, size=N_ATTACK)
    d2a, _ = mahalanobis_sq(attacks, mu_hat, S_inv_half)
    det = float(np.mean(d2a > tau))
    out(f"detection rate on {N_ATTACK} shifted attacks = {det:.3f}")

    # ---- Experiment 2: heavy-tailed data --------------------------------
    Xt = multivariate_t(rng, mu, Sigma, T_DF, N)
    _, S_hat_t, _, d2t, Zt, tau_t = fit_and_score(Xt, ALPHA, D)
    far_t = float(np.mean(d2t > tau_t))
    ks_t = stats.kstest(d2t, "chi2", args=(D,))
    out(f"\n--- Heavy-tailed data (multivariate t, {T_DF} d.f., same mean/covariance) ---")
    out(f"false-alarm rate = {far_t:.4f}  (target {ALPHA}) -> "
        f"{far_t / ALPHA:.1f}x too high")
    out(f"KS test vs chi^2_{D}: statistic = {ks_t.statistic:.4f}, p = {ks_t.pvalue:.2e}")
    out(f"max |cov(Z) - I| = {np.abs(np.cov(Zt.T, bias=True) - np.eye(D)).max():.2e} "
        f"(whitening still works: it only uses the first two moments)")

    # Repeat over many replications to separate signal from noise.
    reps = 200
    far_gauss, far_heavy = [], []
    for _ in range(reps):
        Xg = rng.multivariate_normal(mu, Sigma, size=N)
        Xh = multivariate_t(rng, mu, Sigma, T_DF, N)
        for Xs, store in ((Xg, far_gauss), (Xh, far_heavy)):
            _, _, _, d2s, _, ts = fit_and_score(Xs, ALPHA, D)
            store.append(np.mean(d2s > ts))
    out(f"over {reps} replications: Gaussian false-alarm rate = "
        f"{np.mean(far_gauss):.4f} +/- {np.std(far_gauss, ddof=1):.4f}; "
        f"t(3) false-alarm rate = {np.mean(far_heavy):.4f} +/- "
        f"{np.std(far_heavy, ddof=1):.4f}")

    # ---- Experiment 3: effect of n --------------------------------------
    out("\n--- Effect of sample size n (Gaussian data, 200 replications each) ---")
    ns = [50, 100, 400, 1000, 5000]
    rows_n = []
    for n in ns:
        errs, fars = [], []
        for _ in range(200):
            Xn = rng.multivariate_normal(mu, Sigma, size=n)
            m_n, S_n = mle_gaussian(Xn)
            errs.append(np.linalg.norm(S_n - Sigma) / np.linalg.norm(Sigma))
            # out-of-sample false-alarm rate: fit on n points, test on fresh data
            S_half_n = inv_sqrt_psd(S_n)
            Xnew = rng.multivariate_normal(mu, Sigma, size=2000)
            d2n, _ = mahalanobis_sq(Xnew, m_n, S_half_n)
            fars.append(np.mean(d2n > threshold(ALPHA, D)))
        rows_n.append((n, np.mean(errs), np.mean(fars)))
        out(f"n={n:5d}: rel. error of Sigma_hat = {np.mean(errs):.4f}, "
            f"out-of-sample false-alarm rate = {np.mean(fars):.4f}")

    # ---- Raw data cannot be inverted: p=31 keystroke case in miniature --
    out("\n--- What happens when Sigma_hat is singular (cf. raw keystroke data) ---")
    Xs = rng.multivariate_normal(mu, Sigma, size=N)
    Xs = np.column_stack([Xs, Xs[:, 0] + Xs[:, 1]])  # add a redundant column
    _, S_sing = mle_gaussian(Xs)
    out(f"d=6 data with column 6 = column 1 + column 2: "
        f"smallest eigenvalue of Sigma_hat = {np.linalg.eigvalsh(S_sing).min():.2e}")
    try:
        inv_sqrt_psd(S_sing)
    except np.linalg.LinAlgError as e:
        out(f"inv_sqrt_psd raises: {e}")

    # ---- Link to the real data (optional; skipped if no download possible) --
    out("\n--- Same check on the real keystroke data (one genuine user) ---")
    try:
        from load_data import load_data, feature_columns, feature_groups, user_matrix
        df = load_data()
        cols = feature_columns(df)
        subj = df["subject"].iloc[0]
        groups = feature_groups(cols)
        for label, use in (("all 31 features", cols),
                           ("H + UD (21 features)", groups["H"] + groups["UD"])):
            Xr = user_matrix(df, subj, use)
            m_r, S_r = mle_gaussian(Xr)
            try:
                S_half_r = inv_sqrt_psd(S_r)
                d2r, _ = mahalanobis_sq(Xr, m_r, S_half_r)
                tau_r = threshold(ALPHA, Xr.shape[1])
                out(f"subject {subj}, {label}: Sigma_hat invertible; "
                    f"mean d^2 = {d2r.mean():.2f} (= p by construction), "
                    f"in-sample rate of d^2 > chi2 threshold = "
                    f"{np.mean(d2r > tau_r):.4f} (nominal {ALPHA})")
            except np.linalg.LinAlgError as e:
                out(f"subject {subj}, {label}: {e}")
    except Exception as e:  # no network, missing file, etc.
        out(f"skipped (real data unavailable: {type(e).__name__}: {e})")

    # ---- Table: true vs estimated ---------------------------------------
    iu = np.triu_indices(D)
    rows = []
    for i in range(D):
        rows.append((f"mu_{i + 1}", mu[i], mu_hat[i]))
    for i, j in zip(*iu):
        rows.append((f"Sigma_{i + 1}{j + 1}", Sigma[i, j], S_hat[i, j]))
    rows.append(("alpha (false-alarm rate)", ALPHA, far))
    table = pd.DataFrame(rows, columns=["quantity", "true", "estimate"])
    table["abs_error"] = (table["estimate"] - table["true"]).abs()
    table.to_csv(DOC_DIR / "simulation_table.csv", index=False)
    (DOC_DIR / "simulation_table.tex").write_text(
        table.round(4).to_latex(index=False, float_format="%.4f")
    )
    out(f"\nWrote {DOC_DIR / 'simulation_table.csv'} and .tex "
        f"({len(table)} rows)")

    # ---- Figures ----------------------------------------------------------
    grid = np.linspace(0, np.quantile(d2, 0.999), 400)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d2, bins=60, density=True, alpha=0.6, label="simulated $d^2$")
    ax.plot(grid, stats.chi2.pdf(grid, D), "r-", lw=2, label=rf"$\chi^2_{D}$ density")
    ax.axvline(tau, color="k", ls="--", label=rf"$\tau$ ({1 - ALPHA:.0%} quantile)")
    ax.set(xlabel=r"$d^2$", ylabel="density", title=r"Gaussian data: $d^2$ vs $\chi^2_5$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sim_chi2_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    gmax = np.quantile(d2t, 0.995)
    g2 = np.linspace(0, gmax, 400)
    ax.hist(d2t[d2t <= gmax], bins=80, density=True, alpha=0.6,
            label=f"$d^2$, t({T_DF}) data")
    ax.plot(g2, stats.chi2.pdf(g2, D), "r-", lw=2, label=rf"$\chi^2_{D}$ density")
    ax.axvline(tau, color="k", ls="--", label=r"$\tau$")
    ax.set(xlabel=r"$d^2$ (top 0.5% not shown)", ylabel="density",
           title=f"Heavy tails: false-alarm rate = {far_t:.3f} (target {ALPHA})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sim_heavy_tail.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    arr = np.array(rows_n)
    ax[0].loglog(arr[:, 0], arr[:, 1], "o-")
    ax[0].set(xlabel="n", ylabel=r"rel. error $\|\hat\Sigma-\Sigma\|_F/\|\Sigma\|_F$",
              title="Estimation error vs n")
    ax[1].semilogx(arr[:, 0], arr[:, 2], "o-")
    ax[1].axhline(ALPHA, color="k", ls="--", label=r"target $\alpha$")
    ax[1].set(xlabel="n", ylabel="out-of-sample false-alarm rate",
              title="False alarms vs n")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sim_error_vs_n.png", dpi=150)
    plt.close(fig)

    (DOC_DIR / "simulation_log.txt").write_text("\n".join(log) + "\n")
    out("Saved figures to figures/ and log to docs/simulation_log.txt")


if __name__ == "__main__":
    main()
