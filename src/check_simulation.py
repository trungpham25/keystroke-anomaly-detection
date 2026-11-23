"""check_simulation.py -- Iteration 3 simulation test: recover the known truth.

Run from the repository root:
    python src/check_simulation.py

Uses the Iteration 2 model (d = 5, n = 5000, seed 0, alpha = 0.01) and the
from-scratch functions in src/methods.py. numpy.linalg and scipy are used only
to check the answers.

  1. Truth and library comparisons at n = 5000.
  2. Error against the truth for n = 50, 100, 500, 1000, 5000 (log-log plot).
  3. Stress test A: heavy-tailed data (multivariate t, 3 d.f.).
  4. Stress test B: one feature is an exact copy of another (singular Sigma_hat).
  5. Assumption checks A1-A6 on the simulated data (they should pass).

Outputs: docs/check_simulation_log.txt, docs/verification_simulation.csv,
figures/it3_sim_error_vs_n.png, figures/it3_sim_qq.png
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats            # check only

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods import (my_mle, my_cholesky, my_mahalanobis2, my_inverse,
                     my_cholesky_inverse, my_threshold, my_flag, my_rate,
                     my_condition_number, my_chi2_quantile, my_skew_kurt,
                     my_autocorr)
from simulate import (simulate_normal, simulate_t, simulate_attacks,
                      add_duplicate_column)
import verify
from verify import report

ROOT = Path(__file__).resolve().parents[1]
FIG, DOC = ROOT / "figures", ROOT / "docs"
D, N, SEED, ALPHA = 5, 5000, 0, 0.01
LOG = []


def out(s=""):
    print(s)
    LOG.append(str(s))


def fit(X):
    mu_hat, S_hat = my_mle(X)
    L = my_cholesky(S_hat)
    return mu_hat, S_hat, L


def se_mu(Sigma, n):
    """Standard error of mu_hat_j: sqrt(Sigma_jj / n)."""
    return np.sqrt(np.diag(Sigma) / n)


def se_sigma(Sigma, n):
    """Standard error of Sigma_hat_jk for normal data: sqrt((Sigma_jk^2 + Sigma_jj Sigma_kk) / n)."""
    s = np.diag(Sigma)
    return np.sqrt((Sigma ** 2 + np.outer(s, s)) / n)


def main():
    FIG.mkdir(exist_ok=True)
    DOC.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ 1
    X, mu, Sigma = simulate_normal(D, N, SEED)
    mu_hat, S_hat, L = fit(X)
    d2 = my_mahalanobis2(X, mu_hat, L)
    tau = my_threshold(ALPHA, D)
    far = my_rate(my_flag(d2, tau))
    att = simulate_attacks(D, 50, 4.0)                       # the Iteration 2 attacks
    det = my_rate(my_flag(my_mahalanobis2(att, mu_hat, L), tau))
    att_big = simulate_attacks(D, 20000, 4.0, seed=99, n_before=0)
    det_big = my_rate(my_flag(my_mahalanobis2(att_big, mu_hat, L), tau))

    out(f"=== 1. n = {N}, d = {D}, seed {SEED}, alpha = {ALPHA} ===")
    out("-- against the truth --")
    z_mu = np.abs(mu_hat - mu) / se_mu(Sigma, N)
    z_S = np.abs(S_hat - Sigma) / se_sigma(Sigma, N)
    out(f"||mu_hat - mu||_2 = {np.linalg.norm(mu_hat - mu):.4f};  "
        f"max |mu_hat_j - mu_j| / SE = {z_mu.max():.2f}  (pass if < 3)")
    out(f"||Sigma_hat - Sigma||_F / ||Sigma||_F = "
        f"{np.linalg.norm(S_hat - Sigma) / np.linalg.norm(Sigma):.4f};  "
        f"max |Sigma_hat_jk - Sigma_jk| / SE = {z_S.max():.2f}  (pass if < 3)")
    verify.ROWS.append(dict(data="Simulation (n=5000)", quantity="mu_hat", compared_with="true mu",
                            abs_err=float(np.abs(mu_hat - mu).max()), rel_err=float(z_mu.max()),
                            ok="PASS" if z_mu.max() < 3 else "FAIL"))
    verify.ROWS.append(dict(data="Simulation (n=5000)", quantity="Sigma_hat", compared_with="true Sigma",
                            abs_err=float(np.abs(S_hat - Sigma).max()), rel_err=float(z_S.max()),
                            ok="PASS" if z_S.max() < 3 else "FAIL"))
    se_far = np.sqrt(ALPHA * (1 - ALPHA) / N)
    out(f"false-alarm rate = {far:.4f}; target {ALPHA} +/- 3 SE = "
        f"[{ALPHA - 3 * se_far:.4f}, {ALPHA + 3 * se_far:.4f}]  "
        f"{'PASS' if abs(far - ALPHA) <= 3 * se_far else 'FAIL'}")
    nc = mu.size and float(np.full(D, 4.0) @ np.linalg.solve(Sigma, np.full(D, 4.0)))
    det_true = stats.ncx2.sf(tau, D, nc)
    out(f"detection rate on the 50 Iteration 2 attacks (shift 4) = {det:.3f}; theory "
        f"(non-central chi^2, non-centrality {nc:.2f}) = {det_true:.3f}; "
        f"SE with 50 attacks = {np.sqrt(det_true * (1 - det_true) / 50):.3f}")
    out(f"detection rate on 20000 new attacks = {det_big:.4f} (theory {det_true:.4f}, "
        f"SE {np.sqrt(det_true * (1 - det_true) / 20000):.4f})")
    ks = stats.kstest(d2, "chi2", args=(D,))
    out(f"KS test of d^2 against chi^2_{D}: statistic {ks.statistic:.4f}, p-value {ks.pvalue:.3f}")

    out("-- against libraries (should be at rounding level) --")
    tag = "Simulation (n=5000)"
    report("mu_hat vs X.mean(axis=0)", mu_hat, X.mean(axis=0), 1e-8, tag, "numpy mean")
    report("Sigma_hat vs np.cov(bias=True)", S_hat, np.cov(X.T, bias=True), 1e-8, tag, "np.cov")
    report("Cholesky L vs np.linalg.cholesky", L, np.linalg.cholesky(S_hat), 1e-8, tag,
           "np.linalg.cholesky")
    diff = X - mu_hat
    d2_lib = np.sum(diff @ np.linalg.inv(S_hat) * diff, axis=1)
    report("d^2 vs numpy.linalg.inv", d2, d2_lib, 1e-8, tag, "np.linalg.inv")
    report("Gauss-Jordan inverse vs np.inv", my_inverse(S_hat), np.linalg.inv(S_hat), 1e-8,
           tag, "np.linalg.inv")
    report("Cholesky inverse vs np.inv", my_cholesky_inverse(L), np.linalg.inv(S_hat), 1e-8,
           tag, "np.linalg.inv")
    report("tau vs scipy chi2.ppf", tau, stats.chi2.ppf(1 - ALPHA, D), 1e-8, tag,
           "scipy.stats.chi2")
    report("condition number vs np.linalg.cond", my_condition_number(S_hat),
           np.linalg.cond(S_hat), 1e-8, tag, "np.linalg.cond")

    # ------------------------------------------------------------------ 2
    out("\n=== 2. Error against the truth for several n ===")
    ns = [50, 100, 500, 1000, 5000]
    R = 200
    mean_mu_err, mean_S_err, mean_far = [], [], []
    out(f"{'n':>5} {'seed-0 max z(mu)':>17} {'seed-0 max z(Sigma)':>20} "
        f"{'mean ||mu_hat-mu||':>19} {'mean rel. err Sigma':>20} {'lib max|d^2 diff|':>18}")
    for n in ns:
        Xn, _, _ = simulate_normal(D, n, SEED)
        m_n, S_n, L_n = fit(Xn)
        zmu = (np.abs(m_n - mu) / se_mu(Sigma, n)).max()
        zS = (np.abs(S_n - Sigma) / se_sigma(Sigma, n)).max()
        dn = Xn - m_n
        lib = np.abs(my_mahalanobis2(Xn, m_n, L_n)
                     - np.sum(dn @ np.linalg.inv(S_n) * dn, axis=1)).max()
        e_mu, e_S, f_n = [], [], []
        for r in range(1, R + 1):
            Xr, _, _ = simulate_normal(D, n, seed=r)
            m_r, S_r, L_r = fit(Xr)
            e_mu.append(np.linalg.norm(m_r - mu))
            e_S.append(np.linalg.norm(S_r - Sigma) / np.linalg.norm(Sigma))
            Xnew, _, _ = simulate_normal(D, 2000, seed=10000 + r)
            f_n.append(my_rate(my_flag(my_mahalanobis2(Xnew, m_r, L_r), tau)))
        mean_mu_err.append(np.mean(e_mu))
        mean_S_err.append(np.mean(e_S))
        mean_far.append(np.mean(f_n))
        out(f"{n:5d} {zmu:17.2f} {zS:20.2f} {np.mean(e_mu):19.4f} {np.mean(e_S):20.4f} {lib:18.1e}")
    slope_mu = np.polyfit(np.log(ns), np.log(mean_mu_err), 1)[0]
    slope_S = np.polyfit(np.log(ns), np.log(mean_S_err), 1)[0]
    out(f"log-log slopes: mu error {slope_mu:.3f}, Sigma error {slope_S:.3f}  (theory -0.5)")
    out("out-of-sample false-alarm rate by n (fit on n points, test on 2000 new points): "
        + ", ".join(f"n={n}: {f:.4f}" for n, f in zip(ns, mean_far)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].loglog(ns, mean_mu_err, "o-", label=r"$\|\hat\mu-\mu\|_2$")
    ax[0].loglog(ns, mean_S_err, "s-", label=r"$\|\hat\Sigma-\Sigma\|_F/\|\Sigma\|_F$")
    ref = np.array(ns, float)
    ax[0].loglog(ref, mean_mu_err[0] * np.sqrt(ns[0] / ref), "k--", lw=1, label=r"slope $-1/2$")
    ax[0].set(xlabel="n", ylabel="mean error over 200 runs", title="Error against the truth")
    ax[0].legend()
    ax[1].semilogx(ns, mean_far, "o-")
    ax[1].axhline(ALPHA, color="k", ls="--", label=r"target $\alpha=0.01$")
    ax[1].set(xlabel="n (training points)", ylabel="false-alarm rate on new points",
              title="False alarms against n")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "it3_sim_error_vs_n.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ 3
    out("\n=== 3. Stress test A: heavy tails (multivariate t, 3 d.f.) ===")
    Xt, _, _ = simulate_t(D, N, SEED)
    m_t, S_t, L_t = fit(Xt)
    d2t = my_mahalanobis2(Xt, m_t, L_t)
    far_t = my_rate(my_flag(d2t, tau))
    fars_t = []
    for r in range(1, R + 1):
        Xr, _, _ = simulate_t(D, N, seed=r)
        m_r, S_r, L_r = fit(Xr)
        fars_t.append(my_rate(my_flag(my_mahalanobis2(Xr, m_r, L_r), tau)))
    out(f"false-alarm rate = {far_t:.4f} (target {ALPHA}); over {R} runs "
        f"{np.mean(fars_t):.4f} +/- {np.std(fars_t, ddof=1):.4f}")
    out(f"KS p-value against chi^2_{D}: {stats.kstest(d2t, 'chi2', args=(D,)).pvalue:.1e}")

    probs = (np.arange(1, 200) - 0.5) / 199
    q_theory = my_chi2_quantile(probs, D)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, vals, title in ((ax[0], d2, "Normal data"), (ax[1], d2t, "t data, 3 d.f.")):
        a.plot(q_theory, np.quantile(vals, probs), "o", ms=3)
        top = max(q_theory.max(), np.quantile(vals, probs).max())
        a.plot([0, top], [0, top], "k--", lw=1)
        a.set(xlabel=r"$\chi^2_5$ quantile", ylabel=r"sample quantile of $d^2$",
              title=f"QQ plot: {title}")
    fig.tight_layout()
    fig.savefig(FIG / "it3_sim_qq.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ 4
    out("\n=== 4. Stress test B: column 6 is an exact copy of column 1 ===")
    Xd = add_duplicate_column(X, 0)
    _, S_d = my_mle(Xd)
    try:
        my_cholesky(S_d)
        out("my_cholesky succeeded (unexpected)")
    except ValueError as e:
        out(f"my_cholesky stops: {e}")
    out(f"check: smallest eigenvalue of Sigma_hat (numpy.linalg.eigvalsh) = "
        f"{np.linalg.eigvalsh(S_d).min():.2e}; np.linalg.cond = {np.linalg.cond(S_d):.2e}")
    out(f"rank of Sigma_hat (numpy.linalg.matrix_rank) = {np.linalg.matrix_rank(S_d)} of 6")

    # ------------------------------------------------------------------ 5
    out("\n=== 5. Assumption checks on the simulated data (should pass) ===")
    r1 = [my_autocorr(X[:, j]) for j in range(D)]
    out(f"A1 independence: lag-1 autocorrelation of each feature, max |r| = "
        f"{np.max(np.abs(r1)):.4f}  (3/sqrt(n) = {3 / np.sqrt(N):.4f})")
    g1, g2 = my_skew_kurt(X)
    out(f"A2 normality: max |skewness| = {np.abs(g1).max():.3f} (3 SE = "
        f"{3 * np.sqrt(6 / N):.3f}); max |excess kurtosis| = {np.abs(g2).max():.3f} "
        f"(3 SE = {3 * np.sqrt(24 / N):.3f}); KS p-value of d^2 = {ks.pvalue:.3f}")
    out(f"A3 positive definite: Cholesky succeeds; condition number = "
        f"{my_condition_number(S_hat):.3f}")
    out(f"A4 enough data: n / p = {N // D}")
    m_a, _, L_a = fit(X[: N // 2])
    far_half = my_rate(my_flag(my_mahalanobis2(X[N // 2:], m_a, L_a), tau))
    out(f"A5 stable over time: fit on the first half, false-alarm rate on the second half "
        f"= {far_half:.4f} (target {ALPHA})")
    out(f"A6 anomalies differ: detection rate {det:.3f} against theory {det_true:.3f}")

    (DOC / "check_simulation_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    with open(DOC / "verification_simulation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(verify.ROWS[0]))
        w.writeheader()
        w.writerows(verify.ROWS)


if __name__ == "__main__":
    main()
