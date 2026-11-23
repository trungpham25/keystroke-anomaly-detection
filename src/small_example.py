"""small_example.py -- Iteration 3, Question 1: the tiny example, step by step.

Run from the repository root:
    python src/small_example.py

Four points in R^2 and one new point. Every step uses the same functions as
the simulation and the real data (src/methods.py). Steps 1-3 and the inverse
are computed exactly with fractions.Fraction. The Cholesky route (steps 4-6)
needs square roots, so it runs in floating point and serves as the second check.

Output is also written to docs/small_example_log.txt.
"""
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods import (my_mean, my_center, my_covariance, my_inverse, my_cholesky,
                     my_forward_sub, my_mahalanobis2, my_threshold, my_flag)
from verify import report

ROOT = Path(__file__).resolve().parents[1]
LOG = []


def out(s=""):
    print(s)
    LOG.append(str(s))


def show(name, A):
    A = np.atleast_2d(A)
    out(f"{name} =")
    for row in A:
        out("   [" + "  ".join(f"{str(v):>6}" for v in row) + "]")


def main():
    # data: n = 4 attempts, p = 2 features
    X = np.array([[F(0), F(0)], [F(2), F(1)], [F(2), F(3)], [F(4), F(4)]], dtype=object)
    x_new = np.array([F(4), F(3)], dtype=object)
    n, p = X.shape
    show("X (4 x 2)", X)
    show("new point x", x_new)

    out("\nStep 1: mu_hat = (1/n) X^T 1")
    mu = my_mean(X)
    show("X^T 1", X.T @ np.array([F(1)] * n, dtype=object))
    show("mu_hat", mu)

    out("\nStep 2: X_c = X - 1 mu_hat^T")
    Xc = my_center(X, mu)
    show("X_c", Xc)

    out("\nStep 3: Sigma_hat = (1/n) X_c^T X_c")
    show("X_c^T X_c", Xc.T @ Xc)
    S = my_covariance(Xc)
    show("Sigma_hat", S)
    det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
    out(f"det(Sigma_hat) = {det}  (> 0 and Sigma_11 > 0, so Sigma_hat is positive definite)")

    out("\nInverse by Gauss-Jordan on [Sigma_hat | I] (exact fractions):")
    S_inv = my_inverse(S, verbose=True)
    show("Sigma_hat^{-1}", S_inv)

    out("\nMahalanobis distance of the new point:")
    diff = x_new - mu
    show("x - mu_hat", diff)
    v = S_inv @ diff
    show("Sigma_hat^{-1} (x - mu_hat)", v)
    d2_exact = diff @ v
    out(f"d^2 = (x - mu_hat)^T Sigma_hat^{{-1}} (x - mu_hat) = {d2_exact}")

    out("\n--- Second-way checks ---")
    show("Sigma_hat Sigma_hat^{-1}  (must be I exactly)", S @ S_inv)

    out("\nCholesky route (floating point), steps 4-6 of the algorithm:")
    Sf = S.astype(float)
    L = my_cholesky(Sf)
    out(f"L = [[{L[0, 0]:.6f}, 0], [{L[1, 0]:.6f}, {L[1, 1]:.6f}]]"
        f"   (L_11 = sqrt(2), L_21 = 2/sqrt(2) = sqrt(2), L_22 = sqrt(5/2 - 2) = sqrt(1/2))")
    report("L L^T vs Sigma_hat", L @ L.T, Sf)
    z = my_forward_sub(L, diff.astype(float))
    out(f"z = L^-1 (x - mu_hat) = [{z[0]:.6f}, {z[1]:.6f}],  z^T z = {z @ z:.12f}")
    d2_chol = my_mahalanobis2(x_new.astype(float)[None, :], mu.astype(float), L)[0]
    report("d^2: Cholesky vs exact fraction", d2_chol, float(d2_exact))

    out("\nThreshold for p = 2 (closed form F_2^{-1}(1 - alpha) = -2 log alpha):")
    for alpha in (0.05, 0.01):
        tau = my_threshold(alpha, p)
        report(f"tau(alpha={alpha}) vs -2 log alpha", tau, -2 * np.log(alpha))
        out(f"   tau = {tau:.4f}; d^2 = {float(d2_exact)} > tau ? "
            f"{bool(my_flag(float(d2_exact), tau))}  -> not an anomaly")

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "small_example_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
