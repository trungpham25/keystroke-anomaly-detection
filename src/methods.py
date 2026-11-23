"""methods.py -- From-scratch implementation of the Mahalanobis anomaly detector.

Every function implements one numbered step of the algorithm in the report
(Section "Algorithm", Iteration 3) and uses NumPy arrays and matrix products
only. numpy.linalg, SciPy and scikit-learn are NOT used here; they are only
used in the check scripts to verify these functions.

Symbols follow the report: X is n x p (one row per attempt), mu_hat and
Sigma_hat are the MLEs, L is the Cholesky factor (Sigma_hat = L L^T),
d2 is the squared Mahalanobis distance and tau the chi-squared threshold.

The functions that only use +, -, *, / (my_mean, my_center, my_covariance,
my_mle, my_inverse) also work on arrays of fractions.Fraction, so the small
example can be computed exactly.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Steps 1-3: maximum-likelihood estimates (Theorem 1)
# ---------------------------------------------------------------------------

# Step 1:  mu_hat = (1/n) X^T 1
def my_mean(X):
    n = X.shape[0]
    ones = np.ones(n, dtype=X.dtype) if X.dtype == object else np.ones(n)
    return (X.T @ ones) / n


# Step 2:  X_c = X - 1 mu_hat^T
def my_center(X, mu_hat):
    return X - mu_hat[None, :]


# Step 3:  Sigma_hat = (1/n) X_c^T X_c      (divisor n: the MLE)
def my_covariance(Xc):
    n = Xc.shape[0]
    return (Xc.T @ Xc) / n


# Steps 1-3 together:  (mu_hat, Sigma_hat)
def my_mle(X):
    mu_hat = my_mean(X)
    return mu_hat, my_covariance(my_center(X, mu_hat))


# ---------------------------------------------------------------------------
# Step 4: Cholesky factorization  Sigma_hat = L L^T  (checks assumption A3)
# ---------------------------------------------------------------------------

# L_jj = sqrt( S_jj - sum_{k<j} L_jk^2 )
# L_ij = ( S_ij - sum_{k<j} L_ik L_jk ) / L_jj      for i > j
# If some S_jj - sum_{k<j} L_jk^2 <= 0 (up to rounding), S is not positive
# definite and no Cholesky factor exists.
def my_cholesky(S, tol=1e-12):
    S = np.asarray(S, dtype=float)
    p = S.shape[0]
    L = np.zeros((p, p))
    scale = np.max(np.diag(S))
    for j in range(p):
        pivot = S[j, j] - L[j, :j] @ L[j, :j]
        if pivot <= tol * scale:
            raise ValueError(
                f"matrix is not positive definite: pivot {j + 1} is {pivot:.3e}")
        L[j, j] = np.sqrt(pivot)
        L[j + 1:, j] = (S[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    return L


# ---------------------------------------------------------------------------
# Step 5: triangular solves
# ---------------------------------------------------------------------------

# Forward substitution, solves L Z = B for lower-triangular L:
#   Z_i = ( B_i - sum_{k<i} L_ik Z_k ) / L_ii
def my_forward_sub(L, B):
    B = np.asarray(B, dtype=float)
    Z = np.zeros_like(B)
    for i in range(L.shape[0]):
        Z[i] = (B[i] - L[i, :i] @ Z[:i]) / L[i, i]
    return Z


# Back substitution, solves U Z = B for upper-triangular U:
#   Z_i = ( B_i - sum_{k>i} U_ik Z_k ) / U_ii
def my_back_sub(U, B):
    B = np.asarray(B, dtype=float)
    Z = np.zeros_like(B)
    for i in range(U.shape[0] - 1, -1, -1):
        Z[i] = (B[i] - U[i, i + 1:] @ Z[i + 1:]) / U[i, i]
    return Z


# ---------------------------------------------------------------------------
# Step 6: squared Mahalanobis distance (Theorem 2, Proposition 2)
# ---------------------------------------------------------------------------

# z_j = L^{-1}(y_j - mu_hat)  (forward substitution),   d2_j = z_j^T z_j
# Since Sigma_hat^{-1} = L^{-T} L^{-1}, this equals (y - mu)^T Sigma^{-1} (y - mu).
def my_mahalanobis2(Y, mu_hat, L):
    D = np.atleast_2d(np.asarray(Y, dtype=float)) - mu_hat[None, :]
    Z = my_forward_sub(L, D.T)          # p x m, column j is z_j
    return np.sum(Z ** 2, axis=0)


# ---------------------------------------------------------------------------
# Matrix inverse from scratch (Iteration 3, Question 2)
# ---------------------------------------------------------------------------

# Gauss-Jordan elimination with partial pivoting on [A | I] -> [I | A^{-1}].
# Works on floats and on fractions.Fraction (exact). With verbose=True the
# augmented matrix is printed after every row operation (small example).
def my_inverse(A, verbose=False):
    A = np.array(A)
    exact = A.dtype == object
    p = A.shape[0]
    if exact:
        I = np.array([[type(A[0, 0])(int(i == j)) for j in range(p)] for i in range(p)],
                     dtype=object)
        M = np.concatenate([A, I], axis=1)
    else:
        M = np.concatenate([A.astype(float), np.eye(p)], axis=1)

    def show(msg):
        if verbose:
            print(msg)
            for row in M:
                print("   [" + "  ".join(f"{str(v):>6}" for v in row[:p]) + "  |  "
                      + "  ".join(f"{str(v):>6}" for v in row[p:]) + "]")

    show("start: [A | I]")
    for c in range(p):
        r = c + int(np.argmax([abs(M[i, c]) for i in range(c, p)]))
        if M[r, c] == 0:
            raise ValueError("matrix is singular")
        if r != c:
            M[[c, r]] = M[[r, c]]
            show(f"swap rows {c + 1} and {r + 1}")
        M[c] = M[c] / M[c, c]
        show(f"divide row {c + 1} by its pivot")
        for i in range(p):
            if i != c and M[i, c] != 0:
                f = M[i, c]
                M[i] = M[i] - f * M[c]
                show(f"row {i + 1} <- row {i + 1} - ({f}) x row {c + 1}")
    return M[:, p:]


# Inverse from the Cholesky factor:  Sigma^{-1} = L^{-T} L^{-1}
def my_cholesky_inverse(L):
    Linv = my_forward_sub(L, np.eye(L.shape[0]))
    return Linv.T @ Linv


# ---------------------------------------------------------------------------
# Step 7: chi-squared threshold  tau = F_p^{-1}(1 - alpha)
# ---------------------------------------------------------------------------

# log Gamma(s) for s a positive multiple of 1/2, from Gamma(1) = 1,
# Gamma(1/2) = sqrt(pi) and Gamma(s) = (s - 1) Gamma(s - 1).
def my_log_gamma_half(s):
    k = int(round(2 * s))
    if k <= 0 or abs(2 * s - k) > 1e-12:
        raise ValueError("s must be a positive multiple of 1/2")
    out = 0.0 if k % 2 == 0 else 0.5 * np.log(np.pi)
    t = 1.0 if k % 2 == 0 else 0.5
    while t < s - 1e-12:
        out += np.log(t)
        t += 1.0
    return out


# Chi-squared CDF with p degrees of freedom:
#   F_p(x) = P(p/2, x/2),  P(s, y) = y^s e^{-y} sum_{k>=0} y^k / Gamma(s + k + 1)
# (series for the regularized lower incomplete gamma function).
def my_chi2_cdf(x, p):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    s = p / 2.0
    y = np.maximum(x / 2.0, 0.0)
    out = np.zeros_like(y)
    pos = y > 0
    yp = y[pos]
    log_term = s * np.log(yp) - yp - my_log_gamma_half(s + 1.0)
    term = np.exp(log_term)
    total = term.copy()
    k = 1
    while k < 100000:
        term = term * yp / (s + k)
        total += term
        if np.all(term <= 1e-17 * np.maximum(total, 1e-300)) and np.all(yp < s + k):
            break
        k += 1
    vals = np.minimum(total, 1.0)
    vals[(log_term < -700) & (yp > s)] = 1.0     # far right tail underflows
    out[pos] = vals
    return out


# Quantile F_p^{-1}(q) by bisection on the CDF (F_p is continuous and increasing).
def my_chi2_quantile(q, p, iters=200):
    q = np.atleast_1d(np.asarray(q, dtype=float))
    lo = np.zeros_like(q)
    hi = np.full_like(q, p + 10.0 * np.sqrt(2.0 * p) + 10.0)
    while np.any(my_chi2_cdf(hi, p) < q):
        hi = np.where(my_chi2_cdf(hi, p) < q, 2.0 * hi, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = my_chi2_cdf(mid, p) < q
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
        if np.all(hi - lo <= 1e-13 * np.maximum(hi, 1.0)):
            break
    return 0.5 * (lo + hi)


# Step 7:  tau = F_p^{-1}(1 - alpha)
def my_threshold(alpha, p):
    return float(my_chi2_quantile(1.0 - alpha, p)[0])


# ---------------------------------------------------------------------------
# Steps 8-9: decision and error rates
# ---------------------------------------------------------------------------

# Step 8: flag attempt j as an anomaly if d2_j > tau
def my_flag(d2, tau):
    return np.asarray(d2) > tau


# Step 9: rate = (number flagged) / (number of attempts).
# On genuine attempts this is the false-alarm rate, on impostor attempts the
# detection rate.
def my_rate(flags):
    flags = np.asarray(flags)
    return float(flags.sum() / flags.size)


# ---------------------------------------------------------------------------
# Condition number (assumption A3, Iteration 3 Question 5)
# ---------------------------------------------------------------------------

# Largest eigenvalue by power iteration:  v <- S v / ||S v||,  lambda = v^T S v
def my_lambda_max(S, tol=1e-14, max_iter=200000, seed=0):
    v = np.random.default_rng(seed).normal(size=S.shape[0])
    v = v / np.sqrt(v @ v)
    lam = v @ S @ v
    for _ in range(max_iter):
        w = S @ v
        v = w / np.sqrt(w @ w)
        new = v @ S @ v
        if abs(new - lam) <= tol * abs(new):
            return float(new)
        lam = new
    return float(lam)


# Smallest eigenvalue by inverse iteration with the Cholesky factor:
#   solve S w = v  (L u = v, then L^T w = u),  v <- w / ||w||,  lambda = v^T S v
def my_lambda_min(L, tol=1e-14, max_iter=200000, seed=0):
    S = L @ L.T
    v = np.random.default_rng(seed).normal(size=L.shape[0])
    v = v / np.sqrt(v @ v)
    lam = v @ S @ v
    for _ in range(max_iter):
        w = my_back_sub(L.T, my_forward_sub(L, v))
        v = w / np.sqrt(w @ w)
        new = v @ S @ v
        if abs(new - lam) <= tol * abs(new):
            return float(new)
        lam = new
    return float(lam)


# kappa(S) = lambda_max / lambda_min   (2-norm condition number of an SPD matrix)
def my_condition_number(S):
    L = my_cholesky(S)
    return my_lambda_max(S) / my_lambda_min(L)


# ---------------------------------------------------------------------------
# Helpers for the assumption checks
# ---------------------------------------------------------------------------

# Log transform of the selected columns:  x_ij <- log x_ij  (columns in `cols`)
def my_log_columns(X, cols):
    X = np.array(X, dtype=float)
    X[:, cols] = np.log(X[:, cols])
    return X


# Sample skewness  g1 = m3 / m2^{3/2}  and excess kurtosis  g2 = m4 / m2^2 - 3,
# where m_k = (1/n) sum (x_i - xbar)^k, column by column.
def my_skew_kurt(X):
    Xc = X - X.mean(axis=0)
    m2 = (Xc ** 2).mean(axis=0)
    m3 = (Xc ** 3).mean(axis=0)
    m4 = (Xc ** 4).mean(axis=0)
    return m3 / m2 ** 1.5, m4 / m2 ** 2 - 3.0


# Lag-k autocorrelation  r_k = sum (x_t - xbar)(x_{t+k} - xbar) / sum (x_t - xbar)^2
def my_autocorr(x, lag=1):
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    return float((xc[:-lag] @ xc[lag:]) / (xc @ xc))


# Standard normal quantile from the chi-squared quantile with 1 d.f.:
# if Z ~ N(0,1) then Z^2 ~ chi^2_1, so z_q = sign(q - 1/2) sqrt(F_1^{-1}(|2q - 1|)).
def my_normal_quantile(q):
    q = np.atleast_1d(np.asarray(q, dtype=float))
    return np.sign(q - 0.5) * np.sqrt(my_chi2_quantile(np.abs(2 * q - 1), 1))
