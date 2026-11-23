"""simulate.py -- Data with known truth (same model and seed as Iteration 2).

True model: mu = 0 and Sigma = B B^T + d I, where B is d x d standard normal,
drawn with seed 0. simulate_normal(d=5, n=5000, seed=0) returns exactly the
Iteration 2 data set (same random stream: B first, then X).

For other seeds the true (mu, Sigma) stay the seed-0 values, so runs with
different seeds or sample sizes all estimate the same truth.
"""
import numpy as np

TRUTH_SEED = 0


def make_truth(d=5):
    """Return (mu, Sigma) with Sigma = B B^T + d I (positive definite)."""
    B = np.random.default_rng(TRUTH_SEED).standard_normal((d, d))
    return np.zeros(d), B @ B.T + d * np.eye(d)


def _data_rng(d, seed):
    """Random stream for the data. It skips the d*d draws used for B, so that
    seed 0 continues exactly where Iteration 2 drew X."""
    rng = np.random.default_rng(seed)
    rng.standard_normal((d, d))
    return rng


def simulate_normal(d=5, n=5000, seed=0):
    """n draws from N(mu, Sigma). Returns X (n x d), mu, Sigma."""
    mu, Sigma = make_truth(d)
    X = _data_rng(d, seed).multivariate_normal(mu, Sigma, size=n)
    return X, mu, Sigma


def simulate_t(d=5, n=5000, seed=0, df=3):
    """Stress test (A2): multivariate t with `df` degrees of freedom, scaled so
    that its mean and covariance are the same mu and Sigma as the normal model.
    x = mu + sqrt((df-2)/df) * C z / sqrt(w/df),  C C^T = Sigma,
    z ~ N(0, I), w ~ chi^2_df."""
    mu, Sigma = make_truth(d)
    rng = _data_rng(d, seed)
    C = np.linalg.cholesky(Sigma)   # data generation only, not the method
    z = rng.standard_normal((n, d))
    w = rng.chisquare(df, size=n)
    scale = np.sqrt((df - 2) / df) / np.sqrt(w / df)
    return mu + (z @ C.T) * scale[:, None], mu, Sigma


def simulate_attacks(d=5, n=50, shift=4.0, seed=0, n_before=5000):
    """Anomalies: n draws from N(mu + shift * 1, Sigma).
    The stream first skips B and the n_before genuine points, so with the
    defaults these are exactly the 50 attacks of Iteration 2."""
    mu, Sigma = make_truth(d)
    rng = _data_rng(d, seed)
    rng.multivariate_normal(mu, Sigma, size=n_before)
    return rng.multivariate_normal(mu + shift, Sigma, size=n)


def add_duplicate_column(X, j=0):
    """Stress test (A3): append an exact copy of column j."""
    return np.column_stack([X, X[:, j]])
