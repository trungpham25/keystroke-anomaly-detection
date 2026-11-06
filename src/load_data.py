"""
load_data.py -- Load and describe the CMU Keystroke Dynamics Benchmark.

Run from the repository root:
    python src/load_data.py

The dataset is downloaded automatically into data/ on the first run.

Outputs (written to figures/ and docs/):
    figures/hist_hold_raw_vs_log.png    skew of a hold time, raw vs log
    figures/corr_heatmap_user.png       feature correlations for one user
    figures/eigenvalues_user.png        eigenvalues of the sample covariance
    docs/data_summary.txt               numbers to copy into the report
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download_data import ensure_data  # sits next to this file in src/

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "DSL-StrongPasswordData.csv"
FIG_DIR = ROOT / "figures"
DOC_DIR = ROOT / "docs"

ID_COLS = ["subject", "sessionIndex", "rep"]


def load_data(path=DATA_PATH):
    """Return the full dataframe (20,400 rows x 34 columns).

    If the CSV is not in data/ yet, it is downloaded automatically
    (see src/download_data.py).
    """
    if Path(path) == DATA_PATH:
        ensure_data(verbose=True)
    elif not Path(path).exists():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)


def feature_columns(df):
    """The 31 timing features (everything except the three ID columns)."""
    return [c for c in df.columns if c not in ID_COLS]


def feature_groups(cols):
    """Split features into hold (H), down-down (DD) and up-down (UD) timings."""
    return {
        "H": [c for c in cols if c.startswith("H.")],
        "DD": [c for c in cols if c.startswith("DD.")],
        "UD": [c for c in cols if c.startswith("UD.")],
    }


def user_matrix(df, subject, cols=None, log=False):
    """Data matrix X (400 x p) for one subject. Rows = typing attempts."""
    cols = cols or feature_columns(df)
    X = df.loc[df["subject"] == subject, cols].to_numpy(dtype=float)
    return np.log(X) if log else X


def describe(df):
    """Build a plain-text description of the data for the report."""
    cols = feature_columns(df)
    groups = feature_groups(cols)
    lines = []
    add = lines.append

    add("=== DATA DESCRIPTION: CMU Keystroke Dynamics Benchmark ===")
    add(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
    add(f"Subjects: {df['subject'].nunique()}   "
        f"Sessions per subject: {df['sessionIndex'].nunique()}   "
        f"Reps per session: {df['rep'].max()}")
    per_subj = df.groupby("subject").size()
    add(f"Rows per subject: min={per_subj.min()}, max={per_subj.max()}")
    add("")
    add("Variable types:")
    add("  subject (string ID), sessionIndex (int), rep (int)")
    add(f"  {len(cols)} timing features (float, seconds): "
        f"{len(groups['H'])} hold (H), {len(groups['DD'])} down-down (DD), "
        f"{len(groups['UD'])} up-down (UD)")
    add("")
    add(f"Missing values (total): {int(df.isna().sum().sum())}")
    add(f"Non-positive timing entries: {int((df[cols] <= 0).sum().sum())}")
    add(f"Negative UD (key-up to next key-down) entries: "
        f"{int((df[groups['UD']] < 0).sum().sum())}")
    add("  -> log is defined for H and DD (all > 0) but NOT for UD (can be negative).")
    add("")

    X = df[cols]
    summ = X.agg(["mean", "std", "min", "max", "skew"]).T
    add("Pooled summary of selected features:")
    add(summ.loc[["H.period", "H.Return", "DD.period.t", "UD.period.t"]]
        .round(4).to_string())
    add(f"Median skewness, hold times:        {X[groups['H']].skew().median():.2f}")
    add(f"Median skewness, log hold times:    "
        f"{np.log(X[groups['H']]).skew().median():.2f}")
    add("")

    # Exact linear dependence: DD(a,b) = H(a) + UD(a,b).
    # (Key names such as "Shift.r" contain dots, so pair columns by order:
    #  the k-th DD and k-th UD column belong to the k-th hold column.)
    resid = []
    for h, dd, ud in zip(groups["H"][:-1], groups["DD"], groups["UD"]):
        assert dd[3:] == ud[3:], "DD/UD columns are not aligned"
        resid.append((df[dd] - df[h] - df[ud]).abs().max())
    add("Linear dependence check  DD(a,b) = H(a) + UD(a,b):")
    add(f"  max |DD - H - UD| over all rows and all {len(resid)} key pairs = "
        f"{max(resid):.2e}")

    # Rank and conditioning for one genuine user.
    subj = df["subject"].iloc[0]
    Xu = user_matrix(df, subj, cols)
    S = np.cov(Xu.T)
    rank = np.linalg.matrix_rank(Xu - Xu.mean(axis=0))
    ev = np.linalg.eigvalsh(S)
    add("")
    add(f"Covariance of subject {subj} (n={Xu.shape[0]}, p={Xu.shape[1]}), all 31 features:")
    add(f"  rank = {rank}  (full rank would be {len(cols)})")
    add(f"  eigenvalues below 1e-12: {int((ev < 1e-12).sum())}  -> S is singular, "
        f"S^-1 does not exist")

    keep = groups["H"] + groups["UD"]
    Xr = user_matrix(df, subj, keep)
    Sr = np.cov(Xr.T)
    evr = np.linalg.eigvalsh(Sr)
    add(f"After dropping the DD columns ({len(keep)} features left, H + UD):")
    add(f"  rank = {np.linalg.matrix_rank(Xr - Xr.mean(axis=0))}, "
        f"smallest eigenvalue = {evr.min():.3e}, condition number = "
        f"{evr.max() / evr.min():.3e}")
    return "\n".join(lines)


def make_plots(df):
    FIG_DIR.mkdir(exist_ok=True)
    cols = feature_columns(df)
    subj = df["subject"].iloc[0]

    # 1) Skew: raw vs log of one hold time (pooled over all subjects).
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    h = df["H.period"].to_numpy()
    ax[0].hist(h, bins=60, color="steelblue")
    ax[0].set(title="H.period (raw)", xlabel="seconds", ylabel="count")
    ax[1].hist(np.log(h), bins=60, color="darkorange")
    ax[1].set(title="H.period (log)", xlabel="log seconds")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hist_hold_raw_vs_log.png", dpi=150)
    plt.close(fig)

    # 2) Correlation heatmap for one user, all 31 features.
    Xu = user_matrix(df, subj, cols)
    C = np.corrcoef(Xu.T)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=6)
    ax.set_yticklabels(cols, fontsize=6)
    ax.set_title(f"Feature correlations, subject {subj}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "corr_heatmap_user.png", dpi=150)
    plt.close(fig)

    # 3) Eigenvalues of the covariance (shows the 10 zero eigenvalues).
    ev = np.sort(np.linalg.eigvalsh(np.cov(Xu.T)))[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(np.arange(1, len(ev) + 1), np.clip(ev, 1e-20, None), "o-")
    ax.set(title=f"Covariance eigenvalues, subject {subj} (31 features)",
           xlabel="index", ylabel="eigenvalue (log scale)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eigenvalues_user.png", dpi=150)
    plt.close(fig)


def main():
    df = load_data()
    text = describe(df)
    print(text)
    DOC_DIR.mkdir(exist_ok=True)
    (DOC_DIR / "data_summary.txt").write_text(text + "\n")
    make_plots(df)
    print(f"\nSaved plots to {FIG_DIR} and summary to {DOC_DIR / 'data_summary.txt'}")


if __name__ == "__main__":
    main()
