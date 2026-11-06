"""Load and describe the CMU keystroke data (Iteration 2, section 7: real data).

Usage (from the repository root):
    python src/load_data.py

Prints the size, column types, missing values and per-subject counts of the data,
checks the linear dependency UD = DD - H, and saves two plots to figures/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "DSL-StrongPasswordData.csv")
FIGS = os.path.join(ROOT, "figures")


def load():
    """Return the full DataFrame and the lists of H, DD and UD feature columns."""
    if not os.path.exists(DATA):
        raise FileNotFoundError("Dataset not found. Run:  python src/download_data.py")
    df = pd.read_csv(DATA)
    H = [c for c in df.columns if c.startswith("H.")]
    DD = [c for c in df.columns if c.startswith("DD.")]
    UD = [c for c in df.columns if c.startswith("UD.")]
    return df, H, DD, UD


def user_matrix(df, subject, columns, log=False):
    """Data matrix X (n x d) for one subject: one row = one typing, one column = one feature."""
    X = df.loc[df["subject"] == subject, columns].to_numpy(dtype=float)
    return np.log(X) if log else X


def describe(df, H, DD, UD):
    feats = H + DD + UD
    print("=== Shape ===")
    print(f"{df.shape[0]} rows x {df.shape[1]} columns "
          f"(3 ID columns + {len(feats)} timing features: {len(H)} H, {len(DD)} DD, {len(UD)} UD)")
    print("\n=== Column types ===")
    print(df.dtypes.value_counts().to_string())
    print(f"subject: {df['subject'].dtype} (e.g. {df['subject'].iloc[0]}), "
          f"sessionIndex: 1-{df['sessionIndex'].max()}, rep: 1-{df['rep'].max()}")
    print("\n=== Missing values ===")
    print(f"total missing entries: {int(df.isna().sum().sum())}")
    print("\n=== Subjects ===")
    counts = df.groupby("subject").size()
    print(f"{counts.size} subjects, rows per subject: min {counts.min()}, max {counts.max()}")
    print("\n=== Timing features (seconds) ===")
    print(df[feats].describe().T[["mean", "std", "min", "max"]].round(4).to_string())
    print(f"\nnegative values: H {int((df[H] < 0).sum().sum())}, "
          f"DD {int((df[DD] < 0).sum().sum())}, UD {int((df[UD] < 0).sum().sum())} "
          "(UD < 0 means the next key was pressed before the previous key was released)")

    # UD_{a,b} = DD_{a,b} - H_a  ->  the UD columns are linear combinations of H and DD
    gap = np.abs(df[UD].to_numpy() - (df[DD].to_numpy() - df[H[:-1]].to_numpy())).max()
    X = df[feats].to_numpy()
    rank = np.linalg.matrix_rank(X - X.mean(axis=0))
    print(f"\n=== Linear dependency ===\nmax |UD - (DD - H)| = {gap:.1e}; "
          f"rank of centred {X.shape[0]}x{X.shape[1]} feature matrix = {rank}")


def plots(df, H):
    os.makedirs(FIGS, exist_ok=True)
    col = H[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(df[col], bins=80, color="#4C72B0")
    ax[0].set(title=f"{col} (seconds)", xlabel="hold time (s)", ylabel="count")
    ax[1].hist(np.log(df[col]), bins=80, color="#55A868")
    ax[1].set(title=f"log {col}", xlabel="log hold time", ylabel="count")
    fig.suptitle("Hold times are right-skewed; the log transform makes them more symmetric")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "hold_time_skew.png"), dpi=150)
    plt.close(fig)

    subjects = sorted(df["subject"].unique())[:8]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.boxplot([np.log(df.loc[df["subject"] == s, col]) for s in subjects], showfliers=False)
    ax.set_xticks(range(1, len(subjects) + 1), subjects)
    ax.set(title=f"log {col} for the first 8 subjects: each person has their own typical value",
           xlabel="subject", ylabel=f"log {col}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "hold_time_by_subject.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved figures/hold_time_skew.png and figures/hold_time_by_subject.png")


if __name__ == "__main__":
    df, H, DD, UD = load()
    describe(df, H, DD, UD)
    plots(df, H)
