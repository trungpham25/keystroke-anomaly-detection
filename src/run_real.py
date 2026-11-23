"""run_real.py -- Iteration 3: run the detector on the real keystroke data and verify.

Run from the repository root:
    python src/run_real.py

Protocol (Project 9 box of the Iteration 3 handout):
  genuine subject s002: fit on repetitions 1-200, test on repetitions 201-400;
  impostors: repetitions 1-50 of each of the other 50 subjects;
  alpha = 0.01 and 0.05, with and without a log transform of the hold times.
Features: the 21 columns H + UD (the 10 DD columns are dropped because
DD = H + UD exactly, see Iteration 2).

Also: library comparisons, Iteration 3 Questions 3-5, the assumption checks
A1-A6, and the same protocol repeated with every subject as the genuine user.

Outputs: docs/run_real_log.txt, docs/verification_real.csv,
figures/it3_real_qq.png, figures/it3_real_feature_qq.png,
figures/it3_real_skewness.png
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
from load_data import load_data, feature_columns, feature_groups
from methods import (my_mle, my_cholesky, my_mahalanobis2, my_inverse,
                     my_cholesky_inverse, my_threshold, my_flag, my_rate,
                     my_condition_number, my_chi2_quantile, my_log_columns,
                     my_skew_kurt, my_autocorr, my_normal_quantile)
import verify
from verify import report

ROOT = Path(__file__).resolve().parents[1]
FIG, DOC = ROOT / "figures", ROOT / "docs"
GENUINE = "s002"
ALPHAS = (0.01, 0.05)
LOG = []


def out(s=""):
    print(s)
    LOG.append(str(s))


def prepare():
    """Load, clean and split the data. Returns the cleaned frame and column info."""
    df = load_data()
    out("=== Cleaning steps ===")
    out(f"rows {len(df)}, columns {df.shape[1]}, missing values {int(df.isna().sum().sum())}")
    dup = int(df.duplicated(["subject", "sessionIndex", "rep"]).sum())
    out(f"duplicate (subject, session, rep) rows: {dup}; rows removed: 0")
    df = df.sort_values(["subject", "sessionIndex", "rep"]).reset_index(drop=True)
    df["r"] = (df["sessionIndex"] - 1) * 50 + df["rep"]      # repetition index 1..400
    cols = feature_columns(df.drop(columns="r"))
    g = feature_groups(cols)
    use = g["H"] + g["UD"]
    out(f"features kept: {len(g['H'])} H + {len(g['UD'])} UD = {len(use)}; "
        f"dropped {len(g['DD'])} DD columns (DD = H + UD exactly)")
    out(f"smallest hold time {df[g['H']].to_numpy().min():.4f} s (> 0, so log H is defined); "
        f"negative UD entries {int((df[g['UD']] < 0).to_numpy().sum())} (log UD not defined)")
    return df, use, list(range(len(g["H"])))      # H columns come first in `use`


def split(df, use, subject):
    """Train = reps 1-200 of `subject`; test = reps 201-400; impostors = reps 1-50 of others."""
    me = df["subject"] == subject
    train = df.loc[me & (df["r"] <= 200), use].to_numpy(float)
    test = df.loc[me & (df["r"] > 200), use].to_numpy(float)
    test_sess = df.loc[me & (df["r"] > 200), "sessionIndex"].to_numpy()
    imp_mask = (~me) & (df["r"] <= 50)
    imp = df.loc[imp_mask, use].to_numpy(float)
    imp_subj = df.loc[imp_mask, "subject"].to_numpy()
    return train, test, test_sess, imp, imp_subj


def run_detector(train, test, imp, p):
    mu_hat, S_hat = my_mle(train)
    L = my_cholesky(S_hat)
    d2_train = my_mahalanobis2(train, mu_hat, L)
    d2_test = my_mahalanobis2(test, mu_hat, L)
    d2_imp = my_mahalanobis2(imp, mu_hat, L)
    rates = {}
    for a in ALPHAS:
        tau = my_threshold(a, p)
        rates[a] = (my_rate(my_flag(d2_test, tau)), my_rate(my_flag(d2_imp, tau)),
                    my_rate(my_flag(d2_train, tau)))
    return mu_hat, S_hat, L, d2_train, d2_test, d2_imp, rates


def main():
    FIG.mkdir(exist_ok=True)
    DOC.mkdir(exist_ok=True)
    df, use, hcols = prepare()
    p = len(use)
    train0, test0, test_sess, imp0, imp_subj = split(df, use, GENUINE)
    out(f"genuine {GENUINE}: train {train0.shape[0]} attempts, test {test0.shape[0]} attempts; "
        f"impostors {imp0.shape[0]} attempts from {len(set(imp_subj))} subjects")

    variants = {"raw": (train0, test0, imp0),
                "log H": tuple(my_log_columns(A, hcols) for A in (train0, test0, imp0))}
    res = {}
    out("\n=== Results for the genuine subject ===")
    out(f"{'features':8s} {'alpha':>6s} {'tau':>8s} {'false-alarm (test)':>19s} "
        f"{'detection':>10s} {'false-alarm (train)':>20s}")
    for name, (tr, te, im) in variants.items():
        res[name] = run_detector(tr, te, im, p)
        for a in ALPHAS:
            fa, de, fa_tr = res[name][6][a]
            out(f"{name:8s} {a:6.2f} {my_threshold(a, p):8.3f} {fa:19.3f} {de:10.3f} {fa_tr:20.3f}")

    out("\n=== Library comparisons (genuine subject) ===")
    for name, (tr, te, im) in variants.items():
        mu_hat, S_hat, L, d2_tr, d2_te, d2_im, _ = res[name]
        tag = f"Real data ({name})"
        report(f"[{name}] Sigma_hat vs np.cov(bias=True)", S_hat, np.cov(tr.T, bias=True),
               1e-8, tag, "np.cov")
        report(f"[{name}] Cholesky L vs np.linalg.cholesky", L, np.linalg.cholesky(S_hat),
               1e-8, tag, "np.linalg.cholesky")
        diff = np.vstack([te, im]) - mu_hat
        d2_lib = np.sum(diff @ np.linalg.inv(S_hat) * diff, axis=1)
        report(f"[{name}] d^2 vs numpy.linalg.inv", np.concatenate([d2_te, d2_im]), d2_lib,
               1e-8, tag, "np.linalg.inv")
        report(f"[{name}] Gauss-Jordan inverse vs np.inv", my_inverse(S_hat),
               np.linalg.inv(S_hat), 1e-8, tag, "np.linalg.inv")
        report(f"[{name}] Cholesky inverse vs np.inv", my_cholesky_inverse(L),
               np.linalg.inv(S_hat), 1e-8, tag, "np.linalg.inv")
        report(f"[{name}] condition number vs np.linalg.cond", my_condition_number(S_hat),
               np.linalg.cond(S_hat), 1e-8, tag, "np.linalg.cond")
    for a in ALPHAS:
        report(f"tau (p={p}, alpha={a}) vs scipy", my_threshold(a, p),
               stats.chi2.ppf(1 - a, p), 1e-8, "Real data", "scipy.stats.chi2")

    # ---------------- Question 3: QQ plot of d^2 against chi^2_21
    out("\n=== Q3: d^2 against chi^2_21 ===")
    probs = (np.arange(1, 200) - 0.5) / 199
    q_theory = my_chi2_quantile(probs, p)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a_, name in zip(ax, variants):
        _, _, _, d2_tr, d2_te, _, _ = res[name]
        for vals, lab, mk in ((d2_tr, "training (reps 1-200)", "o"),
                              (d2_te, "test (reps 201-400)", "s")):
            a_.plot(q_theory, np.quantile(vals, probs), mk, ms=3, label=lab)
        top = q_theory.max() * 1.15
        a_.plot([0, top], [0, top], "k--", lw=1, label="y = x")
        ymax = 1.05 * max(np.quantile(d2_te, probs).max(), np.quantile(d2_tr, probs).max())
        a_.set(xlim=(0, top), ylim=(0, ymax), xlabel=r"$\chi^2_{21}$ quantile",
               ylabel=r"sample quantile of $d^2$", title=f"{GENUINE}, {name} features")
        a_.legend(fontsize=8)
        ks_tr = stats.kstest(d2_tr, "chi2", args=(p,)).pvalue
        ks_te = stats.kstest(d2_te, "chi2", args=(p,)).pvalue
        out(f"[{name}] median d^2: training {np.median(d2_tr):.1f}, test {np.median(d2_te):.1f} "
            f"(chi^2_21 median {my_chi2_quantile(0.5, p)[0]:.1f}); 99th percentile: training "
            f"{np.quantile(d2_tr, .99):.1f}, test {np.quantile(d2_te, .99):.1f} "
            f"(chi^2_21: {my_chi2_quantile(0.99, p)[0]:.1f}); KS p-value training {ks_tr:.1e}, "
            f"test {ks_te:.1e}")
    fig.tight_layout()
    fig.savefig(FIG / "it3_real_qq.png", dpi=150)
    plt.close(fig)

    # ---------------- Question 4: normality of each feature
    out("\n=== Q4: normality of each feature (genuine subject, training data) ===")
    g1_raw, g2_raw = my_skew_kurt(variants["raw"][0])
    g1_log, g2_log = my_skew_kurt(variants["log H"][0])
    n_tr = train0.shape[0]
    out(f"3 SE for n = {n_tr}: skewness {3 * np.sqrt(6 / n_tr):.2f}, "
        f"excess kurtosis {3 * np.sqrt(24 / n_tr):.2f}")
    out(f"{'feature':18s} {'skew raw':>9s} {'kurt raw':>9s} {'skew log':>9s} {'kurt log':>9s}")
    for j, c in enumerate(use):
        lg = "" if j in hcols else "   (not logged)"
        out(f"{c:18s} {g1_raw[j]:9.2f} {g2_raw[j]:9.2f} {g1_log[j]:9.2f} {g2_log[j]:9.2f}{lg}")
    bad_raw = [use[j] for j in range(p) if abs(g1_raw[j]) > 1 or abs(g2_raw[j]) > 3]
    bad_log = [use[j] for j in range(p) if abs(g1_log[j]) > 1 or abs(g2_log[j]) > 3]
    out(f"features with |skewness| > 1 or |excess kurtosis| > 3: raw {len(bad_raw)} of {p} "
        f"({', '.join(bad_raw)}); log H {len(bad_log)} of {p} ({', '.join(bad_log)})")
    out(f"median |skewness| of the H features: raw {np.median(np.abs(g1_raw[hcols])):.2f}, "
        f"log {np.median(np.abs(g1_log[hcols])):.2f}; of the UD features: "
        f"{np.median(np.abs(g1_raw[len(hcols):])):.2f}")

    fig, ax = plt.subplots(3, 7, figsize=(16, 7))
    zq = my_normal_quantile(probs)
    for j, a_ in enumerate(ax.ravel()):
        for A, col, lab in ((variants["raw"][0], "C0", "raw"), (variants["log H"][0], "C1", "log")):
            if lab == "log" and j not in hcols:
                continue
            x = A[:, j]
            a_.plot(zq, (np.quantile(x, probs) - x.mean()) / x.std(), ".", ms=2, color=col,
                    label=lab)
        a_.plot([-3, 3], [-3, 3], "k--", lw=0.8)
        a_.set_title(use[j], fontsize=8)
        a_.tick_params(labelsize=6)
    ax[0, 0].legend(fontsize=6)
    fig.suptitle(f"Normal QQ plots of the 21 features ({GENUINE}, training data); "
                 "orange = log hold time", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "it3_real_feature_qq.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    xs = np.arange(p)
    ax.bar(xs - 0.2, g1_raw, 0.4, label="raw")
    ax.bar(xs + 0.2, g1_log, 0.4, label="log hold times")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs, use, rotation=70, fontsize=7)
    ax.set(ylabel="skewness", title=f"Skewness of each feature ({GENUINE}, training data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "it3_real_skewness.png", dpi=150)
    plt.close(fig)

    # ---------------- Question 5: singular or nearly singular Sigma_hat
    out("\n=== Q5: singular and nearly singular Sigma_hat ===")
    all31 = feature_columns(df.drop(columns="r"))
    tr31 = df.loc[(df["subject"] == GENUINE) & (df["r"] <= 200), all31].to_numpy(float)
    _, S31 = my_mle(tr31)
    try:
        my_cholesky(S31)
        out("all 31 features: Cholesky succeeded (unexpected)")
    except ValueError as e:
        out(f"all 31 features: my_cholesky stops: {e}")
    out(f"check: rank {np.linalg.matrix_rank(S31)} of 31, np.linalg.cond = {np.linalg.cond(S31):.1e}")
    for name in variants:
        S_hat = res[name][1]
        out(f"[{name}] 21 features: condition number {my_condition_number(S_hat):.1f}")
    tr, te, im = variants["log H"]
    mu_hat, S_hat = my_mle(tr)
    for eps_rel in (1e-3, 1e-2, 1e-1):
        eps = eps_rel * np.trace(S_hat) / p
        S_r = S_hat + eps * np.eye(p)
        L_r = my_cholesky(S_r)
        tau = my_threshold(0.01, p)
        fa = my_rate(my_flag(my_mahalanobis2(te, mu_hat, L_r), tau))
        de = my_rate(my_flag(my_mahalanobis2(im, mu_hat, L_r), tau))
        out(f"[log H] ridge Sigma_hat + eps I, eps = {eps_rel:g} x trace/p: condition number "
            f"{my_condition_number(S_r):.1f}; alpha = 0.01: false-alarm {fa:.3f}, detection {de:.3f}")
    mu31, S31 = my_mle(tr31)
    eps = 1e-3 * np.trace(S31) / 31
    L31 = my_cholesky(S31 + eps * np.eye(31))
    out(f"all 31 features with ridge eps = 1e-3 x trace/p: Cholesky succeeds, condition number "
        f"{my_condition_number(S31 + eps * np.eye(31)):.1e}")

    # ---------------- Assumption checks A1-A6
    out("\n=== Assumption checks on the real data (genuine subject) ===")
    for name, (tr, te, im) in variants.items():
        mu_hat, S_hat, L, d2_tr, d2_te, d2_im, rates = res[name]
        out(f"-- {name} features --")
        r_d2 = my_autocorr(d2_tr)
        r_feat = [my_autocorr(tr[:, j]) for j in range(p)]
        out(f"A1 independence: lag-1 autocorrelation of d^2 over the 200 training attempts "
            f"= {r_d2:.3f}; of the features: median {np.median(r_feat):.3f}, max "
            f"{np.max(r_feat):.3f}  (3/sqrt(n) = {3 / np.sqrt(len(d2_tr)):.3f})")
        means = [d2_tr[(k - 1) * 50: k * 50].mean() for k in range(1, 5)]
        out(f"    mean d^2 in training sessions 1-4: " + ", ".join(f"{m:.1f}" for m in means))
        fa01 = rates[0.01][0]
        out(f"A2 normality: false-alarm rate on test {fa01:.3f} against alpha 0.01; "
            f"on training {rates[0.01][2]:.3f}; see Q3 and Q4")
        out(f"A3 positive definite: Cholesky succeeds; condition number "
            f"{my_condition_number(S_hat):.1f}")
        tau = my_threshold(0.01, p)
        fa_by_n = []
        for n_fit in (50, 100, 200):
            m_n, S_n = my_mle(tr[:n_fit])
            L_n = my_cholesky(S_n)
            fa_by_n.append(my_rate(my_flag(my_mahalanobis2(te, m_n, L_n), tau)))
        out(f"A4 enough data: n = 200, p = {p}, n/p = {200 / p:.1f}; false-alarm (alpha 0.01) "
            f"when fitting on the first 50 / 100 / 200 attempts: "
            + " / ".join(f"{f:.3f}" for f in fa_by_n))
        per_sess = [my_rate(my_flag(d2_te[test_sess == s], tau)) for s in (5, 6, 7, 8)]
        out(f"A5 stable over time: false-alarm (alpha 0.01) in test sessions 5-8: "
            + ", ".join(f"{f:.2f}" for f in per_sess))
        det_each = np.array([my_rate(my_flag(d2_im[imp_subj == s], tau))
                             for s in sorted(set(imp_subj))])
        out(f"A6 impostors differ: detection (alpha 0.01) per impostor: min {det_each.min():.2f}, "
            f"median {np.median(det_each):.2f}, max {det_each.max():.2f}; "
            f"{int((det_each < 0.5).sum())} of {len(det_each)} impostors detected less than half "
            f"the time")

    # ---------------- Every subject as the genuine user
    out("\n=== Same protocol with each of the 51 subjects as the genuine user ===")
    subjects = sorted(df["subject"].unique())
    table = {name: {a: [] for a in ALPHAS} for name in variants}
    for s in subjects:
        tr_s, te_s, _, im_s, _ = split(df, use, s)
        for name in variants:
            if name == "log H":
                tr_v, te_v, im_v = (my_log_columns(A, hcols) for A in (tr_s, te_s, im_s))
            else:
                tr_v, te_v, im_v = tr_s, te_s, im_s
            rates = run_detector(tr_v, te_v, im_v, p)[6]
            for a in ALPHAS:
                table[name][a].append(rates[a][:2])
    for name in variants:
        for a in ALPHAS:
            arr = np.array(table[name][a])
            out(f"{name:6s} alpha {a:.2f}: mean false-alarm {arr[:, 0].mean():.3f} "
                f"(median {np.median(arr[:, 0]):.3f}), mean detection {arr[:, 1].mean():.3f} "
                f"(median {np.median(arr[:, 1]):.3f})")

    (DOC / "run_real_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    with open(DOC / "verification_real.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(verify.ROWS[0]))
        w.writeheader()
        w.writerows(verify.ROWS)


if __name__ == "__main__":
    main()
