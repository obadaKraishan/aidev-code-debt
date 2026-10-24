"""
============================================================================
 stats.py, Statistical Tests + Effect Sizes for All RQ Analyses
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : One vetted implementation of every test the paper uses, so all
           five RQ scripts report statistics identically:
             - descriptives (n, mean, SD, median, IQR)
             - Mann-Whitney U + rank-biserial r + Cliff's delta (+ boot CI)
             - Kruskal-Wallis + epsilon-squared + Dunn post-hoc (BH)
             - chi-square / Fisher + Cramer's V + odds ratio (+ CI)
             - Benjamini-Hochberg adjustment helper
 Design  : Non-parametric by default, software metrics are heavy-tailed.
 Usage   : from utils.stats import *
============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config  # noqa: E402

RNG = np.random.default_rng(config.RANDOM_SEED)


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------
def describe(x, label: str = "") -> dict:
    """n, mean, SD, median, IQR, min, max for one group (NaNs dropped)."""
    x = pd.Series(x).dropna().astype(float)
    if len(x) == 0:
        return {"group": label, "n": 0}
    q1, q3 = np.percentile(x, [25, 75])
    return {
        "group": label, "n": int(len(x)),
        "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "median": float(x.median()), "q1": float(q1), "q3": float(q3),
        "iqr": float(q3 - q1), "min": float(x.min()), "max": float(x.max()),
    }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------
def cliffs_delta(a, b) -> float:
    """Cliff's delta via the U statistic: delta = 2U/(n1*n2) - 1.
    Positive delta => group A stochastically larger than group B."""
    a = pd.Series(a).dropna().astype(float).values
    b = pd.Series(b).dropna().astype(float).values
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u, _ = sps.mannwhitneyu(a, b, alternative="two-sided")
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def cliffs_delta_magnitude(delta: float) -> str:
    """Map |delta| to Romano et al. (2006) bands from config."""
    if np.isnan(delta):
        return "undefined"
    for threshold, label in config.CLIFFS_DELTA_BANDS:
        if abs(delta) < threshold:
            return label
    return "large"


def bootstrap_ci(stat_fn, *groups, n_boot=None, ci=None) -> tuple:
    """Percentile bootstrap CI for any statistic over >=1 groups (seeded)."""
    n_boot = n_boot or config.N_BOOTSTRAP
    ci = ci or config.CI_LEVEL
    groups = [pd.Series(g).dropna().astype(float).values for g in groups]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        resampled = [g[RNG.integers(0, len(g), len(g))] for g in groups]
        boots[i] = stat_fn(*resampled)
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Two-group comparison (Mann-Whitney)
# ---------------------------------------------------------------------------
def mann_whitney(a, b, label_a="A", label_b="B", boot_ci=True) -> dict:
    """Two-sided Mann-Whitney U with rank-biserial r, Cliff's delta (+CI)."""
    a_ = pd.Series(a).dropna().astype(float)
    b_ = pd.Series(b).dropna().astype(float)
    u, p = sps.mannwhitneyu(a_, b_, alternative="two-sided")
    delta = cliffs_delta(a_, b_)
    rank_biserial = delta                       # identical for MWU two-group
    out = {
        "test": "Mann-Whitney U", "group_a": label_a, "group_b": label_b,
        "n_a": int(len(a_)), "n_b": int(len(b_)),
        "U": float(u), "p": float(p),
        "cliffs_delta": delta,
        "rank_biserial_r": rank_biserial,
        "magnitude": cliffs_delta_magnitude(delta),
    }
    if boot_ci and len(a_) > 1 and len(b_) > 1:
        lo, hi = bootstrap_ci(cliffs_delta, a_, b_)
        out["delta_ci_low"], out["delta_ci_high"] = lo, hi
    return out


# ---------------------------------------------------------------------------
# k-group comparison (Kruskal-Wallis + Dunn post-hoc)
# ---------------------------------------------------------------------------
def kruskal_wallis(groups: dict) -> dict:
    """Kruskal-Wallis H across a dict {label: values}, with epsilon-squared."""
    labels = list(groups.keys())
    arrays = [pd.Series(groups[l]).dropna().astype(float).values for l in labels]
    arrays = [a for a in arrays if len(a) > 0]
    h, p = sps.kruskal(*arrays)
    n = sum(len(a) for a in arrays)
    k = len(arrays)
    # Epsilon-squared: (H - k + 1) / (n - k)  (Tomczak & Tomczak, 2014)
    eps_sq = float((h - k + 1) / (n - k)) if n > k else np.nan
    return {
        "test": "Kruskal-Wallis H", "groups": labels,
        "k": k, "n_total": int(n),
        "H": float(h), "df": k - 1, "p": float(p),
        "epsilon_squared": eps_sq,
    }


def dunn_posthoc(groups: dict) -> pd.DataFrame:
    """Dunn's test (z on mean ranks, tie-corrected) for all pairs, BH-adjusted.
    Standard follow-up to a significant Kruskal-Wallis."""
    labels = [l for l in groups if len(pd.Series(groups[l]).dropna()) > 0]
    data = pd.concat([pd.Series(groups[l]).dropna().astype(float)
                      .to_frame("value").assign(g=l) for l in labels])
    data["rank"] = sps.rankdata(data["value"])
    n = len(data)
    # tie correction term
    _, tie_counts = np.unique(data["value"], return_counts=True)
    tie_term = ((tie_counts ** 3 - tie_counts).sum() / (12 * (n - 1))
                if n > 1 else 0.0)
    mean_ranks = data.groupby("g")["rank"].mean()
    ns = data.groupby("g")["rank"].count()
    rows = []
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            se = np.sqrt((n * (n + 1) / 12.0 - tie_term)
                         * (1.0 / ns[la] + 1.0 / ns[lb]))
            z = (mean_ranks[la] - mean_ranks[lb]) / se if se > 0 else np.nan
            p = 2 * sps.norm.sf(abs(z))
            rows.append({"group_a": la, "group_b": lb,
                         "mean_rank_a": float(mean_ranks[la]),
                         "mean_rank_b": float(mean_ranks[lb]),
                         "z": float(z), "p_raw": float(p),
                         "cliffs_delta": cliffs_delta(groups[la], groups[lb])})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_adj"] = adjust_pvalues(df["p_raw"].tolist())
        df["significant"] = df["p_adj"] < config.ALPHA
        df["magnitude"] = df["cliffs_delta"].map(cliffs_delta_magnitude)
    return df


# ---------------------------------------------------------------------------
# Categorical comparison (chi-square / Fisher, Cramer's V, odds ratio)
# ---------------------------------------------------------------------------
def chi_square(table: pd.DataFrame) -> dict:
    """Chi-square on a contingency table with Cramer's V (bias-corrected).
    Falls back to Fisher's exact for 2x2 with small expected counts."""
    obs = table.values.astype(float)
    chi2, p, dof, expected = sps.chi2_contingency(obs)
    n = obs.sum()
    r, c = obs.shape
    # bias-corrected Cramer's V (Bergsma, 2013)
    phi2 = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    r_c = r - (r - 1) ** 2 / (n - 1)
    c_c = c - (c - 1) ** 2 / (n - 1)
    v = float(np.sqrt(phi2 / max(min(r_c - 1, c_c - 1), 1e-12)))
    out = {"test": "Chi-square", "chi2": float(chi2), "df": int(dof),
           "p": float(p), "n": int(n), "cramers_v": v}
    if obs.shape == (2, 2):
        if (expected < 5).any():
            odds, p_f = sps.fisher_exact(obs)
            out.update({"test": "Fisher exact", "p": float(p_f)})
        a, b_, c_, d = obs.ravel()
        # Haldane-Anscombe correction if any zero cell
        if 0 in (a, b_, c_, d):
            a, b_, c_, d = a + .5, b_ + .5, c_ + .5, d + .5
        or_ = (a * d) / (b_ * c_)
        se = np.sqrt(1/a + 1/b_ + 1/c_ + 1/d)
        z = sps.norm.ppf(1 - (1 - config.CI_LEVEL) / 2)
        out.update({"odds_ratio": float(or_),
                    "or_ci_low": float(np.exp(np.log(or_) - z * se)),
                    "or_ci_high": float(np.exp(np.log(or_) + z * se))})
    return out


# ---------------------------------------------------------------------------
# Multiple-comparison adjustment
# ---------------------------------------------------------------------------
def adjust_pvalues(pvals: list, method: str = None) -> list:
    """Benjamini-Hochberg (default from config) adjusted p-values."""
    if not pvals:
        return []
    _, p_adj, _, _ = multipletests(pvals, alpha=config.ALPHA,
                                   method=method or config.P_ADJUST_METHOD)
    return [float(p) for p in p_adj]


def format_p(p: float) -> str:
    """APA-style p formatting: p < .001 or p = .034 (no leading zero)."""
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")
