"""
============================================================================
 robustness_models.py, Adjusted Models and Robustness Checks (RQ1, RQ3, RQ4)
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : The models and checks the paper reports next to the rank tests:
   A. Size-adjusted logistic models with repository-clustered errors:
      any security smell (RQ1, pooled and per agent), CWE-798 and CWE-95
      presence (RQ4), and reverts (RQ3, per agent).
   B. Within-repository comparisons for reverts: conditional logistic
      regression (repository fixed effects, plus log size) and the
      Mantel-Haenszel odds ratio, over repositories that contain both the
      agent's and human merged PRs.
   C. Concentration checks for the maintenance population.
   D. RQ3 with the corrected post-merge measures from
      data_collection/04_recompute_followup.py (strictly after merge, the
      PR's own commits excluded, three revert definitions).
 Output  : analysis/robustness/outputs/robustness_report.txt
           analysis/robustness/outputs/robustness_results.json
 Usage   : python analysis/robustness/robustness_models.py
============================================================================
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.contingency_tables import StratifiedTable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from setup import config                         # noqa: E402
from utils.io_helpers import load_master         # noqa: E402
from utils import stats as st                    # noqa: E402

warnings.filterwarnings("ignore")
OUT = config.ANALYSIS_DIR / "robustness" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
H = config.HUMAN_LABEL
FILE_RECOMPUTED = config.DATA_INTERIM / "followup_recomputed.csv"
LINES, RESULTS = [], {}


def say(s=""):
    print(s)
    LINES.append(s)


def fmt_or(o, lo, hi, p):
    return f"{o:.2f} [{lo:.2f}, {hi:.2f}], {st.format_p(p)}"


def clustered_logit(formula, d, terms):
    """Logit with repository-clustered errors; returns {term: (OR, lo, hi, p)}."""
    r = smf.logit(formula, d).fit(disp=0, cov_type="cluster",
                                  cov_kwds={"groups": pd.factorize(d["repo"])[0]})
    ci = r.conf_int()
    return {t: (float(np.exp(r.params[t])), float(np.exp(ci.loc[t, 0])),
                float(np.exp(ci.loc[t, 1])), float(r.pvalues[t])) for t in terms}


def within_repo(d, outcome, agent):
    """Agent vs human inside repositories that contain both (fixed effects + MH)."""
    x = d[d.author_group.isin([agent, H])]
    shared = set(x.loc[x.author_group == agent, "repo"]) & set(x.loc[x.author_group == H, "repo"])
    x = x[x.repo.isin(shared)].copy()
    x["is_agent"] = (x.author_group == agent).astype(int)
    y = x[outcome].astype(int)
    fe = ConditionalLogit(y, x[["is_agent", "logsize"]], groups=x["repo"]).fit(disp=0)
    b, se = fe.params["is_agent"], fe.bse["is_agent"]
    tabs = []
    for _, g in x.groupby("repo"):
        tabs.append(np.array([[((g.is_agent == 1) & (g[outcome] == 1)).sum(), ((g.is_agent == 1) & (g[outcome] == 0)).sum()],
                              [((g.is_agent == 0) & (g[outcome] == 1)).sum(), ((g.is_agent == 0) & (g[outcome] == 0)).sum()]]))
    mh = StratifiedTable(tabs)
    informative = int(x.groupby("repo")[outcome].nunique().gt(1).sum())
    return {"agent": agent, "shared_repos": len(shared), "informative_repos": informative,
            "n_agent": int(x.is_agent.sum()), "n_human": int((x.is_agent == 0).sum()),
            "rate_agent": float(x.loc[x.is_agent == 1, outcome].mean()),
            "rate_human": float(x.loc[x.is_agent == 0, outcome].mean()),
            "fe_or": float(np.exp(b)), "fe_lo": float(np.exp(b - 1.96 * se)),
            "fe_hi": float(np.exp(b + 1.96 * se)), "fe_p": float(fe.pvalues["is_agent"]),
            "mh_or": float(mh.oddsratio_pooled)}


def revert_block(d, outcome, label):
    """Raw OR (chi-square, BH), clustered adjusted OR, and within-repository OR for one revert definition."""
    say(f"  {label}")
    say(f"  {'group':<15s} {'n':>6s} {'rate':>7s}  {'raw OR':<28s} {'adjusted OR (clustered)':<28s} within-repository OR")
    h = d[d.author_group == H]
    raw = []
    for a in config.AGENTS:
        s = d[d.author_group == a]
        tab = pd.DataFrame([[s[outcome].sum(), (s[outcome] == 0).sum()],
                            [h[outcome].sum(), (h[outcome] == 0).sum()]]).astype(int)
        raw.append((a, len(s), s[outcome].mean(), st.chi_square(tab)))
    padj = st.adjust_pvalues([r[3]["p"] for r in raw])
    terms = [f"C(author_group, Treatment('{H}'))[T.{a}]" for a in config.AGENTS]
    adj = clustered_logit(f"{outcome} ~ C(author_group, Treatment('{H}')) + logsize", d, terms + ["logsize"])
    wr = [within_repo(d, outcome, a) for a in config.AGENTS]
    wr_padj = st.adjust_pvalues([w["fe_p"] for w in wr])
    rows = []
    for (a, n, rate, cs), pa, t, w, wpa in zip(raw, padj, terms, wr, wr_padj):
        o = adj[t]
        say(f"  {a:<15s} {n:>6,d} {rate:>6.1%}  {fmt_or(cs['odds_ratio'], cs['or_ci_low'], cs['or_ci_high'], pa):<28s} "
            f"{fmt_or(*o):<28s} {fmt_or(w['fe_or'], w['fe_lo'], w['fe_hi'], w['fe_p'])} (BH {st.format_p(wpa)}; "
            f"{w['shared_repos']} shared repos, agent {w['rate_agent']:.1%} vs human {w['rate_human']:.1%})")
        rows.append({"agent": a, "n": n, "rate": float(rate), "raw_or": cs["odds_ratio"], "raw_lo": cs["or_ci_low"],
                     "raw_hi": cs["or_ci_high"], "raw_p_adj": pa, "adj_or": o[0], "adj_lo": o[1], "adj_hi": o[2],
                     "adj_p": o[3], **{f"wr_{k}": v for k, v in w.items() if k != "agent"}, "wr_p_adj": wpa})
    say(f"  {'Human':<15s} {len(h):>6,d} {h[outcome].mean():>6.1%}")
    say(f"  size: OR per log unit {fmt_or(*adj['logsize'])}")
    say("")
    return rows


def main():
    say("=" * 78)
    say(" Adjusted models and robustness checks")
    say("=" * 78)
    df = load_master(config.FILE_MASTER)

    # ---- A. RQ1 / RQ4 adjusted models ----------------------------------------
    a = df[df.static_status == "ok"].copy()
    a["any_smell"] = (a.security_smells_total > 0).astype(int)
    a["logsize"] = np.log1p(a.added_lines_analyzed)
    a["agent"] = (a.author_group != H).astype(int)
    say("\nA. Size-adjusted logistic models, errors clustered by repository "
        f"({a.repo.nunique():,} repositories, {len(a):,} analyzed PRs)")
    pooled = clustered_logit("any_smell ~ agent + logsize", a, ["agent", "logsize"])
    say(f"  any security smell, pooled agents vs human: {fmt_or(*pooled['agent'])}; size {fmt_or(*pooled['logsize'])}")
    terms = [f"C(author_group, Treatment('{H}'))[T.{g}]" for g in config.AGENTS]
    per = clustered_logit(f"any_smell ~ C(author_group, Treatment('{H}')) + logsize", a, terms)
    for g, t in zip(config.AGENTS, terms):
        say(f"    {g:<15s} {fmt_or(*per[t])}")
    RESULTS["rq1_adjusted"] = {"pooled": pooled["agent"], "size": pooled["logsize"],
                               **{g: per[t] for g, t in zip(config.AGENTS, terms)}}
    for c in ["has_CWE-798", "has_CWE-95"]:
        a["y"] = a[c].astype(int)
        r = clustered_logit("y ~ agent + logsize", a, ["agent"])["agent"]
        say(f"  {c} pooled agents vs human: {fmt_or(*r)}")
        RESULTS[f"rq4_adjusted_{c}"] = r

    # ---- B/C. RQ3 maintenance population (published measures) ----------------
    m = df[(df.is_merged == 1) & df.post_merge_commits_90d.notna()].copy()
    m["logsize"] = np.log1p(m.changed_lines.fillna(0))
    m["revert_published"] = m.revert_detected.fillna(0).astype(int)
    say(f"\nB. Reverts as published ({len(m):,} merged PRs, {m.repo.nunique():,} repositories)")
    RESULTS["rq3_published"] = revert_block(m, "revert_published", "revert flag as published")

    say("C. Concentration")
    for g in config.AUTHOR_GROUPS:
        s = m[m.author_group == g]
        vc = s.repo.value_counts()
        one = s.sample(frac=1, random_state=config.RANDOM_SEED).drop_duplicates("repo")
        say(f"  {g:<15s} {len(s):>6,d} PRs in {vc.size:>4d} repos; largest repo {vc.iloc[0] / len(s):5.1%}; "
            f"revert without it {s[s.repo != vc.index[0]].revert_published.mean():.1%}; "
            f"one PR per repo {one.revert_published.mean():.1%}")
    c = m[m.author_group == "OpenAI Codex"]
    hrep = set(m.loc[m.author_group == H, "repo"])
    say(f"  Codex in repos with baseline human PRs: {c[c.repo.isin(hrep)].revert_published.mean():.1%} "
        f"(n = {c.repo.isin(hrep).sum():,}); elsewhere {c[~c.repo.isin(hrep)].revert_published.mean():.1%} "
        f"(n = {(~c.repo.isin(hrep)).sum():,})")

    # ---- D. corrected post-merge measures --------------------------------------
    if FILE_RECOMPUTED.exists():
        rc = pd.read_csv(FILE_RECOMPUTED)
        rc = rc[rc.status == "ok"]
        d = m.merge(rc[["id", "followup_commits_90d", "followup_commits_90d_original", "revert_original",
                        "revert_any_word", "revert_format", "revert_of_this_pr"]], on="id", how="inner")
        fid = d.followup_commits_90d_original.eq(d.post_merge_commits_90d) & d.revert_original.eq(d.revert_published)
        say(f"\nD. Corrected post-merge measures ({len(d):,} of {len(m):,} PRs recomputed; "
            f"original values reproduced exactly for {fid.mean():.1%})")
        d["followup_density"] = 100 * d.followup_commits_90d / d.changed_lines.replace(0, np.nan)
        for metric, label in [("followup_commits_90d", "follow-up commits (90 days)"),
                              ("followup_density", "follow-up commits per 100 changed lines")]:
            groups = {g: d.loc[d.author_group == g, metric].dropna() for g in config.AUTHOR_GROUPS}
            kw = st.kruskal_wallis(groups)
            say(f"  {label}: H({kw['df']}) = {kw['H']:.1f}, {st.format_p(kw['p'])}, eps2 = {kw['epsilon_squared']:.3f}")
            res = [st.mann_whitney(groups[g], groups[H], g, H) for g in config.AGENTS]
            pad = st.adjust_pvalues([r["p"] for r in res])
            for g, r, pa in zip(config.AGENTS, res, pad):
                say(f"    {g:<15s} delta = {r['cliffs_delta']:+.3f} [{r['delta_ci_low']:+.3f}, {r['delta_ci_high']:+.3f}] "
                    f"({r['magnitude']}), BH {st.format_p(pa)}; median {groups[g].median():.2f} vs human {groups[H].median():.2f}")
            RESULTS[f"rq3_corrected_{metric}"] = {"kw": {k: v for k, v in kw.items() if k != 'groups'},
                                                  "pairs": [{**r, "p_adj": pa} for r, pa in zip(res, pad)]}
        say("")
        for col, label in [("revert_format", "revert commit on the file (primary corrected definition)"),
                           ("revert_any_word", "any 'revert' word, own commits excluded"),
                           ("revert_of_this_pr", "revert of this PR")]:
            d[col] = d[col].astype(int)
            try:
                RESULTS[f"rq3_corrected_{col}"] = revert_block(d, col, label)
            except Exception as e:  # sparse outcomes can defeat the fixed-effects fit
                say(f"  {label}: model not estimable ({e})")
                say(d.groupby("author_group")[col].agg(["sum", "mean"]).to_string())
        say("  sensitivity: PRs whose original values were reproduced exactly")
        RESULTS["rq3_corrected_revert_format_fidelity"] = revert_block(d[fid], "revert_format", "revert commit on the file")
    else:
        say(f"\nD. skipped: {FILE_RECOMPUTED} not found (run data_collection/04_recompute_followup.py)")

    (OUT / "robustness_report.txt").write_text("\n".join(LINES) + "\n")
    (OUT / "robustness_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
    print(f"  [SAVED] {OUT}")


if __name__ == "__main__":
    main()
