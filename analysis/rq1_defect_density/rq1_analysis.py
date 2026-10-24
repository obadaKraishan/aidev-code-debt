"""
============================================================================
 rq1_analysis.py, RQ1: Defect/Issue Density, Agent vs Human PRs
============================================================================
 Question: Do agent-authored PRs carry higher defect/issue density than
           human PRs, and by how much?
 Approach: On statically analyzed PRs (target-language diffs):
   (a) security_smell_density + maintainability_smell_density:
       Kruskal-Wallis omnibus, then each agent vs Human (MWU, Cliff's
       delta + bootstrap CI, BH-adjusted family)
   (b) proportion of PRs with >=1 security smell: chi-square + OR
   (c) sensitivity: size-bucket-stratified comparisons (fairness check)
 Input   : data/processed/master.csv
 Output  : analysis/rq1_defect_density/outputs/rq1_results.{csv,json}
           analysis/rq1_defect_density/outputs/rq1_apa_report.txt
           analysis/rq1_defect_density/figures/*.png
 Usage   : python analysis/rq1_defect_density/rq1_analysis.py
============================================================================
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from setup import config                                     # noqa: E402
from utils.io_helpers import load_master, save_csv, save_json, save_txt  # noqa: E402
from utils import stats as st                                # noqa: E402
from utils.reporting import ReportBuilder                    # noqa: E402

RQ = "rq1"
OUT, FIG = config.rq_output_dir(RQ), config.rq_figure_dir(RQ)
METRICS = ["security_smell_density", "maintainability_smell_density"]
sns.set_theme(style="whitegrid", context="paper")


def main():
    print("=" * 76)
    print(" RQ1 — DEFECT/ISSUE DENSITY: AGENT vs HUMAN")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)
    d = df[df["static_status"] == "ok"].copy()
    print(f"  analyzed PRs (static ok): {len(d):,} "
          f"({len(d)/len(df):.1%} of corpus)")

    rb = ReportBuilder(RQ)
    rb.text(f"Analysis population: {len(d):,} PRs with target-language "
            f"diffs (static_status='ok') out of {len(df):,} total.",
            "Coverage by group is reported in validation_report.txt and "
            "acknowledged as a threat to validity.")
    rows, results_rows = [], []

    for metric in METRICS:
        groups = {g: d.loc[d.author_group == g, metric].dropna()
                  for g in config.AUTHOR_GROUPS}
        groups = {g: v for g, v in groups.items() if len(v) >= 30}
        print(f"\n  [{metric}]")
        rb.descriptives_table([st.describe(v, g) for g, v in groups.items()],
                              metric)
        kw = st.kruskal_wallis(groups)
        rb.kruskal_result(kw, metric)
        print(f"    KW H={kw['H']:.1f} {st.format_p(kw['p'])} "
              f"eps2={kw['epsilon_squared']:.4f}")
        results_rows.append({"metric": metric, **{k: v for k, v in kw.items()
                             if k != "groups"}})

        # family: each agent vs Human, BH-adjusted
        mwu = [st.mann_whitney(groups[a], groups["Human"], a, "Human")
               for a in config.AGENTS if a in groups]
        p_adj = st.adjust_pvalues([r["p"] for r in mwu])
        for r, pa in zip(mwu, p_adj):
            r["p_adj"], r["metric"] = pa, metric
            rb.mann_whitney_result(r, metric, p_adj=pa)
            rows.append(r)
            print(f"    {r['group_a']:<15s} vs Human: delta="
                  f"{r['cliffs_delta']:+.3f} ({r['magnitude']}), "
                  f"adj {st.format_p(pa)}")

    # (b) any-security-smell proportion: pooled agents vs human
    d["has_sec"] = (d["security_smells_total"].fillna(0) > 0).astype(int)
    d["is_agent"] = (d["author_group"] != "Human").astype(int)
    tab = pd.crosstab(d["is_agent"], d["has_sec"])
    cs = st.chi_square(tab)
    rb.chi_square_result(cs, "Presence of >=1 security smell "
                             "(agent-pooled vs human PRs)")
    prop = d.groupby("author_group")["has_sec"].mean().sort_values()
    rb.dataframe_table(prop.rename("share_with_security_smell").reset_index(),
                       "Share of analyzed PRs with >=1 security smell")
    results_rows.append({"metric": "has_security_smell", **cs})

    # (c) sensitivity: size-stratified deltas for security density
    rb.section("Sensitivity — size-bucket-stratified (security density)")
    strat = []
    for bucket in ["XS", "S", "M", "L", "XL"]:
        sub = d[d["size_bucket"] == bucket]
        a = sub.loc[sub.is_agent == 1, "security_smell_density"].dropna()
        h = sub.loc[sub.is_agent == 0, "security_smell_density"].dropna()
        if len(a) >= 30 and len(h) >= 30:
            r = st.mann_whitney(a, h, f"Agents[{bucket}]", f"Human[{bucket}]",
                                boot_ci=False)
            strat.append({"bucket": bucket, "n_agent": r["n_a"],
                          "n_human": r["n_b"], "delta": r["cliffs_delta"],
                          "p": r["p"]})
    sdf = pd.DataFrame(strat)
    if len(sdf):
        sdf["p_adj"] = st.adjust_pvalues(sdf["p"].tolist())
        rb.dataframe_table(sdf, "Agent-vs-human delta within PR size buckets")
    rb.assumptions_note()

    # ---- figures ---------------------------------------------------------------
    order = [g for g in config.AUTHOR_GROUPS if g in d.author_group.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, ttl in zip(
            axes, METRICS, ["Security smells /100 lines",
                            "Maintainability smells /100 lines"]):
        sns.boxplot(data=d, x="author_group", y=metric, order=order,
                    showfliers=False, ax=ax, color="#7fa8d9")
        ax.set_title(ttl); ax.set_xlabel(""); ax.set_ylabel("density")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(FIG / "rq1_smell_density_boxplots.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    prop.plot.barh(ax=ax, color="#c0504d")
    ax.set_xlabel("share of PRs with >=1 security smell")
    ax.set_title("Security-smell prevalence by author group")
    plt.tight_layout()
    fig.savefig(FIG / "rq1_security_prevalence.png", dpi=300)
    plt.close(fig)
    print(f"\n  [SAVED] figures -> {FIG}")

    # ---- outputs ------------------------------------------------------------------
    save_csv(pd.DataFrame(rows), OUT / "rq1_results.csv", "rq1 MWU results")
    save_json({"kruskal_and_chi": results_rows,
               "pairwise_mwu": rows}, OUT / "rq1_results.json", "rq1 results")
    save_txt(rb.render(), OUT / "rq1_apa_report.txt", "rq1 APA report")
    print("  RQ1 COMPLETE")


if __name__ == "__main__":
    main()
