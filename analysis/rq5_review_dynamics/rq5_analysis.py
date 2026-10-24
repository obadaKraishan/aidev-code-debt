"""
============================================================================
 rq5_analysis.py, RQ5: Human Review Latency and Depth
============================================================================
 Question: Does human review latency/depth differ for agent vs human PRs?
 Approach: On PRs with >=1 HUMAN review (bots excluded upstream), compare
           groups on review_latency_hours, n_reviews, n_change_requests,
           n_issue_comments, n_bot_reviews: KW omnibus + each agent vs
           Human (MWU family, BH). Coverage is reported explicitly because
           AIDev's review tables cover agentic PRs more than human ones.
 Output  : analysis/rq5_review_dynamics/outputs/rq5_results.{csv,json}
           rq5_apa_report.txt + figures/
 Usage   : python analysis/rq5_review_dynamics/rq5_analysis.py
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

RQ = "rq5"
OUT, FIG = config.rq_output_dir(RQ), config.rq_figure_dir(RQ)
METRICS = {
    "review_latency_hours": "hours to first human review",
    "n_reviews": "human reviews per PR",
    "n_change_requests": "CHANGES_REQUESTED reviews per PR",
    "n_issue_comments": "issue comments per PR",
    "n_bot_reviews": "bot reviews per PR",
}
sns.set_theme(style="whitegrid", context="paper")


def main():
    print("=" * 76)
    print(" RQ5 — REVIEW LATENCY & DEPTH")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)

    rb = ReportBuilder(RQ)
    # coverage note
    cov = (df.assign(has_rev=df["n_reviews"].notna())
           .groupby("author_group")["has_rev"].mean())
    rb.section("Review-data coverage by group (threat to validity)")
    rb.dataframe_table(cov.rename("share_with_human_review_record")
                       .reset_index().round(4), "Coverage")
    print("  coverage (share with human-review record):")
    for g, v in cov.items():
        print(f"    {g:<15s} {v:.1%}")

    rows, omnibus = [], []
    for metric, label in METRICS.items():
        if metric not in df.columns:
            continue
        # population depends on metric: latency needs a review to exist
        base = df[df["n_reviews"].notna()] if metric != "n_bot_reviews" else df
        groups = {g: base.loc[base.author_group == g, metric].dropna()
                  for g in config.AUTHOR_GROUPS}
        groups = {g: v for g, v in groups.items() if len(v) >= 30}
        if len(groups) < 2:
            print(f"  [{metric}] insufficient coverage — skipped")
            continue
        rb.descriptives_table([st.describe(v, g) for g, v in groups.items()],
                              label)
        kw = st.kruskal_wallis(groups)
        rb.kruskal_result(kw, label)
        omnibus.append({"metric": metric, **{k: v for k, v in kw.items()
                        if k != "groups"}})
        print(f"  [{metric:<20s}] KW H={kw['H']:7.1f} {st.format_p(kw['p'])} "
              f"(k={kw['k']})")
        if "Human" in groups:
            mwu = [st.mann_whitney(groups[a], groups["Human"], a, "Human")
                   for a in config.AGENTS if a in groups]
            p_adj = st.adjust_pvalues([r["p"] for r in mwu])
            for r, pa in zip(mwu, p_adj):
                r["p_adj"], r["metric"] = pa, metric
                rb.mann_whitney_result(r, label, p_adj=pa)
                rows.append(r)
        else:
            rb.text("NOTE: Human group lacks coverage for this metric; "
                    "agent-vs-human contrast not computed (see coverage "
                    "table). Cross-agent differences reported via omnibus.")
    rb.assumptions_note()

    # ---- figure ----------------------------------------------------------------
    plotd = df[df["n_reviews"].notna()]
    order = [g for g in config.AUTHOR_GROUPS
             if g in plotd.author_group.unique()]
    if len(order) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        sns.boxplot(data=plotd[plotd.review_latency_hours.notna()],
                    x="author_group", y="review_latency_hours", order=order,
                    showfliers=False, ax=axes[0], color="#7fa8d9")
        axes[0].set_title("Hours to first human review")
        axes[0].set_xlabel(""); axes[0].tick_params(axis="x", rotation=30)
        axes[0].set_yscale("symlog")
        sns.boxplot(data=plotd, x="author_group", y="n_reviews", order=order,
                    showfliers=False, ax=axes[1], color="#9bbb59")
        axes[1].set_title("Human reviews per PR")
        axes[1].set_xlabel(""); axes[1].tick_params(axis="x", rotation=30)
        plt.tight_layout()
        fig.savefig(FIG / "rq5_review_dynamics.png", dpi=300)
        plt.close(fig)
        print(f"  [SAVED] figure -> {FIG}")

    save_csv(pd.DataFrame(rows) if rows else pd.DataFrame(),
             OUT / "rq5_results.csv", "rq5 results")
    save_json({"coverage": cov.reset_index().to_dict("records"),
               "omnibus": omnibus, "pairwise": rows},
              OUT / "rq5_results.json", "rq5 results")
    save_txt(rb.render(), OUT / "rq5_apa_report.txt", "rq5 APA report")
    print("  RQ5 COMPLETE")


if __name__ == "__main__":
    main()
