"""
============================================================================
 rq3_analysis.py, RQ3: Post-Merge Maintenance Overhead
============================================================================
 Question: What is the post-merge maintenance overhead of agent code
           (churn, reverts) relative to human code?
 Approach: Merged PRs with follow-up data:
   (a) post_merge_commits_90d + churn_per_100_lines: KW omnibus +
       each agent vs Human (MWU family, BH)
   (b) revert rate: per-agent 2x2 vs Human (chi-square/Fisher + OR, BH)
 Output  : analysis/rq3_maintenance_overhead/outputs/rq3_results.{csv,json}
           rq3_apa_report.txt + figures/
 Usage   : python analysis/rq3_maintenance_overhead/rq3_analysis.py
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

RQ = "rq3"
OUT, FIG = config.rq_output_dir(RQ), config.rq_figure_dir(RQ)
sns.set_theme(style="whitegrid", context="paper")


def main():
    print("=" * 76)
    print(" RQ3 — POST-MERGE MAINTENANCE OVERHEAD")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)
    d = df[(df["is_merged"] == 1)
           & df["post_merge_commits_90d"].notna()].copy()
    print(f"  merged PRs with 90-day follow-up: {len(d):,}")

    # v1.1.0: use the corrected post-merge measures when they exist
    # (data_collection/04_recompute_followup.py): commits strictly after the
    # merge with the PR's own commits excluded, and a revert commit proper.
    recomputed = config.DATA_INTERIM / "followup_recomputed.csv"
    note = "published v1.0.0 measures"
    if recomputed.exists():
        rc = pd.read_csv(recomputed)
        rc = rc[rc.status == "ok"][["id", "followup_commits_90d", "revert_format"]]
        d = d.merge(rc, on="id", how="inner")
        d["post_merge_commits_90d"] = d["followup_commits_90d"]
        d["churn_per_100_lines"] = 100 * d["followup_commits_90d"] / d["changed_lines"].replace(0, np.nan)
        d["revert_detected"] = d["revert_format"]
        note = ("corrected measures from followup_recomputed.csv (own commits excluded; "
                "revert = a revert commit on the file)")
        print(f"  using corrected measures for {len(d):,} PRs")

    rb = ReportBuilder(RQ)
    rb.text(f"Analysis population: {len(d):,} merged PRs with a 90-day "
            "post-merge observation window on the PR's most-changed file; "
            f"{note}.")
    rows, extra = [], []

    # (a) churn metrics
    for metric, label in [("post_merge_commits_90d",
                           "post-merge commits touching top file (90d)"),
                          ("churn_per_100_lines",
                           "post-merge commits per 100 changed lines")]:
        groups = {g: d.loc[d.author_group == g, metric].dropna()
                  for g in config.AUTHOR_GROUPS}
        groups = {g: v for g, v in groups.items() if len(v) >= 30}
        rb.descriptives_table([st.describe(v, g) for g, v in groups.items()],
                              label)
        kw = st.kruskal_wallis(groups)
        rb.kruskal_result(kw, label)
        extra.append({"metric": metric, **{k: v for k, v in kw.items()
                      if k != "groups"}})
        print(f"  [{metric:<26s}] KW H={kw['H']:7.1f} {st.format_p(kw['p'])}")
        mwu = [st.mann_whitney(groups[a], groups["Human"], a, "Human")
               for a in config.AGENTS if a in groups]
        p_adj = st.adjust_pvalues([r["p"] for r in mwu])
        for r, pa in zip(mwu, p_adj):
            r["p_adj"], r["metric"] = pa, metric
            rb.mann_whitney_result(r, label, p_adj=pa)
            rows.append(r)
            print(f"    {r['group_a']:<15s} vs Human: delta="
                  f"{r['cliffs_delta']:+.3f} ({r['magnitude']}), "
                  f"adj {st.format_p(pa)}")

    # (b) revert rates: each agent vs human 2x2
    rb.section("Revert rates — each agent vs Human (2x2, BH-adjusted)")
    h = d[d.author_group == "Human"]
    rev_rows = []
    for a in config.AGENTS:
        sub = d[d.author_group == a]
        if len(sub) < 30:
            continue
        tab = pd.DataFrame([
            [sub.revert_detected.fillna(0).sum(),
             (sub.revert_detected.fillna(0) == 0).sum()],
            [h.revert_detected.fillna(0).sum(),
             (h.revert_detected.fillna(0) == 0).sum()]]).astype(int)
        cs = st.chi_square(tab)
        rev_rows.append({"agent": a, "n_agent": len(sub),
                         "revert_rate_agent": float(sub.revert_detected
                                                    .fillna(0).mean()),
                         "revert_rate_human": float(h.revert_detected
                                                    .fillna(0).mean()),
                         "test": cs["test"], "chi2": cs["chi2"],
                         "p_raw": cs["p"],
                         "odds_ratio": cs.get("odds_ratio"),
                         "or_ci_low": cs.get("or_ci_low"),
                         "or_ci_high": cs.get("or_ci_high")})
    rev = pd.DataFrame(rev_rows)
    rev["p_adj"] = st.adjust_pvalues(rev["p_raw"].tolist())
    rb.dataframe_table(rev.round(4),
                       "Revert-rate comparisons (OR < 1 = agent reverted "
                       "LESS than human)")
    for _, r in rev.iterrows():
        print(f"  revert {r['agent']:<15s} {r['revert_rate_agent']:.1%} vs "
              f"human {r['revert_rate_human']:.1%}  OR={r['odds_ratio']:.2f} "
              f"adj {st.format_p(r['p_adj'])}")
    rb.assumptions_note()

    # ---- figures ----------------------------------------------------------------
    order = [g for g in config.AUTHOR_GROUPS if g in d.author_group.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.boxplot(data=d, x="author_group", y="post_merge_commits_90d",
                order=order, showfliers=False, ax=axes[0], color="#7fa8d9")
    axes[0].set_title("Post-merge commits (90d)"); axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)
    rr = d.groupby("author_group")["revert_detected"].mean().reindex(order)
    rr.plot.bar(ax=axes[1], color="#c0504d")
    axes[1].set_title("Revert rate"); axes[1].set_xlabel("")
    axes[1].set_ylabel("share of PRs reverted")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(FIG / "rq3_maintenance_overhead.png", dpi=300)
    plt.close(fig)
    print(f"  [SAVED] figure -> {FIG}")

    save_csv(pd.concat([pd.DataFrame(rows), rev], axis=0),
             OUT / "rq3_results.csv", "rq3 results")
    save_json({"omnibus": extra, "pairwise": rows,
               "reverts": rev.to_dict("records")},
              OUT / "rq3_results.json", "rq3 results")
    save_txt(rb.render(), OUT / "rq3_apa_report.txt", "rq3 APA report")
    print("  RQ3 COMPLETE")


if __name__ == "__main__":
    main()
