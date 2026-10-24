"""
============================================================================
 rq2_analysis.py, RQ2: Maintainability of Code Across Agents
============================================================================
 Question: Which agents produce the most maintainable code (structural
           complexity, comments, smells)?
 Approach: On statically analyzed PRs, compare all six author groups on
           max_nesting, branch_density, comment_ratio, and
           maintainability_smell_density: Kruskal-Wallis omnibus with
           epsilon-squared, Dunn post-hoc (BH-adjusted, Cliff's delta per
           pair), plus a median profile table + ranking.
 Output  : analysis/rq2_maintainability/outputs/rq2_results.{csv,json}
           rq2_apa_report.txt + figures/
 Usage   : python analysis/rq2_maintainability/rq2_analysis.py
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

RQ = "rq2"
OUT, FIG = config.rq_output_dir(RQ), config.rq_figure_dir(RQ)
METRICS = {
    "max_nesting": "max nesting depth (levels)",
    "branch_density": "branching keywords per added line",
    "comment_ratio": "comment lines / added lines",
    "maintainability_smell_density": "TODO+long-line smells /100 lines",
}
sns.set_theme(style="whitegrid", context="paper")


def main():
    print("=" * 76)
    print(" RQ2 — MAINTAINABILITY ACROSS AGENTS")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)
    d = df[df["static_status"] == "ok"].copy()
    print(f"  analyzed PRs: {len(d):,}")

    rb = ReportBuilder(RQ)
    all_pairs, omnibus = [], []

    for metric, label in METRICS.items():
        groups = {g: d.loc[d.author_group == g, metric].dropna()
                  for g in config.AUTHOR_GROUPS}
        groups = {g: v for g, v in groups.items() if len(v) >= 30}
        rb.descriptives_table([st.describe(v, g) for g, v in groups.items()],
                              label)
        kw = st.kruskal_wallis(groups)
        rb.kruskal_result(kw, label)
        omnibus.append({"metric": metric, **{k: v for k, v in kw.items()
                        if k != "groups"}})
        print(f"  [{metric:<32s}] KW H={kw['H']:7.1f} "
              f"{st.format_p(kw['p'])} eps2={kw['epsilon_squared']:.4f}")
        dunn = st.dunn_posthoc(groups)
        dunn.insert(0, "metric", metric)
        rb.dataframe_table(
            dunn[dunn.significant].round(4),
            f"Dunn post-hoc — significant pairs only ({label})")
        all_pairs.append(dunn)

    # median profile + composite rank (lower complexity/smells better,
    # higher comment_ratio better)
    med = d.groupby("author_group")[list(METRICS)].median()
    ranks = med.rank()
    ranks["comment_ratio"] = med["comment_ratio"].rank(ascending=False)
    med["composite_rank"] = ranks.mean(axis=1)
    med = med.sort_values("composite_rank")
    rb.dataframe_table(med.round(4).reset_index(),
                       "Median maintainability profile + composite rank "
                       "(1 = most maintainable)")
    rb.assumptions_note()

    # ---- figure: median heatmap (z-scored per metric for comparability) ----
    z = (med[list(METRICS)] - med[list(METRICS)].mean()) / med[list(METRICS)].std()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.heatmap(z, annot=med[list(METRICS)].round(3), fmt="",
                cmap="RdYlGn_r", center=0, cbar_kws={"label": "z (red = worse)"},
                ax=ax)
    ax.set_title("Maintainability medians by author group")
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(FIG / "rq2_maintainability_heatmap.png", dpi=300)
    plt.close(fig)
    print(f"  [SAVED] figure -> {FIG}")

    pairs = pd.concat(all_pairs, ignore_index=True)
    save_csv(pairs, OUT / "rq2_results.csv", "rq2 Dunn pairs")
    save_json({"omnibus": omnibus,
               "median_profile": med.reset_index().to_dict("records")},
              OUT / "rq2_results.json", "rq2 results")
    save_txt(rb.render(), OUT / "rq2_apa_report.txt", "rq2 APA report")
    print("  RQ2 COMPLETE")


if __name__ == "__main__":
    main()
