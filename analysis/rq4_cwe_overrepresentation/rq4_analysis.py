"""
============================================================================
 rq4_analysis.py, RQ4: CWE / Vulnerability Class Over-Representation
============================================================================
 Question: Are specific CWE classes over-represented in agent code vs
           human code?
 Approach: On statically analyzed PRs, for each CWE class build an
           agent-pooled vs human 2x2 (presence per PR): chi-square/Fisher,
           odds ratio + CI, BH-adjusted across the CWE family. Also a
           per-agent x CWE prevalence matrix for the paper's heatmap.
 Output  : analysis/rq4_cwe_overrepresentation/outputs/rq4_results.{csv,json}
           rq4_apa_report.txt + figures/
 Usage   : python analysis/rq4_cwe_overrepresentation/rq4_analysis.py
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

RQ = "rq4"
OUT, FIG = config.rq_output_dir(RQ), config.rq_figure_dir(RQ)
sns.set_theme(style="whitegrid", context="paper")


def main():
    print("=" * 76)
    print(" RQ4 — CWE OVER-REPRESENTATION")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)
    d = df[df["static_status"] == "ok"].copy()
    has_cols = sorted([c for c in d.columns if c.startswith("has_CWE")])
    print(f"  analyzed PRs: {len(d):,}   CWE classes: {len(has_cols)}")
    d["is_agent"] = (d["author_group"] != "Human").astype(int)

    rb = ReportBuilder(RQ)
    rb.text(f"Analysis population: {len(d):,} statically analyzed PRs.",
            "Each CWE class is tested as presence/absence per PR, agent-"
            "pooled vs human, with Benjamini-Hochberg correction across "
            f"the {len(has_cols)} classes. OR>1 => more common in agent PRs.")

    rows = []
    for c in has_cols:
        cwe = c.replace("has_", "")
        tab = pd.crosstab(d["is_agent"], d[c])
        for col in (0, 1):
            if col not in tab.columns:
                tab[col] = 0
        tab = tab[[0, 1]]
        cs = st.chi_square(pd.DataFrame(
            [[tab.loc[1, 1], tab.loc[1, 0]],
             [tab.loc[0, 1], tab.loc[0, 0]]]).astype(int))
        rows.append({
            "cwe": cwe,
            "agent_prevalence": float(d.loc[d.is_agent == 1, c].mean()),
            "human_prevalence": float(d.loc[d.is_agent == 0, c].mean()),
            "n_agent_hits": int(d.loc[d.is_agent == 1, c].sum()),
            "n_human_hits": int(d.loc[d.is_agent == 0, c].sum()),
            "test": cs["test"], "chi2": cs["chi2"], "p_raw": cs["p"],
            "odds_ratio": cs.get("odds_ratio"),
            "or_ci_low": cs.get("or_ci_low"),
            "or_ci_high": cs.get("or_ci_high")})
    res = pd.DataFrame(rows)
    res["p_adj"] = st.adjust_pvalues(res["p_raw"].tolist())
    res["significant"] = res["p_adj"] < config.ALPHA
    res = res.sort_values("odds_ratio", ascending=False)

    rb.dataframe_table(res.round(4),
                       "CWE over-representation (agent-pooled vs human)")
    for _, r in res.iterrows():
        star = "*" if r["significant"] else " "
        orv = r["odds_ratio"] if pd.notna(r["odds_ratio"]) else float("nan")
        rb.text(f" {star} {r['cwe']:<32s} agent {r['agent_prevalence']:.3%} "
                f"vs human {r['human_prevalence']:.3%}  OR={orv:.2f}  "
                f"adj {st.format_p(r['p_adj'])}")
        print(f"  {star} {r['cwe']:<30s} OR={orv:6.2f}  adj "
              f"{st.format_p(r['p_adj'])}")

    # per-agent prevalence matrix
    mat = d.groupby("author_group")[has_cols].mean()
    mat.columns = [c.replace("has_", "") for c in mat.columns]
    rb.dataframe_table((mat * 100).round(3).reset_index(),
                       "Per-agent CWE prevalence (% of analyzed PRs)")
    rb.assumptions_note()

    # ---- figure: prevalence heatmap ------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap((mat * 100).reindex(
        [g for g in config.AUTHOR_GROUPS if g in mat.index]),
        annot=True, fmt=".2f", cmap="Reds",
        cbar_kws={"label": "% of PRs with class"}, ax=ax)
    ax.set_title("CWE prevalence by author group (% of analyzed PRs)")
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(FIG / "rq4_cwe_prevalence_heatmap.png", dpi=300)
    plt.close(fig)
    print(f"  [SAVED] figure -> {FIG}")

    save_csv(res, OUT / "rq4_results.csv", "rq4 CWE tests")
    save_json({"cwe_tests": res.to_dict("records"),
               "per_agent_prevalence":
               (mat * 100).reset_index().to_dict("records")},
              OUT / "rq4_results.json", "rq4 results")
    save_txt(rb.render(), OUT / "rq4_apa_report.txt", "rq4 APA report")
    print("  RQ4 COMPLETE")


if __name__ == "__main__":
    main()
