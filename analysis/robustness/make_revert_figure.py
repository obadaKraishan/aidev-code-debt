"""
============================================================================
 make_revert_figure.py, Fig. 3 of the paper (90-day revert rate by agent)
============================================================================
 Uses the corrected revert definition from data_collection/04_recompute_followup.py
 (a revert commit on the PR's most-changed file within 90 days of merge, the PR's
 own commits excluded). Rates with 95% Wilson intervals; the dashed line and band
 mark the human rate; asterisks mark BH-adjusted p < .05 against humans.
 Output: analysis/robustness/figures/fig_reverts.pdf
 Usage : python analysis/robustness/make_revert_figure.py [--column revert_format]
============================================================================
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from setup import config                 # noqa: E402
from utils import stats as st            # noqa: E402

COLORS = {"OpenAI Codex": "#2563A8", "Devin": "#C2413B", "GitHub Copilot": "#3A8A3F",
          "Cursor": "#D4951A", "Claude Code": "#7856A8"}


def main(column):
    m = pd.read_csv(config.FILE_MASTER, low_memory=False)
    rc = pd.read_csv(config.DATA_INTERIM / "followup_recomputed.csv")
    d = m[["id", "author_group"]].merge(rc[rc.status == "ok"][["id", column]], on="id")
    h = d[d.author_group == config.HUMAN_LABEL][column]
    hk, hn = int(h.sum()), len(h)
    hlo, hhi = proportion_confint(hk, hn, method="wilson")
    rows = []
    for a in config.AGENTS:
        s = d[d.author_group == a][column]
        k, n = int(s.sum()), len(s)
        lo, hi = proportion_confint(k, n, method="wilson")
        cs = st.chi_square(pd.DataFrame([[k, n - k], [hk, hn - hk]]))
        rows.append([a, k / n, lo, hi, cs["odds_ratio"], cs["p"]])
    r = pd.DataFrame(rows, columns=["agent", "rate", "lo", "hi", "odds", "p"])
    r["p_adj"] = st.adjust_pvalues(r.p.tolist())
    r = r.sort_values("rate")

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    ax.axvspan(100 * hlo, 100 * hhi, color="#E6E6E6", zorder=0)
    ax.axvline(100 * hk / hn, color="#333333", ls="--", lw=0.9, zorder=1)
    lab_x = max(100 * r.hi.max(), 100 * hhi) + 0.8
    for i, x in enumerate(r.itertuples()):
        c = COLORS[x.agent]
        ax.errorbar(100 * x.rate, i, xerr=[[100 * (x.rate - x.lo)], [100 * (x.hi - x.rate)]], fmt="o", ms=3.5,
                    color=c, ecolor=c, elinewidth=1.4, capsize=2.5, zorder=3)
        ax.text(lab_x, i, f"OR {x.odds:.2f}" + ("*" if x.p_adj < config.ALPHA else ""),
                va="center", fontsize=7.5)
    ax.set_yticks(range(len(r)), r.agent)
    ax.set_xlabel("PRs with a revert commit within 90 days (%)")
    ax.set_xlim(0, lab_x + 3.2)
    ax.set_xticks([0, 2.5, 5, 7.5, 10])
    ax.grid(axis="x", color="#DDDDDD", lw=0.6)
    ax.set_axisbelow(True)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    plt.tight_layout(pad=0.3)
    out = config.ANALYSIS_DIR / "robustness" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["pdf.fonttype"] = 42
    fig.savefig(out / "fig_reverts.pdf")
    fig.savefig(out / "fig_reverts.png", dpi=300)
    print(r.round(4).to_string(index=False), f"\nhuman {hk}/{hn} = {hk / hn:.4f}")
    print(f"  [SAVED] {out / 'fig_reverts.pdf'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--column", default="revert_format")
    main(ap.parse_args().column)
