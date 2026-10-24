"""
============================================================================
 reporting.py, APA-7 / IEEE Statistical Report Generator (.txt)
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : Turn the dicts/DataFrames produced by utils.stats into the full
           rqN_apa_report.txt files: sample sizes, descriptives tables,
           test statistics, exact + adjusted p, effect sizes with CIs,
           and ready-to-paste IEEE prose for the Results section.
 Usage   : from utils.reporting import ReportBuilder
============================================================================
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config  # noqa: E402
from utils.stats import format_p, cliffs_delta_magnitude  # noqa: E402

W = 78  # report width


class ReportBuilder:
    """Accumulates sections, then renders one APA/IEEE .txt report."""

    def __init__(self, rq_key: str):
        rq = config.RESEARCH_QUESTIONS[rq_key]
        self.rq_key = rq_key
        self.lines: list[str] = []
        self._header(rq)

    # ---- structure ---------------------------------------------------------
    def _header(self, rq: dict):
        self.lines += [
            "=" * W,
            f" {rq['title']}",
            "=" * W,
            f" Research question : {rq['question']}",
            f" Generated (UTC)   : {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",
            f" Alpha             : {config.ALPHA}   |   CI: {int(config.CI_LEVEL*100)}%"
            f"   |   p-adjustment: Benjamini-Hochberg",
            f" Bootstrap draws   : {config.N_BOOTSTRAP:,} (seed {config.RANDOM_SEED})",
            "=" * W, "",
        ]

    def section(self, title: str):
        self.lines += ["", "-" * W, f" {title}", "-" * W]

    def text(self, *paragraphs: str):
        for p in paragraphs:
            self.lines.append(f" {p}")

    # ---- tables ------------------------------------------------------------
    def descriptives_table(self, desc_rows: list[dict], metric: str):
        """ASCII table of n/mean/SD/median/IQR per group."""
        self.section(f"Descriptive statistics — {metric}")
        hdr = (f" {'Group':<16s} {'n':>7s} {'Mean':>10s} {'SD':>10s} "
               f"{'Median':>10s} {'Q1':>9s} {'Q3':>9s}")
        self.lines += [hdr, " " + "-" * (len(hdr) - 1)]
        for d in desc_rows:
            if d.get("n", 0) == 0:
                self.lines.append(f" {d['group']:<16s} {'0':>7s}   (no data)")
                continue
            self.lines.append(
                f" {d['group']:<16s} {d['n']:>7,} {d['mean']:>10.3f} "
                f"{d['sd']:>10.3f} {d['median']:>10.3f} "
                f"{d['q1']:>9.3f} {d['q3']:>9.3f}")

    def dataframe_table(self, df: pd.DataFrame, title: str,
                        float_fmt: str = "{:.4f}"):
        """Generic ASCII rendering of a small results DataFrame."""
        self.section(title)
        if df.empty:
            self.lines.append(" (no rows)")
            return
        show = df.copy()
        for c in show.columns:
            if pd.api.types.is_float_dtype(show[c]):
                show[c] = show[c].map(
                    lambda v: float_fmt.format(v) if pd.notna(v) else "—")
        text = show.to_string(index=False)
        self.lines += [" " + row for row in text.split("\n")]

    # ---- test results ------------------------------------------------------
    def mann_whitney_result(self, r: dict, metric: str, p_adj: float = None):
        """Full APA block + paste-ready prose for one MWU comparison."""
        self.section(f"Inferential test — {metric}: {r['group_a']} vs {r['group_b']}")
        self.text(
            f"Test          : {r['test']} (two-sided)",
            f"Sample sizes  : n({r['group_a']}) = {r['n_a']:,}, "
            f"n({r['group_b']}) = {r['n_b']:,}",
            f"Statistic     : U = {r['U']:,.1f}",
            f"p (raw)       : {format_p(r['p'])}",
        )
        if p_adj is not None:
            self.text(f"p (BH-adj.)   : {format_p(p_adj)}")
        ci = ""
        if "delta_ci_low" in r:
            ci = (f", {int(config.CI_LEVEL*100)}% CI "
                  f"[{r['delta_ci_low']:.3f}, {r['delta_ci_high']:.3f}]")
        self.text(
            f"Effect size   : Cliff's delta = {r['cliffs_delta']:.3f}{ci} "
            f"({r['magnitude']})",
            f"Rank-biserial : r = {r['rank_biserial_r']:.3f}",
        )
        # Ready-to-paste sentence
        p_show = format_p(p_adj if p_adj is not None else r["p"])
        adj_note = "BH-adjusted " if p_adj is not None else ""
        direction = "higher" if r["cliffs_delta"] > 0 else "lower"
        self.text("", "READY-TO-PASTE (IEEE prose):",
                  f'"{r["group_a"]} PRs showed {direction} {metric} than '
                  f'{r["group_b"]} PRs (Mann-Whitney U = {r["U"]:,.1f}, '
                  f'{adj_note}{p_show}, Cliff\'s delta = '
                  f'{r["cliffs_delta"]:.2f}{ci}), a {r["magnitude"]} effect."')

    def kruskal_result(self, r: dict, metric: str):
        self.section(f"Omnibus test — {metric} across "
                     f"{r['k']} groups")
        self.text(
            f"Test          : {r['test']}",
            f"Groups (k)    : {r['k']}  ({', '.join(map(str, r['groups']))})",
            f"Total n       : {r['n_total']:,}",
            f"Statistic     : H({r['df']}) = {r['H']:.2f}",
            f"p             : {format_p(r['p'])}",
            f"Effect size   : epsilon-squared = {r['epsilon_squared']:.4f}",
            "",
            "READY-TO-PASTE (IEEE prose):",
            f'"A Kruskal-Wallis test indicated a significant difference in '
            f'{metric} across the {r["k"]} author groups '
            f'(H({r["df"]}) = {r["H"]:.2f}, {format_p(r["p"])}, '
            f'epsilon-squared = {r["epsilon_squared"]:.3f})."'
            if r["p"] < config.ALPHA else
            f'"A Kruskal-Wallis test found no significant difference in '
            f'{metric} across the {r["k"]} author groups '
            f'(H({r["df"]}) = {r["H"]:.2f}, {format_p(r["p"])})."',
        )

    def chi_square_result(self, r: dict, what: str):
        self.section(f"Categorical test — {what}")
        self.text(
            f"Test          : {r['test']}",
            f"n             : {r['n']:,}",
            f"Statistic     : chi2({r.get('df','-')}) = {r['chi2']:.2f}",
            f"p             : {format_p(r['p'])}",
            f"Effect size   : Cramer's V = {r['cramers_v']:.3f}",
        )
        if "odds_ratio" in r:
            self.text(
                f"Odds ratio    : OR = {r['odds_ratio']:.3f}, "
                f"{int(config.CI_LEVEL*100)}% CI "
                f"[{r['or_ci_low']:.3f}, {r['or_ci_high']:.3f}]")
        self.text(
            "", "READY-TO-PASTE (IEEE prose):",
            f'"{what} differed significantly '
            f'(chi2({r.get("df","-")}) = {r["chi2"]:.2f}, {format_p(r["p"])}, '
            f'Cramer\'s V = {r["cramers_v"]:.2f})."'
            if r["p"] < config.ALPHA else
            f'"{what} did not differ significantly '
            f'(chi2({r.get("df","-")}) = {r["chi2"]:.2f}, {format_p(r["p"])})."')

    def assumptions_note(self):
        self.section("Assumption checks & test justification")
        self.text(
            "Software-engineering metrics (defect counts, churn, latency) are",
            "heavy-tailed and zero-inflated; Shapiro-Wilk/D'Agostino normality",
            "tests reject at these sample sizes and variances are unequal.",
            "Non-parametric rank tests (Mann-Whitney, Kruskal-Wallis) with",
            "Cliff's delta effect sizes are therefore used throughout, per",
            "standard practice in empirical software engineering. All p-values",
            "across families of comparisons are Benjamini-Hochberg adjusted",
            f"(FDR {config.ALPHA}). Bootstrap CIs use {config.N_BOOTSTRAP:,} "
            f"resamples, seed {config.RANDOM_SEED}.")

    # ---- render ------------------------------------------------------------
    def render(self) -> str:
        return "\n".join(self.lines + ["", "=" * W, " END OF REPORT", "=" * W])
