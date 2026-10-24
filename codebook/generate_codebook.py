"""
============================================================================
 generate_codebook.py, Variable Dictionary for the Master Table
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : Emit a codebook: every master.csv variable
           with type, definition, unit, source stage, and coverage.
           Definitions live in a curated dictionary; anything undocumented
           is flagged loudly so the codebook can never silently drift.
 Input   : data/processed/master.csv
 Output  : codebook/codebook.csv / .json / .txt
 Usage   : python codebook/generate_codebook.py
============================================================================
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                    # noqa: E402
from utils.io_helpers import load_master, save_csv, save_json, save_txt  # noqa: E402

D = {
    # identity / provenance
    "id": ("PR unique id (GitHub, via AIDev)", "-", "AIDev"),
    "number": ("PR number within its repository", "-", "AIDev"),
    "title": ("PR title text", "-", "AIDev"),
    "agent": ("Raw AIDev author label", "-", "AIDev"),
    "author_group": ("Canonical author group (5 agents + Human)", "-", "clean_and_merge"),
    "user": ("PR author login", "-", "AIDev"),
    "repo": ("owner/name parsed from html_url", "-", "clean_and_merge"),
    "html_url": ("PR web URL", "-", "AIDev"),
    "state": ("PR state (open/closed)", "-", "AIDev"),
    "created_at": ("PR creation timestamp", "UTC", "AIDev"),
    "closed_at": ("PR close timestamp", "UTC", "AIDev"),
    "merged_at": ("PR merge timestamp (raw)", "UTC", "AIDev"),
    "merged_at_dt": ("Parsed merge timestamp", "UTC", "clean_and_merge"),
    "is_merged": ("1 if PR merged", "0/1", "clean_and_merge"),
    # enrichment
    "enrich_status": ("GitHub enrichment outcome (ok/pr_gone/bad_url)", "-", "enrich_github"),
    "n_files": ("Files changed in PR", "count", "enrich_github"),
    "additions": ("Lines added", "lines", "enrich_github"),
    "deletions": ("Lines deleted", "lines", "enrich_github"),
    "changed_lines": ("additions + deletions", "lines", "clean_and_merge"),
    "changed_files_list": ("First 50 changed file paths", ";-sep", "enrich_github"),
    "languages_touched": ("Target languages present in diff", ";-sep", "enrich_github"),
    "post_merge_commits_90d": ("Commits touching PR's top file in 90d after merge", "count", "enrich_github"),
    "post_merge_authors_90d": ("Distinct authors of those commits", "count", "enrich_github"),
    "revert_detected": ("1 if revert-style commit found in follow-up window", "0/1", "enrich_github"),
    "is_revert_flagged": ("revert_detected with NaN->0", "0/1", "feature_engineering"),
    "size_bucket": ("PR size class XS/S/M/L/XL by changed_lines", "-", "feature_engineering"),
    # static analysis
    "static_status": ("Patch analysis outcome (ok/no_cache/no_target_lang)", "-", "static_analysis"),
    "added_lines_analyzed": ("Added diff lines in target languages", "lines", "static_analysis"),
    "languages_analyzed": ("Languages actually analyzed", ";-sep", "static_analysis"),
    "oversize_diff": ("1 if diff exceeded MAX_DIFF_LINES cap", "0/1", "static_analysis"),
    "todo_count": ("TODO/FIXME/HACK/XXX markers in added lines", "count", "static_analysis"),
    "long_lines": ("Added lines > 120 chars", "count", "static_analysis"),
    "comment_lines": ("Added comment lines", "count", "static_analysis"),
    "comment_ratio": ("comment_lines / added lines", "0-1", "static_analysis"),
    "branch_count": ("Branching keywords in added lines", "count", "static_analysis"),
    "branch_density": ("branch_count / added lines", "per line", "static_analysis"),
    "max_nesting": ("Max indent depth (4-space units) in added lines", "levels", "static_analysis"),
    "security_smells_total": ("Sum over CWE catalog hits", "count", "static_analysis"),
    "maintainability_smells_total": ("todo_count + long_lines", "count", "static_analysis"),
    "security_smell_density": ("Security smells per 100 added lines", "/100 lines", "feature_engineering"),
    "maintainability_smell_density": ("Maintainability smells per 100 added lines", "/100 lines", "feature_engineering"),
    "churn_per_100_lines": ("Post-merge commits per 100 changed lines", "/100 lines", "feature_engineering"),
    # review dynamics
    "n_reviews": ("Human reviews on PR (bots excluded)", "count", "feature_engineering"),
    "n_bot_reviews": ("Bot reviews on PR", "count", "feature_engineering"),
    "first_review_at": ("First human review timestamp", "UTC", "feature_engineering"),
    "n_change_requests": ("Reviews with CHANGES_REQUESTED", "count", "feature_engineering"),
    "review_latency_hours": ("Hours from PR creation to first human review", "hours", "feature_engineering"),
    "n_issue_comments": ("Issue-style comments on PR", "count", "feature_engineering"),
}
# CWE columns get generated definitions
CWE_DEF = ("Hits of {} pattern in added lines", "count", "static_analysis")


def define(col: str):
    if col in D:
        return D[col]
    if col.startswith("CWE-"):
        return (CWE_DEF[0].format(col), CWE_DEF[1], CWE_DEF[2])
    if col.startswith("has_CWE"):
        return (f"1 if any {col[4:]} hit in PR", "0/1", "feature_engineering")
    return None


def main():
    print("=" * 76)
    print(" CODEBOOK GENERATION")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)
    rows, undocumented = [], []
    for col in df.columns:
        d = define(col)
        if d is None:
            undocumented.append(col)
            d = ("!!! UNDOCUMENTED — add to codebook dictionary !!!", "?", "?")
        rows.append({
            "variable": col,
            "dtype": str(df[col].dtype),
            "definition": d[0], "unit": d[1], "source_stage": d[2],
            "non_null": int(df[col].notna().sum()),
            "coverage": round(float(df[col].notna().mean()), 4),
            "example": str(df[col].dropna().iloc[0])[:40] if df[col].notna().any() else "",
        })
    cb = pd.DataFrame(rows)

    lines = ["=" * 100, " CODEBOOK — master.csv variable dictionary",
             f" Generated: {datetime.now(timezone.utc).isoformat()}",
             f" Variables: {len(cb)}   PRs: {len(df):,}", "=" * 100,
             f" {'Variable':<34s} {'Type':<9s} {'Unit':<10s} {'Cover':>6s}  Definition",
             "-" * 100]
    for _, r in cb.iterrows():
        lines.append(f" {r['variable']:<34s} {r['dtype']:<9s} "
                     f"{r['unit']:<10s} {r['coverage']:>6.1%}  {r['definition']}")
    lines.append("=" * 100)

    save_csv(cb, config.CODEBOOK_DIR / "codebook.csv", "codebook")
    save_json(cb.to_dict("records"), config.CODEBOOK_DIR / "codebook.json",
              "codebook")
    save_txt("\n".join(lines), config.CODEBOOK_DIR / "codebook.txt", "codebook")

    if undocumented:
        print(f"  [WARN] {len(undocumented)} undocumented columns: {undocumented}")
        print("         add them to the D dictionary in this script")
    else:
        print(f"  [ OK ] all {len(cb)} variables documented")
    print("  Next: run the RQ analyses in analysis/")


if __name__ == "__main__":
    main()
