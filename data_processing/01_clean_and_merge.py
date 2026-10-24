"""
============================================================================
 01_clean_and_merge.py, Normalize, Dedupe, Join Agent + Human PRs
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : Produce ONE clean PR table with canonical author labels:
             1. load agent PRs (pull_request) + matched human baseline
             2. map AIDev agent labels -> canonical display names
             3. concat, dedupe, basic sanity filters
             4. left-join the GitHub enrichment columns
 Input   : data/raw/pull_request.parquet, data/interim/human_baseline.csv,
           data/interim/pr_enriched.csv
 Output  : data/interim/merged.csv (+ .json summary + merge_report.txt)
 Usage   : python data_processing/01_clean_and_merge.py
============================================================================
"""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                              # noqa: E402
from utils.io_helpers import save_csv, save_json, save_txt  # noqa: E402

# AIDev raw labels -> canonical display names (config.AUTHOR_GROUPS)
AGENT_LABEL_MAP = {
    "OpenAI_Codex": "OpenAI Codex",
    "Devin": "Devin",
    "Copilot": "GitHub Copilot",
    "Cursor": "Cursor",
    "Claude_Code": "Claude Code",
    "Human": "Human",
}


def parse_repo(html_url: str):
    m = re.match(r"https://github\.com/([^/]+/[^/]+)/pull/\d+", str(html_url))
    return m.group(1) if m else None


def main():
    print("=" * 76)
    print(" STEP 1/3 (PROCESSING) — CLEAN & MERGE")
    print("=" * 76)

    # ---- load ----------------------------------------------------------------
    agent_p = config.DATA_RAW / "pull_request.parquet"
    if not agent_p.exists():
        sys.exit(f"[FAIL] {agent_p} missing — run 01_download_aidev.py")
    if not config.FILE_HUMAN_BASELINE.exists():
        sys.exit(f"[FAIL] {config.FILE_HUMAN_BASELINE} missing — run "
                 "03_collect_human_baseline.py")
    agent = pd.read_parquet(agent_p)
    human = pd.read_csv(config.FILE_HUMAN_BASELINE)
    print(f"  loaded agent PRs : {len(agent):,}")
    print(f"  loaded human PRs : {len(human):,} (matched baseline)")

    # ---- harmonize schemas -----------------------------------------------------
    keep = ["id", "number", "title", "agent", "user", "state",
            "created_at", "closed_at", "merged_at", "html_url"]
    agent = agent[[c for c in keep if c in agent.columns]].copy()
    human = human[[c for c in keep if c in human.columns]].copy()
    df = pd.concat([agent, human], ignore_index=True)
    df["author_group"] = df["agent"].map(AGENT_LABEL_MAP)
    unmapped = df["author_group"].isna().sum()
    if unmapped:
        print(f"  [WARN] {unmapped} rows with unmapped agent labels "
              f"({df.loc[df.author_group.isna(),'agent'].unique()[:5]}) — dropped")
        df = df[df["author_group"].notna()]
    df["repo"] = df["html_url"].map(parse_repo)

    # ---- sanity filters ---------------------------------------------------------
    n0 = len(df)
    df = df.drop_duplicates(subset="id")
    print(f"  dedupe by id           : {n0:,} -> {len(df):,}")
    n0 = len(df)
    df = df[df["repo"].notna()]
    print(f"  valid html_url         : {n0:,} -> {len(df):,}")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True,
                                      errors="coerce", format="mixed")
    df["merged_at_dt"] = pd.to_datetime(df["merged_at"], utc=True,
                                        errors="coerce", format="mixed")
    df["is_merged"] = df["merged_at_dt"].notna().astype(int)

    # ---- join enrichment ---------------------------------------------------------
    if config.FILE_PR_ENRICHED.exists():
        enr = pd.read_csv(config.FILE_PR_ENRICHED)
        enr = enr.drop_duplicates(subset="pr_id")
        enr_cols = ["pr_id", "enrich_status", "n_files", "additions",
                    "deletions", "changed_files_list", "languages_touched",
                    "post_merge_commits_90d", "post_merge_authors_90d",
                    "revert_detected"]
        enr = enr[[c for c in enr_cols if c in enr.columns]]
        df = df.merge(enr, left_on="id", right_on="pr_id", how="left")
        df = df.drop(columns=["pr_id"])
        cov = df["enrich_status"].notna().mean()
        ok = (df["enrich_status"] == "ok").mean()
        print(f"  enrichment joined      : coverage {cov:.1%}, status-ok {ok:.1%}")
    else:
        print("  [WARN] pr_enriched.csv missing — enrichment columns skipped")

    df["changed_lines"] = df[["additions", "deletions"]].sum(axis=1, min_count=1)

    # ---- report -------------------------------------------------------------------
    by_group = df.groupby("author_group").agg(
        n=("id", "count"), merged=("is_merged", "sum"),
        enriched_ok=("enrich_status", lambda s: (s == "ok").sum()),
        med_changed=("changed_lines", "median"),
    ).reindex(config.AUTHOR_GROUPS).fillna(0).astype(
        {"n": int, "merged": int, "enriched_ok": int})

    lines = ["=" * 76, " CLEAN & MERGE REPORT",
             f" Generated: {datetime.now(timezone.utc).isoformat()}",
             "=" * 76,
             f" Total PRs               : {len(df):,}",
             f" Repos                   : {df['repo'].nunique():,}",
             f" Date range              : {df['created_at'].min():%Y-%m-%d} .. "
             f"{df['created_at'].max():%Y-%m-%d}",
             "-" * 76,
             f" {'Group':<16s} {'PRs':>8s} {'Merged':>8s} {'Enrich-OK':>10s} "
             f"{'Med.lines':>10s}"]
    for g, r in by_group.iterrows():
        lines.append(f" {g:<16s} {int(r['n']):>8,} {int(r['merged']):>8,} "
                     f"{int(r['enriched_ok']):>10,} {r['med_changed']:>10.0f}")
    lines.append("=" * 76)

    save_csv(df, config.FILE_MERGED, "merged PR table")
    save_json({"n": len(df), "by_group": by_group.reset_index()
               .to_dict("records")},
              config.DATA_INTERIM / "merged_summary.json", "merge summary")
    save_txt("\n".join(lines), config.DATA_INTERIM / "merge_report.txt",
             "merge report")
    print("\n".join(lines))
    print("  Next: python data_processing/02_run_static_analysis.py")


if __name__ == "__main__":
    main()
