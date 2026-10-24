"""
============================================================================
 03_collect_human_baseline.py, Build the Matched Human PR Baseline
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : AIDev already ships a human_pull_request table sampled from the
           SAME repositories as the agentic PRs (>500-star repos). This
           script turns it into a documented, matched baseline:
             1. load human PRs + agent PRs
             2. keep human PRs from repos that also have agent PRs
             3. restrict to the shared time window
             4. cap PRs per repo (no single repo dominates)
             5. report matching quality (repo overlap, time overlap)
 Input   : data/raw/human_pull_request.parquet, data/raw/pull_request.parquet
 Output  : data/interim/human_baseline.csv / .json
           data/interim/human_baseline_report.txt
 Usage   : python data_collection/03_collect_human_baseline.py
============================================================================
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config  # noqa: E402


def parse_repo(html_url: str) -> str | None:
    m = re.match(r"https://github\.com/([^/]+/[^/]+)/pull/\d+", str(html_url))
    return m.group(1) if m else None


def main():
    print("=" * 76)
    print(" STEP 3/3 (COLLECTION) — MATCHED HUMAN BASELINE")
    print("=" * 76)

    # ---- load ---------------------------------------------------------------
    paths = {
        "human": config.DATA_RAW / "human_pull_request.parquet",
        "agent": config.DATA_RAW / "pull_request.parquet",
    }
    for k, p in paths.items():
        if not p.exists():
            print(f"  [FAIL] missing {p} — run 01_download_aidev.py first")
            sys.exit(1)
    human = pd.read_parquet(paths["human"])
    agent = pd.read_parquet(paths["agent"])
    print(f"  loaded human PRs : {len(human):,}")
    print(f"  loaded agent PRs : {len(agent):,}")

    human["repo"] = human["html_url"].map(parse_repo)
    agent["repo"] = agent["html_url"].map(parse_repo)

    # ---- 1. repo overlap ----------------------------------------------------
    agent_repos = set(agent["repo"].dropna())
    before = len(human)
    human = human[human["repo"].isin(agent_repos)].copy()
    print(f"\n  [match 1] same-repo filter : {before:,} -> {len(human):,} "
          f"(repos shared with agents: {human['repo'].nunique():,})")

    # ---- 2. shared time window ----------------------------------------------
    human["created_dt"] = pd.to_datetime(human["created_at"], utc=True,
                                         errors="coerce", format="mixed")
    agent["created_dt"] = pd.to_datetime(agent["created_at"], utc=True,
                                         errors="coerce", format="mixed")
    lo, hi = agent["created_dt"].min(), agent["created_dt"].max()
    before = len(human)
    human = human[human["created_dt"].between(lo, hi)]
    print(f"  [match 2] time window     : {before:,} -> {len(human):,} "
          f"({lo:%Y-%m-%d} .. {hi:%Y-%m-%d})")

    # ---- 3. per-repo cap (deterministic sample, seeded) ----------------------
    rng = np.random.default_rng(config.RANDOM_SEED)
    before = len(human)
    human = (human.sample(frac=1.0, random_state=config.RANDOM_SEED)
                  .groupby("repo", group_keys=False)
                  .head(config.HUMAN_PRS_PER_REPO * 4))   # generous cap; final
    print(f"  [match 3] per-repo cap    : {before:,} -> {len(human):,} "
          f"(max {config.HUMAN_PRS_PER_REPO * 4}/repo, seed={config.RANDOM_SEED})")

    # ---- matching-quality metrics --------------------------------------------
    h_repos = set(human["repo"])
    overlap = len(h_repos & agent_repos)
    agent_in_h = agent[agent["repo"].isin(h_repos)]
    merged_rate_h = human["merged_at"].notna().mean()
    merged_rate_a = agent["merged_at"].notna().mean()

    # ---- save -----------------------------------------------------------------
    human = human.drop(columns=["created_dt"])
    human.to_csv(config.FILE_HUMAN_BASELINE, index=False)
    human.to_json(config.FILE_HUMAN_BASELINE.with_suffix(".json"),
                  orient="records", indent=2)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_human_prs": int(len(human)),
        "n_repos": int(human["repo"].nunique()),
        "repo_overlap_with_agents": int(overlap),
        "agent_prs_in_matched_repos": int(len(agent_in_h)),
        "time_window": [str(lo), str(hi)],
        "merged_rate_human": round(float(merged_rate_h), 4),
        "merged_rate_agent": round(float(merged_rate_a), 4),
        "per_repo_cap": config.HUMAN_PRS_PER_REPO * 4,
        "seed": config.RANDOM_SEED,
    }
    lines = [
        "=" * 76,
        " HUMAN BASELINE MATCHING REPORT",
        f" Generated: {summary['generated_at_utc']}",
        "=" * 76,
        f" Human PRs in baseline      : {summary['n_human_prs']:,}",
        f" Distinct repos             : {summary['n_repos']:,}",
        f" Repos shared with agents   : {summary['repo_overlap_with_agents']:,}",
        f" Agent PRs in those repos   : {summary['agent_prs_in_matched_repos']:,}",
        f" Time window                : {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}",
        f" Merged rate  human / agent : {merged_rate_h:.1%} / {merged_rate_a:.1%}",
        f" Per-repo cap               : {summary['per_repo_cap']} (seed {config.RANDOM_SEED})",
        "-" * 76,
        " NOTE: AIDev's human PRs come from the same repositories as agentic",
        " PRs (>500-star subset), collected by the dataset authors. This",
        " script documents and tightens the match; per-metric size matching",
        " (PR size buckets) happens later in 03_feature_engineering.py where",
        " additions/deletions are available for both groups.",
        "=" * 76,
    ]
    report = config.DATA_INTERIM / "human_baseline_report.txt"
    report.write_text("\n".join(lines))
    (config.DATA_INTERIM / "human_baseline_summary.json").write_text(
        json.dumps(summary, indent=2))

    print("\n".join(lines))
    print(f"  CSV   : {config.FILE_HUMAN_BASELINE}")
    print(f"  JSON  : {config.FILE_HUMAN_BASELINE.with_suffix('.json')}")
    print(f"  Report: {report}")
    print("  Next  : python data_processing/01_clean_and_merge.py")


if __name__ == "__main__":
    main()
