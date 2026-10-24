"""
============================================================================
 03_feature_engineering.py, Build the Master Analysis Table
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : Join everything into ONE analysis-ready table (one row per PR)
           with every feature the five RQs need:
             - size features + PR size buckets (XS..XL, GitHub-style)
             - RQ1  defect proxies  : security/maintainability smell density
             - RQ2  maintainability : nesting, branch density, comment ratio
             - RQ3  maintenance     : post-merge churn, reverts
             - RQ4  CWE flags       : per-class presence indicators
             - RQ5  review dynamics : latency, review/comment depth
                    (from AIDev pr_reviews / pr_comments where available)
 Input   : data/interim/merged.csv, static_metrics.csv,
           data/raw/pr_reviews.parquet, pr_comments.parquet (optional)
 Output  : data/processed/master.csv (+ master_summary.json + report txt)
 Usage   : python data_processing/03_feature_engineering.py
============================================================================
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                    # noqa: E402
from utils.io_helpers import save_csv, save_json, save_txt  # noqa: E402


def size_bucket(changed):
    if pd.isna(changed):
        return None
    for label, cap in config.PR_SIZE_BUCKETS:
        if changed <= cap:
            return label
    return "XL"


def per100(numer, denom):
    """Rate per 100 lines with safe division."""
    return np.where((denom > 0) & numer.notna(), numer / denom * 100, np.nan)


def main():
    print("=" * 76)
    print(" STEP 3/3 (PROCESSING) — FEATURE ENGINEERING -> MASTER TABLE")
    print("=" * 76)
    if not config.FILE_MERGED.exists():
        sys.exit("[FAIL] merged.csv missing — run 01_clean_and_merge.py")
    df = pd.read_csv(config.FILE_MERGED, low_memory=False)
    print(f"  merged table        : {len(df):,} PRs")

    # ---- static metrics join ---------------------------------------------------
    if config.FILE_STATIC_METRICS.exists():
        sm = pd.read_csv(config.FILE_STATIC_METRICS, low_memory=False).drop_duplicates("id")
        df = df.merge(sm, on="id", how="left")
        print(f"  static metrics join : "
              f"{(df['static_status']=='ok').mean():.1%} analyzed ok")
    else:
        print("  [WARN] static_metrics.csv missing — smell features skipped")

    # ---- size features ------------------------------------------------------------
    df["size_bucket"] = df["changed_lines"].map(size_bucket)

    # ---- RQ1: defect proxies (densities per 100 added lines) -----------------------
    if "security_smells_total" in df:
        df["security_smell_density"] = per100(
            df["security_smells_total"], df["added_lines_analyzed"])
        df["maintainability_smell_density"] = per100(
            df["maintainability_smells_total"], df["added_lines_analyzed"])

    # ---- RQ3: maintenance overhead ----------------------------------------------
    if "post_merge_commits_90d" in df:
        df["churn_per_100_lines"] = per100(
            df["post_merge_commits_90d"], df["changed_lines"])

    # ---- RQ4: CWE presence flags ----------------------------------------------------
    cwe_cols = [c for c in df.columns if c.startswith("CWE-")]
    for c in cwe_cols:
        df[f"has_{c.split('_')[0]}"] = (df[c].fillna(0) > 0).astype(int)

    # ---- RQ5: review dynamics from AIDev tables (coverage varies) -------------------
    reviews_p = config.DATA_RAW / "pr_reviews.parquet"
    comments_p = config.DATA_RAW / "pr_comments.parquet"
    if reviews_p.exists():
        rv = pd.read_parquet(reviews_p)
        rv["submitted_at"] = pd.to_datetime(rv["submitted_at"], utc=True,
                                            errors="coerce", format="mixed")
        # Bots (CI, Copilot-review, etc.) respond in seconds and would
        # contaminate review-latency; keep HUMAN reviewers for RQ5 metrics,
        # but retain bot counts as their own feature.
        if "user_type" in rv.columns:
            bot_agg = (rv[rv["user_type"] != "User"].groupby("pr_id").size()
                       .rename("n_bot_reviews").reset_index())
            df = df.merge(bot_agg, left_on="id", right_on="pr_id", how="left")
            df = df.drop(columns=["pr_id"], errors="ignore")
            n_before = len(rv)
            rv = rv[rv["user_type"] == "User"]
            print(f"  pr_reviews: kept {len(rv):,}/{n_before:,} human reviews "
                  f"(bots -> n_bot_reviews feature)")
        agg = rv.groupby("pr_id").agg(
            n_reviews=("id", "count"),
            first_review_at=("submitted_at", "min"),
            n_change_requests=("state",
                               lambda s: (s == "CHANGES_REQUESTED").sum()),
        ).reset_index()
        df = df.merge(agg, left_on="id", right_on="pr_id", how="left")
        df = df.drop(columns=["pr_id"], errors="ignore")
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True,
                                          errors="coerce", format="mixed")
        df["review_latency_hours"] = (
            (df["first_review_at"] - df["created_at"])
            .dt.total_seconds() / 3600)
        df.loc[df["review_latency_hours"] < 0, "review_latency_hours"] = np.nan
        cov = df["n_reviews"].notna().groupby(df["author_group"]).mean()
        print("  review coverage by group (pr_reviews table):")
        for g, v in cov.items():
            print(f"    {g:<16s} {v:6.1%}")
    if comments_p.exists():
        cm = pd.read_parquet(comments_p)
        agg = cm.groupby("pr_id").size().rename("n_issue_comments").reset_index()
        df = df.merge(agg, left_on="id", right_on="pr_id", how="left")
        df = df.drop(columns=["pr_id"], errors="ignore")

    # ---- final master ------------------------------------------------------------
    df["is_revert_flagged"] = df.get(
        "revert_detected", pd.Series(np.nan, index=df.index)).fillna(0).astype(int)
    save_csv(df, config.FILE_MASTER, "MASTER table")

    by = df.groupby("author_group")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_prs": int(len(df)),
        "n_features": int(len(df.columns)),
        "groups": {g: int(n) for g, n in by.size().items()},
        "feature_columns": list(df.columns),
    }
    save_json(summary, config.FILE_MASTER_JSON, "master summary")

    lines = ["=" * 76, " MASTER TABLE REPORT",
             f" Generated: {summary['generated_at_utc']}", "=" * 76,
             f" Rows (PRs)    : {len(df):,}",
             f" Features      : {len(df.columns)}",
             "-" * 76,
             f" {'Group':<16s} {'PRs':>8s} {'Sec.dens(med)':>14s} "
             f"{'Churn(med)':>11s} {'Revert%':>8s} {'RevLat.h(med)':>14s}"]
    for g in config.AUTHOR_GROUPS:
        sub = df[df["author_group"] == g]
        if len(sub) == 0:
            continue
        sd = sub.get("security_smell_density", pd.Series(dtype=float)).median()
        ch = sub.get("post_merge_commits_90d", pd.Series(dtype=float)).median()
        rv_ = sub["is_revert_flagged"].mean() * 100
        rl = sub.get("review_latency_hours", pd.Series(dtype=float)).median()
        lines.append(f" {g:<16s} {len(sub):>8,} "
                     f"{sd if pd.notna(sd) else float('nan'):>14.3f} "
                     f"{ch if pd.notna(ch) else float('nan'):>11.1f} "
                     f"{rv_:>7.1f}% "
                     f"{rl if pd.notna(rl) else float('nan'):>14.1f}")
    lines += ["-" * 76,
              " NOTE: review-dynamics coverage differs by group (AIDev review",
              " tables cover agentic PRs); RQ5 reports coverage explicitly and",
              " compares within covered strata.", "=" * 76]
    save_txt("\n".join(lines), config.DATA_PROCESSED / "master_report.txt",
             "master report")
    print("\n".join(lines))
    print("  Next: python data_validation/validate_dataset.py")


if __name__ == "__main__":
    main()
