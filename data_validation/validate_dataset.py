"""
============================================================================
 validate_dataset.py, Quality Gate Before Any Analysis Runs
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : Assert the master table is analysis-ready. Checks schema,
           uniqueness, ranges, group sizes, and per-RQ feature coverage.
           EXITS NONZERO if any blocking check fails, run this before
           trusting any RQ output.
 Input   : data/processed/master.csv
 Output  : data/processed/validation_report.txt / .json (+ console)
 Usage   : python data_validation/validate_dataset.py
============================================================================
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                    # noqa: E402
from utils.io_helpers import load_master, save_json, save_txt  # noqa: E402

PASS, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"
checks, failures, warnings_ = [], [], []


def check(name: str, ok: bool, detail: str, blocking: bool = True):
    tag = PASS if ok else (FAIL if blocking else WARN)
    line = f"  {tag} {name:<42s} {detail}"
    print(line); checks.append(line)
    if not ok:
        (failures if blocking else warnings_).append(f"{name}: {detail}")


def main():
    print("=" * 76)
    print(" VALIDATION GATE — master.csv")
    print("=" * 76)
    df = load_master(config.FILE_MASTER)

    # ---- 1. schema ------------------------------------------------------------
    required = ["id", "author_group", "repo", "created_at", "is_merged",
                "changed_lines", "size_bucket"]
    missing = [c for c in required if c not in df.columns]
    check("required columns present", not missing,
          f"missing: {missing}" if missing else f"{len(df.columns)} columns")

    # ---- 2. uniqueness ----------------------------------------------------------
    dup = df["id"].duplicated().sum()
    check("PR ids unique", dup == 0, f"{dup} duplicates")

    # ---- 3. author groups --------------------------------------------------------
    groups = set(df["author_group"].dropna().unique())
    expected = set(config.AUTHOR_GROUPS)
    check("author groups match config", groups <= expected,
          f"unexpected: {groups - expected}" if groups - expected
          else f"{len(groups)}/{len(expected)} groups present")
    sizes = df.groupby("author_group").size()
    too_small = sizes[sizes < 30]
    check("every group n >= 30", len(too_small) == 0,
          f"small: {dict(too_small)}" if len(too_small) else
          f"min group n = {sizes.min():,}", blocking=False)

    # ---- 4. value ranges ----------------------------------------------------------
    neg = (df["changed_lines"].dropna() < 0).sum()
    check("changed_lines >= 0", neg == 0, f"{neg} negative")
    if "review_latency_hours" in df:
        neg_lat = (df["review_latency_hours"].dropna() < 0).sum()
        check("review latency >= 0", neg_lat == 0, f"{neg_lat} negative")
    if "comment_ratio" in df:
        bad = (~df["comment_ratio"].dropna().between(0, 1)).sum()
        check("comment_ratio in [0,1]", bad == 0, f"{bad} out of range")
    dates = pd.to_datetime(df["created_at"], utc=True, errors="coerce",
                           format="mixed")
    check("created_at parseable", dates.notna().mean() > 0.99,
          f"{dates.isna().sum()} unparseable")

    # ---- 5. per-RQ feature coverage (informative, non-blocking) --------------------
    print("-" * 76)
    print("  Per-RQ feature coverage (non-null share of relevant PRs):")
    rq_features = {
        "RQ1": ["security_smell_density", "maintainability_smell_density"],
        "RQ2": ["max_nesting", "branch_density", "comment_ratio"],
        "RQ3": ["post_merge_commits_90d", "revert_detected"],
        "RQ4": [c for c in df.columns if c.startswith("has_CWE")],
        "RQ5": ["review_latency_hours", "n_reviews"],
    }
    cov_summary = {}
    for rq, cols in rq_features.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            check(f"{rq} features exist", False, "NONE FOUND", blocking=True)
            continue
        cov = df[cols].notna().mean().mean()
        cov_summary[rq] = round(float(cov), 4)
        check(f"{rq} coverage ({len(cols)} features)", cov > 0.05,
              f"{cov:.1%}", blocking=(cov == 0))
        for g in config.AUTHOR_GROUPS:
            sub = df[df["author_group"] == g]
            if len(sub):
                gcov = sub[cols].notna().mean().mean()
                print(f"          {g:<16s} {gcov:6.1%}")

    # ---- verdict ---------------------------------------------------------------------
    verdict = "MASTER TABLE VALID — safe to run analyses" if not failures \
        else "VALIDATION FAILED — fix before running analyses"
    lines = (["=" * 76, " VALIDATION REPORT",
              f" Generated: {datetime.now(timezone.utc).isoformat()}",
              "=" * 76] + checks +
             ["-" * 76, f" Failures: {len(failures)}  Warnings: {len(warnings_)}",
              f" VERDICT : {verdict}", "=" * 76])
    save_txt("\n".join(lines),
             config.DATA_PROCESSED / "validation_report.txt", "validation")
    save_json({"failures": failures, "warnings": warnings_,
               "rq_coverage": cov_summary,
               "n_rows": len(df)},
              config.DATA_PROCESSED / "validation_report.json", "validation")
    print("-" * 76)
    print(f"  VERDICT: {verdict}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
