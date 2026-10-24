"""
============================================================================
 01_download_aidev.py, Download the AIDev Dataset Tables (Hugging Face)
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : Pull the AIDev normalized parquet tables from hao-li/AIDev and
           store raw snapshots locally, with a manifest of exactly what
           was downloaded (rows, columns, agents, date coverage).
 Input   : Hugging Face hub (public dataset, no token required)
 Output  : data/raw/<table>.parquet          (one file per table)
           data/raw/aidev_manifest.json      (machine-readable manifest)
           data/raw/aidev_download_report.txt(human-readable summary)
 Usage   : python data_collection/01_download_aidev.py            # essential
           python data_collection/01_download_aidev.py --all      # + optional
           python data_collection/01_download_aidev.py --tables pull_request
============================================================================
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config  # noqa: E402

# ---------------------------------------------------------------------------
# AIDev tables (verified against the dataset card, Aug-10-2025 refresh).
# ESSENTIAL = what the pipeline needs.  OPTIONAL = heavier extras.
# NOTE: use pr_review_comments_v2 (v1 is incomplete per the dataset card).
# ---------------------------------------------------------------------------
ESSENTIAL_TABLES = [
    "pull_request",            # AIDev-pop agentic PRs (>100-star repos)
    "repository",              # AIDev-pop repo metadata (stars, language, license)
    "human_pull_request",      # human PRs from same repos (>500 stars) !!
    "pr_timeline",             # PR event history (open/close/merge/label)
    "pr_reviews",              # review states + timestamps
    "pr_comments",             # issue-style comments on PRs
    "pr_review_comments_v2",   # inline review comments (v2 = complete)
]
OPTIONAL_TABLES = [
    "pr_commit_details",       # commits + patches (LARGE, needed for static analysis)
    "human_pr_task_type",      # LLM task-type labels for human PRs
    "all_pull_request",        # full 932K agentic PRs (population stats)
    "all_repository",          # full repo metadata
    "all_user",                # de-identified user table
]

HF_BASE = f"hf://datasets/{config.HF_DATASET_ID}"


def human_size(n_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def download_table(table: str) -> dict:
    """Download one parquet table, save locally, return manifest entry."""
    url = f"{HF_BASE}/{table}.parquet"
    local = config.DATA_RAW / f"{table}.parquet"
    print(f"\n  -> Downloading '{table}' ...")
    print(f"     from : {url}")
    t0 = time.time()
    try:
        df = pd.read_parquet(url)
    except Exception as e:
        print(f"     [FAIL] {type(e).__name__}: {e}")
        return {"table": table, "status": "failed", "error": str(e)}
    elapsed = time.time() - t0

    df.to_parquet(local, index=False)
    size = local.stat().st_size
    print(f"     [ OK ] {len(df):,} rows x {len(df.columns)} cols "
          f"in {elapsed:.1f}s  ({human_size(size)})")
    print(f"     saved: {local}")
    print(f"     cols : {', '.join(list(df.columns)[:12])}"
          f"{' ...' if len(df.columns) > 12 else ''}")

    entry = {
        "table": table,
        "status": "ok",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "n_columns": int(len(df.columns)),
        "file_size_bytes": int(size),
        "local_path": str(local),
        "seconds": round(elapsed, 1),
    }

    # Extra profiling for the PR tables (agent mix + date coverage)
    agent_col = next((c for c in df.columns if c.lower() in
                      ("agent", "agent_label", "bot", "author_type")), None)
    if agent_col:
        counts = df[agent_col].astype(str).value_counts()
        entry["agent_distribution"] = {str(k): int(v) for k, v in counts.items()}
        print("     agent mix:")
        for k, v in counts.items():
            print(f"        {k:<20s} {v:>10,}")
    date_col = next((c for c in df.columns if "created" in c.lower()), None)
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce", utc=True, format="mixed")
            entry["date_min"] = str(dates.min())
            entry["date_max"] = str(dates.max())
            print(f"     dates: {dates.min()}  ..  {dates.max()}")
        except Exception:
            pass
    return entry


def main():
    parser = argparse.ArgumentParser(description="Download AIDev tables")
    parser.add_argument("--all", action="store_true",
                        help="also download OPTIONAL (large) tables")
    parser.add_argument("--tables", nargs="+", default=None,
                        help="explicit table list (overrides --all)")
    args = parser.parse_args()

    tables = (args.tables if args.tables
              else ESSENTIAL_TABLES + (OPTIONAL_TABLES if args.all else []))

    print("=" * 76)
    print(" STEP 1/3 (COLLECTION) — DOWNLOAD AIDEV FROM HUGGING FACE")
    print("=" * 76)
    print(f"  Dataset : {config.HF_DATASET_ID}")
    print(f"  Tables  : {len(tables)} requested -> {', '.join(tables)}")
    print(f"  Target  : {config.DATA_RAW}")
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    entries, t0 = [], time.time()
    for i, table in enumerate(tables, 1):
        print(f"\n[{i}/{len(tables)}] {table}")
        entries.append(download_table(table))

    ok = [e for e in entries if e["status"] == "ok"]
    failed = [e for e in entries if e["status"] != "ok"]
    total_rows = sum(e.get("rows", 0) for e in ok)
    total_bytes = sum(e.get("file_size_bytes", 0) for e in ok)

    manifest = {
        "dataset": config.HF_DATASET_ID,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_tables_ok": len(ok),
        "n_tables_failed": len(failed),
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "elapsed_seconds": round(time.time() - t0, 1),
        "tables": entries,
    }
    config.FILE_AIDEV_MANIFEST.write_text(json.dumps(manifest, indent=2))

    # Human-readable TXT report
    lines = [
        "=" * 76,
        " AIDEV DOWNLOAD REPORT",
        f" Generated: {manifest['downloaded_at_utc']}",
        "=" * 76,
        f" Dataset      : {config.HF_DATASET_ID}",
        f" Tables OK    : {len(ok)}   |   Failed: {len(failed)}",
        f" Total rows   : {total_rows:,}",
        f" Total size   : {human_size(total_bytes)}",
        f" Elapsed      : {manifest['elapsed_seconds']}s",
        "-" * 76,
    ]
    for e in ok:
        lines.append(f" [OK]   {e['table']:<24s} {e['rows']:>10,} rows  "
                     f"{e['n_columns']:>3} cols  {human_size(e['file_size_bytes'])}")
        if "date_min" in e:
            lines.append(f"        dates {e['date_min'][:10]} .. {e['date_max'][:10]}")
        if "agent_distribution" in e:
            for k, v in e["agent_distribution"].items():
                lines.append(f"          {k:<22s} {v:>10,}")
    for e in failed:
        lines.append(f" [FAIL] {e['table']:<24s} {e.get('error','')[:60]}")
    lines.append("=" * 76)
    report = config.DATA_RAW / "aidev_download_report.txt"
    report.write_text("\n".join(lines))

    print("\n" + "=" * 76)
    print(" DOWNLOAD COMPLETE" if not failed else " DOWNLOAD FINISHED WITH FAILURES")
    print("=" * 76)
    print(f"  OK      : {len(ok)}/{len(tables)} tables, {total_rows:,} rows, "
          f"{human_size(total_bytes)}")
    print(f"  Manifest: {config.FILE_AIDEV_MANIFEST}")
    print(f"  Report  : {report}")
    if failed:
        print(f"  [WARN] failed tables: {[e['table'] for e in failed]}")
        print("         re-run with --tables <name> to retry just those")
    print(f"  Next    : python data_collection/02_enrich_github.py")
    sys.exit(1 if failed and not ok else 0)


if __name__ == "__main__":
    main()
