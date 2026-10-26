"""
============================================================================
 02_enrich_github.py, Enrich PRs via the GitHub REST API (resumable)
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : For each PR (agentic + human) add what AIDev tables don't carry:
             - files changed (paths, additions, deletions, patch presence)
             - post-merge maintenance signals (later commits touching the
               same files within FOLLOWUP_DAYS, revert-commit detection)
           Every API response is cached to disk, so the script is fully
           RESUMABLE: re-running skips completed PRs instantly.
 Input   : data/raw/pull_request.parquet, data/raw/human_pull_request.parquet
 Output  : data/interim/pr_enriched.csv / .json
           data/interim/enrichment_report.txt
 Usage   : python data_collection/02_enrich_github.py                 # all PRs
           python data_collection/02_enrich_github.py --max-prs 500   # budget
           python data_collection/02_enrich_github.py --merged-only
============================================================================
 API budget: ~2-3 requests per merged PR (files, follow-up commits).
 At 5,000 req/h authenticated: ~1,700-2,500 PRs per hour.
============================================================================
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config  # noqa: E402

load_dotenv(config.PROJECT_ROOT / ".env")
TOKEN = os.getenv("GITHUB_TOKEN", "")

HEADERS = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

FOLLOWUP_DAYS = 90          # window for post-merge maintenance signals
REVERT_RE = re.compile(r"\brevert(s|ed|ing)?\b", re.IGNORECASE)
# a revert commit proper: "Revert ...", "Reverts owner/repo#N", or git's "This reverts commit"
REVERT_COMMIT_RE = re.compile(r"^\s*Revert\b|^\s*Reverts\s+\S+#\d+|This reverts commit",
                              re.IGNORECASE | re.MULTILINE)
CHECKPOINT_EVERY = 200      # write partial CSV every N PRs


# ---------------------------------------------------------------------------
# Cached GET, the heart of resumability
# ---------------------------------------------------------------------------
def cached_get(url: str, params: dict | None = None) -> dict | list | None:
    """GET with on-disk cache + rate-limit handling. Returns parsed JSON."""
    key = hashlib.sha1(
        (url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    cache_file = config.CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            cache_file.unlink()  # corrupted cache entry -> refetch

    for attempt in range(config.GITHUB_MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, params=params,
                             timeout=config.GITHUB_TIMEOUT_S)
        except requests.RequestException as e:
            print(f"\n    [WARN] network error ({e}); retry {attempt+1}")
            time.sleep(2 ** attempt)
            continue

        # Rate limit: sleep until reset, then retry
        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time(), 0) + 5
            print(f"\n    [RATE LIMIT] sleeping {wait/60:.1f} min "
                  f"(resumes {datetime.fromtimestamp(reset):%H:%M:%S})")
            time.sleep(wait)
            continue
        if r.status_code in (404, 410, 451):
            # 404 deleted, 410 gone, 451 DMCA/legal block, skip & remember
            cache_file.write_text("null")
            return None
        if r.status_code == 401:
            print("\n    [FATAL] 401 Unauthorized — token invalid/expired. "
                  "Fix GITHUB_TOKEN in .env")
            sys.exit(1)
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if r.status_code >= 400:
            # any other client error: log once, skip this URL, NEVER crash the run
            print(f"\n    [SKIP] HTTP {r.status_code} on {url}")
            cache_file.write_text("null")
            return None
        data = r.json()
        cache_file.write_text(json.dumps(data))
        return data
        data = r.json()
        cache_file.write_text(json.dumps(data))
        return data
    print(f"\n    [FAIL] gave up on {url}")
    return None


def parse_repo(html_url: str) -> str | None:
    """https://github.com/owner/repo/pull/123 -> 'owner/repo'"""
    m = re.match(r"https://github\.com/([^/]+/[^/]+)/pull/\d+", str(html_url))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Per-PR enrichment
# ---------------------------------------------------------------------------
def enrich_pr(row: pd.Series) -> dict:
    repo = parse_repo(row["html_url"])
    number = int(row["number"])
    out = {
        "pr_id": row["id"], "repo": repo, "number": number,
        "agent": row["agent"],
        "enrich_status": "ok",
        "n_files": None, "additions": None, "deletions": None,
        "changed_files_list": None, "languages_touched": None,
        "post_merge_commits_90d": None, "post_merge_authors_90d": None,
        "revert_detected": None, "revert_commit": None, "revert_of_pr": None,
    }
    if repo is None:
        out["enrich_status"] = "bad_url"
        return out

    # ---- 1. files changed --------------------------------------------------
    files = cached_get(
        f"{config.GITHUB_API_BASE}/repos/{repo}/pulls/{number}/files",
        params={"per_page": config.GITHUB_PER_PAGE})
    if files is None:
        out["enrich_status"] = "pr_gone"
        return out
    paths = [f.get("filename", "") for f in files]
    out["n_files"] = len(paths)
    out["additions"] = sum(f.get("additions", 0) for f in files)
    out["deletions"] = sum(f.get("deletions", 0) for f in files)
    out["changed_files_list"] = ";".join(paths[:50])
    langs = {config.LANGUAGE_EXTENSIONS.get(Path(p).suffix)
             for p in paths if Path(p).suffix in config.LANGUAGE_EXTENSIONS}
    out["languages_touched"] = ";".join(sorted(l for l in langs if l))

    # ---- 2. post-merge maintenance (merged PRs only) ------------------------
    merged_at = row.get("merged_at")
    if pd.isna(merged_at) or not merged_at:
        return out
    merged_dt = pd.to_datetime(merged_at, utc=True)
    until = merged_dt + timedelta(days=FOLLOWUP_DAYS)
    # commits touching the single most-changed file after merge (1 request,
    # a good proxy that keeps the API budget linear in PRs, not files)
    if paths:
        top_file = max(files, key=lambda f: f.get("changes", 0)).get("filename")
        commits = cached_get(
            f"{config.GITHUB_API_BASE}/repos/{repo}/commits",
            params={"path": top_file,
                    "since": merged_dt.isoformat(),
                    "until": until.isoformat(),
                    "per_page": config.GITHUB_PER_PAGE}) or []
        # keep commits strictly after the merge and drop the PR's own commits.
        # v1.0.0 compared the timestamps as text ("...Z" vs "...+00:00"), which kept
        # the PR's own squash or merge commit; parse them instead (fixed in v1.1.0).
        own_tag = f"#{number}"

        def authored(c):
            return pd.to_datetime(c.get("commit", {}).get("author", {}).get("date"),
                                  utc=True, errors="coerce")

        after = [c for c in commits
                 if pd.notna(authored(c)) and authored(c) > merged_dt]
        later = [c for c in after
                 if own_tag not in c.get("commit", {}).get("message", "")]
        out["post_merge_commits_90d"] = len(later)
        out["post_merge_authors_90d"] = len({
            (c.get("author") or {}).get("login")
            or c.get("commit", {}).get("author", {}).get("name")
            for c in later})
        out["revert_detected"] = int(any(
            REVERT_RE.search(c.get("commit", {}).get("message", ""))
            for c in later))
        out["revert_commit"] = int(any(
            REVERT_COMMIT_RE.search(c.get("commit", {}).get("message", ""))
            for c in later))
        title = str(row.get("title") or "").strip().lower()
        out["revert_of_pr"] = int(any(
            REVERT_COMMIT_RE.search(m) and (own_tag in m or (len(title) > 12 and title[:60] in m.lower()))
            for m in (c.get("commit", {}).get("message", "") for c in after)))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Enrich PRs via GitHub API")
    ap.add_argument("--max-prs", type=int, default=None,
                    help="cap total PRs enriched this run (API budget)")
    ap.add_argument("--merged-only", action="store_true",
                    help="only enrich merged PRs")
    args = ap.parse_args()

    print("=" * 76)
    print(" STEP 2/3 (COLLECTION) — GITHUB ENRICHMENT (cached + resumable)")
    print("=" * 76)
    if not TOKEN:
        print("  [FAIL] GITHUB_TOKEN missing — run setup_environment.py first")
        sys.exit(1)

    # Load both PR tables (agent + human) downloaded by script 01
    frames = []
    for name in ["pull_request", "human_pull_request"]:
        p = config.DATA_RAW / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            frames.append(df)
            print(f"  loaded {name:<22s} {len(df):>8,} PRs")
        else:
            print(f"  [WARN] missing {p} — run 01_download_aidev.py")
    if not frames:
        sys.exit(1)
    prs = pd.concat(frames, ignore_index=True)

    if args.merged_only:
        prs = prs[prs["merged_at"].notna()]
        print(f"  filter merged-only        {len(prs):>8,} PRs remain")

    # Resume: skip PRs already in the output CSV
    done_ids = set()
    if config.FILE_PR_ENRICHED.exists():
        prev = pd.read_csv(config.FILE_PR_ENRICHED)
        done_ids = set(prev["pr_id"].astype(str))
        print(f"  resume: {len(done_ids):,} PRs already enriched — skipping")
    todo = prs[~prs["id"].astype(str).isin(done_ids)]
    if args.max_prs:
        todo = todo.head(args.max_prs)
    print(f"  to do : {len(todo):,} PRs this run "
          f"(~{len(todo)*2.2/5000:.1f} h of API budget)")

    results, t0 = [], time.time()
    for i, (_, row) in enumerate(tqdm(todo.iterrows(), total=len(todo),
                                      desc="  enriching", unit="PR")):
        results.append(enrich_pr(row))
        if (i + 1) % CHECKPOINT_EVERY == 0:
            _flush(results)
            results = []
    _flush(results)

    # Final report
    full = pd.read_csv(config.FILE_PR_ENRICHED)
    ok = (full["enrich_status"] == "ok").sum()
    report_lines = [
        "=" * 76,
        " GITHUB ENRICHMENT REPORT",
        f" Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 76,
        f" PRs enriched total : {len(full):,}",
        f"   status ok        : {ok:,}",
        f"   pr_gone (404)    : {(full['enrich_status']=='pr_gone').sum():,}",
        f"   bad_url          : {(full['enrich_status']=='bad_url').sum():,}",
        f" merged w/ follow-up: {full['post_merge_commits_90d'].notna().sum():,}",
        f" reverts detected   : {int(full['revert_detected'].fillna(0).sum()):,}",
        f" elapsed this run   : {(time.time()-t0)/60:.1f} min",
        "=" * 76,
    ]
    report = config.DATA_INTERIM / "enrichment_report.txt"
    report.write_text("\n".join(report_lines))
    full.to_json(config.FILE_PR_ENRICHED.with_suffix(".json"),
                 orient="records", indent=2)

    print("\n".join(report_lines))
    print(f"  CSV   : {config.FILE_PR_ENRICHED}")
    print(f"  JSON  : {config.FILE_PR_ENRICHED.with_suffix('.json')}")
    print(f"  Report: {report}")
    print("  Next  : python data_collection/03_collect_human_baseline.py")


def _flush(results: list):
    """Append a batch of results to the output CSV (checkpointing)."""
    if not results:
        return
    df = pd.DataFrame(results)
    header = not config.FILE_PR_ENRICHED.exists()
    df.to_csv(config.FILE_PR_ENRICHED, mode="a", header=header, index=False)


if __name__ == "__main__":
    main()
