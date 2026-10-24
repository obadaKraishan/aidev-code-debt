"""
============================================================================
 02_run_static_analysis.py, Patch-Level Static Analysis from the API Cache
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : Analyze the ADDED LINES of every PR's diff for maintainability
           and security signals. Patches are read from the enrichment
           cache (data/.api_cache), where every /pulls/N/files response
           already contains per-file patch text -> ZERO extra API calls,
           works identically for agent and human PRs.
 Metrics : per PR, added lines analyzed, code-smell counts (TODO/FIXME,
           long lines, deep nesting, branch density, comment ratio) and
           security-smell counts mapped to CWE classes (regex catalog).
 Input   : data/interim/merged.csv + data/.api_cache/*.json
 Output  : data/interim/static_metrics.csv (+ .json + static_report.txt)
 Usage   : python data_processing/02_run_static_analysis.py
============================================================================
 Method note (for the paper): pattern-based smell detection on added diff
 lines is standard in MSR studies when full-project builds are infeasible
 at corpus scale; we report the catalog and its limitations explicitly.
============================================================================
"""

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config                                    # noqa: E402
from utils.io_helpers import save_csv, save_json, save_txt  # noqa: E402

# ---------------------------------------------------------------------------
# Security-smell catalog: CWE class -> compiled regex (checked on ADDED lines)
# ---------------------------------------------------------------------------
CWE_CATALOG = {
    "CWE-798_hardcoded_secret": re.compile(
        r"(?i)(password|passwd|secret|api[_-]?key|auth[_-]?token)\s*[:=]\s*[\"'][^\"']{4,}[\"']"),
    "CWE-89_sql_injection": re.compile(
        r"(?i)(execute|executemany|query)\s*\(\s*(f[\"']|[\"'].*[\"']\s*(\+|%)|.*\.format\()"),
    "CWE-78_cmd_injection": re.compile(
        r"(?i)(os\.system\s*\(|shell\s*=\s*True|child_process\.exec\s*\()"),
    "CWE-95_eval_injection": re.compile(
        r"(?<![\w.])(eval|exec)\s*\("),
    "CWE-502_unsafe_deserialization": re.compile(
        r"(pickle\.loads?\s*\(|yaml\.load\s*\((?!.*SafeLoader))"),
    "CWE-327_weak_crypto": re.compile(
        r"(?i)(hashlib\.(md5|sha1)\s*\(|createHash\s*\(\s*[\"'](md5|sha1)[\"']|\bDES\b|MODE_ECB)"),
    "CWE-79_xss": re.compile(
        r"(\.innerHTML\s*=|dangerouslySetInnerHTML|document\.write\s*\()"),
    "CWE-22_path_traversal": re.compile(
        r"(?i)(open|readFile|readFileSync|fopen)\s*\([^)]*(\+\s*\w|%s|\$\{|\.format\()"),
}
TODO_RE = re.compile(r"(?i)\b(TODO|FIXME|HACK|XXX)\b")
BRANCH_RE = re.compile(r"(?<![\w])(if|for|while|case|catch|except|elif)(?![\w])")
COMMENT_RE = re.compile(r"^\s*(#|//|/\*|\*)")
LONG_LINE = 120


def cache_key(url: str, params: dict) -> Path:
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    return config.CACHE_DIR / f"{key}.json"


def load_cached_files(repo: str, number) -> list | None:
    """Recreate the exact cache key 02_enrich_github used and read it."""
    url = f"{config.GITHUB_API_BASE}/repos/{repo}/pulls/{int(number)}/files"
    p = cache_key(url, {"per_page": config.GITHUB_PER_PAGE})
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def analyze_patch_lines(added: list[str]) -> dict:
    """All metrics over one PR's added lines."""
    n = len(added)
    out = {"added_lines_analyzed": n}
    if n == 0:
        return out
    out["todo_count"] = sum(bool(TODO_RE.search(l)) for l in added)
    out["long_lines"] = sum(len(l) > LONG_LINE for l in added)
    out["comment_lines"] = sum(bool(COMMENT_RE.match(l)) for l in added)
    out["comment_ratio"] = out["comment_lines"] / n
    out["branch_count"] = sum(len(BRANCH_RE.findall(l)) for l in added)
    out["branch_density"] = out["branch_count"] / n
    indents = [(len(l) - len(l.lstrip(" "))) // 4 for l in added if l.strip()]
    out["max_nesting"] = max(indents) if indents else 0
    for cwe, rx in CWE_CATALOG.items():
        out[cwe] = sum(bool(rx.search(l)) for l in added)
    out["security_smells_total"] = sum(out[c] for c in CWE_CATALOG)
    out["maintainability_smells_total"] = out["todo_count"] + out["long_lines"]
    return out


def main():
    print("=" * 76)
    print(" STEP 2/3 (PROCESSING) — PATCH-LEVEL STATIC ANALYSIS (from cache)")
    print("=" * 76)
    if not config.FILE_MERGED.exists():
        sys.exit("[FAIL] merged.csv missing — run 01_clean_and_merge.py first")
    df = pd.read_csv(config.FILE_MERGED, low_memory=False)
    print(f"  PRs in merged table : {len(df):,}")
    print(f"  cache directory     : {config.CACHE_DIR} "
          f"({len(list(config.CACHE_DIR.glob('*.json'))):,} cached responses)")

    rows, no_cache, no_patch = [], 0, 0
    lang_exts = set(config.LANGUAGE_EXTENSIONS)
    for _, pr in tqdm(df.iterrows(), total=len(df), desc="  analyzing", unit="PR"):
        files = load_cached_files(pr["repo"], pr["number"])
        rec = {"id": pr["id"], "static_status": "ok"}
        if files is None:
            rec["static_status"] = "no_cache"; no_cache += 1
            rows.append(rec); continue
        added, langs, n_patchless = [], set(), 0
        total_lines_seen = 0
        for f in files:
            ext = Path(f.get("filename", "")).suffix
            patch = f.get("patch")
            if patch is None:
                n_patchless += 1
                continue
            file_added = [l[1:] for l in patch.split("\n")
                          if l.startswith("+") and not l.startswith("+++")]
            total_lines_seen += len(file_added)
            if total_lines_seen > config.MAX_DIFF_LINES:
                rec["oversize_diff"] = 1
                break
            if ext in lang_exts:                      # analyze target langs only
                added.extend(file_added)
                langs.add(config.LANGUAGE_EXTENSIONS[ext])
        if not added:
            rec["static_status"] = "no_target_lang"; no_patch += 1
            rows.append(rec); continue
        rec.update(analyze_patch_lines(added))
        rec["languages_analyzed"] = ";".join(sorted(langs))
        rows.append(rec)

    sm = pd.DataFrame(rows)
    ok = sm[sm["static_status"] == "ok"]

    lines = ["=" * 76, " STATIC ANALYSIS REPORT",
             f" Generated: {datetime.now(timezone.utc).isoformat()}", "=" * 76,
             f" PRs processed          : {len(sm):,}",
             f"   analyzed ok          : {len(ok):,} "
             f"({len(ok)/max(len(sm),1):.1%})",
             f"   no cached patch      : {no_cache:,}",
             f"   no target-lang lines : {no_patch:,}",
             f" Added lines analyzed   : {int(ok['added_lines_analyzed'].sum()):,}",
             f" Security smells found  : {int(ok['security_smells_total'].sum()):,}",
             f" PRs w/ >=1 sec smell   : {(ok['security_smells_total']>0).sum():,} "
             f"({(ok['security_smells_total']>0).mean():.1%} of analyzed)",
             "-" * 76, " CWE class breakdown (total hits):"]
    for cwe in CWE_CATALOG:
        lines.append(f"   {cwe:<36s} {int(ok[cwe].sum()):>8,}")
    lines.append("=" * 76)

    save_csv(sm, config.FILE_STATIC_METRICS, "static metrics")
    save_json({"n": len(sm), "analyzed_ok": int(len(ok)),
               "cwe_totals": {c: int(ok[c].sum()) for c in CWE_CATALOG}},
              config.DATA_INTERIM / "static_summary.json", "static summary")
    save_txt("\n".join(lines), config.DATA_INTERIM / "static_report.txt",
             "static report")
    print("\n".join(lines))
    print("  Next: python data_processing/03_feature_engineering.py")


if __name__ == "__main__":
    main()
