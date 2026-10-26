"""
============================================================================
 setup_environment.py, One-Time Environment Health Check & Bootstrap
============================================================================
 Project : Not All Agents Are Equal (IEEE BigData 2026)
 Purpose : Verify EVERYTHING the pipeline needs before you run any script:
             1. Python version
             2. Required Python packages (with versions)
             3. Secrets loaded from .env (GITHUB_TOKEN, optional HF_TOKEN)
             4. GitHub API reachability + remaining rate limit
             5. Project folder tree (creates anything missing)
 Usage   : python setup/setup_environment.py
 Output  : Console health report + setup/environment_report.txt
============================================================================
"""

import importlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make 'setup.config' importable when run from the project root OR from setup/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from setup import config  # noqa: E402

# ---------------------------------------------------------------------------
# What we require
# ---------------------------------------------------------------------------
MIN_PYTHON = (3, 10)

# (import name, pip name, why it's needed)
REQUIRED_PACKAGES = [
    ("pandas",        "pandas",         "dataframes everywhere"),
    ("numpy",         "numpy",          "numeric computing"),
    ("scipy",         "scipy",          "statistical tests"),
    ("statsmodels",   "statsmodels",    "BH correction, regression"),
    ("matplotlib",    "matplotlib",     "figures"),
    ("seaborn",       "seaborn",        "publication plots"),
    ("requests",      "requests",       "GitHub REST API"),
    ("dotenv",        "python-dotenv",  ".env secret loading"),
    ("datasets",      "datasets",       "Hugging Face download"),
    ("pyarrow",       "pyarrow",        "parquet read/write"),
    ("tqdm",          "tqdm",           "progress bars"),
]

REPORT_PATH = Path(__file__).resolve().parent / "environment_report.txt"

PASS, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"
lines = []          # everything printed is also saved to the report
failures = []       # blocking problems
warnings_ = []      # non-blocking problems


def log(msg=""):
    print(msg)
    lines.append(msg)


def section(title):
    log()
    log("-" * 76)
    log(f" {title}")
    log("-" * 76)


# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------
def check_python():
    section("1/5  Python version")
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    tag = PASS if ok else FAIL
    log(f"  {tag} Python {v.major}.{v.minor}.{v.micro} "
        f"(need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    if not ok:
        failures.append("Python version too old")


# ---------------------------------------------------------------------------
# 2. Packages
# ---------------------------------------------------------------------------
def check_packages():
    section("2/5  Required Python packages")
    for import_name, pip_name, why in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            log(f"  {PASS} {pip_name:<15s} v{ver:<10s} ({why})")
        except ImportError:
            log(f"  {FAIL} {pip_name:<15s} MISSING     ({why})")
            failures.append(f"pip install {pip_name}")


# ---------------------------------------------------------------------------
# 3. Secrets from .env
# ---------------------------------------------------------------------------
def check_secrets():
    section("3/5  Secrets (.env)")
    env_path = config.PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv
        loaded = load_dotenv(env_path)
        log(f"  {PASS if loaded else WARN} .env file "
            f"{'loaded from <project root>/.env' if loaded else 'not found at <project root>/.env'}")
        if not loaded:
            warnings_.append("Copy .env.example to .env and fill in tokens")
    except ImportError:
        log(f"  {FAIL} python-dotenv not installed — cannot load .env")
        failures.append("pip install python-dotenv")
        return

    gh = os.getenv("GITHUB_TOKEN", "")
    if gh:
        log(f"  {PASS} GITHUB_TOKEN present ({len(gh)} chars)")
    else:
        log(f"  {FAIL} GITHUB_TOKEN missing — enrichment (5,000 req/h) needs it")
        failures.append("Set GITHUB_TOKEN in .env")

    hf = os.getenv("HF_TOKEN", "")
    if hf:
        log(f"  {PASS} HF_TOKEN present")
    else:
        log(f"  {WARN} HF_TOKEN missing — fine if the dataset is public")
        warnings_.append("HF_TOKEN empty (only needed for gated datasets)")


# ---------------------------------------------------------------------------
# 4. GitHub API reachability + rate limit
# ---------------------------------------------------------------------------
def check_github_api():
    section("4/5  GitHub API connectivity")
    try:
        import requests
    except ImportError:
        log(f"  {FAIL} requests not installed — skipping API check")
        return
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(f"{config.GITHUB_API_BASE}/rate_limit",
                         headers=headers, timeout=config.GITHUB_TIMEOUT_S)
        core = r.json()["resources"]["core"]
        reset = datetime.fromtimestamp(core["reset"], tz=timezone.utc)
        auth = "authenticated" if token else "UNAUTHENTICATED (60 req/h!)"
        tag = PASS if core["limit"] >= 5000 else WARN
        log(f"  {tag} API reachable — {auth}")
        log(f"         limit={core['limit']}/h  remaining={core['remaining']}  "
            f"resets={reset:%H:%M UTC}")
        if core["limit"] < 5000:
            warnings_.append("Unauthenticated GitHub limit; enrichment will crawl")
    except Exception as e:
        log(f"  {WARN} Could not reach GitHub API: {e}")
        warnings_.append("GitHub API unreachable (offline?) — retry before collection")


# ---------------------------------------------------------------------------
# 5. Folder tree
# ---------------------------------------------------------------------------
def check_folders():
    section("5/5  Project folder tree")
    created = 0
    for d in config.ALL_DIRS:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created += 1
            log(f"  {PASS} created  {d.relative_to(config.PROJECT_ROOT)}")
        else:
            log(f"  {PASS} exists   {d.relative_to(config.PROJECT_ROOT)}")
    # RQ output/figure folders from the registry
    for key in config.RESEARCH_QUESTIONS:
        for d in (config.rq_output_dir(key), config.rq_figure_dir(key)):
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                created += 1
                log(f"  {PASS} created  {d.relative_to(config.PROJECT_ROOT)}")
    log(f"  -> {created} folder(s) created, rest already in place")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 76)
    log(" ENVIRONMENT SETUP & HEALTH CHECK — AIDev Code-Debt Project")
    log(f" Run at: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}")
    log("=" * 76)

    check_python()
    check_packages()
    check_secrets()
    check_github_api()
    check_folders()

    section("SUMMARY")
    if failures:
        log(f"  {FAIL} {len(failures)} blocking issue(s):")
        for f in failures:
            log(f"     - {f}")
    else:
        log(f"  {PASS} No blocking issues.")
    if warnings_:
        log(f"  {WARN} {len(warnings_)} warning(s):")
        for w in warnings_:
            log(f"     - {w}")

    verdict = "ENVIRONMENT READY — proceed to data_collection/01_download_aidev.py" \
        if not failures else "FIX THE FAILURES ABOVE, THEN RE-RUN THIS SCRIPT"
    log()
    log("=" * 76)
    log(f" {verdict}")
    log("=" * 76)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFull report saved to: {REPORT_PATH}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
