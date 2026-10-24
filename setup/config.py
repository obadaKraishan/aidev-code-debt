"""
============================================================================
 config.py, Single Source of Truth for the AIDev Code-Debt Project
============================================================================
 Project : Not All Agents Are Equal, A Large-Scale Empirical Study of
           Code Across Five Autonomous Coding Agents (IEEE BigData 2026)
 Purpose : Central constants imported by EVERY script in the pipeline.
           Nothing (paths, labels, thresholds, seeds) is hard-coded twice.
 Usage   : from setup.config import *          (from project root)
           python setup/config.py              (standalone: prints config)
============================================================================
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PROJECT PATHS (all derived from the repo root, portable on any machine)
# ---------------------------------------------------------------------------
# config.py lives in <root>/setup/, so the root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
CODEBOOK_DIR = PROJECT_ROOT / "codebook"
PAPER_DIR = PROJECT_ROOT / "paper"
PAPER_TABLES = PAPER_DIR / "tables"
PAPER_FIGURES = PAPER_DIR / "figures"

# Cache for GitHub API responses (lets 02_enrich_github.py resume cheaply)
CACHE_DIR = DATA_DIR / ".api_cache"

ALL_DIRS = [
    DATA_RAW, DATA_INTERIM, DATA_PROCESSED, CACHE_DIR,
    ANALYSIS_DIR, CODEBOOK_DIR, PAPER_TABLES, PAPER_FIGURES,
]

# ---------------------------------------------------------------------------
# 2. DATASET / SOURCE CONSTANTS
# ---------------------------------------------------------------------------
# Primary corpus: AIDev on Hugging Face (agentic PRs with per-agent labels)
HF_DATASET_ID = "hao-li/AIDev"

# The five coding agents labeled in AIDev. Canonical names used EVERYWHERE
# (plots, tables, reports) so group labels never drift between scripts.
AGENTS = [
    "OpenAI Codex",
    "Devin",
    "GitHub Copilot",
    "Cursor",
    "Claude Code",
]

# Label used for the matched human-authored baseline sample
HUMAN_LABEL = "Human"

# All author groups in canonical display order (human last = reference group)
AUTHOR_GROUPS = AGENTS + [HUMAN_LABEL]

# GitHub REST API
GITHUB_API_BASE = "https://api.github.com"
GITHUB_PER_PAGE = 100          # max page size for list endpoints
GITHUB_TIMEOUT_S = 30          # per-request timeout
GITHUB_MAX_RETRIES = 5         # retries on 5xx / rate-limit
GITHUB_SLEEP_ON_LIMIT_S = 60   # base sleep when rate-limited

# ---------------------------------------------------------------------------
# 3. SAMPLING / MATCHING RULES (human baseline)
# ---------------------------------------------------------------------------
# Human PRs are sampled from the SAME repositories and the SAME time window
# as the agent PRs, matched on PR size bucket, for a fair comparison.
MIN_REPO_STARS = 100           # focus on the curated 'popular' slice
HUMAN_PRS_PER_REPO = 5         # cap so no single repo dominates the baseline
PR_SIZE_BUCKETS = [            # (label, max changed lines), XS..XL like GitHub
    ("XS", 10),
    ("S", 50),
    ("M", 250),
    ("L", 1000),
    ("XL", float("inf")),
]
TIME_WINDOW_START = "2024-01-01"   # AIDev coverage window (collected to Aug 2025)
TIME_WINDOW_END = "2025-08-01"

# ---------------------------------------------------------------------------
# 4. STATIC ANALYSIS SETTINGS
# ---------------------------------------------------------------------------
# Start with the top languages by PR volume; language is a covariate in models.
TARGET_LANGUAGES = ["Python", "JavaScript", "TypeScript"]

# File extensions mapped to language (used when routing diffs to linters)
LANGUAGE_EXTENSIONS = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
}

# Maximum diff size to analyze (very large auto-generated diffs are skipped
# and flagged, not silently analyzed, they distort complexity metrics)
MAX_DIFF_LINES = 5000

# ---------------------------------------------------------------------------
# 5. STATISTICS SETTINGS (used by utils/stats.py and every RQ script)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42               # ONE seed for all sampling / bootstrap
ALPHA = 0.05                   # significance level
N_BOOTSTRAP = 5000             # bootstrap resamples for effect-size CIs
CI_LEVEL = 0.95                # confidence-interval level
P_ADJUST_METHOD = "fdr_bh"     # Benjamini-Hochberg across multiple comparisons

# Cliff's delta magnitude thresholds (Romano et al., 2006 conventions)
CLIFFS_DELTA_BANDS = [
    (0.147, "negligible"),
    (0.33, "small"),
    (0.474, "medium"),
    (float("inf"), "large"),
]

# ---------------------------------------------------------------------------
# 6. RESEARCH QUESTION REGISTRY (drives folders, filenames, report titles)
# ---------------------------------------------------------------------------
RESEARCH_QUESTIONS = {
    "rq1": {
        "dir": "rq1_defect_density",
        "title": "RQ1 — Defect / Issue Density: Agent vs Human PRs",
        "question": ("Do agent-authored PRs carry higher defect/issue density "
                     "than human PRs, and by how much?"),
    },
    "rq2": {
        "dir": "rq2_maintainability",
        "title": "RQ2 — Maintainability Across Agents",
        "question": ("Which agents produce the most maintainable code "
                     "(code smells, complexity, duplication)?"),
    },
    "rq3": {
        "dir": "rq3_maintenance_overhead",
        "title": "RQ3 — Post-Merge Maintenance Overhead",
        "question": ("What is the post-merge maintenance overhead of agent "
                     "code (churn, reverts, time-to-fix)?"),
    },
    "rq4": {
        "dir": "rq4_cwe_overrepresentation",
        "title": "RQ4 — CWE / Vulnerability Class Over-Representation",
        "question": ("Are specific CWE/vulnerability classes over-represented "
                     "in agent code?"),
    },
    "rq5": {
        "dir": "rq5_review_dynamics",
        "title": "RQ5 — Human Review Latency and Depth",
        "question": ("Does human review latency/depth differ for agent vs "
                     "human PRs?"),
    },
}

def rq_output_dir(rq_key: str) -> Path:
    """Return analysis/<rq_dir>/outputs for a given RQ key (e.g. 'rq1')."""
    return ANALYSIS_DIR / RESEARCH_QUESTIONS[rq_key]["dir"] / "outputs"

def rq_figure_dir(rq_key: str) -> Path:
    """Return analysis/<rq_dir>/figures for a given RQ key (e.g. 'rq1')."""
    return ANALYSIS_DIR / RESEARCH_QUESTIONS[rq_key]["dir"] / "figures"

# ---------------------------------------------------------------------------
# 7. KEY FILE LOCATIONS (the pipeline contract between scripts)
# ---------------------------------------------------------------------------
FILE_AIDEV_RAW = DATA_RAW / "aidev_prs.parquet"            # 01_download_aidev
FILE_AIDEV_MANIFEST = DATA_RAW / "aidev_manifest.json"     # 01_download_aidev
FILE_PR_ENRICHED = DATA_INTERIM / "pr_enriched.csv"        # 02_enrich_github
FILE_HUMAN_BASELINE = DATA_INTERIM / "human_baseline.csv"  # 03_collect_human
FILE_MERGED = DATA_INTERIM / "merged.csv"                  # 01_clean_and_merge
FILE_STATIC_METRICS = DATA_INTERIM / "static_metrics.csv"  # 02_run_static
FILE_MASTER = DATA_PROCESSED / "master.csv"                # 03_feature_eng
FILE_MASTER_JSON = DATA_PROCESSED / "master_summary.json"  # 03_feature_eng

# ---------------------------------------------------------------------------
# Standalone: print the resolved configuration so anyone can sanity-check it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 76)
    print(" AIDev Code-Debt Project — Resolved Configuration")
    print("=" * 76)
    print(f"  PROJECT_ROOT     : {PROJECT_ROOT}")
    print(f"  DATA (raw)       : {DATA_RAW}")
    print(f"  DATA (interim)   : {DATA_INTERIM}")
    print(f"  DATA (processed) : {DATA_PROCESSED}")
    print(f"  HF dataset       : {HF_DATASET_ID}")
    print(f"  Agents           : {', '.join(AGENTS)}")
    print(f"  Human label      : {HUMAN_LABEL}")
    print(f"  Languages        : {', '.join(TARGET_LANGUAGES)}")
    print(f"  Time window      : {TIME_WINDOW_START} .. {TIME_WINDOW_END}")
    print(f"  Random seed      : {RANDOM_SEED}")
    print(f"  Alpha / CI       : {ALPHA} / {CI_LEVEL}")
    print(f"  Bootstrap draws  : {N_BOOTSTRAP}")
    print(f"  p-adjust method  : {P_ADJUST_METHOD}")
    print("-" * 76)
    print("  Research questions:")
    for key, rq in RESEARCH_QUESTIONS.items():
        print(f"    {key.upper():4s} -> analysis/{rq['dir']}/")
    print("=" * 76)
    print("  Config OK — import with:  from setup.config import *")
