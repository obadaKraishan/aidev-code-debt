[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23627054.svg)](https://doi.org/10.5281/zenodo.23627054)

# Not All Agents Are Equal

### Code Quality and Post-Merge Maintenance Across Five Autonomous Coding Agents in the Wild

Replication package for the paper accepted at the **IEEE International
Conference on Big Data 2026**. It contains the pipeline that joins the AIDev
provenance labels with GitHub enrichment, runs patch-level static analysis
over the added lines of every pull request, follows each merged PR for 90
days, and produces every number, table, and figure in the paper.

**Corpus:** 37,623 provenance-labeled PRs (33,596 from OpenAI Codex, Devin,
GitHub Copilot, Cursor, and Claude Code; 4,027 matched human PRs) across
2,807 repositories, December 2024 to July 2025, built on the
[AIDev dataset](https://huggingface.co/datasets/hao-li/AIDev) and enriched
with 58,792 cached GitHub REST API responses.

## What is in this repository

| Folder               | Contents                                                                                                                                                                                                                                           |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `setup/`           | `config.py`, the single source of truth: agent labels, human-baseline matching rules, languages, diff-size limits, CWE smell catalog, statistical settings. `setup_environment.py` checks the environment before any collection.               |
| `data_collection/` | AIDev download, matched human baseline, GitHub enrichment (cached, resumable), and `04_recompute_followup.py`, which rebuilds the 90-day post-merge measures from each repository's git history (v1.1.0).                                         |
| `data_processing/` | Clean and merge, patch-level static analysis, feature engineering into `master.csv`.                                                                                                                                                             |
| `data_validation/` | Quality gate: schema, ranges, group coverage. Exits non-zero on failure.                                                                                                                                                                           |
| `codebook/`        | Generated variable dictionary for all 64 columns of `master.csv`.                                                                                                                                                                                |
| `analysis/rq*/`    | One self-contained folder per research question. Each writes `rqN_results.csv`, `rqN_results.json`, a statistical report with sample sizes, test statistics, raw and BH-adjusted p-values, effect sizes with bootstrap intervals, and figures. |
| `data/`            | `interim/` (enriched and merged PR tables, static metrics, human baseline) and `processed/` (`master.csv` and reports).                                                                                                                      |
| `utils/`           | I/O helpers, the statistics engine (Mann-Whitney with Cliff's delta, Kruskal-Wallis with epsilon-squared and Dunn, chi-square and Fisher with odds ratios, bootstrap), and the report builder.                                                     |

| Paper RQ                                | Folder                                   |
| --------------------------------------- | ---------------------------------------- |
| RQ1 Defect-prone patterns in added code | `analysis/rq1_defect_density/`         |
| RQ2 Maintainability across agents       | `analysis/rq2_maintainability/`        |
| RQ3 Post-merge maintenance overhead     | `analysis/rq3_maintenance_overhead/`   |
| RQ4 Vulnerability classes               | `analysis/rq4_cwe_overrepresentation/` |
| RQ5 Human review behavior               | `analysis/rq5_review_dynamics/`        |
| Size-adjusted and within-repository models, concentration checks, Fig. 3 | `analysis/robustness/` |

## Reproduce the results from the shipped tables

No credentials are needed for this path. It re-runs validation, the codebook,
and all five analyses on `data/processed/master.csv`.

```bash
git clone https://github.com/obadaKraishan/aidev-code-debt.git && cd aidev-code-debt
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --analysis-only
```

Bootstrap intervals (5,000 resamples, seed 42) make this take a few minutes;
the output is deterministic.

## Rebuild from AIDev and GitHub

The collection stage downloads the AIDev tables from Hugging Face and needs a
GitHub token with public-repo scope for enrichment, which takes roughly ten
hours once and is resumable. The static-analysis stage reads patches from the
enrichment cache, so it needs that cache to exist.

```bash
cp .env.example .env            # add GITHUB_TOKEN (and HF_TOKEN if required)
python setup/setup_environment.py
python run_all.py               # or --skip-enrich to reuse an existing cache
```

## Version 1.1.0 (November 2026)

- **Post-merge measures corrected.** In v1.0.0, `02_enrich_github.py` compared commit timestamps as text
  (`...Z` sorts after `...+00:00` at the same second), so each PR's own squash or merge commit was counted as a
  follow-up commit, and the revert flag fired on any message containing the word "revert". The new
  `data_collection/04_recompute_followup.py` rebuilds the measures from git history for every merged PR, excludes the
  PR's own commits, and reports three revert definitions (any "revert" word; a revert commit; a revert of this PR).
  Run with the old rules, it reproduces the v1.0.0 values, which validates the method. Output:
  `data/interim/followup_recomputed.csv`. `02_enrich_github.py` is fixed for future runs.
- **RQ3 uses the corrected measures** when `followup_recomputed.csv` is present (the camera-ready numbers).
- **`analysis/robustness/`** adds the size-adjusted, repository-clustered logistic models quoted in the paper (RQ1,
  RQ3, RQ4), within-repository comparisons (conditional logit and Mantel-Haenszel), concentration checks, and the
  script for Fig. 3.
- Notes on shipped reports: `data/interim/enrichment_report.txt` describes every PR submitted to enrichment before the
  human baseline was matched (a superset of the paper's corpus); the paper's counts come from `master.csv`.
  AIDev's human PR sample comes from repositories with more than 500 stars.
- Removed a stale `data/raw/` download report left by a failed test run; `01_download_aidev.py` writes a fresh one.

## Citation

See `CITATION.cff`. Please also cite the AIDev dataset (Li, Zhang, and
Hassan, MSR 2026).

## License

Code: MIT. Data tables under `data/`: CC BY 4.0. See `LICENSE`.
