"""
============================================================================
 run_all.py, One-Command Pipeline Orchestrator
============================================================================
 Runs the full study in order, stopping on the first failure:
   collection -> processing -> validation -> codebook -> all 5 RQ analyses
 Usage:
   python run_all.py                  # everything (enrichment ~10 h once)
   python run_all.py --analysis-only  # skip collection+processing
   python run_all.py --skip-enrich    # skip only the long GitHub step
============================================================================
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

COLLECTION = [
    ["python", "data_collection/01_download_aidev.py"],
    ["python", "data_collection/03_collect_human_baseline.py"],
    ["python", "data_collection/02_enrich_github.py", "--merged-only"],
]
PROCESSING = [
    ["python", "data_processing/01_clean_and_merge.py"],
    ["python", "data_processing/02_run_static_analysis.py"],
    ["python", "data_processing/03_feature_engineering.py"],
]
GATE = [
    ["python", "data_validation/validate_dataset.py"],
    ["python", "codebook/generate_codebook.py"],
]
ANALYSES = [
    ["python", "analysis/rq1_defect_density/rq1_analysis.py"],
    ["python", "analysis/rq2_maintainability/rq2_analysis.py"],
    ["python", "analysis/rq3_maintenance_overhead/rq3_analysis.py"],
    ["python", "analysis/rq4_cwe_overrepresentation/rq4_analysis.py"],
    ["python", "analysis/rq5_review_dynamics/rq5_analysis.py"],
]


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 76)
    print(f" RUN: {' '.join(cmd)}")
    print("=" * 76, flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=ROOT)
    dt = (time.time() - t0) / 60
    if rc != 0:
        print(f"\n[ABORT] step failed (exit {rc}) after {dt:.1f} min: "
              f"{' '.join(cmd)}")
        sys.exit(rc)
    print(f"[ OK ] finished in {dt:.1f} min")


def main():
    ap = argparse.ArgumentParser(description="Run the full study pipeline")
    ap.add_argument("--analysis-only", action="store_true",
                    help="only validation, codebook, and the 5 analyses")
    ap.add_argument("--skip-enrich", action="store_true",
                    help="run everything except the long GitHub enrichment")
    args = ap.parse_args()

    t0 = time.time()
    if not args.analysis_only:
        steps = COLLECTION + PROCESSING
        if args.skip_enrich:
            steps = [s for s in steps if "02_enrich_github" not in s[1]]
        for cmd in steps:
            run(cmd)
    for cmd in GATE + ANALYSES:
        run(cmd)
    print("\n" + "=" * 76)
    print(f" PIPELINE COMPLETE in {(time.time()-t0)/60:.1f} min")
    print(" Results: analysis/rq*/outputs/  |  figures: analysis/rq*/figures/")
    print("=" * 76)


if __name__ == "__main__":
    main()
